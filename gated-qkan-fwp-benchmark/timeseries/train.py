"""Fast gradient-training harness for the time-series benchmark (paper Tables 3-6).

One "cell" is (dataset, window, seed).  A cell is trained with mini-batch gradient
descent on pre-tensorised data (no DataLoader), and after EVERY epoch the test MSE
is computed exactly as the paper's protocol defines it (``protocol.test_mse``).
The reported metric is the FINAL-epoch test MSE; the curve is kept for diagnosis
only and is never used for model selection.

Sub-commands
    run         train every (dataset, window, seed) cell of a sweep, then aggregate
    aggregate   print mean +- std per cell next to the paper's best, mark win / lose
    models      list the registered models

Examples
    python train.py run --tag gqkan_repro --model gqkan_qkanfwp \\
        --datasets bessel_j2 narma_5 --windows 16 --seeds 0 1 2 3 4
    python train.py aggregate --tag gqkan_repro --paper-model GQKAN-QKANFWP

Results: ``runs/timeseries/<tag>/cells/<dataset>_w<window>_s<seed>.json`` plus
``summary.json`` / ``summary.md`` written by the aggregator.

Defaults are the paper's recipe: 50 epochs, batch 4, Adam, lr 1e-3, no schedule.
With ``--shuffle reference`` (default) the mini-batch order is the one the authors'
DataLoader produces for the same seed, so a re-implemented reference model follows
the authors' training trajectory (see ``test_train.py``).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import models
import protocol

torch.set_num_threads(1)

# Epochs at which the authors' trainer iterates one extra DataLoader (prediction dump).
_REFERENCE_PREDICTION_EPOCHS = {1, 15, 30, 50, 100}


@dataclass
class TrainConfig:
    """Everything that defines how a cell is trained (stored with every result)."""

    model: str = "gqkan_qkanfwp"
    model_kwargs: dict = field(default_factory=dict)
    epochs: int = protocol.PAPER_EPOCHS
    batch_size: int = protocol.PAPER_BATCH_SIZE  # 0 = full batch
    lr: float = protocol.PAPER_LR
    optimizer: str = "adam"  # adam | adamw | sgd | rmsprop | lbfgs
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    adam_eps: float = 1e-8  # adam / adamw epsilon
    momentum: float = 0.0  # sgd / rmsprop
    scheduler: str = "none"  # none | cosine | linear | exponential
    warmup_epochs: float = 0.0  # linear warm-up from 0, in epochs (may be fractional)
    final_lr_ratio: float = 0.0  # lr at the end of the schedule, relative to --lr
    grad_clip: float = 0.0  # max gradient norm, 0 = off
    shuffle: str = "reference"  # reference | simple | none

    def follows_paper_recipe(self) -> bool:
        """True if the optimisation recipe is exactly the paper's (Section 6.1)."""
        return (self.epochs == protocol.PAPER_EPOCHS and self.batch_size == protocol.PAPER_BATCH_SIZE
                and self.lr == protocol.PAPER_LR and self.optimizer == "adam" and self.scheduler == "none"
                and self.weight_decay == 0.0 and self.warmup_epochs == 0.0 and self.grad_clip == 0.0
                and (self.beta1, self.beta2, self.adam_eps) == (0.9, 0.999, 1e-8))


# ------------------------------------------------------------------ mini-batch order
def _draw_int64(generator=None) -> int:
    """One 64-bit draw, as DataLoader / RandomSampler do (global RNG when ``generator`` is None)."""
    return int(torch.empty((), dtype=torch.int64).random_(generator=generator).item())


class BatchOrder:
    """Yields the training permutation of every epoch.

    ``reference``  replays the RNG usage of the authors' trainer: per epoch the train
                   DataLoader draws an iterator seed and its RandomSampler a permutation
                   seed; the test DataLoader (and, at epochs 1/15/30/50/100, the
                   prediction-dump DataLoader) draw one iterator seed each.  The draws
                   come from torch's global RNG, or from ``reference_stream`` when given
                   (a private generator that continues a seed's global stream, used when
                   several seeds are trained at once).
    ``simple``     ``torch.randperm`` from a private generator seeded with the run seed.
    ``none``       no shuffling.
    """

    def __init__(self, mode: str, n: int, seed: int, reference_stream: torch.Generator | None = None):
        if mode not in ("reference", "simple", "none"):
            raise ValueError(f"unknown shuffle mode {mode!r}")
        self.mode, self.n = mode, n
        self.generator = torch.Generator().manual_seed(seed)
        self.reference_stream = reference_stream

    def train_permutation(self) -> torch.Tensor:
        if self.mode == "none":
            return torch.arange(self.n)
        if self.mode == "simple":
            return torch.randperm(self.n, generator=self.generator)
        _draw_int64(self.reference_stream)  # train DataLoader iterator seed
        sampler_seed = _draw_int64(self.reference_stream)  # RandomSampler seed
        return torch.randperm(self.n, generator=torch.Generator().manual_seed(sampler_seed))

    def after_epoch(self, epoch: int) -> None:
        if self.mode == "reference":
            _draw_int64(self.reference_stream)  # test DataLoader iterator seed
            if epoch in _REFERENCE_PREDICTION_EPOCHS:
                _draw_int64(self.reference_stream)  # prediction-dump DataLoader iterator seed


# ------------------------------------------------------------ optimiser and schedule
def make_optimizer(config: TrainConfig, parameters):
    name = config.optimizer
    if name == "adam":
        return torch.optim.Adam(parameters, lr=config.lr, betas=(config.beta1, config.beta2),
                                eps=config.adam_eps, weight_decay=config.weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=config.lr, betas=(config.beta1, config.beta2),
                                 eps=config.adam_eps, weight_decay=config.weight_decay)
    if name == "sgd":
        return torch.optim.SGD(parameters, lr=config.lr, momentum=config.momentum,
                               weight_decay=config.weight_decay)
    if name == "rmsprop":
        return torch.optim.RMSprop(parameters, lr=config.lr, momentum=config.momentum,
                                   weight_decay=config.weight_decay)
    if name == "lbfgs":
        return torch.optim.LBFGS(parameters, lr=config.lr, max_iter=20, history_size=50,
                                 line_search_fn="strong_wolfe")
    raise ValueError(f"unknown optimizer {name!r}")


def lr_factor(config: TrainConfig, step: int, total_steps: int, steps_per_epoch: int) -> float:
    """Learning-rate multiplier for optimisation step ``step`` (0-based)."""
    warmup_steps = int(round(config.warmup_epochs * steps_per_epoch))
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    if config.scheduler == "none":
        return 1.0
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps - 1)
    final = config.final_lr_ratio
    if config.scheduler == "cosine":
        return final + (1.0 - final) * 0.5 * (1.0 + math.cos(math.pi * progress))
    if config.scheduler == "linear":
        return 1.0 - (1.0 - final) * progress
    if config.scheduler == "exponential":
        if final <= 0.0:
            raise ValueError("the exponential schedule needs --final_lr_ratio > 0")
        return final ** progress
    raise ValueError(f"unknown scheduler {config.scheduler!r}")


# ------------------------------------------------------------------------ one cell
def count_parameters(model) -> tuple[int, int]:
    """(requires_grad, effective).  ``effective`` counts parameters that actually received
    a gradient, which is what the authors log and what Table 2 reports."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    effective = sum(p.numel() for p in model.parameters() if p.grad is not None)
    return trainable, effective


@torch.no_grad()
def predict(model, x: torch.Tensor, batch_size: int = 0) -> np.ndarray:
    """Model predictions in eval mode (``batch_size`` 0 = one batch)."""
    model.eval()
    if batch_size <= 0:
        return model(x).reshape(-1).numpy()
    chunks = [model(x[i:i + batch_size]).reshape(-1) for i in range(0, len(x), batch_size)]
    return torch.cat(chunks).numpy()


def train_cell(config: TrainConfig, dataset: str, window: int, seed: int, verbose: bool = False) -> dict:
    """Train one (dataset, window, seed) cell and return its result record."""
    x_train, y_train, x_test, y_test = protocol.load_torch(dataset, window)
    y_test_np = y_test.numpy()
    n_train = len(y_train)

    # Same seeding as the authors' train.py, then model construction.
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = models.build_model(config.model, window, **config.model_kwargs)
    if hasattr(model, "prepare"):
        model.prepare(x_train, y_train)  # training split only
    optimizer = make_optimizer(config, [p for p in model.parameters() if p.requires_grad])

    batch_size = config.batch_size if config.batch_size > 0 else n_train
    steps_per_epoch = math.ceil(n_train / batch_size)
    total_steps = steps_per_epoch * config.epochs
    scheduled = config.scheduler != "none" or config.warmup_epochs > 0
    order = BatchOrder(config.shuffle, n_train, seed)
    parameters = [p for p in model.parameters() if p.requires_grad]

    curve = {"epoch": [], "train_loss": [], "test_mse": []}
    params_trainable = params_effective = None
    step = 0
    start = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train()
        permutation = order.train_permutation()
        running = 0.0
        for first in range(0, n_train, batch_size):
            index = permutation[first:first + batch_size]
            x_batch, y_batch = x_train[index], y_train[index]
            if scheduled:
                factor = lr_factor(config, step, total_steps, steps_per_epoch)
                for group in optimizer.param_groups:
                    group["lr"] = config.lr * factor

            def closure():
                optimizer.zero_grad()
                prediction = model(x_batch)
                if prediction.shape != y_batch.shape:
                    raise ValueError(f"model must return shape (B,), got {tuple(prediction.shape)}")
                loss = F.mse_loss(prediction, y_batch)
                loss.backward()
                if config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(parameters, config.grad_clip)
                return loss

            loss = optimizer.step(closure)
            running += loss.item() * len(index)  # as the authors: batch loss weighted by batch size
            step += 1
            if params_effective is None:
                params_trainable, params_effective = count_parameters(model)
        train_loss = running / n_train
        test = protocol.test_mse(predict(model, x_test), y_test_np)  # the paper's metric
        order.after_epoch(epoch)
        curve["epoch"].append(epoch)
        curve["train_loss"].append(train_loss)
        curve["test_mse"].append(test)
        if verbose:
            print(f"    epoch {epoch:3d}  train {train_loss:.3e}  test {test:.3e}", flush=True)
        if not math.isfinite(train_loss):
            break  # diverged: the remaining epochs cannot recover a finite metric
    seconds = time.perf_counter() - start
    return _result_record(config, dataset, window, seed, "loop", n_train, len(y_test_np), params_trainable,
                          params_effective, curve, seconds, step / seconds)


def _result_record(config, dataset, window, seed, engine, n_train, n_test, params_trainable, params_effective,
                   curve, seconds, steps_per_second) -> dict:
    finished = len(curve["epoch"]) == config.epochs
    return {
        "dataset": dataset, "window": window, "seed": seed, "config": asdict(config), "engine": engine,
        "follows_paper_recipe": config.follows_paper_recipe(),
        "is_paper_cell": protocol.is_paper_cell(dataset, window),
        "n_train": n_train, "n_test": n_test,
        "params_trainable": params_trainable, "params_effective": params_effective,
        "final_test_mse": curve["test_mse"][-1] if finished else float("nan"),
        "final_train_loss": curve["train_loss"][-1],
        "min_test_mse_any_epoch_DIAGNOSTIC_ONLY": float(np.nanmin(curve["test_mse"])),
        "curve": curve, "seconds": seconds, "steps_per_second": steps_per_second,
    }


def train_cells_vmap(config: TrainConfig, dataset: str, window: int, seeds) -> list[dict]:
    """Train several seeds of one (dataset, window) cell at once with ``torch.func.vmap``.

    Mathematically the same as calling ``train_cell`` per seed: every seed keeps its own
    initial weights and its own mini-batch order, and the optimiser acts element-wise on the
    seed-stacked parameters, so the seeds never interact.  It is faster because the python /
    autograd overhead of a step is shared by all seeds.  Requirements: the model's forward
    must be vmap-compatible (no data-dependent python control flow, no BatchNorm) and must
    not draw random numbers; the optimiser must be element-wise (not lbfgs).
    """
    from torch.func import functional_call, grad_and_value, vmap

    if config.optimizer == "lbfgs":
        raise ValueError("the vmap engine does not support lbfgs; use the loop engine")
    x_train, y_train, x_test, y_test = protocol.load_torch(dataset, window)
    y_test_np = y_test.numpy()
    n_train, n_seeds = len(y_train), len(seeds)

    nets, orders = [], []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        net = models.build_model(config.model, window, **config.model_kwargs)
        if hasattr(net, "prepare"):
            net.prepare(x_train, y_train)  # training split only
        nets.append(net)
        stream = torch.Generator()
        stream.set_state(torch.get_rng_state())  # continue this seed's global RNG stream privately
        orders.append(BatchOrder(config.shuffle, n_train, seed, reference_stream=stream))
    base = nets[0]
    # parameter counts from one ordinary backward pass (grad of unused parameters stays None)
    F.mse_loss(base(x_train[:4]), y_train[:4]).backward()
    params_trainable, params_effective = count_parameters(base)
    base.zero_grad()

    def stack(name_to_tensor_per_net, trainable):
        return {name: torch.stack([tensors[name].detach() for tensors in name_to_tensor_per_net]).requires_grad_(trainable)
                for name in name_to_tensor_per_net[0]}

    trainable = stack([{n: p for n, p in net.named_parameters() if p.requires_grad} for net in nets], True)
    # Non-trainable tensors: those equal for all seeds (e.g. constants fitted by prepare()) are
    # shared as they are; only the ones that differ between seeds get a seed dimension.
    constants = [{**{n: p for n, p in net.named_parameters() if not p.requires_grad}, **dict(net.named_buffers())}
                 for net in nets]
    shared = {n: t.detach() for n, t in constants[0].items() if all(torch.equal(t, c[n]) for c in constants)}
    frozen = stack([{n: t for n, t in c.items() if n not in shared} for c in constants], False)

    def forward_one_seed(weights, fixed, x):
        return functional_call(base, {**weights, **fixed, **shared}, (x,))

    def loss_of_one_seed(weights, fixed, x, y):
        prediction = forward_one_seed(weights, fixed, x)
        if prediction.shape != y.shape:
            raise ValueError(f"model must return shape (B,), got {tuple(prediction.shape)}")
        return F.mse_loss(prediction, y)

    grads_and_losses = vmap(grad_and_value(loss_of_one_seed))
    predict_all_seeds = vmap(forward_one_seed, in_dims=(0, 0, None))
    optimizer = make_optimizer(config, list(trainable.values()))

    batch_size = config.batch_size if config.batch_size > 0 else n_train
    steps_per_epoch = math.ceil(n_train / batch_size)
    total_steps = steps_per_epoch * config.epochs
    scheduled = config.scheduler != "none" or config.warmup_epochs > 0
    curves = [{"epoch": [], "train_loss": [], "test_mse": []} for _ in seeds]
    alive = [True] * n_seeds  # a diverged seed stops being recorded, like the loop engine
    step = 0
    start = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        base.train()
        permutations = torch.stack([order.train_permutation() for order in orders])  # (seeds, n_train)
        running = torch.zeros(n_seeds, dtype=torch.float64)
        for first in range(0, n_train, batch_size):
            index = permutations[:, first:first + batch_size]
            if scheduled:
                factor = lr_factor(config, step, total_steps, steps_per_epoch)
                for group in optimizer.param_groups:
                    group["lr"] = config.lr * factor
            with torch.no_grad():  # the inner grad transform is unaffected by the outer no_grad
                grads, losses = grads_and_losses(trainable, frozen, x_train[index], y_train[index])
                if config.grad_clip > 0:  # per-seed global gradient norm, as clip_grad_norm_ per model
                    squared = sum(g.reshape(n_seeds, -1).pow(2).sum(dim=1) for g in grads.values())
                    scale = (config.grad_clip / (squared.sqrt() + 1e-6)).clamp(max=1.0)
                    grads = {n: g * scale.view(-1, *[1] * (g.dim() - 1)) for n, g in grads.items()}
            for name, parameter in trainable.items():
                parameter.grad = grads[name]
            optimizer.step()
            running += losses.double() * index.shape[1]
            step += 1
        base.eval()
        with torch.no_grad():
            predictions = predict_all_seeds(trainable, frozen, x_test).numpy()
        for i, order in enumerate(orders):
            order.after_epoch(epoch)
            if not alive[i]:
                continue
            train_loss = running[i].item() / n_train
            curves[i]["epoch"].append(epoch)
            curves[i]["train_loss"].append(train_loss)
            curves[i]["test_mse"].append(protocol.test_mse(predictions[i], y_test_np))  # the paper's metric
            alive[i] = math.isfinite(train_loss)
    seconds = time.perf_counter() - start
    return [_result_record(config, dataset, window, seed, "vmap", n_train, len(y_test_np), params_trainable,
                           params_effective, curves[i], seconds / n_seeds, step * n_seeds / seconds)
            for i, seed in enumerate(seeds)]


# --------------------------------------------------------------------------- sweeps
def tag_dir(tag: str) -> Path:
    return protocol.RUNS_ROOT / tag


def cell_path(tag: str, dataset: str, window: int, seed: int) -> Path:
    return tag_dir(tag) / "cells" / f"{dataset}_w{window}_s{seed}.json"


def _write_cell(tag: str, result: dict) -> str:
    """Store one cell record and return a one-line summary."""
    result["tag"] = tag
    path = cell_path(tag, result["dataset"], result["window"], result["seed"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result))
    tmp.rename(path)
    return (f"{result['dataset']:24s} w{result['window']:<3d} seed {result['seed']}  "
            f"final test MSE {result['final_test_mse']:.3e}  params {result['params_effective']}  "
            f"{result['seconds']:.1f}s ({result['steps_per_second']:.0f} steps/s)")


def _run_job(job) -> list[str]:
    """Worker: train the seeds of one (dataset, window) group and write their JSON files."""
    tag, config, dataset, window, seeds, engine = job
    torch.set_num_threads(1)
    if engine == "vmap":
        results = train_cells_vmap(config, dataset, window, seeds)
    else:
        results = [train_cell(config, dataset, window, seed) for seed in seeds]
    return [_write_cell(tag, result) for result in results]


def run_sweep(tag: str, config: TrainConfig, datasets, windows, seeds, jobs: int = 1, force: bool = False,
              engine: str = "loop"):
    """Train all missing cells of a sweep (resumable).

    ``engine`` "loop" trains one seed at a time (works for every model); "vmap" trains all
    seeds of a (dataset, window) cell together (see ``train_cells_vmap``).
    """
    if engine not in ("loop", "vmap"):
        raise ValueError(f"unknown engine {engine!r}")
    directory = tag_dir(tag)
    directory.mkdir(parents=True, exist_ok=True)
    config_file = directory / "config.json"
    if config_file.exists() and not force:
        try:  # options added to TrainConfig after the tag was created take their default value
            stored = asdict(TrainConfig(**json.loads(config_file.read_text())))
        except TypeError:
            stored = None
        if stored != asdict(config):
            raise SystemExit(f"tag {tag!r} already holds results for a different configuration "
                             f"({config_file}); use another --tag or --force")
    config_file.write_text(json.dumps(asdict(config), indent=2))

    todo = []
    for dataset in datasets:
        for window in windows:
            missing = [s for s in seeds if force or not cell_path(tag, dataset, window, s).exists()]
            if missing and engine == "vmap":
                todo.append((tag, config, dataset, window, missing, engine))
            else:
                todo += [(tag, config, dataset, window, [s], engine) for s in missing]
    n_cells = sum(len(job[4]) for job in todo)
    print(f"[{tag}] {n_cells} cells to train ({config.model}, engine {engine}, {jobs} job(s))", flush=True)
    start = time.time()
    if jobs <= 1:
        for job in todo:
            for line in _run_job(job):
                print("  " + line, flush=True)
    else:
        import multiprocessing

        todo.sort(key=lambda job: -protocol.load(job[2], job[3])[0].size)  # longest cells first
        with multiprocessing.get_context("fork").Pool(jobs) as pool:
            for lines in pool.imap_unordered(_run_job, todo):
                for line in lines:
                    print("  " + line, flush=True)
    print(f"[{tag}] done in {time.time() - start:.0f}s", flush=True)


def load_cells(tag: str) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted((tag_dir(tag) / "cells").glob("*.json"))]


def aggregate(tag: str, paper_model: str | None = None, write: bool = True) -> dict:
    """Mean +- std of the final-epoch test MSE per (dataset, window) vs the paper.

    The std is the population std (ddof=0): with that convention the authors' own code
    reproduces the paper's "mean +- std" digits (see reference_repro.py report).
    """
    cells = load_cells(tag)
    if not cells:
        raise SystemExit(f"no results under {tag_dir(tag)}")
    groups = {}
    for cell in cells:
        groups.setdefault((cell["dataset"], cell["window"]), []).append(cell)
    order = {d: i for i, d in enumerate(protocol.ALL_DATASETS)}
    rows, lines = [], []
    header = (f"| dataset | N | seeds | params | final test MSE (mean ± std) | paper best | best paper model "
              f"| ours / paper best | result |")
    if paper_model:
        header += f" paper {paper_model} |"
    lines.append(header)
    lines.append("|" + "---|" * (header.count("|") - 1))
    wins = targets = 0
    for (dataset, window) in sorted(groups, key=lambda k: (order[k[0]], k[1])):
        group = sorted(groups[(dataset, window)], key=lambda c: c["seed"])
        values = np.array([c["final_test_mse"] for c in group], dtype=float)
        mean, std = float(values.mean()), float(values.std(ddof=0))
        row = {"dataset": dataset, "window": window, "seeds": [c["seed"] for c in group],
               "final_test_mse": values.tolist(), "mean": mean, "std": std,
               "params_effective": group[0]["params_effective"],
               "follows_paper_recipe": all(c["follows_paper_recipe"] for c in group)}
        line = f"| {dataset} | {window} | {len(group)} | {row['params_effective']} | {mean:.3e} ± {std:.3e} |"
        if protocol.is_paper_cell(dataset, window):
            best = protocol.paper_best(dataset, window)
            verdict = protocol.compare_with_paper(mean, best)
            targets += 1
            wins += verdict == "WIN"
            row.update(paper_best=best, paper_best_model=protocol.paper_best_model(dataset, window),
                       ratio=mean / best, win=verdict == "WIN", verdict=verdict)
            line += f" {best:.1e} | {row['paper_best_model']} | {mean / best:.3g} | {verdict} |"
            if paper_model and paper_model in protocol.paper_results(dataset, window):
                p_mean, p_std = protocol.paper_model(paper_model, dataset, window)
                row.update(paper_model_mean=p_mean, paper_model_std=p_std)
                line += f" {p_mean:.6f} ± {p_std:.6f} |"
            elif paper_model:
                line += " - |"
        else:
            line += " non-target | - | - | - |" + (" - |" if paper_model else "")
        rows.append(row)
        lines.append(line)
    config = cells[0]["config"]
    title = (f"Tag `{tag}`: model `{config['model']}` {config['model_kwargs'] or ''}, {config['epochs']} epochs, "
             f"batch {config['batch_size']}, {config['optimizer']} lr {config['lr']}, scheduler {config['scheduler']}; "
             f"paper recipe: {all(r['follows_paper_recipe'] for r in rows)}")
    footer = f"Wins on paper cells: {wins} / {targets} evaluated (24 paper cells in total)."
    text = "\n".join([title, ""] + lines + ["", footer])
    print(text)
    summary = {"tag": tag, "config": config, "rows": rows, "wins": wins, "paper_cells_evaluated": targets}
    if write:
        (tag_dir(tag) / "summary.json").write_text(json.dumps(summary, indent=2))
        (tag_dir(tag) / "summary.md").write_text(text + "\n")
    return summary


# ------------------------------------------------------------------------------ CLI
def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="train a sweep of cells")
    run.add_argument("--tag", required=True, help="name of the experiment (directory under runs/timeseries/)")
    run.add_argument("--model", required=True, help="registered model name (see: train.py models)")
    run.add_argument("--model_kwargs", default="{}", help='JSON dict passed to the model factory, e.g. \'{"hidden_size": 8}\'')
    run.add_argument("--datasets", nargs="+", default=list(protocol.PAPER_DATASETS), choices=list(protocol.ALL_DATASETS))
    run.add_argument("--windows", nargs="+", type=int, default=list(protocol.WINDOWS))
    run.add_argument("--seeds", nargs="+", type=int, default=list(protocol.PAPER_SEEDS))
    defaults = TrainConfig()
    for name in ("epochs", "batch_size"):
        run.add_argument(f"--{name}", type=int, default=getattr(defaults, name))
    for name in ("lr", "weight_decay", "beta1", "beta2", "adam_eps", "momentum", "warmup_epochs", "final_lr_ratio",
                 "grad_clip"):
        run.add_argument(f"--{name}", type=float, default=getattr(defaults, name))
    run.add_argument("--optimizer", default=defaults.optimizer, choices=["adam", "adamw", "sgd", "rmsprop", "lbfgs"])
    run.add_argument("--scheduler", default=defaults.scheduler, choices=["none", "cosine", "linear", "exponential"])
    run.add_argument("--shuffle", default=defaults.shuffle, choices=["reference", "simple", "none"])
    run.add_argument("--engine", default="loop", choices=["loop", "vmap"],
                     help="loop: one seed at a time (any model); vmap: all seeds of a cell at once (faster)")
    run.add_argument("--jobs", type=int, default=1, help="worker processes (machine rule: at most 2)")
    run.add_argument("--force", action="store_true", help="retrain cells that already have results")
    run.add_argument("--paper-model", default=None, help="also show this paper model's row, e.g. GQKAN-QKANFWP")

    agg = sub.add_parser("aggregate", help="summarise a finished or partial sweep")
    agg.add_argument("--tag", required=True)
    agg.add_argument("--paper-model", default=None)

    sub.add_parser("models", help="list registered models")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "models":
        for name, factory in models.MODEL_REGISTRY.items():
            print(f"{name:18s} {(factory.__doc__ or '').strip().splitlines()[0] if factory.__doc__ else ''}")
        return
    if args.command == "run":
        config = TrainConfig(
            model=args.model, model_kwargs=json.loads(args.model_kwargs), epochs=args.epochs,
            batch_size=args.batch_size, lr=args.lr, optimizer=args.optimizer, weight_decay=args.weight_decay,
            beta1=args.beta1, beta2=args.beta2, adam_eps=args.adam_eps, momentum=args.momentum,
            scheduler=args.scheduler,
            warmup_epochs=args.warmup_epochs, final_lr_ratio=args.final_lr_ratio, grad_clip=args.grad_clip,
            shuffle=args.shuffle)
        run_sweep(args.tag, config, args.datasets, args.windows, args.seeds, jobs=args.jobs, force=args.force,
                  engine=args.engine)
    aggregate(args.tag, paper_model=args.paper_model)


if __name__ == "__main__":
    main()
