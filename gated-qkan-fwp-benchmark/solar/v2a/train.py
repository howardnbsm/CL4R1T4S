"""Training of the v2a model (one fit = one seed on one fit / selection split).

``fit_model`` is the only training routine: AdamW + warm-up + cosine schedule,
mini-batches, gradient clipping, optional EMA weights, optional peak-aware
loss terms.  The selection split is used for checkpoint selection only
(``select="best"``) or not at all (``select="last"``); it never contributes a
gradient.  At most 100 epochs (asserted).
"""

from __future__ import annotations

import copy
import time

import numpy as np
import torch

import common as C
from model import GatedQKANFWP2, predict_numpy

torch.set_num_threads(1)

TRAIN_DEFAULTS = dict(
    epochs=60,
    batch=64,
    lr=3e-3,
    wd=1e-2,
    warmup=0.05,
    clip=1.0,
    alpha=0.0,         # v1's peak-aware weighting: mean((p - y)^2 * (1 + alpha * y))
    peak_w=0.0,        # weight of the smooth-max amplitude term (see peak_terms)
    peak_temp=0.02,    # temperature of the smooth maximum, in scaled units
    time_w=0.0,        # weight of the peak-timing term (cross-entropy of softmax(pred / temp) vs the
                       # Gaussian-smoothed position of the target maximum)
    time_sigma=6.0,    # months
    ema=0.0,           # EMA decay of the weights (0 = off)
    noise=0.0,         # multiplicative amplitude jitter of (x, y) pairs (std of log-scale)
    tok_noise=0.0,     # std of Gaussian noise added to the token / summary level features (training only)
    warp=0.0,          # time-warp augmentation: segments of 660*r months, log r ~ U(-warp, warp), resampled
    oof=False,         # train the nonlinear part on OUT-OF-FOLD ridge baselines of the fit windows
    oof_blocks=6,
    select="last",     # "last" | "best" (best selection-split loss, evaluated every epoch)
    select_metric="scaled_mse",
    select_from=0.5,   # only epochs after this fraction of training are eligible for "best"
)


def make_train_config(**overrides):
    cfg = dict(TRAIN_DEFAULTS)
    unknown = set(overrides) - set(cfg)
    if unknown:
        raise KeyError(f"unknown train config keys: {sorted(unknown)}")
    cfg.update(overrides)
    return cfg


def loss_fn(pred, y, tcfg):
    weight = 1.0 + tcfg["alpha"] * y if tcfg["alpha"] else 1.0
    loss = ((pred - y) ** 2 * weight).mean()
    if tcfg["peak_w"] or tcfg["time_w"]:
        temp = tcfg["peak_temp"]
        soft = torch.softmax(pred / temp, dim=1)
        if tcfg["peak_w"]:
            # smooth maximum of the forecast vs the maximum of the (noisy monthly) target
            smooth_max = temp * torch.logsumexp(pred / temp, dim=1)
            loss = loss + tcfg["peak_w"] * ((smooth_max - y.max(dim=1).values) ** 2).mean()
        if tcfg["time_w"]:
            lead = torch.arange(pred.shape[1], dtype=pred.dtype)
            dist = lead[None, :] - y.argmax(dim=1, keepdim=True).to(pred.dtype)
            target = torch.exp(-0.5 * (dist / tcfg["time_sigma"]) ** 2)
            target = target / target.sum(dim=1, keepdim=True)
            log_soft = torch.log_softmax(pred / temp, dim=1)
            loss = loss + tcfg["time_w"] * (-(target * log_soft).sum(dim=1)).mean()
    return loss


def _index(feats, idx):
    return {k: v[idx] for k, v in feats.items()}


def fit_model(cfg, tcfg, x_fit, y_fit, x_sel=None, y_sel=None, seed=0, log=None):
    """Train one model.  Returns ``(model, info)``; the model holds the selected weights.

    Args:
        cfg / tcfg: model and training configs (dicts of overrides).
        x_fit, y_fit: scaled windows (n, 528) and targets (n, 132) used for the gradient.
        x_sel, y_sel: selection split (checkpoint selection / monitoring only).
        seed: seeds torch and numpy; one seed = one independent repetition.
    """
    tcfg = make_train_config(**tcfg)
    assert tcfg["epochs"] <= 100, "the protocol allows at most 100 epochs"
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = GatedQKANFWP2(cfg)
    xf = torch.from_numpy(np.asarray(x_fit, dtype=np.float32))
    yf = torch.from_numpy(np.asarray(y_fit, dtype=np.float32))
    model.set_decoder_constants(xf, yf)
    feats = model.features(xf)
    if tcfg["oof"]:
        assert not tcfg["warp"] and not tcfg["noise"], "oof baselines exist for the original fit windows only"
        feats["base"] = model.oof_baseline(xf, yf, tcfg["oof_blocks"])

    series = None
    if tcfg["warp"]:
        # the fit windows are contiguous stride-1 windows -> rebuild the monthly series they cover
        xa, ya = np.asarray(x_fit, dtype=np.float64), np.asarray(y_fit, dtype=np.float64)
        assert np.array_equal(xa[1:, :-1], xa[:-1, 1:]), "warp augmentation needs contiguous stride-1 windows"
        series = np.concatenate([xa[0], xa[1:, -1], ya[-1]])
        assert len(series) == len(xa) + xa.shape[1] + ya.shape[1] - 1
    n_content = 1 + int(model.cfg["slope_feature"])

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if (p.dim() <= 1 or "gate" in name or "freq" in name) else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": tcfg["wd"]},
                             {"params": no_decay, "weight_decay": 0.0}], lr=tcfg["lr"])
    n = len(xf)
    steps_per_epoch = int(np.ceil(n / tcfg["batch"]))
    total = steps_per_epoch * tcfg["epochs"]
    warm = max(1, int(tcfg["warmup"] * total))

    def lr_at(step):
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1.0 + np.cos(np.pi * (step - warm) / max(1, total - warm)))

    ema = copy.deepcopy(model) if tcfg["ema"] else None
    history, best, step = [], None, 0
    t0 = time.time()
    for epoch in range(tcfg["epochs"]):
        model.train()
        perm = rng.permutation(n)
        running = 0.0
        for s in range(steps_per_epoch):
            idx = torch.from_numpy(perm[s * tcfg["batch"]:(s + 1) * tcfg["batch"]])
            if series is not None:
                total_len = xf.shape[1] + yf.shape[1]
                ratio = np.exp(rng.uniform(-tcfg["warp"], tcfg["warp"], size=len(idx)))
                seg = np.minimum(np.round(total_len * ratio), len(series) - 1)
                start = rng.uniform(0, len(series) - seg)
                pos = start[:, None] + (np.arange(total_len)[None, :] + 0.5) * (seg[:, None] / total_len) - 0.5
                warped = np.interp(pos, np.arange(len(series)), series).astype(np.float32)
                batch = model.features(torch.from_numpy(warped[:, :xf.shape[1]]))
                target = torch.from_numpy(warped[:, xf.shape[1]:])
            else:
                batch = _index(feats, idx)
                target = yf[idx]
            if tcfg["tok_noise"]:
                tokens = batch["tokens"].clone()
                tokens[..., :n_content] += tcfg["tok_noise"] * torch.randn_like(tokens[..., :n_content])
                batch = dict(batch, tokens=tokens)
            if tcfg["noise"]:
                # amplitude jitter: with norm="mean" the model is scale-equivariant, so this is
                # only active for norm="none" models (scales x and y consistently)
                factor = torch.exp(tcfg["noise"] * torch.randn(len(idx), 1))
                batch = model.features(xf[idx] * factor)
                target = target * factor
            for group in opt.param_groups:
                group["lr"] = tcfg["lr"] * lr_at(step)
            loss = loss_fn(model.forward_features(batch), target, tcfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if tcfg["clip"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["clip"])
            opt.step()
            step += 1
            running += float(loss.detach()) * len(idx)
            if ema is not None:
                with torch.no_grad():
                    d = min(tcfg["ema"], (1 + step) / (10 + step))
                    for pe, pm in zip(ema.parameters(), model.parameters()):
                        pe.mul_(d).add_(pm.detach(), alpha=1 - d)
        record = {"epoch": epoch + 1, "train_loss": running / n}
        current = ema if ema is not None else model
        if x_sel is not None:
            m = C.score(predict_numpy(current, x_sel), y_sel)
            record.update({k: m[k] for k in ("scaled_mse", "pae", "pte", "peak_aware_mse")})
            eligible = (epoch + 1) >= tcfg["select_from"] * tcfg["epochs"]
            if tcfg["select"] == "best" and eligible and (best is None or m[tcfg["select_metric"]] < best[0]):
                best = (m[tcfg["select_metric"]], epoch + 1, copy.deepcopy(current.state_dict()))
        history.append(record)
        if log is not None and ((epoch + 1) % 10 == 0 or epoch == 0):
            log(f"  ep {epoch + 1:3d} train {record['train_loss']:.5f} "
                + (f"sel {record['scaled_mse']:.4f}/{record['pae']:.1f}/{record['pte']:.1f}" if x_sel is not None else ""))
    final = ema if ema is not None else model
    selected_epoch = tcfg["epochs"]
    if tcfg["select"] == "best" and best is not None:
        final.load_state_dict(best[2])
        selected_epoch = best[1]
    final.eval()
    info = {"history": history, "selected_epoch": selected_epoch, "seconds": time.time() - t0,
            "params_trainable": final.n_trainable(), "params_data_derived": final.n_data_derived()}
    return final, info
