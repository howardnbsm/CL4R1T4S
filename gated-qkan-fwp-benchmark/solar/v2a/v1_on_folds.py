"""v1 (the authors' model and training recipe) on the rolling-origin folds, for a per-fold v1-vs-v2 comparison.

    python solar/v2a/v1_on_folds.py --folds cv1 cv2 cv3 cv4 --seeds 0 [--epochs 100]
    -> runs/solar/v2a/v1_on_folds.jsonl  (one line per fold / seed, restart-safe)

The model is the authors' GQKAN-QKANFWP built by ``cpu_backend.build_model`` with the ``run.sh``
hyper-parameters (12,474 parameters).  The loop mirrors the reference trainer: peak-aware MSE
(alpha = 1), Adam lr 2.5e-3, keras decay 1/(1 + 1e-6 step) per batch, batch 32 with shuffling,
100 epochs, checkpoint = the epoch with the lowest peak-aware loss on the evaluation block.
Folds ``far1..far4`` replicate the official protocol shifted back in time: fit windows 0..v0-261, a
selection block v0-260..v0-1 (the role of the official validation split: checkpoint selection only)
and an evaluation block v0..v0+259 (the role of the test split).  Reported per fold:
  "sel"    evaluation-block metrics of the checkpoint with the lowest selection-block loss (= v1's protocol);
  "oracle" evaluation-block metrics of the checkpoint with the lowest evaluation-block loss (optimistic);
  "last"   evaluation-block metrics after the last epoch.
For ``cv*`` folds there is no selection block and "sel" equals "oracle".
The test split is never touched (sealed protocol data only).
"""
from __future__ import annotations

import argparse
import json
import random
import time

import numpy as np
import torch

import common as C
import cpu_backend
from reference_cpu import reference_args

torch.set_num_threads(1)
OUT = C.RUNS_DIR / "v1_on_folds.jsonl"
KEYS = ("scaled_mse", "pae", "pte", "peak_aware_mse")


@torch.no_grad()
def predict(model, x):
    model.eval()
    out = [model(torch.from_numpy(x[i:i + 64]).unsqueeze(-1))[-1] for i in range(0, len(x), 64)]
    return torch.cat(out).numpy()


def run(fold, seed, epochs):
    d = C.data()
    fit, ev = C.folds((fold,))[fold]
    torch.manual_seed(seed), np.random.seed(seed), random.seed(seed)
    model = cpu_backend.build_model(reference_args(seed=seed, epochs=epochs), impl="fast")
    n_par = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    xf, yf = torch.from_numpy(d.X[fit]).unsqueeze(-1), torch.from_numpy(d.Y[fit])
    opt = torch.optim.Adam(model.parameters(), lr=2.5e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda step: 1.0 / (1.0 + 1e-6 * step))
    sel_idx = np.arange(ev[0] - C.FOLD_VAL_SIZE, ev[0]) if fold.startswith("far") else ev
    best, sel, t0 = None, None, time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(xf))
        for start in range(0, len(xf), 32):
            idx = perm[start:start + 32]
            out = model(xf[idx])[-1]
            loss = ((out.squeeze() - yf[idx]) ** 2 * (1.0 + yf[idx])).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        m = C.score(predict(model, d.X[ev]), d.Y[ev])
        if best is None or m["peak_aware_mse"] < best["peak_aware_mse"]:
            best = dict(m, epoch=epoch)
        sel_loss = m["peak_aware_mse"] if sel_idx is ev else C.score(predict(model, d.X[sel_idx]), d.Y[sel_idx])["peak_aware_mse"]
        if sel is None or sel_loss < sel["sel_loss"]:
            sel = dict(m, epoch=epoch, sel_loss=sel_loss)
        if epoch % 10 == 0:
            print(f"  {fold} s{seed} epoch {epoch:3d} eval {C.short(m)} best@{best['epoch']} {C.short(best)} "
                  f"{time.time() - t0:.0f}s", flush=True)
    rec = {"fold": fold, "seed": seed, "epochs": epochs, "params": n_par, "n_fit": int(len(fit)),
           "sel": {k: sel[k] for k in KEYS}, "sel_epoch": sel["epoch"], "sel_loss": sel["sel_loss"],
           "oracle": {k: best[k] for k in KEYS}, "oracle_epoch": best["epoch"], "last": {k: m[k] for k in KEYS},
           "seconds": round(time.time() - t0, 1)}
    with open(OUT, "a") as handle:
        handle.write(json.dumps(rec) + "\n")
    print(f"v1 {fold} s{seed} sel@{sel['epoch']} {C.short(sel)} oracle@{best['epoch']} {C.short(best)} last {C.short(m)} "
          f"{rec['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="+", default=["far1", "far2", "far3", "far4"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--epochs", type=int, default=100)
    cli = ap.parse_args()
    done = set()
    if OUT.is_file():
        done = {(r["fold"], r["seed"]) for r in map(json.loads, OUT.read_text().splitlines())}
    for seed in cli.seeds:
        for fold in cli.folds:
            if (fold, seed) not in done:
                run(fold, seed, cli.epochs)
