"""v1 (GQKAN-QKANFWP, 12,474 parameters) reference numbers for approach B. The test split is never touched.

    python v1_reference.py --val            # 5-seed reproduction checkpoints (best validation epoch) scored on VAL
                                            # -> runs/solar/v2b/v1_on_val.json
    python v1_reference.py --fold cv1 [--seed 0] [--epochs 100]
                                            # v1 trained from scratch on the fit windows of a rolling-origin fold with the
                                            # authors' recipe (peak-aware loss alpha 1, Adam 2.5e-3, keras decay per batch,
                                            # batch 32, shuffled), scored on the fold's evaluation block after every epoch
                                            # -> runs/solar/v2b/v1_folds/{fold}_seed{seed}.json

Fold runs report (i) the last epoch and (ii) the epoch with the best peak-aware loss ON THE EVALUATION BLOCK itself -
an ORACLE checkpoint choice that favours v1 (v1's protocol selects the checkpoint on a validation block; inside a
fold no such block exists without removing fit windows).
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import torch

import common as C

sys.path.insert(0, str(C.PROJECT_DIR / "solar"))
import analyze_reference as AR  # noqa: E402  (read-only helper: load_reference_model, predict)
import cpu_backend  # noqa: E402
import reference_cpu  # noqa: E402

torch.set_num_threads(1)


def on_val():
    d = C.data()
    xva, yva = d.split("val")
    rows, preds = {}, []
    for seed in range(5):
        model = AR.load_reference_model(seed, AR.DEFAULT_OUT_ROOT)
        pva = AR.predict(model, xva)
        preds.append(pva)
        rows[f"seed{seed}"] = dict(C.score(pva, yva),
                                   params=int(sum(p.numel() for p in model.parameters() if p.requires_grad)))
        print(seed, "val", C.short(rows[f"seed{seed}"]), flush=True)
    agg = {}
    for k in ("scaled_mse", "pae", "pte", "peak_aware_mse"):
        v = np.array([rows[f"seed{s}"][k] for s in range(5)])
        agg[k] = dict(mean=float(v.mean()), std_population=float(v.std()), std_sample=float(v.std(ddof=1)))
    rows["aggregate_val"] = agg
    rows["seed_mean_forecast_val"] = C.score(np.mean(preds, axis=0), yva)
    (C.RUNS_DIR / "v1_on_val.json").write_text(json.dumps(rows, indent=1))
    print("aggregate:", {k: (round(v["mean"], 5), round(v["std_sample"], 5)) for k, v in agg.items()})


def on_fold(fold, seed, epochs):
    d = C.data()
    fit, ev = C.folds((fold,))[fold]
    X = torch.from_numpy(d.X[fit]).unsqueeze(-1)
    Y = torch.from_numpy(d.Y[fit])
    args = reference_cpu.reference_args(seed=seed, epochs=epochs)
    reference_cpu.seed_everything(seed)
    model = cpu_backend.build_model(args, impl="fast")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda step: 1.0 / (1.0 + 1e-6 * step))
    gen = torch.Generator().manual_seed(seed)
    curve, t0 = [], time.time()
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(X), generator=gen)
        for i in range(0, len(perm), args.batch_size):
            idx = perm[i:i + args.batch_size]
            out = model(X[idx])[-1]
            loss = torch.mean((out.squeeze() - Y[idx]) ** 2 * (1.0 + args.alpha * Y[idx]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        model.eval()
        m = C.score(AR.predict(model, d.X[ev]), d.Y[ev])
        curve.append([ep, m["scaled_mse"], m["pae"], m["pte"], m["peak_aware_mse"]])
        if ep % 10 == 0 or ep == 1:
            print(f"{fold} s{seed} ep {ep} {C.short(m)} pa {m['peak_aware_mse']:.4f} {time.time() - t0:.0f}s", flush=True)
            out_dir = C.RUNS_DIR / "v1_folds"
            out_dir.mkdir(parents=True, exist_ok=True)
            c = np.array(curve)
            best = c[c[:, 4].argmin()]
            (out_dir / f"{fold}_seed{seed}.json").write_text(json.dumps(dict(
                fold=fold, seed=seed, epochs_done=ep, n_fit=int(len(fit)),
                last=dict(scaled_mse=c[-1, 1], pae=c[-1, 2], pte=c[-1, 3]),
                oracle_best_epoch=dict(epoch=int(best[0]), scaled_mse=best[1], pae=best[2], pte=best[3]),
                min_mse_any_epoch=float(c[:, 1].min()), curve=curve, seconds=time.time() - t0), indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", action="store_true")
    ap.add_argument("--fold", nargs="+", default=[])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100)
    a = ap.parse_args()
    if a.val:
        on_val()
    for f in a.fold:
        on_fold(f, a.seed, a.epochs)
