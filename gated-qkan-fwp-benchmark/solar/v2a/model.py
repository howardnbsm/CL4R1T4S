"""Gated QKAN-FWP v2, approach A (solar): the model.

Backbone kept from v1: a QKAN SLOW PROGRAMMER reads the input window step by
step (all steps in parallel), writes rank-1 angle updates into GATED FAST
WEIGHTS, and the folded fast weights are the rotation angles of a QKAN FAST
PROGRAMMER that is evaluated once per window on a query.

What changed with respect to v1 (each item is a config switch, see DEFAULTS):

  tokens      the window is cut into ``patch``-month patches; a step sees the
              patch mean, the within-patch slope and position-from-the-end /
              11-year-phase features instead of one raw monthly value;
  scale       amplitude normalisation by the window mean (the forecast is
              scale-equivariant by construction);
  gates       ``n_gates`` gate channels initialised to different memory
              horizons matched to the window (v1: one gate, bias +2 = 8 steps);
              gate g programs the fast-layer output rows o with o % n_gates == g;
  query       the fast programmer is queried with a learned projection of a
              summary of the recent history (12-month means of the last
              ``query_months`` months), not with the last raw value;
  decoder     "basis": the fast programmer's outputs are mapped to the
              coefficients of a small smooth basis over the 132 forecast months
              (principal components of the scale-normalised training targets),
              instead of BatchNorm -> Dropout -> Linear(20, 132);
              "leadtime": the fast programmer is queried once per lead time
              (query phase = recent-history projection + frequency * lead time)
              and a Linear(O, 1) read-out gives the forecast value at that lead;
  observable  the physical <Z> (qkan ``fast_measure=False``) in both QKAN layers.

All QKAN layers are evaluated with ``daruan.py`` (verified identical to the
qkan package layer by ``test_daruan.py``).

Ablation switches (for the honest "is the QKAN load-bearing" report):
  slow="linear"  the slow QKAN layer is replaced by a Linear(q, q);
  fast="linear"  the programmed QKAN layer is replaced by a programmed LINEAR
                 layer z_o = sum_i Theta[o, i] * query_i (classical fast weights).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

import daruan

SEQ_LEN, HORIZON = 528, 132

DEFAULTS = dict(
    # featurisation
    patch=12,              # months per token
    norm="mean",           # "mean": divide by the window mean; "none"
    phase_features=True,   # cos/sin of 2 pi * (months before window end) / 132
    slope_feature=True,
    query_months=132,      # recent history summarised for the query
    query_patch=12,
    # slow programmer
    slow="qkan",           # "qkan" | "linear" (ablation)
    q=8,                   # slow QKAN width
    h=16,                  # slow hidden width
    # fast weights
    n_gates=3,
    gate_horizons=(3.0, 11.0, 44.0),   # in tokens
    theta0=False,          # learnable static angle offset
    # fast programmer
    fast="qkan",           # "qkan" | "linear" (ablation)
    I=6,
    O=12,
    reps=2,
    fast_base=False,       # static SiLU base path in the fast layer
    observable="z",
    # decoder
    decoder="basis",       # "basis" | "leadtime"
    K=8,                   # basis rank
    skip=False,            # linear skip: recent summary -> basis coefficients
    head_dropout=0.0,
    n_freq_init=(0.5, 1.0, 1.5, 2.0, 3.0, 4.0),  # leadtime decoder: cycles per 132 months
    lead_stride=3,         # leadtime decoder: query every ``lead_stride`` months, linear interpolation between
    freq_max=0.0,          # leadtime decoder: cap of the query frequencies in cycles per 132 months (0 = none)
    lead_value=False,      # leadtime decoder: the query also contains beta_i * (baseline forecast value at that lead)
    lead_rel=False,        # ... and gamma_i * (baseline value at that lead / maximum of the baseline curve)
    prog_gain=1.0,         # initial value of the learnable scalar multiplying the programmed angles (theta0=True only)
    query="summary",       # "summary": query offset = Linear(recent summary); "static": a learned constant vector;
                           # "clock": query_i = w_i * (cycle clock angle) + b_i  (w, b learnable, I = len(clock_freqs))
    clock="minimum",       # cycle clock (query="clock"): "minimum" = 2 pi * (months since the last minimum of the
                           # smoothed window) / 132;  "fourier" = phase of the 132-month harmonic of the last 132 months
    clock_freqs=(1.0, 0.5),  # initial w_i (cycles per 132 months of the clock)
    clock_sigma=8.0,       # "minimum" clock: Gaussian smoothing (months)
    clock_margin=6,        # "minimum" clock: a minimum is accepted once it lies this many months before the window end
    program=True,          # False (ablation): no slow programmer / fast weights, angles = static theta0 only
    # baseline (the linear part of the model, solved in closed form on the fit split)
    baseline="mean",       # "mean": mean normalised target curve; "ridge": ridge forecast from the recent summary
    ridge_lam=1e-2,
    residual="add",        # "add": baseline + r ; "mul": baseline * (1 + r)
    zero_init=False,       # zero-initialise the read-out (the model starts as its baseline)
)


def make_config(**overrides):
    cfg = dict(DEFAULTS)
    unknown = set(overrides) - set(cfg)
    if unknown:
        raise KeyError(f"unknown config keys: {sorted(unknown)}")
    cfg.update(overrides)
    return cfg


class Features:
    """Fixed featurisation of a batch of scaled windows (no parameters)."""

    def __init__(self, cfg):
        self.cfg = cfg
        patch = cfg["patch"]
        assert SEQ_LEN % patch == 0
        self.n_tokens = SEQ_LEN // patch
        centre = (torch.arange(self.n_tokens, dtype=torch.float32) + 0.5) * patch
        age_months = SEQ_LEN - centre                       # months from patch centre to window end
        cols = [age_months / SEQ_LEN]
        if cfg["phase_features"]:
            cols += [torch.cos(2 * math.pi * age_months / HORIZON), torch.sin(2 * math.pi * age_months / HORIZON)]
        self.position = torch.stack(cols, dim=1)            # (T, n_pos)
        t = torch.arange(patch, dtype=torch.float32)
        t = t - t.mean()
        self.slope_kernel = t / (t * t).sum() * patch       # slope expressed as change over one patch
        self.token_dim = 1 + int(cfg["slope_feature"]) + self.position.shape[1]
        self.query_dim = cfg["query_months"] // cfg["query_patch"]
        t_end = torch.arange(HORIZON, dtype=torch.float32) - (HORIZON - 1)       # 0 at the window end
        self.clock_cos = torch.cos(2 * math.pi * t_end / HORIZON) / HORIZON
        self.clock_sin = torch.sin(2 * math.pi * t_end / HORIZON) / HORIZON

    def clock(self, xn: torch.Tensor) -> torch.Tensor:
        """Cycle clock angle (B, 1) in radians, computed from the scale-normalised window."""
        if self.cfg["clock"] == "fourier":
            seg = xn[:, SEQ_LEN - HORIZON:]
            return torch.atan2(seg @ self.clock_sin, seg @ self.clock_cos).unsqueeze(1)
        if self.cfg["clock"] == "minimum":
            since = months_since_last_minimum(xn, sigma=self.cfg["clock_sigma"], margin=self.cfg["clock_margin"])
            return (2 * math.pi / HORIZON) * since.unsqueeze(1)
        raise ValueError(self.cfg["clock"])

    def __call__(self, x: torch.Tensor):
        """x: (B, 528) scaled windows -> dict(tokens (B, T, d), summary (B, n_q), scale (B, 1))."""
        cfg = self.cfg
        x = x.float()
        if cfg["norm"] == "mean":
            scale = x.mean(dim=1, keepdim=True) + 1e-3
        elif cfg["norm"] == "none":
            scale = torch.ones(x.shape[0], 1)
        else:
            raise ValueError(cfg["norm"])
        xn = x / scale
        patches = xn.view(x.shape[0], self.n_tokens, cfg["patch"])
        cols = [patches.mean(dim=2, keepdim=True)]
        if cfg["slope_feature"]:
            cols.append((patches * self.slope_kernel).sum(dim=2, keepdim=True))
        cols.append(self.position.unsqueeze(0).expand(x.shape[0], -1, -1))
        tokens = torch.cat(cols, dim=2)
        recent = xn[:, SEQ_LEN - cfg["query_months"]:]
        summary = recent.view(x.shape[0], self.query_dim, cfg["query_patch"]).mean(dim=2)
        out = {"tokens": tokens, "summary": summary, "scale": scale}
        if cfg["query"] == "clock":
            out["clock"] = self.clock(xn)
        return out


def months_since_last_minimum(xn: torch.Tensor, sigma: float = 8.0, half_width: int = 40, margin: int = 6,
                              cap: float = 220.0) -> torch.Tensor:
    """Months from the last minimum of the Gaussian-smoothed window to the window end, (B,).

    The window is smoothed with a Gaussian of ``sigma`` months (edge-renormalised).  A minimum is a month whose
    smoothed value is the lowest within +-``half_width`` months (neighbourhood truncated at the window end) and
    that lies at least ``margin`` months before the window end and ``half_width`` months after its start.  If
    the window has no such month the cap is returned.  Only the given window is used.
    """
    from scipy.ndimage import gaussian_filter1d, minimum_filter1d

    x = xn.double().numpy()
    smooth = gaussian_filter1d(x, sigma, axis=1, mode="constant", cval=0.0, truncate=4.0)
    smooth = smooth / gaussian_filter1d(np.ones_like(x[:1]), sigma, axis=1, mode="constant", cval=0.0, truncate=4.0)
    local_min = minimum_filter1d(smooth, size=2 * half_width + 1, axis=1, mode="constant", cval=np.inf)
    n_t = x.shape[1]
    is_min = smooth <= local_min
    is_min[:, :half_width] = False
    is_min[:, n_t - margin:] = False
    last = n_t - 1 - np.argmax(is_min[:, ::-1], axis=1)              # index of the last minimum
    since = np.where(is_min.any(axis=1), n_t - 1 - last, cap)
    return torch.from_numpy(np.minimum(since, cap).astype(np.float32))


class GatedQKANFWP2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg = make_config(**cfg)
        self.features = Features(cfg)
        q, h, n_in, n_out, reps = cfg["q"], cfg["h"], cfg["I"], cfg["O"], cfg["reps"]
        self.n_layers = reps + 1 if cfg["fast"] == "qkan" else 1
        self.n_par = 2 if cfg["fast"] == "qkan" else 1

        # slow programmer
        self.slow_in = nn.Linear(self.features.token_dim, q)
        if cfg["slow"] == "qkan":
            self.slow_qkan = daruan.StaticQKANLayer(q, q, reps=1, preact_trainable=True, ba_trainable=True,
                                                    observable=cfg["observable"])
        elif cfg["slow"] == "linear":
            self.slow_qkan = nn.Linear(q, q)
        else:
            raise ValueError(cfg["slow"])
        self.slow_out = nn.Linear(q, h)
        self.head_a = nn.Linear(h, n_out * self.n_layers)
        self.head_b = nn.Linear(h, n_in * self.n_par)
        self.gate = nn.Linear(h, cfg["n_gates"])
        horizons = list(cfg["gate_horizons"])[: cfg["n_gates"]]
        assert len(horizons) == cfg["n_gates"]
        with torch.no_grad():
            self.gate.weight.mul_(0.1)
            self.gate.bias.copy_(torch.tensor([daruan.gate_bias_for_horizon(v) for v in horizons]))
        self.register_buffer("gate_of_row", torch.arange(n_out) % cfg["n_gates"])
        if cfg["theta0"]:
            self.theta0 = nn.Parameter(0.1 * torch.randn(n_out, n_in, self.n_layers, self.n_par))
            self.prog_gain = nn.Parameter(torch.tensor(float(cfg["prog_gain"])))
        if cfg["fast_base"]:
            self.fast_base = nn.Parameter(torch.zeros(n_out, n_in))

        # query
        if cfg["query"] == "summary":
            self.query = nn.Linear(self.features.query_dim, n_in)
        elif cfg["query"] == "static":
            self.query_bias = nn.Parameter(torch.zeros(n_in))
        elif cfg["query"] == "clock":
            freqs = list(cfg["clock_freqs"])
            assert len(freqs) == n_in, "query='clock' needs len(clock_freqs) == I"
            self.clock_w = nn.Parameter(torch.tensor(freqs, dtype=torch.float32))
            self.clock_b = nn.Parameter(torch.zeros(n_in))
        else:
            raise ValueError(cfg["query"])
        if not cfg["program"]:
            assert cfg["theta0"], "program=False needs theta0=True"
            for module in (self.slow_in, self.slow_qkan, self.slow_out, self.head_a, self.head_b, self.gate):
                for p in module.parameters():
                    p.requires_grad_(False)

        # decoder
        if cfg["decoder"] == "basis":
            self.readout = nn.Linear(n_out, cfg["K"])
            self.register_buffer("basis", torch.zeros(cfg["K"], HORIZON))
            self.register_buffer("mean_curve", torch.zeros(HORIZON))
            if cfg["skip"]:
                self.skip = nn.Linear(self.features.query_dim, cfg["K"], bias=False)
                nn.init.zeros_(self.skip.weight)
        elif cfg["decoder"] == "leadtime":
            freqs = list(cfg["n_freq_init"])
            freqs = (freqs * ((n_in + len(freqs) - 1) // len(freqs)))[:n_in]
            self.freq = nn.Parameter(2 * math.pi * torch.tensor(freqs, dtype=torch.float32))
            stride = cfg["lead_stride"]
            assert HORIZON % stride == 0
            n_nodes = HORIZON // stride
            node_month = torch.arange(n_nodes, dtype=torch.float32) * stride + (stride - 1) / 2.0
            self.register_buffer("lead", (node_month + 0.5) / HORIZON)          # query positions in (0, 1)
            # fixed linear interpolation (linear extrapolation at both ends) from nodes to the 132 months
            month = torch.arange(HORIZON, dtype=torch.float32)
            left = torch.clamp(torch.floor((month - node_month[0]) / stride), 0, max(n_nodes - 2, 0)).long()
            interp = torch.zeros(HORIZON, n_nodes)
            if n_nodes == HORIZON:
                interp = torch.eye(HORIZON)
            else:
                frac = (month - node_month[left]) / stride
                interp[torch.arange(HORIZON), left] = 1.0 - frac
                interp[torch.arange(HORIZON), left + 1] = frac
            self.register_buffer("interp", interp)
            self.readout = nn.Linear(n_out, 1)
            self.register_buffer("mean_curve", torch.zeros(HORIZON))
            if cfg["lead_value"]:
                self.beta = nn.Parameter(torch.randn(n_in))
            if cfg["lead_rel"]:
                self.gamma = nn.Parameter(3.0 * torch.randn(n_in))
        else:
            raise ValueError(cfg["decoder"])
        self.dropout = nn.Dropout(cfg["head_dropout"])
        if cfg["baseline"] == "ridge":
            self.register_buffer("ridge_w", torch.zeros(self.features.query_dim, HORIZON))
            self.register_buffer("ridge_b", torch.zeros(HORIZON))
        elif cfg["baseline"] != "mean":
            raise ValueError(cfg["baseline"])
        if cfg["zero_init"]:
            nn.init.zeros_(self.readout.weight)
            nn.init.zeros_(self.readout.bias)

    # -- data-derived, non-trainable decoder constants (fit split only) --------------------
    @torch.no_grad()
    def set_decoder_constants(self, x_fit: torch.Tensor, y_fit: torch.Tensor):
        """Mean curve and (basis decoder) principal components of the scale-normalised FIT targets."""
        feats = self.features(x_fit)
        scale = feats["scale"]
        yn = (y_fit.float() / scale).double()
        mean = yn.mean(dim=0)
        self.mean_curve.copy_(mean.float())
        if self.cfg["baseline"] == "ridge":
            # closed-form ridge (unpenalised intercept) from the recent-history summary to the target curve
            f = feats["summary"].double()
            mu_f = f.mean(dim=0)
            fc, tc = f - mu_f, yn - mean
            gram = fc.T @ fc + self.cfg["ridge_lam"] * len(f) * torch.eye(f.shape[1], dtype=torch.float64)
            w = torch.linalg.solve(gram, fc.T @ tc)
            self.ridge_w.copy_(w.float())
            self.ridge_b.copy_((mean - mu_f @ w).float())
        if self.cfg["decoder"] == "basis":
            _, sing, vt = torch.linalg.svd(yn - mean, full_matrices=False)
            k = self.cfg["K"]
            basis = vt[:k] * (sing[:k, None] / math.sqrt(len(yn)))   # coefficient targets have unit variance
            self.basis.copy_(basis.float())

    def n_data_derived(self) -> int:
        n = self.mean_curve.numel() if self.cfg["baseline"] == "mean" else 0
        if self.cfg["decoder"] == "basis":
            n += self.basis.numel()
        if self.cfg["baseline"] == "ridge":
            n += self.ridge_w.numel() + self.ridge_b.numel()
        return int(n)

    @torch.no_grad()
    def oof_baseline(self, x_fit: torch.Tensor, y_fit: torch.Tensor, n_blocks: int = 6) -> torch.Tensor:
        """Out-of-fold ridge baseline for the FIT windows (normalised units), for training the nonlinear part.

        The fit windows (contiguous, stride 1) are cut into ``n_blocks`` blocks; the baseline of a block comes
        from a ridge fitted on the windows at least 132 positions away from it, so no target month of the
        block is a target month of that ridge (the same embargo as the rolling-origin folds).  The nonlinear
        part therefore sees honest baseline errors instead of in-sample ones.
        """
        assert self.cfg["baseline"] == "ridge"
        feats = self.features(x_fit)
        f = feats["summary"].double()
        yn = (y_fit.float() / feats["scale"]).double()
        n = len(f)
        out = torch.zeros(n, HORIZON, dtype=torch.float64)
        edges = [round(j * n / n_blocks) for j in range(n_blocks + 1)]
        index = torch.arange(n)
        for lo, hi in zip(edges[:-1], edges[1:]):
            keep = (index < lo - (HORIZON - 1)) | (index >= hi + (HORIZON - 1))
            fk, tk = f[keep], yn[keep]
            mu_f, mu_t = fk.mean(dim=0), tk.mean(dim=0)
            fc = fk - mu_f
            gram = fc.T @ fc + self.cfg["ridge_lam"] * len(fk) * torch.eye(f.shape[1], dtype=torch.float64)
            w = torch.linalg.solve(gram, fc.T @ (tk - mu_t))
            out[lo:hi] = (f[lo:hi] - mu_f) @ w + mu_t
        return out.float()

    def baseline_curve(self, feats: dict) -> torch.Tensor:
        """The linear part of the model in normalised units, (B, 132)."""
        if self.cfg["baseline"] == "ridge":
            return feats["summary"] @ self.ridge_w + self.ridge_b
        return self.mean_curve.expand(feats["summary"].shape[0], -1)

    # -- forward ---------------------------------------------------------------------------
    def fast_weights(self, tokens: torch.Tensor):
        """Slow programmer + gated fold -> programmed angles (B, O, I, layers, par) and diagnostics."""
        cfg = self.cfg
        batch, n_tok, _ = tokens.shape
        flat = tokens.reshape(batch * n_tok, -1)
        hidden = self.slow_out(self.slow_qkan(self.slow_in(flat)))
        n_out, n_in = cfg["O"], cfg["I"]
        a = self.head_a(hidden).view(batch, n_tok, n_out, self.n_layers)
        b = self.head_b(hidden).view(batch, n_tok, n_in * self.n_par)
        log_g = torch.nn.functional.logsigmoid(self.gate(hidden)).view(batch, n_tok, cfg["n_gates"])
        log_1mg = torch.nn.functional.logsigmoid(-self.gate(hidden)).view(batch, n_tok, cfg["n_gates"])
        # weight of step t in the final fast weights: (1 - g_t) * prod_{s > t} g_s   (theta_0 = 0)
        tail = torch.flip(torch.cumsum(torch.flip(log_g, dims=[1]), dim=1), dims=[1]) - log_g
        coef = torch.exp(log_1mg + tail)                                    # (B, T, G)
        coef_rows = coef[:, :, self.gate_of_row]                            # (B, T, O)
        a_w = (a * coef_rows.unsqueeze(-1)).reshape(batch, n_tok, n_out * self.n_layers)
        folded = torch.bmm(a_w.transpose(1, 2), b)                          # (B, O*L, I*P)
        theta = folded.view(batch, n_out, self.n_layers, n_in, self.n_par).permute(0, 1, 3, 2, 4)
        if cfg["theta0"]:
            theta = self.prog_gain * theta + self.theta0
        return theta, coef

    def fast_layer(self, query: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        if cfg["fast"] == "qkan":
            base = self.fast_base if cfg["fast_base"] else None
            return daruan.programmed_qkan(query, theta, cfg["observable"], base)
        weight = theta[..., 0, 0]                                           # (B, O, I)
        if query.dim() == 2:
            return torch.einsum("boi,bi->bo", weight, query)
        return torch.einsum("boi,bqi->bqo", weight, query)

    def query_vector(self, feats: dict) -> torch.Tensor:
        """The query of the fast programmer, (B, I)."""
        if self.cfg["query"] == "summary":
            return self.query(feats["summary"])
        if self.cfg["query"] == "clock":
            return feats["clock"] * self.clock_w + self.clock_b
        return self.query_bias.unsqueeze(0).expand(feats["summary"].shape[0], -1)

    def forward_features(self, feats: dict) -> torch.Tensor:
        cfg = self.cfg
        batch = feats["summary"].shape[0]
        if cfg["program"]:
            theta, _ = self.fast_weights(feats["tokens"])
        else:
            theta = self.theta0.unsqueeze(0).expand(batch, -1, -1, -1, -1)
        q0 = self.query_vector(feats)                                       # (B, I)
        base = feats["base"] if "base" in feats else self.baseline_curve(feats)
        if cfg["decoder"] == "basis":
            z = self.dropout(self.fast_layer(q0, theta))                     # (B, O)
            coef = self.readout(z)
            if cfg["skip"]:
                coef = coef + self.skip(feats["summary"] - 1.0)
            resid = coef @ self.basis
        else:
            freq = self.freq
            if cfg["freq_max"]:
                freq = freq.clamp(max=2 * math.pi * cfg["freq_max"])
            queries = q0[:, None, :] + self.lead[None, :, None] * freq[None, None, :]   # (B, nodes, I)
            if cfg["lead_value"] or cfg["lead_rel"]:
                base_nodes = base.view(batch, self.lead.shape[0], cfg["lead_stride"]).mean(dim=2)
                if cfg["lead_value"]:
                    queries = queries + base_nodes[:, :, None] * self.beta[None, None, :]
                if cfg["lead_rel"]:
                    rel = base_nodes / base_nodes.max(dim=1, keepdim=True).values.clamp(min=0.05)
                    queries = queries + rel[:, :, None] * self.gamma[None, None, :]
            z = self.dropout(self.fast_layer(queries, theta))                # (B, nodes, O)
            resid = self.readout(z).squeeze(-1) @ self.interp.T
        yn = base * (1.0 + resid) if cfg["residual"] == "mul" else base + resid
        return yn * feats["scale"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 528) scaled windows -> (B, 132) scaled forecasts."""
        return self.forward_features(self.features(x))

    def n_trainable(self) -> int:
        return daruan.count_parameters(self)


def predict_numpy(model: GatedQKANFWP2, x: np.ndarray, batch_size: int = 512) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            out.append(model(torch.from_numpy(np.asarray(x[start:start + batch_size], dtype=np.float32))))
    return torch.cat(out).numpy()
