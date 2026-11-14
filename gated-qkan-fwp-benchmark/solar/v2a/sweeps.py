"""Named sweeps for run_cv.py: ``SWEEPS[name] = [(config_name, model_overrides, train_overrides), ...]``."""

SWEEPS = {}

# -- s1: first look.  Does the backbone train, and how do the two decoders compare? ----------------
SWEEPS["s1"] = [
    ("basis", dict(), dict()),
    ("basis_skip", dict(skip=True), dict()),
    ("leadtime", dict(decoder="leadtime"), dict()),
    ("basis_nonorm", dict(norm="none"), dict()),
]

# -- s2: regularisation study (seed 0, CV folds): what controls the overfitting seen in s1? ---------
_LT = dict(decoder="leadtime")
SWEEPS["s2"] = [
    ("L0", dict(**_LT), dict()),
    ("L_small", dict(**_LT, q=6, h=8, I=4, O=6), dict()),
    ("L_toknoise", dict(**_LT), dict(tok_noise=0.2)),
    ("L_warp", dict(**_LT), dict(warp=0.2)),
    ("L_warp_toknoise", dict(**_LT), dict(warp=0.2, tok_noise=0.2)),
    ("L_ep30", dict(**_LT), dict(epochs=30)),
    ("L_wd0.3", dict(**_LT), dict(wd=0.3)),
    ("L_shortmem", dict(**_LT, gate_horizons=(2.0, 5.0, 11.0)), dict()),
    ("L_reps1", dict(**_LT, reps=1), dict()),
    ("L_1gate", dict(**_LT, n_gates=1, gate_horizons=(11.0,)), dict()),
    ("B_warp", dict(), dict(warp=0.2)),
    ("B_toknoise", dict(), dict(tok_noise=0.2)),
    ("B_warp_toknoise_small", dict(q=6, h=8, I=4, O=6), dict(warp=0.2, tok_noise=0.2)),
]

# -- s3: hybrid.  Closed-form ridge baseline (linear part) + QKAN-FWP residual starting at zero. -----
_H = dict(decoder="leadtime", baseline="ridge", zero_init=True)
_T = dict(epochs=40, lr=1e-3)
SWEEPS["s3"] = [
    ("H_mse", dict(**_H), dict(**_T)),
    ("H_peak0.3", dict(**_H), dict(**_T, peak_w=0.3)),
    ("H_peak1", dict(**_H), dict(**_T, peak_w=1.0)),
    ("H_mul_peak0.3", dict(**_H, residual="mul"), dict(**_T, peak_w=0.3)),
    ("H_peak0.3_warp", dict(**_H), dict(**_T, peak_w=0.3, warp=0.2)),
    ("H_peak0.3_time", dict(**_H), dict(**_T, peak_w=0.3, time_w=0.003)),
    ("HB_peak0.3", dict(baseline="ridge", zero_init=True), dict(**_T, peak_w=0.3)),
]

# -- s4: calibration-style fast programmer.  Query per lead time = (baseline value at that lead, lead time). ---
_C = dict(decoder="leadtime", baseline="ridge", zero_init=True, lead_value=True, theta0=True,
          I=4, O=6, n_freq_init=(0.0, 0.25, 0.5, 1.0))
_TC = dict(epochs=40, lr=2e-3, peak_w=0.3, time_w=0.003)
SWEEPS["s4"] = [
    ("C_static", dict(**_C, program=False, query="static"), dict(**_TC)),
    ("C_static_mse", dict(**_C, program=False, query="static"), dict(epochs=40, lr=2e-3)),
    ("C_prog", dict(**_C, query="static"), dict(**_TC)),
    ("C_prog_sumq", dict(**_C), dict(**_TC)),
    ("C_static_peak1", dict(**_C, program=False, query="static"), dict(epochs=40, lr=2e-3, peak_w=1.0, time_w=0.01)),
    ("C_static_notime", dict(**_C, program=False, query="static"), dict(epochs=40, lr=2e-3, peak_w=0.3)),
]

# -- s5: calibrator with out-of-fold baselines, relative-height input, multiplicative residual. ------
_C5 = dict(decoder="leadtime", baseline="ridge", zero_init=True, lead_value=True, lead_rel=True, theta0=True,
           I=4, O=6, n_freq_init=(0.0, 0.25, 0.5, 1.0), residual="mul")
_ST = dict(program=False, query="static")
_T5 = dict(epochs=40, lr=2e-3, oof=True)
SWEEPS["s5"] = [
    ("S_mse", dict(**_C5, **_ST), dict(**_T5)),
    ("S_p0.3", dict(**_C5, **_ST), dict(**_T5, peak_w=0.3)),
    ("S_p1", dict(**_C5, **_ST), dict(**_T5, peak_w=1.0)),
    ("S_p1_t0.01", dict(**_C5, **_ST), dict(**_T5, peak_w=1.0, time_w=0.01)),
    ("S_p3_t0.01", dict(**_C5, **_ST), dict(**_T5, peak_w=3.0, time_w=0.01)),
    ("P_p1_t0.01", dict(**_C5, query="static", prog_gain=0.1), dict(**_T5, peak_w=1.0, time_w=0.01)),
    ("P_p1_t0.01_tn", dict(**_C5, query="static", prog_gain=0.1), dict(**_T5, peak_w=1.0, time_w=0.01, tok_noise=0.3)),
    ("S_p1_inbase", dict(**_C5, **_ST), dict(epochs=40, lr=2e-3, peak_w=1.0)),
    ("S_p1_add", dict(**dict(_C5, residual="add"), **_ST), dict(**_T5, peak_w=1.0)),
]

# ======================================================================================================
# CVVP sweeps (run_cvvp.py): (config_name, model overrides, cvvp overrides, ensemble members)
# ======================================================================================================
CVVP_SWEEPS = {}
_V = dict(O=16, I=8, theta0=True)
CVVP_SWEEPS["v1"] = [
    ("rand", dict(**_V), dict(steps=0), 1),
    ("t20", dict(**_V), dict(steps=20), 1),
    ("t60", dict(**_V), dict(steps=60), 1),
    ("t60_lam0.3", dict(**_V), dict(steps=60, lam=0.3), 1),
    ("t60_small", dict(O=12, I=6, q=6, h=8, theta0=True), dict(steps=60), 1),
    ("t60_noprog", dict(**_V, program=False), dict(steps=60), 1),
    ("t60_shortmem", dict(**_V, gate_horizons=(2.0, 5.0, 11.0)), dict(steps=60), 1),
    ("t60_O32", dict(O=32, I=8, theta0=True), dict(steps=60), 1),
]

CVVP_SWEEPS["v2"] = [
    ("fwd_t60", dict(**_V), dict(steps=60, scheme="forward"), 1),
    ("fwd_t30", dict(**_V), dict(steps=30, scheme="forward"), 1),
    ("fwd_t60_lam0.3", dict(**_V), dict(steps=60, scheme="forward", lam=0.3), 1),
    ("fwd_t60_b10", dict(**_V), dict(steps=60, scheme="forward", blocks=10), 1),
]

# -- v3: candidates (2-member ensembles) and QKAN ablations, seeds 0-1(-2) on all folds --------------
_F60 = dict(steps=60, scheme="forward")
CVVP_SWEEPS["v3"] = [
    ("E2_fwd60", dict(**_V), dict(**_F60), 2),
    ("E2_rand", dict(**_V), dict(steps=0), 2),
    ("E2_two60", dict(**_V), dict(steps=60), 2),
    ("abl_linear_both", dict(**_V, slow="linear", fast="linear"), dict(**_F60), 2),
    ("abl_fast_linear", dict(**_V, fast="linear"), dict(**_F60), 2),
    ("abl_slow_linear", dict(**_V, slow="linear"), dict(**_F60), 2),
    ("abl_no_program", dict(**_V, program=False), dict(**_F60), 2),
    ("E4_small_fwd60", dict(O=8, I=6, q=6, h=8, theta0=True), dict(**_F60), 4),
    ("E2_fwd60_nonorm", dict(**_V, norm="none"), dict(**_F60), 2),
]

# -- v4: linear skip.  Read-out = ridge on [standardised recent summary (11), z_gain * QKAN-FWP features] -------
_SK = dict(skip=True, lam=0.03)
CVVP_SWEEPS["v4"] = [
    ("K_rand_g0.3", dict(**_V), dict(steps=0, z_gain=0.3, **_SK), 1),
    ("K_rand_g1", dict(**_V), dict(steps=0, z_gain=1.0, **_SK), 1),
    ("K_fwd60_g0.3", dict(**_V), dict(steps=60, scheme="forward", z_gain=0.3, **_SK), 1),
    ("K_fwd60_learn", dict(**_V), dict(steps=60, scheme="forward", z_gain=0.3, learn_gain=True, **_SK), 1),
    ("K_two60_learn", dict(**_V), dict(steps=60, z_gain=0.3, learn_gain=True, **_SK), 1),
    ("K_fwd60_learn_O8", dict(O=8, I=6, theta0=True), dict(steps=60, scheme="forward", z_gain=0.3, learn_gain=True, **_SK), 1),
]

# -- v5: CLOCK query.  The fast programmer is queried with the cycle clock (I = 2 learnable frequencies); read-out =
#        ridge on [recent summary, z_gain * QKAN-FWP features].  Small programmer (q=6, h=8, O=4, reps=1). ---------------
_CK = dict(query="clock", O=4, I=2, reps=1, q=6, h=8, theta0=True)
_SKC = dict(skip=True, lam=0.03, z_gain=0.3)
_TR = dict(steps=60, lr=1e-2, learn_gain=True)
CVVP_SWEEPS["v5"] = [
    ("CK_min_static_rand", dict(**_CK, program=False), dict(steps=0, **_SKC), 1),
    ("CK_min_static", dict(**_CK, program=False), dict(**_TR, **_SKC), 1),
    ("CK_min_prog", dict(**_CK), dict(**_TR, **_SKC), 1),
    ("CK_min_prog_fwd", dict(**_CK), dict(**_TR, **_SKC, scheme="forward"), 1),
    ("CK_fou_static", dict(**_CK, program=False, clock="fourier"), dict(**_TR, **_SKC), 1),
    ("CK_fou_prog", dict(**_CK, clock="fourier"), dict(**_TR, **_SKC), 1),
]

# -- v6: static-vs-programmed clock variants with random / trained angles (cheap; 5 seeds, all folds) -----------------
_CKS = dict(**_CK, program=False)
_R = dict(steps=0, skip=True, lam=0.03)
CVVP_SWEEPS["v6"] = [
    ("S_rand_g0.3", dict(**_CKS), dict(**_R, z_gain=0.3), 1),
    ("S_rand_g0.5", dict(**_CKS), dict(**_R, z_gain=0.5), 1),
    ("S_rand_g0.3_m12", dict(**_CKS, clock_margin=12), dict(**_R, z_gain=0.3), 1),
    ("S_rand_g0.3_O8", dict(**dict(_CKS, O=8)), dict(**_R, z_gain=0.3), 1),
    ("S_rand_g0.3_f1", dict(**dict(_CKS, I=1, clock_freqs=(1.0,))), dict(**_R, z_gain=0.3), 1),
    ("S_rand_g0.3_reps2", dict(**dict(_CKS, reps=2)), dict(**_R, z_gain=0.3), 1),
    ("S_rand_g0.3_E4", dict(**_CKS), dict(**_R, z_gain=0.3), 4),
    ("P_rand_g0.3", dict(**_CK), dict(**_R, z_gain=0.3), 1),
    ("P_rand_g0.3_pg0.1", dict(**_CK, prog_gain=0.1), dict(**_R, z_gain=0.3), 1),
    ("P_rand_g0.3_pg0.1_E4", dict(**_CK, prog_gain=0.1), dict(**_R, z_gain=0.3), 4),
    ("P_rand_g0.3_E4", dict(**_CK), dict(**_R, z_gain=0.3), 4),
    ("S_rand_g0.3_fourier", dict(**_CKS, clock="fourier"), dict(**_R, z_gain=0.3), 1),
    ("S_rand_g0.2", dict(**_CKS), dict(**_R, z_gain=0.2), 1),
    ("S_rand_g0.3_sig12", dict(**_CKS, clock_sigma=12.0), dict(**_R, z_gain=0.3), 1),
    ("S_rand_g0.3_f3", dict(**dict(_CKS, I=3, clock_freqs=(1.0, 0.5, 2.0))), dict(**_R, z_gain=0.3), 1),
]

# -- v7: can a TRAINED programmer be made harmless?  (programming starts small: prog_gain 0.1) --------------------------
_CKP = dict(**_CK, prog_gain=0.1)
_SK3 = dict(skip=True, lam=0.03, z_gain=0.3, learn_gain=True)
CVVP_SWEEPS["v7"] = [
    ("P_t30_lr3e-3", dict(**_CKP), dict(steps=30, lr=3e-3, **_SK3), 1),
    ("P_t60_lr1e-3", dict(**_CKP), dict(steps=60, lr=1e-3, **_SK3), 1),
    ("P_t60_lr3e-3_fwd", dict(**_CKP), dict(steps=60, lr=3e-3, scheme="forward", **_SK3), 1),
    ("S_t60_lr1e-2", dict(**_CKS), dict(steps=60, lr=1e-2, **_SK3), 1),
    ("S_t60_lr1e-2_sig12", dict(**_CKS, clock_sigma=12.0), dict(steps=60, lr=1e-2, **_SK3), 1),
]
