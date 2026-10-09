#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reproduce.py  (v2)  --  Practical Limits and Numerical Reliability in ED of
Quantum Spin Chains.

Changes relative to v1 (see README_v2.md):
  * single script (no duplicated cell), CLI + __main__ guard
  * ONE timing experiment: numpy / scipy / scipy-subset / sparse are timed
    INTERLEAVED (rotating order) on the same matrices; every repetition is
    stored in raw_timings.csv; tables use median (+min, IQR, bootstrap CI)
  * v0-seed and tol sensitivity (bench_seeds.csv); K via LinearOperator,
    time via plain csr eigsh (no eigenvectors, no python matvec overhead)
  * fixed v0 everywhere; TFI closed form used as independent reference
  * sector dimensions verified by orbit enumeration; ground-state sector
    verified by <psi|R|psi>, <psi|P|psi>
  * entropy fits for BOTH branches with standard errors and dof
  * session_numbers.json: every number quoted in the paper, with a format tag
    (consumed by aggregate_numbers.py -> numbers.tex)
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse, csv, gc, json, math, platform, socket, subprocess, sys, time
from datetime import datetime
from math import comb, log, pi, sin
from pathlib import Path

import numpy as np
import scipy
import scipy.linalg as la
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh, LinearOperator
from threadpoolctl import threadpool_limits, threadpool_info

SPARSE_SEED = 12345          # v0 seed for all timing/K tables
NUM = {}                     # key -> {"v": value, "fmt": tag}
RAW = []                     # every single timing repetition
SESSION = ""


def num(key, v, fmt):
    NUM[key] = {"v": None if v is None else float(v), "fmt": fmt}


def banner(msg, ch="=", n=78):
    print(ch * n); print(msg); print(ch * n, flush=True)


def write_csv(d, name, header, rows):
    with open(d / "data" / name, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    print(f"  -> data/{name}  ({len(rows)} rows)", flush=True)


def write_text(d, rel, txt):
    p = d / rel; p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(txt, encoding="utf-8"); print(f"  -> {rel}")


# ----------------------------------------------------------------------------
# session log
# ----------------------------------------------------------------------------
def write_session_log(d, session_id, args):
    L = [f"session_id: {session_id}",
         f"timestamp_local: {datetime.now().isoformat()}",
         f"mode: {args.mode}",
         f"argv: {' '.join(sys.argv)}",
         f"python: {platform.python_version()}",
         f"platform: {platform.platform()}",
         f"processor: {platform.processor()}",
         f"hostname: {socket.gethostname()}",
         f"numpy: {np.__version__}", f"scipy: {scipy.__version__}",
         f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS')}"]
    try:
        L.append("git_commit: " + subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip())
    except Exception:
        L.append("git_commit: unavailable")
    L.append("threadpool_info:")
    for p in threadpool_info():
        L.append(f"  internal_api={p['internal_api']} num_threads={p['num_threads']} "
                 f"version={p.get('version')} filepath={p.get('filepath', 'unknown')}")
    try:
        import psutil
        bat = psutil.sensors_battery(); fr = psutil.cpu_freq()
        L += [f"cpu_logical: {psutil.cpu_count(logical=True)}",
              f"cpu_physical: {psutil.cpu_count(logical=False)}",
              f"ram_total_gb: {psutil.virtual_memory().total/1e9:.1f}",
              f"cpu_freq_mhz_now/max: {None if fr is None else (round(fr.current), round(fr.max))}",
              f"cpu_percent_before_run: {psutil.cpu_percent(interval=1.0)}"]
        if bat:
            L += [f"battery_percent: {bat.percent}", f"power_plugged: {bat.power_plugged}"]
    except ImportError:
        L.append("psutil: not installed (cpu freq / load not recorded)")
    (d / "session_log.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L), flush=True)


def warmup(seconds):
    if seconds <= 0:
        return
    A = np.random.default_rng(0).standard_normal((1200, 1200)); A = A + A.T
    t0 = time.time()
    while time.time() - t0 < seconds:
        np.linalg.eigvalsh(A)


# ----------------------------------------------------------------------------
# Hamiltonians (Pauli convention, open boundaries)
# ----------------------------------------------------------------------------
def tfi_hamiltonian(N, J=1.0, h=1.0):
    D = 1 << N; s = np.arange(D, dtype=np.int64)
    z = 1 - 2 * ((s[:, None] >> np.arange(N)) & 1)
    diag = -J * (z[:, :-1] * z[:, 1:]).sum(axis=1).astype(float)
    r = [np.arange(D, dtype=np.int64)]; c = [np.arange(D, dtype=np.int64)]; v = [diag]
    for i in range(N):
        r.append(np.arange(D, dtype=np.int64)); c.append(s ^ (1 << i)); v.append(np.full(D, -h))
    return csr_matrix((np.concatenate(v), (np.concatenate(r), np.concatenate(c))), shape=(D, D))


def spin_basis(N):
    s_all = np.arange(1 << N, dtype=np.int64); pop = np.zeros_like(s_all)
    for b in range(N):
        pop += (s_all >> b) & 1
    return s_all[pop == N // 2]


def heisenberg_hamiltonian(N, Delta=1.0):
    """XXZ chain, S^z=0 sector for even N (S^z=+1/2 for odd N), Pauli convention."""
    states = spin_basis(N); D = states.size; idx = np.arange(D, dtype=np.int64)
    diag = np.zeros(D); rows = [idx]; cols = [idx]; vals = [diag]
    for i in range(N - 1):
        a = (states >> i) & 1; b = (states >> (i + 1)) & 1; differs = a != b
        diag += Delta * np.where(differs, -1.0, 1.0)
        if differs.any():
            rr = idx[differs]; cc = np.searchsorted(states, states[differs] ^ (3 << i))
            rows.append(rr); cols.append(cc); vals.append(np.full(rr.size, 2.0))
    H = csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(D, D))
    return H, states


def tfi_parity_sector(N, J=1.0, h=1.0, p=+1):
    dim = 1 << (N - 1); mask = (1 << N) - 1; half = 1 << (N - 1)
    s = np.arange(dim, dtype=np.int64)
    z = 1 - 2 * ((s[:, None] >> np.arange(N)) & 1)
    diag = -J * (z[:, :-1] * z[:, 1:]).sum(axis=1).astype(float)
    r = [np.arange(dim, dtype=np.int64)]; c = [np.arange(dim, dtype=np.int64)]; v = [diag]
    for i in range(N):
        t = s ^ (1 << i); flip = t >= half
        tf = np.where(flip, mask ^ t, t); sign = np.where(flip, float(p), 1.0)
        r.append(np.arange(dim, dtype=np.int64)); c.append(tf.astype(np.int64)); v.append(-h * sign)
    return csr_matrix((np.concatenate(v), (np.concatenate(r), np.concatenate(c))), shape=(dim, dim))


def build(model, N):
    return tfi_hamiltonian(N) if model == "TFI" else heisenberg_hamiltonian(N)[0]


def tfi_closed(N):
    return 1.0 - 1.0 / sin(pi / (4 * N + 2))


# ----------------------------------------------------------------------------
# solvers / timing
# ----------------------------------------------------------------------------
def v0_for(D, seed):
    return np.random.default_rng(seed).standard_normal(D)


def ground_state(H, seed=7, dense_limit=512):
    D = H.shape[0]
    if D <= dense_limit:
        w, V = np.linalg.eigh(H.toarray()); return float(w[0]), V[:, 0]
    w, V = eigsh(H.tocsr(), k=1, which="SA", tol=0, v0=v0_for(D, seed))
    return float(w[0]), V[:, 0]


def low_spectrum(H, k=2, dense_limit=512):
    D = H.shape[0]
    if D <= dense_limit:
        return np.linalg.eigvalsh(H.toarray())[:k]
    return np.sort(eigsh(H.tocsr(), k=k, which="SA", tol=0, v0=v0_for(D, 7),
                         return_eigenvectors=False))


def count_matvecs(H, tol, seed):
    Hs = H.tocsr(); D = Hs.shape[0]; cnt = [0]
    def mv(x):
        cnt[0] += 1; return Hs @ x
    A = LinearOperator((D, D), matvec=mv, dtype=Hs.dtype)
    w = eigsh(A, k=1, which="SA", tol=tol, v0=v0_for(D, seed), return_eigenvectors=False)
    return cnt[0], float(w[0])


def sparse_callable(H, tol=0.0, seed=SPARSE_SEED):
    Hs = H.tocsr(); v0 = v0_for(Hs.shape[0], seed)
    return lambda: eigsh(Hs, k=1, which="SA", v0=v0, tol=tol, return_eigenvectors=False)


def time_interleaved(solvers, rounds, inner=None):
    """Round-robin with rotating start; one untimed warm-up per solver.
    Returns {name: [seconds,...]} in acquisition order."""
    names = list(solvers); out = {n: [] for n in names}; inner = inner or {}
    for n in names:
        solvers[n]()
    for r in range(rounds):
        k = r % len(names)
        for n in names[k:] + names[:k]:
            for _ in range(inner.get(n, 1)):
                gc.collect(); t0 = time.perf_counter(); solvers[n]()
                out[n].append(time.perf_counter() - t0)
    return out


def stats(v):
    v = np.asarray(v, float)
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return dict(n=len(v), min=v.min(), med=med, q1=q1, q3=q3, max=v.max())


def boot_ratio(a, b, rng, n=2000):
    a = np.asarray(a); b = np.asarray(b)
    ra = rng.choice(a, (n, len(a))); rb = rng.choice(b, (n, len(b)))
    r = np.median(ra, 1) / np.median(rb, 1)
    return [float(x) for x in np.percentile(r, [2.5, 97.5])]


def crossover(Ds, R):
    pts = sorted((D, r) for D, r in zip(Ds, R) if np.isfinite(r) and r > 0)
    last = float("nan")
    for (D1, r1), (D2, r2) in zip(pts, pts[1:]):
        if r1 < 1 <= r2:
            x = math.log(D1) + (0 - math.log(r1)) * (math.log(D2) - math.log(D1)) / (math.log(r2) - math.log(r1))
            last = math.exp(x)
    return last


def reps_for(D, R):
    return R[0] if D <= 1024 else (R[1] if D <= 2048 else R[2])


def sci_t(v, p=2):
    if v is None or not np.isfinite(v): return "---"
    if v == 0: return "$0$"
    m, e = f"{v:.{p}e}".split("e"); return "$" + m + "\\times10^{" + str(int(e)) + "}$"


def tt(v):
    return "---" if (v is None or not np.isfinite(v)) else f"${v:.3g}$"


def row(*cells):
    return " & ".join(str(c) for c in cells) + " \\\\\n"


# ----------------------------------------------------------------------------
# experiments
# ----------------------------------------------------------------------------
def exp_convergence(d, sizes):
    banner("EXP 1 -- E0/N convergence (dense only for D<=512, else Lanczos tol=0, fixed v0)")
    rows = []
    for model in ("TFI", "Heis", "XXZ0.5", "XXZ2.0"):
        for N in sizes:
            H = (tfi_hamiltonian(N) if model == "TFI" else
                 heisenberg_hamiltonian(N, {"Heis": 1.0, "XXZ0.5": 0.5, "XXZ2.0": 2.0}[model])[0])
            E0, _ = ground_state(H)
            rows.append([model, N, E0 / N, E0, tfi_closed(N) if model == "TFI" else ""])
            print(f"  {model:7s} N={N:2d} E0/N={E0/N:.10f}", flush=True)
    write_csv(d, "bench_convergence.csv", ["model", "N", "E0_per_site", "E0", "E0_closed_form"], rows)
    dev = max(abs(r[3] - r[4]) for r in rows if r[0] == "TFI")
    num("tfiEDvsClosedMaxDev", dev, "sci1")
    return rows


def exp_gap(d, sizes):
    banner("EXP 2 -- TFI gap")
    rows = []
    for N in sizes:
        w = low_spectrum(tfi_hamiltonian(N)); gap = float(w[1] - w[0])
        pred = 4 * sin(pi / (4 * N + 2))
        rows.append([N, gap, gap * N, pred, abs(gap - pred)]); print(f"  N={N:2d} gap*N={gap*N:.6f}")
    write_csv(d, "bench_gap.csv", ["N", "gap", "gap_times_N", "gap_formula", "abs_diff"], rows)
    num("gapFormulaMaxDev", max(r[4] for r in rows), "sci1")
    return rows


def exp_extrapolation(d, conv_rows):
    banner("EXP 3 -- extrapolation fits (2,3,4 param, PBC, pinned N^-4)")
    tfi = sorted((int(r[1]), float(r[2])) for r in conv_rows if r[0] == "TFI")
    Ns = np.array([n for n, _ in tfi], float); Eps = np.array([e for _, e in tfi])
    exact = -4 / pi; n4 = -(pi / 96 + 7 * pi ** 3 / 23040)
    ext, par = [], []
    for Nmax in sorted(set(int(n) for n in Ns if n >= 8)):
        m = Ns <= Nmax; nm, em = Ns[m], Eps[m]; k = int(m.sum())
        def fit(cols, y=em):
            A = np.vstack(cols).T; return np.linalg.lstsq(A, y, rcond=None)[0]
        one = np.ones(k)
        c2 = fit([one, 1 / nm]); c3 = fit([one, 1 / nm, 1 / nm ** 2]); cp = fit([one, 1 / nm ** 2])
        rel = lambda c: abs(c[0] - exact) / abs(exact)
        e4 = d4 = dp = None
        if k >= 4:
            cols4 = [one, 1 / nm, 1 / nm ** 2, 1 / nm ** 3]
            c4 = fit(cols4); e4 = rel(c4); d4 = float(c4[3])
            dp = float(fit(cols4, em - n4 / nm ** 4)[3])
        ext.append([Nmax, k, rel(c2), rel(c3), e4, rel(cp)])
        par.append([Nmax, k, c3[0], c3[1], c3[2], d4, dp])
        print(f"  Nmax={Nmax:2d} k={k} err3={rel(c3):.2e} err4={'--' if e4 is None else format(e4,'.2e')} "
              f"delta4={d4} pinned={dp}")
    write_csv(d, "bench_extrapolation.csv",
              ["N_max", "n_points", "err_2param_OBC", "err_3param_OBC", "err_4param_OBC", "err_2param_PBC"], ext)
    write_csv(d, "bench_extrapolation_params.csv",
              ["N_max", "n_points", "eps_inf_3p", "sigma_3p", "gamma_3p", "delta_4p", "delta_pinned"], par)
    last = ext[-1]; lp = par[-1]; t = pi / 48
    num("extrapNmax", last[0], "int")
    num("extrapErrTwoObc", last[2], "sci1"); num("extrapErrThreeObc", last[3], "sci1")
    num("extrapErrFourObc", last[4], "sci1"); num("extrapErrPbc", last[5], "sci1")
    num("extrapDeltaFree", lp[5], "f4"); num("extrapDeltaPinned", lp[6], "f4")
    num("extrapDeltaFreeOffPct", 100 * (lp[5] - t) / t, "f0"); num("extrapDeltaPinnedOffPct", 100 * (lp[6] - t) / t, "f0")
    num("extrapSigmaThree", lp[3], "f4"); num("extrapGammaThree", lp[4], "f4")
    return ext, par


def exp_numerical(d, heis_sizes, tfi_dense_sizes, tfi_sparse_sizes, heis_dense_max):
    banner("EXP 4 -- numerical agreement; TFI also vs closed form")
    rows = []
    for N in heis_sizes:
        if N > heis_dense_max: continue
        H, _ = heisenberg_hamiltonian(N); D = H.shape[0]
        Ed = float(np.linalg.eigvalsh(H.toarray())[0])
        w, V = eigsh(H.tocsr(), k=1, which="SA", tol=0, v0=v0_for(D, 7)); Es = float(w[0])
        res = float(np.linalg.norm(H @ V[:, 0] - Es * V[:, 0]))
        rows.append(["Heis", N, D, abs(Ed - Es), abs(Ed - Es) / abs(Ed), res, ""])
        print(f"  Heis N={N:2d} dE={abs(Ed-Es):.2e} resid={res:.2e}", flush=True)
    for N in sorted(set(tfi_dense_sizes) | set(tfi_sparse_sizes)):
        H = tfi_hamiltonian(N); D = H.shape[0]; cf = tfi_closed(N)
        w, V = eigsh(H.tocsr(), k=1, which="SA", tol=0, v0=v0_for(D, 7)); Es = float(w[0])
        res = float(np.linalg.norm(H @ V[:, 0] - Es * V[:, 0]))
        if N in tfi_dense_sizes:
            Ed = float(np.linalg.eigvalsh(H.toarray())[0]); dE = abs(Ed - Es); rel = dE / abs(Ed)
        else:
            dE = rel = float("nan")
        rows.append(["TFI", N, D, dE, rel, res, abs(Es - cf)])
        print(f"  TFI  N={N:2d} sparse-vs-closed={abs(Es-cf):.2e} resid={res:.2e}", flush=True)
    write_csv(d, "bench_numerical.csv",
              ["model", "N", "D", "abs_diff_dense_sparse", "rel_diff", "residual", "abs_err_sparse_vs_closed"], rows)
    fin = lambda x: isinstance(x, float) and np.isfinite(x)
    num("precMaxRel", max(r[4] for r in rows if fin(r[4])), "sci1")
    num("precMaxResid", max(r[5] for r in rows), "sci1")
    num("precTfiClosedMax", max(r[6] for r in rows if fin(r[6])), "sci1")
    return rows


def exp_timing(d, plan, R, sparse_inner, rng):
    banner("TIMING -- numpy/scipy/subset/sparse interleaved, all reps stored")
    srows, rrows = [], []
    big = {}
    for model, N, dense in plan:
        H = build(model, N); D = H.shape[0]; rounds = reps_for(D, R)
        solvers = {}
        if dense:
            Hd = H.toarray()
            solvers["numpy_full"] = lambda: np.linalg.eigvalsh(Hd)
            solvers["scipy_full"] = lambda: la.eigvalsh(Hd)
            solvers["scipy_subset"] = lambda: la.eigvalsh(Hd, subset_by_index=[0, 0])
        solvers["sparse"] = sparse_callable(H)
        T = time_interleaved(solvers, rounds, {"sparse": sparse_inner if dense else 1})
        for name, v in T.items():
            for i, t in enumerate(v):
                RAW.append([SESSION, "timing", model, N, D, name, i, t])
            s = stats(v)
            srows.append([model, N, D, name, s["n"], s["min"], s["med"], s["q1"], s["q3"], s["max"]])
        if dense:
            S = {k: stats(v) for k, v in T.items()}
            lo, hi = boot_ratio(T["numpy_full"], T["sparse"], rng)
            rrows.append([model, N, D,
                          S["numpy_full"]["med"] / S["sparse"]["med"], S["numpy_full"]["min"] / S["sparse"]["min"], lo, hi,
                          S["scipy_full"]["med"] / S["sparse"]["med"], S["scipy_full"]["min"] / S["sparse"]["min"],
                          S["numpy_full"]["med"] / S["scipy_full"]["med"],
                          1 - S["scipy_subset"]["med"] / S["scipy_full"]["med"],
                          (S["numpy_full"]["max"] - S["numpy_full"]["min"]) / S["numpy_full"]["min"],
                          (S["sparse"]["max"] - S["sparse"]["min"]) / S["sparse"]["min"],
                          (S["scipy_full"]["max"] - S["scipy_full"]["min"]) / S["scipy_full"]["min"]])
            r = rrows[-1]
            print(f"  {model:4s} N={N:2d} D={D:6d} n={rounds:2d} numpy_med={S['numpy_full']['med']:.4g} "
                  f"sparse_med={S['sparse']['med']:.4g} ratio_med={r[3]:.1f} [CI {lo:.3g},{hi:.3g}] "
                  f"scipy/numpy={1/r[9]:.2f}", flush=True)
            big[model] = (N, D, r, S)
        else:
            print(f"  {model:4s} N={N:2d} D={D:6d} sparse-only med={stats(T['sparse'])['med']:.4g}", flush=True)
        del H; gc.collect()
    write_csv(d, "bench_timing.csv", ["model", "N", "D", "solver", "n", "min_s", "median_s", "q1_s", "q3_s", "max_s"], srows)
    write_csv(d, "bench_ratios.csv",
              ["model", "N", "D", "ratio_numpy_med", "ratio_numpy_min", "ratio_numpy_ci_lo", "ratio_numpy_ci_hi",
               "ratio_scipy_med", "ratio_scipy_min", "numpy_over_scipy_med", "subset_saving_med",
               "spread_numpy", "spread_sparse", "spread_scipy"], rrows)
    # --- numbers
    for model, tag in (("TFI", "Tfi"), ("Heis", "Heis")):
        rs = [r for r in rrows if r[0] == model]
        if not rs: continue
        cx = crossover([r[2] for r in rs], [r[3] for r in rs]); num("cross" + tag, cx, "sci1")
        N, D, r, S = big[model]
        num("bigN" + tag, N, "int"); num("bigD" + tag, D, "int")
        num("ratioNumpy" + tag + "BigMed", r[3], "sci1"); num("ratioNumpy" + tag + "BigMin", r[4], "sci1")
        num("ratioNumpy" + tag + "BigCiLo", r[5], "sci1"); num("ratioNumpy" + tag + "BigCiHi", r[6], "sci1")
        num("ratioScipy" + tag + "BigMed", r[7], "sci1"); num("ratioScipy" + tag + "BigMin", r[8], "sci1")
        num("libFactor" + tag + "Big", r[9], "f2")
        num("spreadNumpy" + tag + "Big", 100 * r[11], "f1"); num("spreadSparse" + tag + "Big", 100 * r[12], "f1")
        num("spreadScipy" + tag + "Big", 100 * r[13], "f1")
    sv = [(r[2], 100 * r[10]) for r in rrows]
    big_sv = [s for D, s in sv if D >= 924]; small_sv = [s for D, s in sv if D <= 512]
    if big_sv: num("subsetSavingLargeLo", min(big_sv), "f0"); num("subsetSavingLargeHi", max(big_sv), "f0")
    if small_sv: num("subsetSavingSmallMax", max(small_sv), "f0")
    lf = [r[9] for r in rrows if r[2] >= 1024]
    if lf: num("libFactorLargeLo", min(lf), "f2"); num("libFactorLargeHi", max(lf), "f2")
    mids = [r for r in rrows if r[0] == "TFI" and 1500 <= r[2] <= 2500]
    if mids: num("ratioNumpyTfiMidMed", mids[0][3], "sci1")
    return srows, rrows


def exp_seeds(d, plan_sizes, n_seeds, reps):
    banner("SEEDS/TOL -- K and time vs v0 seed and tol (tol in 0,1e-10,1e-8)")
    rows = []; tols = (0.0, 1e-10, 1e-8)
    seeds = [SPARSE_SEED] + list(range(1000, 1000 + n_seeds))
    for model, N in plan_sizes:
        H = build(model, N).tocsr(); D = H.shape[0]
        for sd in seeds:
            for tol in tols:
                K, E = count_matvecs(H, tol, sd)
                f = sparse_callable(H, tol, sd); f()
                ts = []
                for _ in range(reps):
                    gc.collect(); t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0)
                    RAW.append([SESSION, "seeds", model, N, D, f"sparse_tol{tol:g}_seed{sd}", len(ts) - 1, ts[-1]])
                rows.append([model, N, D, H.nnz, sd, tol, K, min(ts), float(np.median(ts)), E])
        print(f"  {model} N={N} done", flush=True)
    write_csv(d, "bench_seeds.csv", ["model", "N", "D", "nnz", "seed", "tol", "K", "t_min_s", "t_med_s", "E0"], rows)
    # matvec table (TFI) + numbers
    mv = []
    names = {8: "Eight", 10: "Ten", 12: "Twelve", 14: "Fourteen", 16: "Sixteen"}
    for model, N in plan_sizes:
        if model != "TFI": continue
        sel = lambda tol: [r for r in rows if r[0] == model and r[1] == N and r[5] == tol]
        a, b, c = sel(0.0), sel(1e-10), sel(1e-8)
        K0 = np.array([r[6] for r in a]); K1 = np.array([r[6] for r in b]); K2 = np.array([r[6] for r in c])
        t0 = float(np.median([r[8] for r in a])); t1 = float(np.median([r[8] for r in b]))
        Kmed = float(np.median(K0)); nnz = a[0][3]
        mv.append([N, a[0][2], nnz, Kmed, K0.min(), K0.max(), float(np.median(K1)), K1.min(), K1.max(),
                   float(np.median(K2)), t0, t1, 1e9 * t0 / (Kmed * nnz),
                   [r[6] for r in a if r[4] == SPARSE_SEED][0]])
        if N in names:
            nm = names[N]
            num("kTfi" + nm + "Med", Kmed, "int"); num("kTfi" + nm + "Min", K0.min(), "int"); num("kTfi" + nm + "Max", K0.max(), "int")
            num("kTfi" + nm + "LooseMed", float(np.median(K1)), "int")
            num("tolSaving" + nm, 100 * (1 - t1 / t0), "f0")
    write_csv(d, "bench_matvec.csv",
              ["N", "D", "nnz", "K0_med", "K0_min", "K0_max", "K1e10_med", "K1e10_min", "K1e10_max",
               "K1e8_med", "t0_med_s", "t1e10_med_s", "ns_per_Knnz", "K0_seed12345"], mv)
    if mv:
        sav = [100 * (1 - m[11] / m[10]) for m in mv]
        num("tolSavingLo", min(sav), "f0"); num("tolSavingHi", max(sav), "f0")
        num("kAllMin", min(m[4] for m in mv), "int"); num("kAllMax", max(m[5] for m in mv), "int")
    return mv


def reverse_bits(x, N):
    r = np.zeros_like(x)
    for b in range(N):
        r |= ((x >> b) & 1) << (N - 1 - b)
    return r


def exp_symmetry(d, sizes, gs_max):
    banner("SYMMETRY -- Burnside formula vs orbit enumeration; GS sector via <R>,<P>")
    tr_R = lambda N: comb(N // 2, N // 4) if (N // 2) % 2 == 0 else 0
    rows = []; nnz_rows = []
    for N in sizes:
        states = spin_basis(N); Dsz = states.size; full = (1 << N) - 1
        R = reverse_bits(states, N); P = full ^ states; RP = full ^ R
        canon = np.minimum(np.minimum(states, R), np.minimum(P, RP)); reps = np.unique(canon)
        Rr = reverse_bits(reps, N); sR = Rr == reps; sP = (full ^ reps) == reps; sRP = (full ^ Rr) == reps
        dims = {}
        for r in (1, -1):
            for p in (1, -1):
                ok = (~sR | (r == 1)) & (~sP | (p == 1)) & (~sRP | (r * p == 1)); dims[(r, p)] = int(ok.sum())
        assert sum(dims.values()) == Dsz, "sector dims do not sum to D_Sz"
        formula = (comb(N, N // 2) + tr_R(N) + 2 ** (N // 2)) // 4
        assert formula == dims[(1, 1)], "Burnside formula != orbit enumeration"
        gs = (1 if N % 4 == 0 else -1); Dgs = dims[(gs, gs)]
        Reig = Peig = ""
        if N <= gs_max:
            _, psi = ground_state(heisenberg_hamiltonian(N)[0])
            Reig = float(psi @ psi[np.searchsorted(states, R)]); Peig = float(psi @ psi[np.searchsorted(states, P)])
            assert abs(Reig - gs) < 1e-6 and abs(Peig - gs) < 1e-6, f"GS sector claim fails at N={N}"
        rows.append([N, 2 ** N, Dsz, dims[(1, 1)], dims[(1, -1)], dims[(-1, 1)], dims[(-1, -1)], formula,
                     "++" if gs == 1 else "--", Dgs, Reig, Peig, 2 ** N / Dgs, Dsz / Dgs])
        print(f"  N={N:2d} D_Sz={Dsz:6d} D(++)={dims[(1,1)]:5d} D(--)={dims[(-1,-1)]:5d} GS<R>={Reig} <P>={Peig}", flush=True)
        if N <= 14:
            Hh = heisenberg_hamiltonian(N)[0]; Ht = tfi_hamiltonian(N) if N <= 12 else None
            nnz_rows.append(["Heis", N, Hh.shape[0], Hh.nnz, (N // 2 + 1) * Hh.shape[0], Hh.nnz == (N // 2 + 1) * Hh.shape[0]])
            if Ht is not None:
                nnz_rows.append(["TFI", N, Ht.shape[0], Ht.nnz, (N + 1) * 2 ** N, Ht.nnz == (N + 1) * 2 ** N])
    write_csv(d, "bench_symmetry.csv",
              ["N", "D_full", "D_Sz", "D_pp", "D_pm", "D_mp", "D_mm", "D_pp_formula", "gs_sector", "D_gs_sector",
               "gs_R_eigenvalue", "gs_P_eigenvalue", "ratio_full_over_gs", "ratio_Sz_over_gs"], rows)
    write_csv(d, "bench_nnz.csv", ["model", "N", "D", "nnz_measured", "nnz_formula", "match"], nnz_rows)
    num("nnzAllMatch", int(all(r[5] for r in nnz_rows)), "int")
    return rows


def entanglement_entropy(psi, N, l, basis):
    nL, nR = 1 << l, 1 << (N - l); M = np.zeros((nL, nR)); M[basis & (nL - 1), basis >> l] = psi
    p = np.linalg.svd(M, compute_uv=False) ** 2; p = p[p > 1e-30]
    return -float(np.sum(p * np.log(p)))


def linfit(x, y):
    n = len(x); A = np.vstack([x, np.ones(n)]).T; c = np.linalg.lstsq(A, y, rcond=None)[0]
    r = y - A @ c; dof = n - 2
    cov = (r @ r / dof) * np.linalg.inv(A.T @ A) if dof > 0 else np.full((2, 2), np.nan)
    return float(c[0]), float(np.sqrt(cov[0, 0])), dof


def exp_entropy(d, Ns):
    banner("EXP 7 -- half-chain entropy, c_eff for both N mod 4 branches")
    rows = []
    for N in Ns:
        H, basis = heisenberg_hamiltonian(N); _, psi = ground_state(H)
        S = entanglement_entropy(psi, N, N // 2, basis); rows.append([N, N % 4, S, log(N)])
        print(f"  N={N:2d} S={S:.6f}", flush=True)
    write_csv(d, "bench_entanglement.csv", ["N", "N_mod_4", "S_halfchain", "log_N"], rows)
    fits = []
    for br, tag in ((0, "Zero"), (2, "Two")):
        pts = [(r[3], r[2]) for r in rows if r[1] == br]
        if len(pts) >= 3:
            x = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts]); s, e, dof = linfit(x, y)
            fits.append([br, len(pts), s, e, dof, 6 * s, 6 * e])
            num("cEff" + tag, 6 * s, "f3"); num("cEff" + tag + "Err", 6 * e, "f3"); num("cEff" + tag + "Npts", len(pts), "int")
            print(f"  branch {br}: n={len(pts)} c_eff={6*s:.3f} +- {6*e:.3f} (dof={dof})")
    write_csv(d, "bench_entropy_fits.csv", ["branch_mod4", "n_points", "slope", "slope_se", "dof", "c_eff", "c_eff_se"], fits)
    return rows, fits


def exp_parity(d, Ns_energy, Ns_timing, R, rng):
    banner("PARITY -- energies vs closed form; interleaved timing full / P=+1 / P=-1")
    er = []
    for N in Ns_energy:
        H = tfi_parity_sector(N, p=+1); D = H.shape[0]
        E0 = (float(np.linalg.eigvalsh(H.toarray())[0]) if D <= 512 else
              float(eigsh(H.tocsr(), k=1, which="SA", tol=0, v0=v0_for(D, 7))[0][0]))
        cf = tfi_closed(N); er.append([N, D, E0, cf, abs(E0 - cf)])
        print(f"  N={N:2d} D={D:7d} E0={E0:.12f} |d|={abs(E0-cf):.2e}", flush=True)
    write_csv(d, "bench_tfi_parity.csv", ["N", "D_parity", "E0_numeric", "E0_closed_form", "abs_diff"], er)
    num("tfiParityMaxDev", max(r[4] for r in er), "sci1")
    tr = []
    for N in Ns_timing:
        Hf = tfi_hamiltonian(N); Hp = tfi_parity_sector(N, p=+1); Hm = tfi_parity_sector(N, p=-1)
        rounds = reps_for(Hf.shape[0], R)
        T = time_interleaved({"full": sparse_callable(Hf), "plus": sparse_callable(Hp), "minus": sparse_callable(Hm)}, rounds)
        for name, v in T.items():
            for i, t in enumerate(v): RAW.append([SESSION, "parity", "TFI", N, Hf.shape[0], name, i, t])
        S = {k: stats(v) for k, v in T.items()}
        Kf = count_matvecs(Hf, 0.0, SPARSE_SEED)[0]; Kp = count_matvecs(Hp, 0.0, SPARSE_SEED)[0]; Km = count_matvecs(Hm, 0.0, SPARSE_SEED)[0]
        ratio = S["full"]["med"] / S["plus"]["med"]; model = 2 * Kf / Kp
        tr.append([N, Hf.shape[0], Hp.shape[0], S["full"]["med"], S["plus"]["med"], S["minus"]["med"], ratio,
                   S["full"]["min"] / S["plus"]["min"], Kf, Kp, Km, model, ratio / model])
        print(f"  N={N:2d} ratio_med={ratio:.2f} model(2Kf/Kp)={model:.2f} meas/model={ratio/model:.2f} "
              f"K=({Kf},{Kp},{Km})", flush=True)
        del Hf, Hp, Hm; gc.collect()
    write_csv(d, "bench_tfi_parity_timing.csv",
              ["N", "D_full", "D_parity", "t_full_med", "t_plus_med", "t_minus_med", "ratio_med", "ratio_min",
               "K_full", "K_plus", "K_minus", "model_2Kf_over_Kp", "meas_over_model"], tr)
    if tr:
        num("paritySpeedupLo", min(r[6] for r in tr), "f1"); num("paritySpeedupHi", max(r[6] for r in tr), "f1")
        num("parityModelLo", min(r[11] for r in tr), "f1"); num("parityModelHi", max(r[11] for r in tr), "f1")
        num("parityMeasOverModelLo", min(r[12] for r in tr), "f2"); num("parityMeasOverModelHi", max(r[12] for r in tr), "f2")
    return er, tr


# ----------------------------------------------------------------------------
# tables
# ----------------------------------------------------------------------------
def emit_tables(d, conv, srows, rrows, mv, ext, par, numrows, sym, er, tr, ent_fits):
    banner("LaTeX tables")
    # timing: medians + min
    S = {(r[0], r[1], r[3]): r for r in srows}
    tex = "\\begin{tabular}{lrrrrrrr}\n\\toprule\nModel & $N$ & $D$ & $t^{\\rm med}_{\\rm numpy}$ & $t^{\\rm med}_{\\rm sparse}$ & ratio (med) & ratio (min) & 95\\% CI \\\\\n\\midrule\n"
    for r in rrows:
        m, N, D = r[0], r[1], r[2]
        tex += row(m, N, D, tt(S[(m, N, "numpy_full")][6]), tt(S[(m, N, "sparse")][6]), tt(r[3]), tt(r[4]),
                   "[" + f"{r[5]:.3g}" + ", " + f"{r[6]:.3g}" + "]")
    tex += "\\bottomrule\n\\end{tabular}\n"; write_text(d, "tables/table1_timing.tex", tex)
    # libraries
    tex = "\\begin{tabular}{lrrrrrr}\n\\toprule\nModel & $N$ & $D$ & $t_{\\rm numpy}$ & $t_{\\rm scipy}$ & $t_{\\rm scipy,subset}$ & numpy/scipy \\\\\n\\midrule\n"
    for r in rrows:
        m, N, D = r[0], r[1], r[2]
        tex += row(m, N, D, tt(S[(m, N, "numpy_full")][6]), tt(S[(m, N, "scipy_full")][6]),
                   tt(S[(m, N, "scipy_subset")][6]), tt(r[9]))
    tex += "\\bottomrule\n\\end{tabular}\n"; write_text(d, "tables/table2_libraries.tex", tex)
    # matvec
    tex = "\\begin{tabular}{rrrrrrrr}\n\\toprule\n$N$ & $D$ & nnz & $K(0)$ med [min,max] & $K(10^{-10})$ med & $t(0)$ med & $t(10^{-10})$ med & ns/($K\\,$nnz) \\\\\n\\midrule\n"
    for m in mv:
        tex += row(m[0], m[1], m[2], f"{m[3]:.0f} [{m[4]:.0f},{m[5]:.0f}]", f"{m[6]:.0f}", tt(m[10]), tt(m[11]), f"${m[12]:.1f}$")
    tex += "\\bottomrule\n\\end{tabular}\n"; write_text(d, "tables/table3_matvec.tex", tex)
    # precision
    tex = "\\begin{tabular}{lrrrrr}\n\\toprule\nModel & $N$ & $D$ & $|E^{\\rm dense}-E^{\\rm sparse}|$ & residual & $|E^{\\rm sparse}-E^{\\rm exact}|$ \\\\\n\\midrule\n"
    for r in numrows:
        tex += row(r[0], r[1], r[2], sci_t(r[3], 1), sci_t(r[5], 1), sci_t(r[6], 1) if r[6] != "" else "---")
    tex += "\\bottomrule\n\\end{tabular}\n"; write_text(d, "tables/table4_precision.tex", tex)
    # symmetry
    tex = "\\begin{tabular}{rrrrrrrr}\n\\toprule\n$N$ & $D_{\\rm full}$ & $D_{S^z}$ & $D_{(+,+)}$ & $D_{(-,-)}$ & GS sector & $D_{\\rm GS}$ & $D_{\\rm full}/D_{\\rm GS}$ \\\\\n\\midrule\n"
    for r in sym:
        tex += row(r[0], r[1], r[2], r[3], r[6], "$(" + r[8][0] + "," + r[8][1] + ")$", r[9], f"${r[12]:.2f}$")
    tex += "\\bottomrule\n\\end{tabular}\n"; write_text(d, "tables/table5_symmetry.tex", tex)
    # TFI surface
    eps = -4 / pi; sg = 1 - 2 / pi
    tex = "\\begin{tabular}{rrrrr}\n\\toprule\n$N$ & $E_0/N$ & $g(N)$ & $\\sigma-\\pi/(24N)$ & residual \\\\\n\\midrule\n"
    for r in conv:
        if r[0] != "TFI": continue
        N, e = r[1], r[2]; g = (e - eps) * N; a = sg - pi / (24 * N)
        tex += row(N, f"${e:.6f}$", f"${g:.5f}$", f"${a:.5f}$", f"${g-a:.5f}$")
    tex += row("$\\infty$", "$-4/\\pi$", "$0.36338$", "$0.36338$", "$0$") + "\\bottomrule\n\\end{tabular}\n"
    write_text(d, "tables/table6_tfi_surface.tex", tex)
    # extrapolation
    tex = "\\begin{tabular}{rrrrrr}\n\\toprule\n$N_{\\max}$ & points & 2p OBC & 3p OBC & 4p OBC & 2p PBC \\\\\n\\midrule\n"
    for r in ext:
        tex += row(r[0], r[1], sci_t(r[2]), sci_t(r[3]), "---" if r[4] is None else sci_t(r[4]), sci_t(r[5]))
    tex += "\\bottomrule\n\\end{tabular}\n"; write_text(d, "tables/table7_extrapolation.tex", tex)
    tex = "\\begin{tabular}{rrrrrrr}\n\\toprule\n$N_{\\max}$ & points & $\\epsilon_\\infty$ (3p) & $\\sigma$ (3p) & $\\gamma$ (3p) & $\\delta$ (4p free) & $\\delta$ (4p, $N^{-4}$ pinned) \\\\\n\\midrule\n"
    for r in par:
        tex += row(r[0], r[1], f"${r[2]:.6f}$", f"${r[3]:.6f}$", f"${r[4]:.6f}$",
                   "---" if r[5] is None else f"${r[5]:.5f}$", "---" if r[6] is None else f"${r[6]:.5f}$")
    tex += "\\bottomrule\n\\end{tabular}\n"; write_text(d, "tables/table8_fitparams.tex", tex)
    # Heisenberg surface
    eh = 1 - 4 * log(2)
    tex = "\\begin{tabular}{rrr}\n\\toprule\n$N$ & $E_0/N$ & $(E_0/N-\\epsilon_\\infty)N$ \\\\\n\\midrule\n"
    for r in conv:
        if r[0] == "Heis": tex += row(r[1], f"${r[2]:.6f}$", f"${(r[2]-eh)*r[1]:.6f}$")
    tex += "\\bottomrule\n\\end{tabular}\n"; write_text(d, "tables/table9_heis_surface.tex", tex)
    # parity
    tex = "\\begin{tabular}{rrrrrrrrr}\n\\toprule\n$N$ & $D_{\\rm full}$ & $t_{\\rm full}$ & $t_{P=+1}$ & ratio (med) & $K_{\\rm full}$ & $K_{P=+1}$ & model $2K_f/K_p$ & meas/model \\\\\n\\midrule\n"
    for r in tr:
        tex += row(r[0], r[1], tt(r[3]), tt(r[4]), f"${r[6]:.2f}$", r[8], r[9], f"${r[11]:.2f}$", f"${r[12]:.2f}$")
    tex += "\\bottomrule\n\\end{tabular}\n"; write_text(d, "tables/table10_parity_timing.tex", tex)
    tex = "\\begin{tabular}{rrrr}\n\\toprule\n$N$ & $D_{P=+1}$ & $E_0$ & $|E_0-E_0^{\\rm cf}|$ \\\\\n\\midrule\n"
    for r in er:
        tex += row(r[0], r[1], f"${r[2]:.9f}$", sci_t(r[4], 1))
    tex += "\\bottomrule\n\\end{tabular}\n"; write_text(d, "tables/table11_tfi_parity_energy.tex", tex)
    tex = "\\begin{tabular}{rrrrr}\n\\toprule\nbranch $N\\bmod4$ & points & slope $\\pm$ s.e. & dof & $c_{\\rm eff}\\pm$ s.e. \\\\\n\\midrule\n"
    for f in ent_fits:
        tex += row(f[0], f[1], f"${f[2]:.4f}\\pm{f[3]:.4f}$", f[4], f"${f[5]:.3f}\\pm{f[6]:.3f}$")
    tex += "\\bottomrule\n\\end{tabular}\n"; write_text(d, "tables/table12_entropy_fits.tex", tex)


# ----------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------
def emit_figures(d, srows, rrows, ext, ent_rows, gap_rows, tr, sizes_hilb):
    banner("Figures")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib missing -- figures skipped"); return
    F = d / "figures"
    fig, ax = plt.subplots(figsize=(7, 5))
    for model, mk, col in (("Heis", "o", "C0"), ("TFI", "s", "C3")):
        for solver, ls in (("numpy_full", "-"), ("sparse", "--")):
            rs = sorted([r for r in srows if r[0] == model and r[3] == solver], key=lambda r: r[2])
            if solver == "numpy_full" and not rs: continue
            D = [r[2] for r in rs]; med = np.array([r[6] for r in rs]); q1 = np.array([r[7] for r in rs]); q3 = np.array([r[8] for r in rs])
            ax.errorbar(D, med, yerr=[med - q1, q3 - med], fmt=mk + ls, color=col, ms=4, capsize=2,
                        label=f"{model} {'dense (numpy)' if solver=='numpy_full' else 'sparse'}")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("$D$"); ax.set_ylabel("median time (s), IQR bars")
    ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(F / "Fig1_crossover.pdf"); plt.close(fig)
    Ns = np.array(sizes_hilb); fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(Ns, 2.0 ** Ns, "o-", label="$2^N$"); Dz = np.array([comb(int(n), int(n) // 2) for n in Ns])
    ax.plot(Ns, Dz, "s-", label=r"$\binom{N}{N/2}$"); ax.plot(Ns, Dz / 4, "^--", label=r"$\binom{N}{N/2}/4$")
    ax.set_yscale("log"); ax.legend(); ax.grid(alpha=.3); ax.set_xlabel("$N$"); fig.tight_layout()
    fig.savefig(F / "Fig2_hilbert_scaling.pdf"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.5, 4.5)); Nm = [r[0] for r in ext]
    for j, lab in ((2, "2p OBC"), (3, "3p OBC"), (4, "4p OBC"), (5, "2p PBC")):
        pts = [(r[0], r[j]) for r in ext if r[j] is not None]
        ax.semilogy([p[0] for p in pts], [p[1] for p in pts], "o-", label=lab)
    ax.set_xlabel(r"$N_{\max}$"); ax.set_ylabel("relative error"); ax.legend(); ax.grid(alpha=.3, which="both")
    fig.tight_layout(); fig.savefig(F / "Fig3_extrapolation.pdf"); plt.close(fig)
    if tr:
        fig, ax = plt.subplots(figsize=(6.5, 4.5)); x = [r[0] for r in tr]
        ax.plot(x, [r[6] for r in tr], "o-", label="measured (median)"); ax.plot(x, [r[11] for r in tr], "s--", label=r"model $2K_f/K_p$")
        ax.set_xlabel("$N$"); ax.set_ylabel("full / parity speed-up"); ax.legend(); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(F / "Fig4_tfi_parity.pdf"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5))
    for br, mk in ((0, "o"), (2, "s")):
        pts = [(r[3], r[2]) for r in ent_rows if r[1] == br]
        if pts: ax.plot([p[0] for p in pts], [p[1] for p in pts], mk + "-", label=f"$N\\equiv{br}$ (mod 4)")
    ax.set_xlabel(r"$\ln N$"); ax.set_ylabel(r"$S_{N/2}$"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(F / "Fig5_entanglement.pdf"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([r[0] for r in gap_rows], [r[2] for r in gap_rows], "o-"); ax.axhline(pi, ls="--", color="gray")
    ax.set_xlabel("$N$"); ax.set_ylabel(r"$N\,\Delta E$"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(F / "Fig6_gap_convergence.pdf"); plt.close(fig)
    print("  figures written")


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
PRESETS = {
    "fast":  dict(conv=(4, 6, 8, 10), tfi_dense=range(4, 10), tfi_sp=(), heis_dense=range(4, 11), heis_sp=(),
                  R=(5, 3, 3), inner=3, seeds=2, seed_reps=2, seed_sizes=[("TFI", 6), ("TFI", 8), ("TFI", 10)],
                  ent=range(4, 15), gap=(4, 6, 8, 10), par_e=(10, 12), par_t=(8, 10), heis_prec=10,
                  tfi_prec_dense=(4, 6, 8, 10), tfi_prec_sp=(10, 12), sym=(4, 6, 8, 10), gs_max=10, hilb=(4, 6, 8, 10, 12), warm=0),
    "quick": dict(conv=(4, 6, 8, 10, 12, 14), tfi_dense=range(4, 12), tfi_sp=(12, 14), heis_dense=range(4, 14), heis_sp=(14,),
                  R=(10, 5, 3), inner=5, seeds=5, seed_reps=3, seed_sizes=[("TFI", n) for n in (8, 10, 12, 14)],
                  ent=range(4, 15), gap=(4, 6, 8, 10, 12, 14, 16), par_e=(12, 14, 16), par_t=(10, 12, 14),
                  heis_prec=12, tfi_prec_dense=(4, 6, 8, 10), tfi_prec_sp=(12, 14), sym=(4, 6, 8, 10, 12, 14), gs_max=12,
                  hilb=(4, 6, 8, 10, 12, 14, 16), warm=30),
    "full":  dict(conv=(4, 6, 8, 10, 12, 14, 16), tfi_dense=range(4, 13), tfi_sp=(14, 16), heis_dense=range(4, 15), heis_sp=(16,),
                  R=(15, 9, 5), inner=5, seeds=10, seed_reps=3, seed_sizes=[("TFI", n) for n in (8, 10, 12, 14, 16)] + [("Heis", 12), ("Heis", 14)],
                  ent=range(4, 19), gap=(4, 6, 8, 10, 12, 14, 16, 18), par_e=(12, 14, 16, 17, 18, 19, 20), par_t=(10, 12, 14, 16),
                  heis_prec=14, tfi_prec_dense=(4, 6, 8, 10, 12), tfi_prec_sp=(14, 16), sym=(4, 6, 8, 10, 12, 14, 16), gs_max=14,
                  hilb=(4, 6, 8, 10, 12, 14, 16, 18, 20), warm=60),
}


def main(argv=None):
    global SESSION
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=PRESETS, default="full")
    ap.add_argument("--out", default="results"); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warmup", type=float, default=None, help="seconds of CPU warm-up before timing")
    a = ap.parse_args(argv); P = PRESETS[a.mode]; t_start = time.time()
    SESSION = datetime.now().strftime("%Y%m%d_%H%M%S"); d = Path(a.out) / SESSION
    for sub in ("data", "tables", "figures"): (d / sub).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    with threadpool_limits(limits=1):
        write_session_log(d, SESSION, a)
        banner("Sanity")
        E, _ = ground_state(heisenberg_hamiltonian(4)[0]); T, _ = ground_state(tfi_hamiltonian(4))
        assert abs(E + 6.4641016151) < 1e-8 and abs(T + 4.7587704831) < 1e-8; print("  ok")
        conv = exp_convergence(d, P["conv"]); gap = exp_gap(d, P["gap"])
        ext, par = exp_extrapolation(d, conv)
        numrows = exp_numerical(d, range(4, P["heis_prec"] + 1, 2), P["tfi_prec_dense"], P["tfi_prec_sp"], P["heis_prec"])
        sym = exp_symmetry(d, P["sym"], P["gs_max"])
        ent_rows, ent_fits = exp_entropy(d, P["ent"])
        warmup(P["warm"] if a.warmup is None else a.warmup)
        plan = [("Heis", N, True) for N in P["heis_dense"]] + [("Heis", N, False) for N in P["heis_sp"]] + \
               [("TFI", N, True) for N in P["tfi_dense"]] + [("TFI", N, False) for N in P["tfi_sp"]]
        srows, rrows = exp_timing(d, plan, P["R"], P["inner"], rng)
        mv = exp_seeds(d, P["seed_sizes"], P["seeds"], P["seed_reps"])
        er, tr = exp_parity(d, P["par_e"], P["par_t"], P["R"], rng)
        write_csv(d, "raw_timings.csv", ["session", "experiment", "model", "N", "D", "solver", "rep", "seconds"], RAW)
        emit_tables(d, conv, srows, rrows, mv, ext, par, numrows, sym, er, tr, ent_fits)
        emit_figures(d, srows, rrows, ext, ent_rows, gap, tr, P["hilb"])
    elapsed = (time.time() - t_start) / 60; num("runtimeMinutes", elapsed, "f0")
    (d / "session_numbers.json").write_text(json.dumps({"session": SESSION, "mode": a.mode, "numbers": NUM}, indent=1))
    with open(d / "session_log.txt", "a", encoding="utf-8") as f:
        f.write(f"elapsed_minutes: {elapsed:.1f}\nfinished_local: {datetime.now().isoformat()}\n")
    banner(f"DONE in {elapsed:.1f} min -- {d.resolve()}")
    return d


if __name__ == "__main__":
    main()