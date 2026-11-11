"""
run_local_lambda2_3.py -- reproduces the lambda = 2/3 results of the paper.

Open cascade, geometric kernel lambda = 2/3, coagulation or fragmentation.
Eight independent replicas run in parallel and save their state after every
pass, so a run can be continued.  Prediction: dN/dm ~ m^-p with p = 11/6.

    python run_local_lambda2_3.py --process coag --add-hours 1
    python run_local_lambda2_3.py --process frag --add-hours 1
    python run_local_lambda2_3.py --process coag --stack-only      # redraw, no run

See README.md.
"""

import argparse
import os
import shutil
import sys
import time
import traceback
import warnings

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message=".*gen_max.*")
warnings.filterwarnings("ignore", message=".*NO honest bodies.*")

N_SS = 1.0e6
def mass_edges(g):
    lo = np.log10(min(g["m_inj"], g["m_sink"])) - 1.0
    hi = np.log10(max(g["m_inj"], g["m_sink"])) + 1.0
    return 10.0 ** np.round(np.arange(lo, hi + 0.05, 0.1), 6)

EDGES = 10.0 ** np.round(np.arange(-2.0, 7.05, 0.1), 6)
GEN_MAX = 128
PROBE_EVENTS = 200_000
SNAPSHOTS = 200
REBIN_DEFAULT = 1
REACH_TOL = 0.05

ISO_DT = 0.05
ISO_SPAN = 1.6
ISO_DLOGM = 0.25
ISO_MIN_COUNTS = 1.0e4
ISO_START_SINK = 0

KERNEL_NAME = "kernel_geometric"
LAMBDA = 2.0 / 3.0
BETA_TH = 0.5 * (1.0 + LAMBDA)
ALPHA_TH = -0.5 * (3.0 + LAMBDA)
B_TH = 2.0 / (1.0 - LAMBDA)
P_TH = -ALPHA_TH
COMP = P_TH

def _kernel(BF):
    return getattr(BF, KERNEL_NAME)

def box_mass(g):
    """Box mass of a steady spectrum."""
    lo = min(g["m_inj"], g["m_sink"])
    hi = max(g["m_inj"], g["m_sink"])
    return N_SS * (P_TH - 1.0) / (2.0 - P_TH) * lo ** (P_TH - 1.0) \
        * hi ** (2.0 - P_TH)

def _here():
    return os.path.dirname(os.path.abspath(__file__))

def _imports():
    d = _here()
    if d not in sys.path:
        sys.path.insert(0, d)
    import masscascade as BF
    import analysis as AN
    return BF, AN

M_FLOOR_COAG = 0.1
M_FLOOR_FRAG = 1.0
M_TOP = 1.0e6

CORE_LO, CORE_HI = 30.0, 1.0e4

def geometry(process):
    """Injection mass and wall mass."""
    if process == "coagulation":
        return dict(m_inj=M_FLOOR_COAG, m_sink=M_TOP)
    return dict(m_inj=M_TOP, m_sink=M_FLOOR_FRAG)

def core_window(g):
    """Core window where results are measured."""
    lo = min(g["m_inj"], g["m_sink"])
    hi = max(g["m_inj"], g["m_sink"])
    return lo * 10.0, hi / 30.0

def _q_of(BF, g, process):
    """Injection rate."""
    K = _kernel(BF)
    if process == "coagulation":
        return 0.5 * K(g["m_inj"], g["m_inj"]) * N_SS ** 2
    M_box = box_mass(g)
    return 0.5 * K(g["m_sink"], g["m_sink"]) * N_SS ** 2 \
        * (M_box / (N_SS * g["m_inj"]))

def _tc_of(BF, g, process):
    """Collision time at the injection end."""
    m = g["m_inj"] if process == "coagulation" else g["m_sink"]
    return 1.0 / (_kernel(BF)(m, m) * N_SS)

def iso_age_grid(BF, g, process):
    """Age grid for isochrones."""
    tc = _tc_of(BF, g, process)
    R = max(g["m_inj"], g["m_sink"]) / min(g["m_inj"], g["m_sink"])
    t_max = ISO_SPAN * B_TH * (R ** (1.0 - BETA_TH) - 1.0) * tc
    n = max(int(round(t_max / (ISO_DT * tc))), 8)
    return np.arange(n + 1, dtype=float) * (ISO_DT * tc)

def frag_n0(g):
    """Bodies for a cold fragmentation start."""
    return max(int(round(box_mass(g) / max(g["m_inj"], g["m_sink"]))), 2)

def frag_ic(g, alpha=ALPHA_TH):
    """Seeded power-law start for fragmentation."""
    return {"steady": {"alpha": float(alpha), "m_lo": min(g["m_sink"], g["m_inj"]),
                       "m_hi": max(g["m_sink"], g["m_inj"]), "N": int(N_SS)}}

def _sim_kwargs(BF, g, f, process, ic, max_events, stride, seed, verbose=False,
                q=None):
    if q is None:
        q = _q_of(BF, g, process)
    kw = dict(
        process=process, system="open", kernel=_kernel(BF),
        edges=mass_edges(g),
        ic=ic, injection_rate=q, injection_mass=g["m_inj"],
        sink_mass=g["m_sink"],
        snapshot_mode="events", snapshot_stride=stride, max_events=int(max_events),
        track_generations=True, gen_max=GEN_MAX, track_waiting=False,
        rng=np.random.default_rng(seed), verbose=bool(verbose),
        age_rule="mass_weighted")
    if process == "coagulation":
        kw["coag_min_ratio"] = float(f)
        kw["iso_age_edges"] = iso_age_grid(BF, g, process)
        kw["iso_start_sink"] = ISO_START_SINK
    else:
        kw["frag_min_ratio"] = float(f)
    return kw

def _gen_of(r):
    """Generation histogram of one pass, raw counts."""
    if "gen_counts" not in r:
        return None
    w = float(r.get("meta", {}).get("weight", 1.0)) or 1.0
    tau = np.asarray(r["gen_tau"], float)
    num = np.asarray(r["gen_num"], float)
    return dict(counts=np.asarray(r["gen_counts"], float) / w,
                num=num, tau_sum=np.nan_to_num(tau, nan=0.0) * num,
                snaps=float(r["gen_snapshots"]))

def _snap_weighted(dts, snaps):
    """Mean snapshot spacing over passes."""
    d = np.asarray(dts, float)
    w = np.asarray(snaps, float)
    ok = np.isfinite(d) & np.isfinite(w) & (w > 0) & (d > 0)
    if not np.any(ok):
        return np.nan
    return float(np.sum(d[ok] * w[ok]) / np.sum(w[ok]))

def _iso_of(r):
    """Isochrone histogram of one pass, raw counts."""
    if "iso_counts" not in r:
        return None
    w = float(r.get("meta", {}).get("weight", 1.0)) or 1.0
    t = np.asarray(r["t"], float)
    dt = float(np.median(np.diff(t))) if t.size > 1 else np.nan
    return dict(counts=np.asarray(r["iso_counts"], float) / w,
                edges=np.asarray(r["iso_age_edges"], float),
                snaps=float(r["iso_snapshots"]), dt_snap=dt)

def _iso_overflow(r, iso):
    """Population fraction outside the age grid."""
    try:
        live = np.asarray(r["live"], float)
        n = int(iso["snaps"])
        if n <= 0 or live.size < n:
            return np.nan
        expect = float(live[-n:].sum())
        return float(1.0 - iso["counts"].sum() / max(expect, 1e-300))
    except Exception:
        return np.nan

def _save_atomic(AN, run, path, drop_particles, analysis=None):
    """Write a checkpoint atomically."""
    d = os.path.dirname(path) or "."
    base = os.path.basename(path).replace(".npz", "")
    tmp = AN.add_last_run(run, base + ".__writing__", drop_particles=drop_particles,
                          runs_dir=d, analysis=analysis)
    os.replace(tmp, path)
    return path

def calibrate_q(process, f, rounds=3, burst=15_000_000, seed=31337):
    """Measure the injection rate holding the population."""
    BF, _ = _imports()
    g = geometry(process)
    if process == "coagulation":
        return _q_of(BF, g, process), []
    q = _q_of(BF, g, process)
    hist = []
    for it in range(rounds):
        r = BF.simulate(**_sim_kwargs(BF, g, f, process,
                                      {"m": g["m_inj"], "N": frag_n0(g)},
                                      burst, max(burst // 60, 1), seed + it,
                                      q=q))
        lv = np.asarray(r["live"], float)
        n0 = float(lv[0])
        n1 = float(np.mean(lv[lv.size // 2:]))
        ratio = n1 / max(N_SS, 1.0)
        hist.append((q, n0, n1, float(r["n_out"][-1])))
        if 0.95 < ratio < 1.05:
            break
        q = q / min(max(ratio, 1.0 / 30.0), 30.0)
    return q, hist

def probe(job):
    """Event rate with all workers running."""
    rep, process, f, q = job
    BF, _ = _imports()
    g = geometry(process)
    ic = (frag_ic(g) if process == "fragmentation"
          else {"m": g["m_inj"], "N": int(N_SS)})
    t0 = time.time()
    r = BF.simulate(**_sim_kwargs(BF, g, f, process, ic,
                                  PROBE_EVENTS, 10 ** 12, 7000 + rep, q=q))
    return float(r["events"][-1]) / max(time.time() - t0, 1e-9)

def run_replica(job):
    try:
        return _run(job)
    except Exception:
        return dict(rep=job["rep"], ok=False, error=traceback.format_exc(limit=8))

def _run(cfg):
    BF, AN = _imports()
    g = geometry(cfg["process"])
    rep, f, outdir = cfg["rep"], cfg["f"], cfg["outdir"]
    _sa = float(cfg.get("seed_alpha", ALPHA_TH))
    _st = "" if (cfg["process"] != "fragmentation"
                 or cfg.get("start", "cold") != "seed"
                 or abs(_sa - ALPHA_TH) < 1e-9) else "_a%+.2f" % _sa
    tag = "%s_f%.2f%s_rep%02d" % (cfg["short"], f, _st, rep)
    run_file = os.path.join(outdir, tag + ".npz")
    state_file = os.path.join(outdir, tag + "_state.npz")
    png_file = os.path.join(outdir, tag + ".png")

    done = 0
    if cfg["process"] != "fragmentation":
        ic = {"m": g["m_inj"], "N": int(N_SS)}
    elif cfg.get("start", "cold") == "seed":
        ic = frag_ic(g, cfg.get("seed_alpha", ALPHA_TH))
    else:
        ic = {"m": g["m_inj"], "N": frag_n0(g)}
    if (not cfg["fresh"]) and os.path.exists(state_file):
        prev = AN.load(state_file)
        ic = {"state": BF.state_of(prev, source=state_file)}
        done = int(prev["meta"].get("analysis", {}).get("events_done", 0))
        del prev

    total = (done + int(cfg["events_total"]) if cfg.get("additive")
             else int(cfg["events_total"]))
    per_pass = max(int(cfg["events_pass"]), 1)
    log, npass = [], 0
    while done < total:
        budget = int(min(per_pass, total - done))
        if npass >= 1 and budget < 0.05 * per_pass:
            print("[%s] remainder of %d events dropped: shorter than 5%% of a "
                  "pass" % (tag, budget), flush=True)
            break
        npass += 1
        t0 = time.time()
        r = BF.simulate(**_sim_kwargs(
            BF, g, f, cfg["process"], ic, budget,
            max(budget // SNAPSHOTS, 1), 900_000 + 1_000 * rep + 7 * npass,
            verbose=(rep == 0), q=cfg["q"]))
        done += int(r["events"][-1])
        md = abs(float(r["mass_drift"]))
        if md > 1e-4:
            raise RuntimeError(
                "rep %d: mass_drift = %.3e -- that is a bookkeeping failure, "
                "not round-off" % (rep, md))
        if md > 1e-9:
            print("[rep%02d] note: mass_drift = %.3e (round-off at this scale; "
                  "the run is kept)" % (rep, md), flush=True)

        c = np.asarray(r["centers"], float)
        F = AN.spectrum(r)["F"]
        mg, G = AN.local_slope(c, F)

        drift_half = drift_half_max = drift_half_at = np.nan
        D = np.asarray(r["dndm"], float)
        if D.ndim == 2 and D.shape[0] >= 8:
            cw = core_window(g)
            _, G1 = AN.local_slope(c, D[: D.shape[0] // 2].mean(axis=0))
            _, G2 = AN.local_slope(c, D[D.shape[0] // 2:].mean(axis=0))
            kk = (mg >= cw[0]) & (mg <= cw[1]) & np.isfinite(G1) & np.isfinite(G2)
            if kk.any():
                dd = np.abs(G2[kk] - G1[kk])
                drift_half = float(np.median(dd))
                drift_half_max = float(dd.max())
                drift_half_at = float(mg[kk][int(np.argmax(dd))])
        an = dict(rep=rep, f=float(f), process=cfg["process"],
                  events_done=int(done), events_total=int(total),
                  passes_done=npass, n_ss=N_SS,
                  decades=float(np.log10(max(g["m_inj"], g["m_sink"])
                                          / min(g["m_inj"], g["m_sink"]))),
                  m_inj=g["m_inj"], m_sink=g["m_sink"],
                  t_c=float(_tc_of(BF, g, cfg["process"])),
                  iso_dt=(float(ISO_DT) if cfg["process"] == "coagulation"
                          else None),
                  iso_start_sink=int(ISO_START_SINK),
                  band_bins=float(r["meta"].get("loc_band_bins", -1)))
        _save_atomic(AN, r, run_file, True, an)
        _save_atomic(AN, r, state_file, False, dict(an, checkpoint=True))
        spec_path = os.path.join(outdir, tag + "_spec.npz")
        F_passes, ev_passes, no_passes = [], [], []
        if os.path.exists(spec_path) and not (cfg.get("fresh") and npass == 1):
            try:
                old = np.load(spec_path)
                if "F_passes" in old:
                    F_passes = [np.asarray(x, float) for x in old["F_passes"]]
                    ev_passes = [float(x) for x in old["ev_passes"]]
                    no_passes = ([float(x) for x in old["no_passes"]]
                                 if "no_passes" in old
                                 else [0.0] * len(ev_passes))
            except Exception:
                F_passes, ev_passes, no_passes = [], [], []
        F_passes.append(np.asarray(F, float))
        ev_passes.append(float(r["events"][-1]))
        no_passes.append(float(r["n_out"][-1]))
        Fp = np.asarray(F_passes, float)
        wp = np.asarray(ev_passes, float)
        F_acc = (Fp * wp[:, None]).sum(axis=0) / max(wp.sum(), 1e-300)
        mg_acc, G_acc = AN.local_slope(c, F_acc)

        iso_new = _iso_of(r)
        iso_sum, iso_snaps, iso_stack, iso_snap_list = None, 0.0, [], []
        iso_dt_list = []
        iso_over = np.nan
        if iso_new is not None:
            if os.path.exists(spec_path) and not (cfg.get("fresh")
                                                  and npass == 1):
                try:
                    old = np.load(spec_path)
                    if ("iso_passes" in old and "iso_age_edges" in old
                            and np.array_equal(np.asarray(old["iso_age_edges"],
                                                          float),
                                               iso_new["edges"])):
                        iso_stack = [np.asarray(x, float)
                                     for x in old["iso_passes"]]
                        iso_snap_list = [float(x) for x in old["iso_snaps"]]
                        iso_dt_list = ([float(x) for x in old["iso_dts"]]
                                       if "iso_dts" in old
                                       else [np.nan] * len(iso_snap_list))
                except Exception:
                    iso_stack, iso_snap_list, iso_dt_list = [], [], []
            iso_stack.append(iso_new["counts"])
            iso_snap_list.append(iso_new["snaps"])
            iso_dt_list.append(float(iso_new["dt_snap"]))
            iso_sum = np.asarray(iso_stack, float).sum(axis=0)
            iso_snaps = float(np.sum(iso_snap_list))
            iso_dt_mean = _snap_weighted(iso_dt_list, iso_snap_list)
            iso_over = _iso_overflow(r, iso_new)

        gen_new = _gen_of(r)
        gen_acc = None
        if gen_new is not None:
            gc, gn, gt, gs = (gen_new["counts"], gen_new["num"],
                              gen_new["tau_sum"], gen_new["snaps"])
            if os.path.exists(spec_path) and not (cfg.get("fresh")
                                                  and npass == 1):
                try:
                    old = np.load(spec_path)
                    if ("gen_counts" in old
                            and np.asarray(old["gen_counts"]).shape == gc.shape):
                        gc = gc + np.asarray(old["gen_counts"], float)
                        gn = gn + np.asarray(old["gen_num"], float)
                        gt = gt + np.asarray(old["gen_tau_sum"], float)
                        gs = gs + float(old["gen_snapshots"])
                except Exception:
                    pass
            gen_acc = dict(gen_counts=gc, gen_num=gn, gen_tau_sum=gt,
                           gen_snapshots=float(gs),
                           gen_tau=np.where(gn > 0, gt / np.maximum(gn, 1e-300),
                                            np.nan))
        np.savez_compressed(spec_path,
                            centers=c, F=F_acc, m_slope=mg_acc, G=G_acc,
                            F_passes=Fp, ev_passes=wp,
                            F_last=F, f=float(f),
                            events=float(done),
                            events_in_spectrum=float(wp.sum()),
                            live=float(r["live"][-1]),
                            n_out=float(np.sum(no_passes)),
                            no_passes=np.asarray(no_passes, float),
                            n_out_last=float(r["n_out"][-1]),
                            drift_half=float(drift_half),
                            drift_half_max=float(drift_half_max),
                            drift_half_at=float(drift_half_at),
                            m_inj=float(g["m_inj"]), m_sink=float(g["m_sink"]),
                            t_c=float(_tc_of(BF, g, cfg["process"])),
                            iso_overflow=float(iso_over),
                            edges=mass_edges(g),
                            **({} if iso_sum is None else dict(
                                iso_age_edges=iso_new["edges"],
                                iso_counts=iso_sum,
                                iso_snapshots=float(iso_snaps),
                                iso_passes=np.asarray(iso_stack, float),
                                iso_snaps=np.asarray(iso_snap_list, float),
                                iso_dts=np.asarray(iso_dt_list, float),
                                iso_dt_snap=float(iso_dt_mean))),
                            **({} if gen_acc is None else gen_acc))
        _replica_figure(AN, c, F_acc, mg_acc, G_acc, g, png_file, tag, done,
                        total, int(r["live"][-1]), float(r["n_out"][-1]))
        log.append(dict(p=npass, min=(time.time() - t0) / 60.0,
                        live=int(r["live"][-1]), n_out=float(r["n_out"][-1]),
                        events=int(done)))
        print("[%s] pass %d: %.4g ev, live %d, absorb %.4g total, "
              "half-vs-half |dGamma| med %.4f (worst %.4f at m=%.3g), %.1f min%s"
              % (tag, npass, done, r["live"][-1], float(np.sum(no_passes)),
                 drift_half,
                 drift_half_max, drift_half_at, log[-1]["min"],
                 ("" if not np.isfinite(iso_over) else
                  ", iso %d passes / %.0f snaps, %.2f%% past the age grid"
                  % (len(iso_stack), iso_snaps, 100.0 * iso_over))),
              flush=True)
        ic = {"state": BF.state_of(AN.load(state_file), source=state_file)}

    last = log[-1] if log else {}
    return dict(rep=rep, ok=True, events=done, passes=npass,
                live=last.get("live", 0), n_out=last.get("n_out", 0.0))

def _replica_figure(AN, c, F, mg, G, g, path, tag, done, total, live, n_out):
    lo, hi = min(g["m_inj"], g["m_sink"]), max(g["m_inj"], g["m_sink"])
    cw = core_window(g)
    xl = (lo / 30.0, hi * 30.0)
    fig, ax = plt.subplots(1, 2, figsize=(10.6, 4.0))
    ok = F > 0
    ax[0].loglog(c[ok], F[ok] * c[ok] ** COMP, "o", ms=2.5, color="C0")
    inb = ok & (c >= cw[0]) & (c <= cw[1])
    if inb.any():
        y = F[inb] * c[inb] ** COMP
        ax[0].set_ylim(float(y.min()) / 3.0, float(y.max()) * 3.0)
    for a in ax:
        a.axvspan(cw[0], cw[1], color="C2", alpha=0.09, zorder=0)
        a.set_xscale("log")
        a.set_xlim(*xl)
        a.set_xlabel("m")
    ax[0].set_ylabel(r"$m^{11/6}\,dN/dm$")
    ax[0].set_title("(a) compensated spectrum", fontsize=10)
    k = np.isfinite(G)
    ax[1].plot(mg[k], G[k], "o-", ms=2.5, lw=0.6, color="C0")
    ax[1].axhline(ALPHA_TH, ls="-", lw=1.8, color="C3",
                  label=r"$-11/6$, no fit")
    ax[1].set_ylim(ALPHA_TH - 0.45, ALPHA_TH + 0.45)
    ax[1].set_ylabel(r"$\Gamma=d\log F/d\log m$")
    ax[1].legend(fontsize=7)
    ax[1].set_title("(b) local slope", fontsize=10)
    fig.suptitle("%s | %.3g of %.3g events | live %d | absorptions %.4g"
                 % (tag, done, total, live, n_out), fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)

def iso_load(paths, skip=0):
    """Stack age histograms of all replicas."""
    edges = C = edges_m = None
    snaps = 0.0
    used = refused = 0
    dt_snap = []
    dt_w = []
    for p in paths:
        d = np.load(p)
        if "iso_counts" not in d:
            continue
        e = np.asarray(d["iso_age_edges"], float)
        if "iso_passes" in d and skip > 0:
            P = np.asarray(d["iso_passes"], float)
            S = np.asarray(d["iso_snaps"], float)
            if skip >= P.shape[0]:
                continue
            c, s = P[skip:].sum(axis=0), float(S[skip:].sum())
            dt_here = (_snap_weighted(np.asarray(d["iso_dts"], float)[skip:],
                                      S[skip:]) if "iso_dts" in d
                       else float(d.get("iso_dt_snap", np.nan)))
        else:
            c, s = np.asarray(d["iso_counts"], float), float(d["iso_snapshots"])
            dt_here = (float(d["iso_dt_snap"]) if "iso_dt_snap" in d
                       else np.nan)
        if edges is None:
            edges, C = e, np.zeros_like(c)
            edges_m = np.asarray(d["edges"], float) if "edges" in d else None
        elif e.shape != edges.shape or not np.allclose(e, edges):
            refused += 1
            continue
        C += c
        snaps += s
        used += 1
        dt_snap.append(dt_here)
        dt_w.append(s)
    if edges is None or used == 0:
        return None
    return dict(edges=edges, counts=C, snaps=snaps, n=used, refused=refused,
                dt_snap=_snap_weighted(dt_snap, dt_w),
                edges_m=edges_m)

def gen_load(paths):
    """Stack generation histograms of all replicas."""
    C = N = T = None
    snaps = 0.0
    used = 0
    for p in paths:
        d = np.load(p)
        if "gen_counts" not in d:
            continue
        c = np.asarray(d["gen_counts"], float)
        if C is None:
            C, N, T = np.zeros_like(c), np.zeros(c.shape[0]), np.zeros(c.shape[0])
        elif c.shape != C.shape:
            continue
        C += c
        N += np.asarray(d["gen_num"], float)
        T += np.asarray(d["gen_tau_sum"], float)
        snaps += float(d["gen_snapshots"])
        used += 1
    if C is None or used == 0:
        return None
    with np.errstate(invalid="ignore", divide="ignore"):
        tau = np.where(N > 0, T / np.maximum(N, 1e-300), np.nan)
    return dict(counts=C, num=N, tau=tau, snaps=snaps, n=used)

def iso_regroup(edges, C, centers, m_floor=None, dlogm=ISO_DLOGM,
                min_counts=ISO_MIN_COUNTS, dt_snap=np.nan, edges_m=None):
    """Merge fine age bins into isochrones."""
    C = np.asarray(C, float)
    m = np.asarray(centers, float)
    w = np.diff(edges_m)[:m.size] if edges_m is not None else np.gradient(m)
    use = np.ones(m.size, bool) if m_floor is None else (m >= float(m_floor))
    dropped = float(C[:, ~use].sum()) / max(float(C.sum()), 1e-300)
    Cu, mu, wu = C[:, use], m[use], w[use]

    s_fine = Cu.sum(axis=1)
    ok = np.flatnonzero(s_fine > 0)
    if ok.size < 2:
        return None
    mb_fine = np.full(Cu.shape[0], np.nan)
    mb_fine[ok] = (Cu[ok] * mu[None, :]).sum(axis=1) / s_fine[ok]

    mono = np.maximum.accumulate(np.nan_to_num(mb_fine, nan=0.0))
    i0 = int(ok[0])
    edges_i = [i0]
    target = mono[i0] * 10.0 ** dlogm
    for i in ok[1:]:
        if mono[i] >= target:
            edges_i.append(int(i))
            target = mono[i] * 10.0 ** dlogm
    edges_i.append(int(ok[-1]) + 1)
    groups = [(edges_i[k], edges_i[k + 1] - 1) for k in range(len(edges_i) - 1)]
    groups = [gk for gk in groups if gk[1] >= gk[0]]
    if not groups:
        return None

    merged, k = [], 0
    while k < len(groups):
        a, b = groups[k]
        while (Cu[a:b + 1].sum() < min_counts) and (k + 1 < len(groups)):
            k += 1
            b = groups[k][1]
        merged.append((a, b))
        k += 1
    if len(merged) > 1 and Cu[merged[-1][0]:merged[-1][1] + 1].sum() < min_counts:
        merged[-2] = (merged[-2][0], merged[-1][1])
        merged.pop()

    rows = [Cu[a:b + 1].sum(axis=0) for a, b in merged]
    lo_i = [a for a, _ in merged]
    hi_i = [b for _, b in merged]
    R = np.asarray(rows, float)
    S0 = R.sum(axis=1)
    mbar = (R * mu[None, :]).sum(axis=1) / np.maximum(S0, 1e-300)
    spread2 = (R * (mu[None, :] - mbar[:, None]) ** 2).sum(axis=1)
    tau_lo = edges[np.asarray(lo_i, int)]
    tau_hi = edges[np.asarray(hi_i, int) + 1]
    n_rep = np.ones_like(S0)
    if np.isfinite(dt_snap) and dt_snap > 0:
        n_rep = np.clip((tau_hi - tau_lo) / dt_snap, 1.0, None)
    sigma = np.sqrt(spread2 * n_rep) / np.maximum(S0, 1e-300)
    return dict(rows=R, centers=mu, widths=wu, counts=S0, mbar=mbar,
                sigma=sigma, tau_lo=tau_lo, tau_hi=tau_hi,
                tau=0.5 * (tau_lo + tau_hi), n_rep=n_rep, dropped=dropped,
                n_fine=int(C.shape[0]))

def iso_growth_fit(tau, mbar, sigma=None, tc=1.0):
    """Fit log <m> against log tau: power law."""
    t = np.asarray(tau, float)
    y = np.asarray(mbar, float)
    k = np.isfinite(t) & np.isfinite(y) & (y > 0)
    if k.sum() < 3:
        return None
    k = k & (t > 0)
    if k.sum() < 3:
        return None
    lx, ly = np.log(t[k] / tc), np.log(y[k])
    sl, ic = np.polyfit(lx, ly, 1)
    res = ly - (sl * lx + ic)
    return dict(slope=float(sl), b=float(sl), amp=float(np.exp(ic)),
                rms=float(np.std(res)), n=int(k.sum()))

def iso_figure(I, fit, g, tc, f, path, n_rep_files, F=None, centers=None,
               snaps=1.0):
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    lo, hi = min(g["m_inj"], g["m_sink"]), max(g["m_inj"], g["m_sink"])
    fig, ax = plt.subplots(1, 2, figsize=(11.6, 4.4))
    tau = I["tau"] / tc
    norm = Normalize(vmin=float(tau.min()), vmax=float(tau.max()))
    cmap = plt.get_cmap("viridis")

    S = max(float(snaps), 1.0)
    for i in range(I["rows"].shape[0]):
        dn = I["rows"][i] / I["widths"] / S
        k = I["rows"][i] >= 3.0
        if k.sum() < 2:
            continue
        ax[0].loglog(I["centers"][k], dn[k] * I["centers"][k] ** COMP, "-",
                     lw=1.1, color=cmap(norm(tau[i])))
    tot = I["rows"].sum(axis=0) / I["widths"] / S
    kt = tot > 0
    ax[0].loglog(I["centers"][kt], tot[kt] * I["centers"][kt] ** COMP, "--",
                 lw=1.3, color="C3", label="sum of the isochrones")
    if F is not None and centers is not None:
        ok = F > 0
        ax[0].loglog(centers[ok], F[ok] * centers[ok] ** COMP, "-", lw=2.4,
                     color="0.45", alpha=0.8, zorder=0,
                     label="stationary spectrum")
    ax[0].legend(fontsize=7, loc="lower left")
    ax[0].set_xlim(lo / 3.0, hi * 3.0)
    ax[0].set_xlabel("m")
    ax[0].set_ylabel(r"$m^{11/6}\,dN/dm$")
    ax[0].set_title("(a) isochrones, %d of %d fine age bins merged"
                    % (I["rows"].shape[0], I["n_fine"]), fontsize=10)
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax[0])
    cb.set_label(r"age $\tau / t_c$", fontsize=8)

    ax[1].errorbar(tau, I["mbar"], yerr=I["sigma"], fmt="o", ms=3.4, lw=0.8,
                   elinewidth=0.8, capsize=1.5, color="C0",
                   label=r"$\langle m\rangle(\tau)$, %d replicas" % n_rep_files)
    if fit is not None:
        xs = np.logspace(np.log10(max(tau.min(), 1e-12)),
                         np.log10(tau.max()), 50)
        ax[1].plot(xs, fit["amp"] * xs ** fit["b"], "-", lw=1.8, color="C3",
                   label=r"$\propto \tau^{%.2f}$  (theory $6$)" % fit["b"])
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_ylabel(r"$\langle m\rangle$")
    ax[1].set_xlabel(r"age  $\tau / t_c$   (LOG)")
    ax[1].legend(fontsize=7, loc="upper left")
    ax[1].set_title("(b) growth law", fontsize=10)
    fig.suptitle(r"coagulation, geometric kernel $\lambda=2/3$, locality $f=%.2f$"
                 r"   |   isochrones accumulated over %d replicas" % (f, n_rep_files),
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def rebin(centers, F, k, edges=None):
    """Merge k adjacent mass bins."""
    k = int(k)
    if k <= 1:
        return centers, F
    e = EDGES if edges is None else np.asarray(edges, float)
    n = (e.size - 1) // k * k
    w = np.diff(e)[:n]
    cnt = (np.asarray(F, float)[:n] * w).reshape(-1, k).sum(axis=1)
    e2 = np.concatenate([e[:n:k], [e[n]]])
    return np.sqrt(e2[:-1] * e2[1:]), cnt / np.diff(e2)

def gather(paths, nbin=1, skip=0):
    _, AN = _imports()
    Fs, Gs, ev, live, nout, half = [], [], [], [], [], []
    hmax, hat, ev_spec, npass = [], [], [], []
    raw, raw_c = [], None
    centers = None
    for p in paths:
        d = np.load(p)
        if skip > 0 and "F_passes" in d:
            Fp = np.asarray(d["F_passes"], float)
            wp = np.asarray(d["ev_passes"], float)
            if skip < Fp.shape[0]:
                Fp, wp = Fp[skip:], wp[skip:]
                F_use = (Fp * wp[:, None]).sum(0) / max(wp.sum(), 1e-300)
                ev_spec.append(float(wp.sum()))
            else:
                F_use = np.asarray(d["F"], float)
                ev_spec.append(float(d["events_in_spectrum"])
                               if "events_in_spectrum" in d else np.nan)
        else:
            F_use = np.asarray(d["F"], float)
            ev_spec.append(float(d["events_in_spectrum"])
                           if "events_in_spectrum" in d else np.nan)
        npass.append(int(np.asarray(d["ev_passes"]).size)
                     if "ev_passes" in d else 1)
        e_f = np.asarray(d["edges"], float) if "edges" in d else None
        c2, F2 = rebin(d["centers"], F_use, nbin, e_f)
        centers = c2
        raw_c = np.asarray(d["centers"], float)
        raw.append(np.asarray(F_use, float))
        Fs.append(F2)
        _, G2 = AN.local_slope(c2, F2)
        Gs.append(G2)
        ev.append(float(d["events"]))
        live.append(float(d["live"]))
        nout.append(float(d["n_out"]))
        half.append(float(d["drift_half"]) if "drift_half" in d else np.nan)
        hmax.append(float(d["drift_half_max"]) if "drift_half_max" in d else np.nan)
        hat.append(float(d["drift_half_at"]) if "drift_half_at" in d else np.nan)
    if not Fs:
        return None
    Fs, Gs = np.asarray(Fs, float), np.asarray(Gs, float)
    n = Fs.shape[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        F = np.nanmean(Fs, axis=0)
        mg, G = AN.local_slope(centers, F)
        sd = np.nanstd(Gs, axis=0, ddof=1) if n > 1 else np.full_like(G, np.nan)
    ev = np.asarray(ev, float)
    return dict(centers=centers, mg=mg, F=F, G=G, se=sd / np.sqrt(n), n=n,
                G_reps=Gs, events=float(ev.mean()), ev_all=ev,
                ev_spread=float(ev.max() - ev.min()),
                live=float(np.mean(live)),
                n_out=float(np.mean(nout)), half=float(np.nanmedian(half)),
                half_max=float(np.nanmax(hmax)),
                half_at=float(np.nanmedian(hat)),
                ev_spec=float(np.nanmean(ev_spec)),
                npass=int(np.max(npass)) if npass else 1,
                centers_raw=raw_c,
                F_raw=np.nanmean(np.asarray(raw, float), axis=0),
                F_raw_se=(np.nanstd(np.asarray(raw, float), axis=0, ddof=1)
                          / np.sqrt(n) if n > 1
                          else np.full(np.shape(raw)[1], np.nan)))

def reach(mg, G, tol=REACH_TOL):
    k = np.isfinite(G) & (mg > 0)
    idx = np.flatnonzero(k)
    if idx.size == 0:
        return 0.0, 0.0, 0.0
    ok = np.abs(G[idx] - ALPHA_TH) < tol
    best = cur = 0
    end = -1
    for i, good in enumerate(ok):
        cur = cur + 1 if good else 0
        if cur > best:
            best, end = cur, i
    if best < 2:
        return 0.0, 0.0, 0.0
    lo, hi = mg[idx[end - best + 1]], mg[idx[end]]
    return float(lo), float(hi), float(np.log10(hi / lo))

def report(R, cw):
    """Drift of local slope on the core window."""
    k = (R["mg"] >= cw[0]) & (R["mg"] <= cw[1]) & np.isfinite(R["G"])
    if k.sum() < 4:
        return None
    x = np.log10(R["mg"][k])
    drifts = []
    for Gi in R["G_reps"]:
        ki = k & np.isfinite(Gi)
        if ki.sum() > 3:
            drifts.append(np.polyfit(np.log10(R["mg"][ki]), Gi[ki], 1)[0])
    drifts = np.asarray(drifts, float)
    se_d = (float(drifts.std(ddof=1) / np.sqrt(drifts.size))
            if drifts.size > 1 else np.nan)
    lo, hi, dec = reach(R["mg"], R["G"])
    return dict(mean=float(np.mean(R["G"][k])),
                se=float(np.sqrt(np.nansum(R["se"][k] ** 2)) / k.sum()),
                drift=float(np.polyfit(x, R["G"][k], 1)[0]), drift_se=se_d,
                rms=float(np.std(R["G"][k])), nbin=int(k.sum()),
                reach_lo=lo, reach_hi=hi, reach_dec=dec)

def stacked_figure(R, g, cw, f, process, path):
    lo, hi = min(g["m_inj"], g["m_sink"]), max(g["m_inj"], g["m_sink"])
    xl = (lo / 30.0, hi * 30.0)
    fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.4))
    ok = R["F"] > 0
    ax[0].loglog(R["centers"][ok], R["F"][ok] * R["centers"][ok] ** COMP, "o",
                 ms=2.8, color="C0", label="mean of %d" % R["n"])
    band = ok & (R["centers"] >= cw[0]) & (R["centers"] <= cw[1])
    if band.any():
        amp = float(np.median(R["F"][band]
                              / R["centers"][band] ** ALPHA_TH))
        xs = np.logspace(np.log10(cw[0]), np.log10(cw[1]), 40)
        ax[0].loglog(xs, amp * xs ** 0.0, "-", lw=2.0, color="C3", zorder=5,
                     label=r"$m^{-11/6}$, no fit")
        y = R["F"][band] * R["centers"][band] ** COMP
        ax[0].set_ylim(float(y.min()) / 3.0, float(y.max()) * 3.0)
    ax[0].set_ylabel(r"$m^{11/6}\,dN/dm$")
    ax[0].legend(fontsize=7, loc="lower left")
    ax[0].set_title("(a) compensated spectrum", fontsize=10)

    k = np.isfinite(R["G"])
    ax[1].errorbar(R["mg"][k], R["G"][k], yerr=R["se"][k], fmt="o", ms=2.8,
                   lw=0.7, elinewidth=0.7, capsize=1.3, color="C0",
                   label=r"mean of %d, $\pm$ s.e." % R["n"])
    ax[1].axhline(ALPHA_TH, ls="-", lw=2.0, color="C3",
                  label=r"$-11/6$, no fit")
    ax[1].set_ylim(ALPHA_TH - 0.4, ALPHA_TH + 0.4)
    ax[1].set_ylabel(r"$\Gamma=d\log F/d\log m$")
    ax[1].legend(fontsize=7)
    ax[1].set_title("(b) local slope", fontsize=10)

    for a in ax:
        a.axvspan(cw[0], cw[1], color="C2", alpha=0.09, zorder=0)
        a.set_xscale("log")
        a.set_xlim(*xl)
        a.set_xlabel("m")
    fig.suptitle(r"%s, geometric kernel $\lambda=2/3$, locality $f=%.2f$   |   "
                 r"$m=%g\ldots%g$   |   core %.2g..%.3g   |   %d replicas"
                 % (process, f, lo, hi, cw[0], cw[1], R["n"]), fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser(
        description="Open cascade, geometric kernel lambda = 2/3: "
                    "coagulation (m = 0.1 .. 1e6) or fragmentation "
                    "(m = 1e6 .. 1).")
    ap.add_argument("--process", choices=["coag", "frag"], required=True)
    ap.add_argument("--f", type=float, default=-1.0,
                    help="locality ratio, m_small >= f m_large; negative: "
                         "0 for coag, 0.30 for frag.")
    ap.add_argument("--nreps", type=int, default=8,
                    help="number of replicas.")
    ap.add_argument("--hours", type=float, default=8.0,
                    help="total hours per replica.")
    ap.add_argument("--add-hours", type=float, default=0.0,
                    help="hours to add to each replica; overrides --hours.")
    ap.add_argument("--pass-hours", type=float, default=1.0,
                    help="hours between checkpoints.")
    ap.add_argument("--events", type=float, default=0.0,
                    help="total events per replica; skips the rate probe.")
    ap.add_argument("--add-events", type=float, default=0.0,
                    help="events to add to each replica.")
    ap.add_argument("--start", choices=["cold", "seed"], default="cold",
                    help="fragmentation start: cold (bodies at the injection "
                         "mass) or seed (power law).")
    ap.add_argument("--seed-alpha", type=float, default=ALPHA_TH,
                    help="slope of the seeded power law, with --start seed.")
    ap.add_argument("--skip-passes", type=int, default=0,
                    help="leave out the first k passes of every replica.")
    ap.add_argument("--iso-dlogm", type=float, default=ISO_DLOGM,
                    help="spacing of drawn isochrones, decades of mean mass.")
    ap.add_argument("--iso-min-counts", type=float, default=ISO_MIN_COUNTS,
                    help="minimum particle count per drawn isochrone.")
    ap.add_argument("--iso-mmin", type=float, default=0.0,
                    help="mass floor for isochrones; 0: bottom of core window.")
    ap.add_argument("--q", type=float, default=0.0,
                    help="fragmentation: injection rate; skips the calibration.")
    ap.add_argument("--outdir", default="runs",
                    help="output folder.")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore saved states and start over.")
    ap.add_argument("--stack-only", action="store_true",
                    help="redraw from saved results, no run.")
    args = ap.parse_args()
    if args.f < 0.0:
        args.f = 0.0 if args.process == "coag" else 0.30

    process = "coagulation" if args.process == "coag" else "fragmentation"
    short = args.process
    BF, AN = _imports()
    g = geometry(process)
    cw = core_window(g)
    pr = BF.predict("open", BF.KERNEL_LAMBDA[KERNEL_NAME])
    if abs(pr["alpha"] - ALPHA_TH) > 1e-9 or abs(pr["beta"] - BETA_TH) > 1e-9:
        raise SystemExit(
            "engine predicts alpha = %.6g, beta = %.6g for %s; this file was "
            "built for alpha = %.6g, beta = %.6g"
            % (pr["alpha"], pr["beta"], KERNEL_NAME, ALPHA_TH, BETA_TH))
    outdir = (args.outdir if os.path.isabs(args.outdir)
              else os.path.join(_here(), args.outdir))
    os.makedirs(outdir, exist_ok=True)

    print("=" * 78)
    print("  %s, GEOMETRIC kernel, lambda = %.4f, locality f = %.2f"
          % (process, LAMBDA, args.f))
    print("  theory : beta = %.4f -> Gamma = %.4f, FLAT;  b = %.3f"
          % (pr["beta"], pr["alpha"], pr["b"]))
    print("  box    : injection %g -> wall %g   (%.1f decades)  N_ss = %g"
          % (g["m_inj"], g["m_sink"],
             np.log10(max(g["m_inj"], g["m_sink"])
                      / min(g["m_inj"], g["m_sink"])), N_SS))
    print("  core   : %.4g .. %.4g  (%.1f decades)"
          % (cw[0], cw[1], np.log10(cw[1] / cw[0])))
    print("  note   : below m = %.3g a 0.1 dex bin holds one or two masses"
          % (3.9 * min(g["m_inj"], g["m_sink"])))
    print("  outdir : %s" % outdir)
    if "onedrive" in outdir.lower():
        print("  !! inside OneDrive: exclude this folder from sync, or pause it")
    print("=" * 78)

    if not args.stack_only:
        from concurrent.futures import ProcessPoolExecutor
        try:
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x1)
            print("sleep     : blocked while this process lives")
        except Exception:
            pass
        print("disk      : %.0f GB free" % (shutil.disk_usage(outdir).free / 1e9))

        if process == "fragmentation":
            print("calibrate : tuning the injection rate to hold %.3g bodies"
                  % N_SS)
            if args.q > 0:
                q_use, hist = float(args.q), []
                print("            given: q = %.6g (calibration skipped)" % q_use)
            else:
                q_use, hist = calibrate_q(process, args.f)
            for i, (qq, n0, n1, nout) in enumerate(hist):
                print("            round %d: q = %.4g -> live %d, settled at %d "
                      "(target %.3g), absorbed %.4g"
                      % (i + 1, qq, n0, n1, N_SS, nout))
            print("            chosen q = %.6g" % q_use)
            print("start     : %s"
                  % ("cold, %d bodies at m_inj = %.3g"
                     % (frag_n0(geometry(process)), geometry(process)["m_inj"])
                     if args.start == "cold"
                     else "seeded power law, alpha = %+.2f" % args.seed_alpha))
            if not hist and args.q <= 0:
                print("            !! calibration produced nothing -- check it")
        else:
            q_use = _q_of(BF, geometry(process), process)
            print("injection : q = %.6g" % q_use)

        add = (args.add_hours > 0.0) or (args.add_events > 0.0)
        if args.add_events > 0:
            ev = int(args.add_events)
            print("budget    : +%.4g events ADDED to each replica" % ev)
        elif args.events > 0:
            ev = int(args.events)
            print("budget    : %.4g events per replica, TOTAL (given)" % ev)
        else:
            print("probe     : %d workers at once, measuring the LOADED rate"
                  % args.nreps)
            with ProcessPoolExecutor(max_workers=args.nreps) as pool:
                rates = list(pool.map(probe, [(r, process, args.f, q_use)
                                              for r in range(args.nreps)]))
            rate = float(np.median(rates))
            hrs = args.add_hours if args.add_hours > 0 else args.hours
            ev = int(rate * 3600.0 * hrs)
            print("            %.4g ev/s per worker -> %.4g events for %.1f h %s"
                  % (rate, ev, hrs, "ADDED" if add else "TOTAL"))
        _h = args.add_hours if args.add_hours > 0 else args.hours
        ev_pass = max(int(ev * args.pass_hours / max(_h, 1e-9)), 1)
        print("checkpoint: every %.4g events (about %.2f h)\n"
              % (ev_pass, args.pass_hours))

        jobs = [dict(rep=r, f=args.f, process=process, short=short, q=q_use,
                     outdir=outdir, events_total=ev, events_pass=ev_pass,
                     additive=add, fresh=args.fresh,
                     seed_alpha=args.seed_alpha, start=args.start)
                for r in range(args.nreps)]
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=args.nreps) as pool:
            for res in pool.map(run_replica, jobs):
                if not res["ok"]:
                    print("[rep%02d] FAILED\n%s" % (res["rep"], res["error"]))
        print("\nall replicas finished in %.2f h" % ((time.time() - t0) / 3600.0))

    import glob
    _sa = float(args.seed_alpha)
    _st = "" if (process != "fragmentation" or args.start != "seed"
                 or abs(_sa - ALPHA_TH) < 1e-9) else "_a%+.2f" % _sa
    R = gather(sorted(glob.glob(os.path.join(
        outdir, "%s_f%.2f%s_rep*_spec.npz" % (short, args.f, _st)))),
        REBIN_DEFAULT, args.skip_passes)
    if R is None:
        raise SystemExit("no results in %s" % outdir)
    png = os.path.join(outdir, "%s_f%.2f%s_stacked.png" % (short, args.f, _st))
    stacked_figure(R, g, cw, args.f, process, png)

    st = report(R, cw)
    print("\n" + "=" * 78)
    print("STACKED over %d replicas | %d passes each" % (R["n"], R["npass"]))
    print("  events computed  : %.5g per replica" % R["events"])
    if not np.isfinite(R["ev_spec"]):
        print("  events IN the spectrum: unknown")
    else:
        print("  events IN the spectrum: %.5g per replica%s"
              % (R["ev_spec"],
                 "   (first %d pass(es) skipped)" % args.skip_passes
                 if args.skip_passes else ""))
        if not args.skip_passes and R["ev_spec"] < 0.99 * R["events"]:
            print("  !! the spectrum stands on less than what was computed.")
    print("  live %.4g | absorptions %.4g" % (R["live"], R["n_out"]))
    if R["ev_spread"] > 0.01 * max(R["events"], 1.0):
        print("  !! replicas are NOT in step: %.5g .. %.5g events (spread %.3g)"
              % (R["ev_all"].min(), R["ev_all"].max(), R["ev_spread"]))
        print("     run once with --hours H to level them.")
    elif R["ev_spread"] > 0:
        print("  replicas in step to %.2f%%"
              % (100.0 * R["ev_spread"] / max(R["events"], 1.0)))
    else:
        print("  replicas in step")
    print("-" * 78)
    if st is None:
        print("too few bins in the core window")
    else:
        print("  core window     : %.4g .. %.4g   (%.2f decades, %d bins)"
              % (cw[0], cw[1], np.log10(cw[1] / cw[0]), st["nbin"]))
        print("  <Gamma>         : %+.4f +- %.4f      (theory %+.4f)"
              % (st["mean"], st["se"], pr["alpha"]))
        print("  drift per decade: %+.4f +- %.4f      (%.1f sigma from zero)"
              % (st["drift"], st["drift_se"],
                 abs(st["drift"]) / max(st["drift_se"], 1e-12)))
        print("  scatter rms     :  %.4f" % st["rms"])
        print("  half-vs-half    :  |dGamma| in the core,")
        print("                     median over replicas   %.4f" % R["half"])
        print("                     worst single bin       %.4f  at m = %.3g"
              % (R["half_max"], R["half_at"]))
        print("  plateau reached : %s"
              % ("%.3g .. %.3g   = %.2f decades"
                 % (st["reach_lo"], st["reach_hi"], st["reach_dec"])
                 if st["reach_dec"] > 0 else "none"))
    print("  -> %s" % png)

    if process == "coagulation":
        S = iso_load(sorted(glob.glob(os.path.join(
            outdir, "%s_f%.2f_rep*_spec.npz" % (short, args.f)))),
            args.skip_passes)
        print("-" * 78)
        if S is None:
            print("  isochrones      : none stored")
        else:
            tc = _tc_of(BF, g, process)
            m_floor = args.iso_mmin if args.iso_mmin > 0 else cw[0]
            I = iso_regroup(S["edges"], S["counts"], R["centers_raw"],
                            m_floor=m_floor, dlogm=args.iso_dlogm,
                            min_counts=args.iso_min_counts,
                            dt_snap=S["dt_snap"], edges_m=S.get("edges_m"))
            print("  isochrones      : %d replicas, %.0f snapshots, %.4g particle"
                  " counts" % (S["n"], S["snaps"], S["counts"].sum()))
            if S["refused"]:
                print("  !! %d replica(s) refused: their age grid differs from the"
                      " first" % S["refused"])
            if I is None:
                print("                    too few counts to group")
            else:
                print("  age grid        : %d fine bins of %.3g t_c, merged into %d"
                      % (I["n_fine"], ISO_DT, I["rows"].shape[0]))
                print("  mass floor      : %.4g  (%.2f%% of the counts sit below "
                      "it and are not drawn:" % (m_floor, 100.0 * I["dropped"]))
                print("                    the picket fence, computed and stored, "
                      "not measured)")
                n_g = I["rows"].shape[0]
                sl = slice(1, -1) if n_g > 4 else slice(None)
                fit = iso_growth_fit(I["tau"][sl], I["mbar"][sl], tc=tc)
                print("")
                print("     tau/t_c        <m>        +-        counts   "
                      "dlog(tau)   dlog<m>")
                for i in range(I["rows"].shape[0]):
                    dt = (np.log10(I["tau"][i] / I["tau"][i - 1])
                          if i and I["tau"][i - 1] > 0 else np.nan)
                    dm = (np.log10(I["mbar"][i] / I["mbar"][i - 1])
                          if i else np.nan)
                    print("   %6.2f-%-6.2f %10.4g %9.3g %11.4g %10s %9s"
                          % (I["tau_lo"][i] / tc, I["tau_hi"][i] / tc,
                             I["mbar"][i], I["sigma"][i], I["counts"][i],
                             "--" if i == 0 else "%.3f" % dt,
                             "--" if i == 0 else "%.3f" % dm))
                print("")
                with np.errstate(divide="ignore", invalid="ignore"):
                    dts = np.diff(np.log10(np.maximum(I["tau"], 1e-300)))
                inner = dts[1:-1] if dts.size > 3 else dts
                inner = inner[np.isfinite(inner)]
                if inner.size > 1:
                    print("  age step        : dlog(tau) = %.3f +- %.3f per %.2f"
                          " decades of mass  (%d interior steps;"
                          % (inner.mean(), inner.std(ddof=1), args.iso_dlogm,
                             inner.size))
                    print("                    the first and last groups are the "
                          "mass floor and the wall and are left out)")
                    print("                    a constant step in LOG tau is a "
                          "POWER LAW.  Spread/mean = %.1f%%"
                          % (100.0 * inner.std(ddof=1)
                             / max(abs(inner.mean()), 1e-30)))
                    print("                    expected dlog(tau) = dlogm / b = "
                          "%.3f at b = %.3f" % (args.iso_dlogm / B_TH, B_TH))
                if fit is not None:
                    print("  growth law      : <m> ~ tau^%.3f  (theory b = %.3f),"
                          "  rms of log residual %.4f over %d interior points"
                          % (fit["b"], B_TH, fit["rms"], fit["n"]))
                    print("                    t_c here is 1/(K(m_inj,m_inj) N_ss)"
                          " = %.4g" % tc)
                ipng = os.path.join(outdir, "%s_f%.2f_iso.png" % (short, args.f))
                iso_figure(I, fit, g, tc, args.f, ipng, S["n"],
                           F=R["F_raw"], centers=R["centers_raw"],
                           snaps=S["snaps"])
                print("  -> %s" % ipng)
    print("=" * 78)

if __name__ == "__main__":
    main()
