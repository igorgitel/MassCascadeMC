"""
analysis.py -- measurement layer: spectral index with its uncertainty,
power-law range, generation and age statistics, settling tests.
Works on what `simulate` returns; it never calls the engine.
"""

import numpy as np

BF = None
for _engine_name in ("masscascade", "BF_warm_start_v7", "BF_warm_start_v6", "BF_warm_start_v5", "BF_v_no_resampling_v4",
                     "BF_v_no_resampling_v3", "BF_v_no_resampling_v2"):
    try:
        BF = __import__(_engine_name)
        break
    except ImportError:
        continue
if BF is None:
    raise ImportError("no Majorant_v2 engine found on the path")

local_slope         = BF.local_slope
find_inertial_range = BF.find_inertial_range
guard_band          = BF.guard_band
fit_powerlaw        = BF.fit_powerlaw
superpose           = BF.superpose
predict             = BF.predict
iso_mean_mass_raw   = BF.iso_mean_mass
growth_fit          = BF.growth_fit
sample_powerlaw     = getattr(BF, "sample_powerlaw", None)
steady_N_for_mass   = getattr(BF, "steady_N_for_mass", None)
ENGINE_NAME         = getattr(BF, "__name__", "?")

def load(path):
    """Read a run written by add_last_run."""
    return BF.load_run(path)

def describe(run, name=""):
    """One block of text: configuration, cost, what was stored alongside."""
    m = run["meta"]
    a = m.get("analysis", {})
    out = ["=" * 78, "%s  <-  %s" % (name or m.get("saved_as", "?"), m.get("saved_as", "?")),
           "  %-16s %s / %s, kernel %s (lambda = %s)"
           % ("configuration", m["process"], m["system"], m["kernel"], m["lambda"]),
           "  %-16s m_i = %.4g, N = %g" % ("initial", m["ic"]["m"], m["ic"]["N"]),
           "  %-16s rate = %.4g at m = %.4g" % ("injection", m["injection_rate"],
                                                m["injection_mass"]),
           "  %-16s %s" % ("sink", m["sink_mass"]),
           "  %-16s %s" % ("stopped by", m.get("stop_reason")),
           "  %-16s events = %.4g, acc = %.3f, mass drift = %+.1e"
           % ("cost", float(run["events"][-1]), float(run["acceptance"]),
              float(run["mass_drift"])),
           "  %-16s %d snapshots, %d mass bins"
           % ("arrays", np.asarray(run["dndm"]).shape[0], np.asarray(run["centers"]).size)]
    if m.get("seeded"):
        s = m.get("seed") or {}
        out.append("  %-16s alpha = %+.4f on [%.3g, %.3g], N = %s, <m> = %.4g"
                   % ("SEEDED (v5)", s.get("alpha", np.nan), s.get("m_lo", np.nan),
                      s.get("m_hi", np.nan), s.get("N", "?"), s.get("mean_mass", np.nan)))
    if "honest_num" in run and np.size(run["honest_num"]):
        out.append("  %-16s %.3f by number, %.3f by mass  (1.0 = no seed left)"
                   % ("honest at end", float(np.asarray(run["honest_num"])[-1]),
                      float(np.asarray(run["honest_mass"])[-1])))
    if "wait_mean" in run:
        out.append("  %-16s %.4g stays closed in %d bins"
                   % ("waiting law", float(np.asarray(run["wait_events"]).sum()),
                      int((np.asarray(run["wait_events"]) > 0).sum())))
    if m.get("restarted_from"):
        out.append("  %-16s %s" % ("RESTARTED FROM", m["restarted_from"]))
        out.append("  %-16s %.4g before this run + %.4g in it = %.4g in total"
                   % ("t accumulated", m.get("t_origin", 0.0),
                      float(run.get("final_t_phys", np.nan)),
                      m.get("t_origin", 0.0) + float(run.get("final_t_phys", np.nan))))
    if a.get("checkpoint"):
        out.append("  %-16s %s (%.2f flushes), drift of <m> %+.3f, alpha %+.4f"
                   % ("CHECKPOINT",
                      "STATIONARY" if a.get("stationary") else "continuation point",
                      a.get("flushes", np.nan), a.get("drift_mbar", np.nan),
                      a.get("drift_alpha", np.nan)))
    if a:
        out.append("  stored analysis:")
        for k in sorted(a):
            v = a[k]
            out.append("      %-22s %s" % (k, ("%+.4g" % v) if isinstance(v, float) else v))
    return "\n".join(out)

def scale_factors(run):
    """(w, V, widths): everything needed to go between dndm and raw counts."""
    m = run["meta"]
    w = float(m.get("weight", 1.0))
    V = float(m.get("volume", 1.0))
    return w, V, np.asarray(run["widths"], float)

def raw_counts(run):
    """Particle counts per (snapshot, mass bin), recovered from dndm."""
    w, V, widths = scale_factors(run)
    return np.asarray(run["dndm"], float) * widths[None, :] * V / w

def steady_mask(run, frac=0.5, verbose=False):
    """Which snapshots belong to the steady state."""
    t = np.asarray(run["t"], float)
    meta = run.get("meta", {})
    gate = float(meta.get("iso_start_sink", 0) or 0)
    n_out = np.asarray(run["n_out"], float) if "n_out" in run else None

    ss = None
    if n_out is not None and gate > 0:
        ss = n_out >= gate
        if ss.sum() >= 1 and verbose:
            print("   steady mask: %d of %d snapshots past the sink gate (n_out >= %.3g)"
                  % (ss.sum(), ss.size, gate))
    if ss is None or ss.sum() < 1:
        t0 = float(run["iso_t_begin"]) if "iso_t_begin" in run else np.nan
        if np.isfinite(t0) and t0 <= t.max():
            ss = t >= t0
        else:
            ss = np.zeros(t.size, bool)
            ss[int((1.0 - frac) * t.size):] = True
            if verbose:
                print("   no usable sink gate -- averaging the last %.0f%% of snapshots"
                      % (100 * frac))
    if ss.sum() < 1:
        ss[-1] = True
    return ss

def effective_snapshots(series):
    """Number of INDEPENDENT samples in a correlated series, K (1-rho)/(1+rho)."""
    x = np.asarray(series, float)
    x = x[np.isfinite(x)]
    K = x.size
    if K < 4:
        return float(max(K, 1))
    x = x - x.mean()
    denom = float((x * x).sum())
    if denom <= 0:
        return float(K)
    rho = float((x[:-1] * x[1:]).sum() / denom)
    rho = min(max(rho, 0.0), 0.98)
    return float(np.clip(K * (1.0 - rho) / (1.0 + rho), 1.0, K))

def steady_spectrum(run, frac=0.5, verbose=False):
    """The estimator for an OPEN run and its uncertainty."""
    D = np.asarray(run["dndm"], float)
    n = raw_counts(run)
    ss = steady_mask(run, frac=frac, verbose=verbose)
    Ds, ns = D[ss], n[ss]
    K = int(ss.sum())

    F = Ds.mean(axis=0)
    C = ns.sum(axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_pois = np.where(C > 0, F / np.sqrt(np.maximum(C, 1e-300)), np.nan)

    if K >= 4:
        Keff = np.array([effective_snapshots(Ds[:, b]) for b in range(Ds.shape[1])])
        sigma_emp = Ds.std(axis=0, ddof=1) / np.sqrt(np.maximum(Keff, 1.0))
    else:
        Keff = np.full(Ds.shape[1], float(K))
        sigma_emp = np.full(F.shape, np.nan)

    sigma = np.where(np.isfinite(sigma_emp),
                     np.maximum(sigma_emp, np.where(np.isfinite(sigma_pois), sigma_pois, 0.0)),
                     sigma_pois)
    sigma = np.where(F > 0, sigma, np.nan)

    return dict(F=F, sigma=sigma, sigma_emp=sigma_emp, sigma_pois=sigma_pois,
                counts=C, K=K, K_eff=Keff, mask=ss)

def superposed_spectrum(run):
    """The CLOSED-system estimator: the dt-weighted superposition."""
    F = BF.superpose(run["dndm"], run["t"])
    n = raw_counts(run)
    dt = np.diff(np.asarray(run["t"], float), prepend=float(run["t"][0]))
    w, V, widths = scale_factors(run)
    scale = (w / V) / widths
    var = ((dt[:, None] ** 2) * n).sum(axis=0) * scale ** 2
    with np.errstate(invalid="ignore"):
        sigma = np.where(F > 0, np.sqrt(var), np.nan)
    return dict(F=F, sigma=sigma, sigma_pois=sigma, sigma_emp=np.full(F.shape, np.nan),
                counts=n.sum(axis=0), K=int(n.shape[0]),
                K_eff=np.full(F.shape, float(n.shape[0])), mask=np.ones(n.shape[0], bool))

def spectrum(run, frac=0.5, verbose=False):
    """Pick the estimator from the system recorded in meta."""
    if run["meta"]["system"] == "open":
        return steady_spectrum(run, frac=frac, verbose=verbose)
    return superposed_spectrum(run)

def iso_mean_mass(run, correct_repeats=True, groups=None):
    """<m>(tau) from the accumulated (age, mass) histogram, with its uncertainty."""
    w, V, widths = scale_factors(run)
    c = np.asarray(run["iso_counts"], float) / w
    m = np.asarray(run["centers"], float)
    e = np.asarray(run["iso_age_edges"], float)

    if groups is None:
        lo, hi = e[:-1], e[1:]
    else:
        groups = [np.atleast_1d(np.asarray(g, int)) for g in groups]
        if len(groups) == 0:
            nan = np.zeros(0)
            return dict(tau=nan, mbar=nan, sigma=nan, counts=nan, n_rep=nan,
                        tau_lo=nan, tau_hi=nan)
        c = np.array([c[g].sum(axis=0) for g in groups])
        lo = np.array([e[g.min()] for g in groups])
        hi = np.array([e[g.max() + 1] for g in groups])
    tau = np.sqrt(lo * hi)

    S0 = c.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mbar = np.where(S0 > 0, (c * m[None, :]).sum(axis=1) / np.maximum(S0, 1e-300), np.nan)
        spread2 = np.where(S0 > 0,
                           (c * (m[None, :] - mbar[:, None]) ** 2).sum(axis=1), np.nan)

    n_rep = np.ones_like(tau)
    if correct_repeats:
        ss = steady_mask(run)
        t = np.asarray(run["t"], float)[ss]
        if t.size > 1:
            dt_snap = float(np.median(np.diff(t)))
            if dt_snap > 0:
                n_rep = np.clip((hi - lo) / dt_snap, 1.0, None)

    with np.errstate(invalid="ignore", divide="ignore"):
        sigma = np.where(S0 > 0, np.sqrt(spread2 * n_rep) / np.maximum(S0, 1e-300), np.nan)

    return dict(tau=tau, mbar=mbar, sigma=sigma, counts=S0, n_rep=n_rep,
                tau_lo=lo, tau_hi=hi)

def wls_powerlaw(x, y, sy=None, mask=None):
    """Weighted least squares of ln y against ln x."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    k = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if mask is not None:
        k &= mask
    if sy is not None:
        sy = np.asarray(sy, float)
        k &= np.isfinite(sy) & (sy > 0)
    if k.sum() < 3:
        return dict(p=np.nan, sigma_p=np.nan, A=np.nan, chi2_dof=np.nan,
                    rms_resid=np.nan, n=int(k.sum()))

    X, Y = np.log(x[k]), np.log(y[k])
    W = np.ones_like(X) if sy is None else 1.0 / (sy[k] / y[k]) ** 2

    Sw = W.sum(); Sx = (W * X).sum(); Sy = (W * Y).sum()
    Sxx = (W * X * X).sum(); Sxy = (W * X * Y).sum()
    det = Sw * Sxx - Sx * Sx
    if det == 0:
        return dict(p=np.nan, sigma_p=np.nan, A=np.nan, chi2_dof=np.nan,
                    rms_resid=np.nan, n=int(k.sum()))

    p = (Sw * Sxy - Sx * Sy) / det
    a = (Sxx * Sy - Sx * Sxy) / det
    sigma_p = np.sqrt(Sw / det)
    resid = Y - (a + p * X)
    dof = max(k.sum() - 2, 1)
    chi2_dof = float((W * resid ** 2).sum() / dof)
    rms = float(np.sqrt((resid ** 2).mean()))
    if sy is None:
        sigma_p *= np.sqrt(chi2_dof)
        chi2_dof = np.nan
    return dict(p=float(p), sigma_p=float(sigma_p), A=float(np.exp(a)),
                chi2_dof=chi2_dof, rms_resid=rms, n=int(k.sum()))

def anchor_amplitude(m, F, m_lo, m_hi, alpha):
    """A such that A m^alpha passes through the data over [m_lo, m_hi]: the least-squares intercept at FIXED slope, so the drawn line carries the measured index and nothing else, and the theory line drawn the same way differs from it in slope alone."""
    m = np.asarray(m, float); F = np.asarray(F, float)
    if not (np.isfinite(m_lo) and np.isfinite(m_hi) and np.isfinite(alpha)):
        return np.nan
    k = np.isfinite(F) & (F > 0) & (m >= m_lo) & (m <= m_hi)
    if k.sum() < 3:
        return np.nan
    return 10.0 ** np.mean(np.log10(F[k]) - alpha * np.log10(m[k]))

def spectrum_plateau(run, half=3, tol=0.30, min_decades=0.8, spec=None):
    """The inertial range of the spectrum: the longest flat run of d log F / d log m."""
    s = spec if spec is not None else spectrum(run)
    return BF.find_inertial_range(run["centers"], s["F"], half=half, tol=tol,
                                  min_decades=min_decades)

def growth_plateau(tau, mbar, counts=None, half=3, tol=0.50, min_bins=5,
                   min_slope=0.30, n_min=1e3):
    """The self-similar stretch of <m>(tau): the longest flat run of d log<m> / d log tau whose mean is an actual growth rate."""
    tau = np.asarray(tau, float); mbar = np.asarray(mbar, float)
    ok = np.isfinite(mbar) & (mbar > 0)
    if counts is not None:
        ok &= np.asarray(counts, float) > n_min
    x, y = tau[ok], mbar[ok]
    if x.size < min_bins + 2:
        return dict(b=np.nan, scatter=np.nan, decades=0.0, n_bins=0, tau_lo=np.nan,
                    tau_hi=np.nan, m_lo=np.nan, m_hi=np.nan)

    _, G = BF.local_slope(x, y, half=half)
    good = np.isfinite(G)
    best, i, n = None, 0, x.size
    while i < n:
        if not good[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and good[j + 1]:
            seg = G[i:j + 2]
            if seg.max() - seg.min() > 2 * tol:
                break
            j += 1
        seg = G[i:j + 1]
        if seg.size >= min_bins and abs(seg.mean()) >= min_slope:
            dec = np.log10(x[j] / x[i])
            if best is None or dec > best[0]:
                best = (dec, i, j)
        i = j + 1

    if best is None:
        return dict(b=np.nan, scatter=np.nan, decades=0.0, n_bins=0, tau_lo=np.nan,
                    tau_hi=np.nan, m_lo=np.nan, m_hi=np.nan)
    dec, i, j = best
    seg = G[i:j + 1]
    return dict(b=float(seg.mean()), scatter=float(seg.std()), decades=float(dec),
                n_bins=int(j - i + 1), tau_lo=float(x[i]), tau_hi=float(x[j]),
                m_lo=float(y[i]), m_hi=float(y[j]))

def growth_law(run, correct_repeats=True, **kw):
    """The whole chain in one call: counts -> <m>(tau) with sigma -> plateau -> weighted fit -> b with its uncertainty."""
    iso = iso_mean_mass(run, correct_repeats=correct_repeats)
    pl = growth_plateau(iso["tau"], iso["mbar"], iso["counts"], **kw)
    k = (np.isfinite(iso["mbar"]) & (iso["tau"] >= pl["tau_lo"])
         & (iso["tau"] <= pl["tau_hi"]))
    fit = wls_powerlaw(iso["tau"], iso["mbar"], iso["sigma"], mask=k)
    return iso, pl, fit, k

def iso_groups(run, pl=None, iso=None, n_curves=8, min_counts=1e3,
               inside_plateau=True, correct_repeats=True):
    """Merge the age bins into a handful of wide ones and return them as index groups."""
    empty = dict(rejected=np.array([], int), candidates=np.array([], int),
                 pl=pl, span=0.0, mode="none")
    if "iso_counts" not in run:
        return [], empty
    e = np.asarray(run["iso_age_edges"], float)
    tau = np.sqrt(e[:-1] * e[1:])
    cnt = np.asarray(run["iso_counts"], float).sum(axis=1)
    have = np.flatnonzero(cnt > 0)
    if have.size == 0:
        return [], empty

    mode = "all bins"
    cand = have
    if inside_plateau:
        if pl is None:
            if iso is None:
                iso = iso_mean_mass(run, correct_repeats=correct_repeats)
            pl = growth_plateau(iso["tau"], iso["mbar"], iso["counts"])
        if np.isfinite(pl.get("tau_lo", np.nan)):
            sel = have[(tau[have] >= pl["tau_lo"]) & (tau[have] <= pl["tau_hi"])]
            if sel.size >= 2:
                cand, mode = sel, "growth plateau"
    if cand.size < 2:
        cand, mode = have, "all bins (no plateau found)"

    span = float(np.log10(e[cand[-1] + 1] / e[cand[0]]))
    target = span / max(int(n_curves), 1) if span > 0 else np.inf

    groups, i = [], 0
    while i < cand.size:
        j = i
        while j + 1 < cand.size:
            wide = np.log10(e[cand[j] + 1] / e[cand[i]]) >= target
            full = cnt[cand[i:j + 1]].sum() >= min_counts
            if wide and full:
                break
            j += 1
        groups.append(cand[i:j + 1])
        i = j + 1
    if len(groups) > 1 and cnt[groups[-1]].sum() < min_counts:
        groups[-2] = np.concatenate([groups[-2], groups[-1]])
        groups.pop()

    used = np.concatenate(groups) if groups else np.array([], int)
    info = dict(rejected=np.setdiff1d(have, used), candidates=cand, pl=pl,
                span=span, mode=mode)
    return groups, info

def plateau_line(pl, fit=None, b_theory=None, tag=""):
    """One line stating the window the picture and the fit share, ready to print."""
    if pl is None or not np.isfinite(pl.get("b", np.nan)):
        return "%s  no growth plateau found" % tag
    s = ("%s  plateau: <m> from %.4g to %.4g, tau from %.4g to %.4g, "
         "%d age bins over %.2f dec, b = %.3f +- %.3f"
         % (tag, pl["m_lo"], pl["m_hi"], pl["tau_lo"], pl["tau_hi"],
            pl["n_bins"], pl["decades"], pl["b"], pl["scatter"]))
    if fit is not None and np.isfinite(fit.get("p", np.nan)):
        s += " | weighted fit %.4f +- %.4f" % (fit["p"], fit["sigma_p"])
    if b_theory:
        s += " | theory %.3f" % b_theory
    return s

def pick_isochrones(run, n_show=None, mode="plateau", min_frac=1e-4,
                    min_counts=1e3, pl=None, iso=None):
    """Which isochrones to draw."""
    if "iso_dndm" not in run:
        return [], np.array([]), 0
    e = np.asarray(run["iso_age_edges"], float)
    tau = np.sqrt(e[:-1] * e[1:])
    cnt = np.asarray(run["iso_counts"], float).sum(axis=1)
    if not np.any(cnt > 0):
        return [], tau, 0

    if mode == "plateau":
        groups, info = iso_groups(run, pl=pl, iso=iso,
                                  n_curves=(8 if not n_show else int(n_show)),
                                  min_counts=min_counts)
        return groups, tau, int(info["candidates"].size)

    usable = np.flatnonzero(cnt > min_frac * cnt.max())
    if usable.size == 0:
        return [], tau, 0
    if n_show and n_show < usable.size:
        usable = usable[np.unique(np.linspace(0, usable.size - 1,
                                              int(n_show)).astype(int))]
    return [np.array([k]) for k in usable], tau, int(usable.size)

def draw_isochrones(ax, run, idx, tau, tau_scale=1.0, cbar_label=r"age $\tau$",
                    min_count=3.0, show_rejected=True, rejected=None):
    """Isochrones coloured by age."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    if len(idx) == 0:
        return
    groups = [np.atleast_1d(np.asarray(g, int)) for g in idx]
    c = np.asarray(run["centers"], float)
    cnt = np.asarray(run["iso_counts"], float)
    dn = np.asarray(run["iso_dndm"], float)
    nsnp = max(float(run["iso_snapshots"]), 1.0)

    if show_rejected:
        if rejected is None:
            used = np.concatenate(groups)
            alive = np.flatnonzero(cnt.sum(axis=1) > 0)
            rejected = np.setdiff1d(alive, used)
        for k in np.atleast_1d(rejected):
            y = dn[k] / nsnp
            ok = (y > 0) & (cnt[k] >= min_count)
            if ok.sum() < 2:
                continue
            ax.loglog(c, np.where(ok, y * c ** 2, np.nan), lw=.7, alpha=.55,
                      color="0.78", zorder=0)

    tg = np.array([np.average(tau[g], weights=np.maximum(cnt[g].sum(axis=1), 1e-300))
                   for g in groups])
    lo, hi = tg.min() * tau_scale, tg.max() * tau_scale
    norm = LogNorm(vmin=lo, vmax=max(hi, lo * 1.0001))
    cm = plt.cm.viridis
    for g, tk in zip(groups, tg):
        cg = cnt[g].sum(axis=0)
        y = dn[g].sum(axis=0) / nsnp
        ok = (y > 0) & (cg >= min_count)
        ax.loglog(c, np.where(ok, y * c ** 2, np.nan), lw=1.6, alpha=.95,
                  color=cm(norm(tk * tau_scale)), zorder=1)
    sm = plt.cm.ScalarMappable(cmap=cm, norm=norm); sm.set_array([])
    cb = ax.figure.colorbar(sm, ax=ax, pad=.02, fraction=.05)
    cb.set_label(cbar_label, fontsize=7); cb.ax.tick_params(labelsize=6)

def compensated_ylim(ax, m, F, lo=3e-5, hi=4.0):
    """Frame a compensated panel on the steady state, not on the faintest isochrone."""
    y = np.asarray(F, float) * np.asarray(m, float) ** 2
    top = np.nanmax(np.where(np.isfinite(y) & (y > 0), y, np.nan))
    if np.isfinite(top) and top > 0:
        ax.set_ylim(top * lo, top * hi)

def plot_spectrum(ax, run, spec=None, plateau=None, compensate=True, errorbars=True,
                  color="k", label="measured"):
    """Spectrum with error bars, compensated as m^2 dN/dm by default."""
    s = spec if spec is not None else spectrum(run)
    c = np.asarray(run["centers"], float)
    f = c ** 2 if compensate else np.ones_like(c)
    k = np.isfinite(s["F"]) & (s["F"] > 0)
    if errorbars:
        ax.errorbar(c[k], (s["F"] * f)[k], yerr=(s["sigma"] * f)[k], fmt="o", ms=3,
                    lw=0, elinewidth=.8, capsize=1.5, color=color, zorder=4, label=label)
    else:
        ax.loglog(c[k], (s["F"] * f)[k], "o", ms=3, color=color, zorder=4, label=label)
    ax.set_xscale("log"); ax.set_yscale("log")
    if plateau is not None and np.isfinite(plateau.get("m_lo", np.nan)):
        xs = np.logspace(np.log10(plateau["m_lo"]), np.log10(plateau["m_hi"]), 30)
        A = anchor_amplitude(c, s["F"], plateau["m_lo"], plateau["m_hi"], plateau["alpha"])
        ax.plot(xs, A * xs ** plateau["alpha"] * (xs ** 2 if compensate else 1), "-",
                lw=2.2, color="C3", zorder=5,
                label=r"plateau $\alpha=%.2f$" % plateau["alpha"])
    ax.set_xlabel("m")
    ax.set_ylabel(r"$m^2\,dN/dm$" if compensate else "dN/dm")
    return s

def plot_growth(axs, run, iso=None, pl=None, fit=None, k=None, b_theory=None,
                color="C0", tag=""):
    """Two panels: <m>(tau) with error bars and every age bin visible, and the local slope beside it so the plateau can be seen rather than trusted."""
    if iso is None:
        iso, pl, fit, k = growth_law(run)
    tau, mbar, sig = iso["tau"], iso["mbar"], iso["sigma"]
    have = np.isfinite(mbar) & (mbar > 0) & (iso["counts"] > 0)

    a = axs[0]
    a.errorbar(tau[have], mbar[have], yerr=sig[have], fmt="o", ms=4, mfc="none",
               mec="0.6", mew=.9, lw=0, elinewidth=.7, ecolor="0.75",
               label="all age bins (%d)" % have.sum())
    if k is not None and k.sum():
        a.errorbar(tau[k], mbar[k], yerr=sig[k], fmt="o", ms=4, color=color, lw=0,
                   elinewidth=.9, capsize=1.5, label="on the plateau (%d)" % k.sum())
    if fit is not None and np.isfinite(fit["p"]):
        a.plot(tau[have], fit["A"] * tau[have] ** fit["p"], "--", lw=1, color="0.4",
               label="fit extrapolated")
        a.plot(tau[k], fit["A"] * tau[k] ** fit["p"], "-", lw=2.2, color="C3",
               label=r"fit $\tau^{%.3f\pm%.3f}$" % (fit["p"], fit["sigma_p"]))
    if b_theory and np.isfinite(b_theory) and k is not None and k.sum():
        a.plot(tau[k], mbar[k][0] * (tau[k] / tau[k][0]) ** b_theory, ":", lw=1.6,
               color="C1", label=r"theory $\tau^{%.1f}$" % b_theory)
    a.set_xscale("log"); a.set_yscale("log")
    a.set_ylim(np.nanmin(mbar[have]) / 3.0, np.nanmax(mbar[have]) * 3.0)
    a.set_xlabel(r"age $\tau$"); a.set_ylabel(r"$\langle m\rangle$")
    a.set_title("%s — growth law, all data" % tag, fontsize=9)
    a.legend(fontsize=6, loc="upper left")

    lt, lm = np.log(tau), np.log(mbar)
    idx = np.flatnonzero(have)
    slope = np.full(tau.size, np.nan)
    for j in range(1, idx.size - 1):
        slope[idx[j]] = ((lm[idx[j + 1]] - lm[idx[j - 1]]) /
                         (lt[idx[j + 1]] - lt[idx[j - 1]]))
    a2 = axs[1]
    ok = np.isfinite(slope)
    a2.semilogx(mbar[ok], slope[ok], "o-", ms=3.5, lw=.8, color="0.55",
                label="local slope")
    if k is not None:
        a2.semilogx(mbar[ok & k], slope[ok & k], "o", ms=4.5, color=color,
                    label="plateau")
    if b_theory and np.isfinite(b_theory):
        a2.axhline(b_theory, ls="--", lw=1, color="C1",
                   label="theory $b=%.1f$" % b_theory)
    if fit is not None and np.isfinite(fit["p"]):
        a2.axhline(fit["p"], ls="-", lw=1, color="C3",
                   label=r"fit $b=%.3f\pm%.3f$" % (fit["p"], fit["sigma_p"]))
        if pl is not None and np.isfinite(pl["m_lo"]):
            a2.axvspan(min(pl["m_lo"], pl["m_hi"]), max(pl["m_lo"], pl["m_hi"]),
                       color="C1", alpha=.10)
    a2.set_xlabel(r"$\langle m\rangle$")
    a2.set_ylabel(r"$d\log\langle m\rangle/d\log\tau$")
    a2.set_ylim(-0.5, (b_theory if b_theory else 6) * 1.35)
    a2.set_title("where the slope settles", fontsize=9)
    a2.legend(fontsize=6, loc="upper left")
    return iso, pl, fit, k

def growth_compare(run, groups=None, correct_repeats=True, **kw):
    """The growth law measured twice on the same particles: once on the age grid as recorded, once on the merged age bins the isochrone panel is drawn from."""
    iso = iso_mean_mass(run, correct_repeats=correct_repeats)
    pl = growth_plateau(iso["tau"], iso["mbar"], iso["counts"], **kw)
    k = (np.isfinite(iso["mbar"]) & (iso["tau"] >= pl["tau_lo"])
         & (iso["tau"] <= pl["tau_hi"]))
    fit = wls_powerlaw(iso["tau"], iso["mbar"], iso["sigma"], mask=k)

    if groups is None:
        groups, _ = iso_groups(run, pl=pl, iso=iso, correct_repeats=correct_repeats)
    iso_g = iso_mean_mass(run, correct_repeats=correct_repeats, groups=groups)
    fit_g = (wls_powerlaw(iso_g["tau"], iso_g["mbar"], iso_g["sigma"])
             if iso_g["tau"].size >= 3 else
             dict(p=np.nan, sigma_p=np.nan, A=np.nan, chi2_dof=np.nan,
                  rms_resid=np.nan, n=0))
    return dict(iso=iso, pl=pl, fit=fit, k=k, groups=groups, iso_g=iso_g, fit_g=fit_g)

def plot_growth_compare(axs, run, cmp=None, groups=None, b_theory=None,
                        color="C0", tag=""):
    """The two growth laws side by side: the original fine age bins and the merged ones the isochrones were drawn from."""
    if cmp is None:
        cmp = growth_compare(run, groups=groups)
    iso, pl, fit, k = cmp["iso"], cmp["pl"], cmp["fit"], cmp["k"]
    iso_g, fit_g = cmp["iso_g"], cmp["fit_g"]
    tau, mbar, sig = iso["tau"], iso["mbar"], iso["sigma"]
    have = np.isfinite(mbar) & (mbar > 0) & (iso["counts"] > 0)

    a = axs[0]
    a.errorbar(tau[have], mbar[have], yerr=sig[have], fmt="o", ms=3.5, mfc="none",
               mec="0.72", mew=.8, lw=0, elinewidth=.6, ecolor="0.82",
               label="original bins, all (%d)" % have.sum())
    if k.sum():
        a.errorbar(tau[k], mbar[k], yerr=sig[k], fmt="o", ms=4, color=color, lw=0,
                   elinewidth=.9, capsize=1.5, label="original, on plateau (%d)" % k.sum())
        if np.isfinite(fit["p"]):
            a.plot(tau[k], fit["A"] * tau[k] ** fit["p"], "-", lw=2.2, color="C3",
                   label=r"original: $b=%.3f\pm%.3f$" % (fit["p"], fit["sigma_p"]))
    tg, mg, sg = iso_g["tau"], iso_g["mbar"], iso_g["sigma"]
    if tg.size:
        a.errorbar(tg, mg, yerr=sg, fmt="s", ms=8, mfc="none", mec="C2", mew=1.6,
                   lw=0, elinewidth=1.1, ecolor="C2", capsize=2,
                   label="rebinned (%d curves)" % tg.size)
        if np.isfinite(fit_g["p"]):
            a.plot(tg, fit_g["A"] * tg ** fit_g["p"], "--", lw=1.8, color="C2",
                   label=r"rebinned: $b=%.3f\pm%.3f$" % (fit_g["p"], fit_g["sigma_p"]))
    if b_theory and np.isfinite(b_theory) and k.sum():
        a.plot(tau[k], mbar[k][0] * (tau[k] / tau[k][0]) ** b_theory, ":", lw=1.6,
               color="C1", label=r"theory $\tau^{%.1f}$" % b_theory)
    a.set_xscale("log"); a.set_yscale("log")
    a.set_ylim(np.nanmin(mbar[have]) / 3.0, np.nanmax(mbar[have]) * 3.0)
    a.set_xlabel(r"age $\tau$"); a.set_ylabel(r"$\langle m\rangle$")
    a.set_title("%s — original vs rebinned" % tag, fontsize=9)
    a.legend(fontsize=6, loc="upper left")

    def _slope(x, y):
        s = np.full(np.size(x), np.nan)
        i = np.flatnonzero(np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0))
        for j in range(1, i.size - 1):
            s[i[j]] = ((np.log(y[i[j + 1]]) - np.log(y[i[j - 1]])) /
                       (np.log(x[i[j + 1]]) - np.log(x[i[j - 1]])))
        return s

    a2 = axs[1]
    sf = _slope(tau, mbar); ok = np.isfinite(sf)
    a2.semilogx(mbar[ok], sf[ok], "o-", ms=3.5, lw=.8, color="0.6",
                label="original bins")
    if k.sum():
        a2.semilogx(mbar[ok & k], sf[ok & k], "o", ms=4.5, color=color, label="plateau")
    sg2 = _slope(tg, mg); okg = np.isfinite(sg2)
    if okg.any():
        a2.semilogx(mg[okg], sg2[okg], "s--", ms=7, mfc="none", mec="C2", mew=1.5,
                    lw=1.1, color="C2", label="rebinned")
    if b_theory and np.isfinite(b_theory):
        a2.axhline(b_theory, ls="--", lw=1, color="C1",
                   label="theory $b=%.1f$" % b_theory)
    if np.isfinite(fit["p"]):
        a2.axhline(fit["p"], ls="-", lw=1, color="C3", label="original fit")
    if np.isfinite(fit_g["p"]):
        a2.axhline(fit_g["p"], ls=":", lw=1.4, color="C2", label="rebinned fit")
    if np.isfinite(pl["m_lo"]):
        a2.axvspan(min(pl["m_lo"], pl["m_hi"]), max(pl["m_lo"], pl["m_hi"]),
                       color="C1", alpha=.10)
    a2.set_xlabel(r"$\langle m\rangle$")
    a2.set_ylabel(r"$d\log\langle m\rangle/d\log\tau$")
    a2.set_ylim(-0.5, (b_theory if b_theory else 6) * 1.35)
    a2.set_title("the same slope, sampled twice", fontsize=9)
    a2.legend(fontsize=6, loc="upper left")
    return cmp

def check_grid(edges, *masses):
    """Refuse to start if a characteristic mass falls outside the bin grid."""
    e = np.asarray(edges, float)
    bad = [float(m) for m in masses if m is not None and not (e[0] < m < e[-1])]
    if bad:
        raise ValueError("masses %s fall outside the grid (%.3g, %.3g) -- widen EDGES"
                         % (bad, e[0], e[-1]))

def _jsonable(v):
    if v is None or isinstance(v, (str, bool)):
        return v
    try:
        return float(v)
    except (TypeError, ValueError):
        return str(v)

def steady_mean_mass(alpha, m_lo, m_hi):
    """<m> = M/N for a truncated power law F ~ m^alpha on [m_lo, m_hi]."""
    a_, lo, hi = float(alpha), float(m_lo), float(m_hi)

    def _int(e):
        return (np.log(hi / lo) if abs(e + 1.0) < 1e-12
                else (hi ** (e + 1.0) - lo ** (e + 1.0)) / (e + 1.0))

    return _int(a_ + 1.0) / _int(a_)

def waiting_band(run, pad_decades=(1.0, 0.15)):
    """The default fitting window for <dt>(m), and it is DELIBERATELY ASYMMETRIC."""
    meta = run["meta"]
    m_inj = float(meta.get("injection_mass") or meta["ic"]["m"])
    m_sink = float(meta.get("sink_mass") or 0.0)
    try:
        p_lo, p_hi = float(pad_decades[0]), float(pad_decades[1])
    except (TypeError, IndexError):
        p_lo = p_hi = float(pad_decades)
    if m_sink <= 0 or not np.isfinite(m_sink):
        c = np.asarray(run["centers"], float)
        return float(c.min()), float(c.max())
    if m_sink < m_inj:
        return m_sink * 10.0 ** p_lo, m_inj / 10.0 ** p_hi
    return m_inj * 10.0 ** p_hi, m_sink / 10.0 ** p_lo

def waiting_law(run, estimator="exposure", gated=None, band=None,
                pad_decades=(1.0, 0.15), min_events=200.0, weighted=False,
                verbose=False, _cross=True):
    """Fit <dt>(m) = A m^(1/b) and return 1/b, b and the diagnostics."""
    if "wait_mean" not in run:
        raise ValueError("this run carries no waiting law "
                         "(engine older than v5, or track_waiting=False)")
    c = np.asarray(run["centers"], float)
    meta = run["meta"]
    m_inj = float(meta.get("injection_mass") or meta["ic"]["m"])
    m_sink = float(meta.get("sink_mass") or 0.0)

    def _pair(g):
        sfx = "_gated" if g else ""
        return (np.asarray(run["wait_mean" + sfx], float),
                np.asarray(run["wait_sem" + sfx], float),
                np.asarray(run["wait_events" + sfx], float))

    if gated is None:
        _, _, n_g = _pair(True)
        _, _, n_u = _pair(False)
        gated = bool(n_g.sum() >= 0.5 * n_u.sum() and n_g.sum() > 0)
    y, sy, n = _pair(gated)
    if estimator == "interval":
        sfx = "_gated" if gated else ""
        y = np.asarray(run["wait_mean_interval" + sfx], float)
        sy = np.asarray(run.get("wait_sem_interval", np.full_like(y, np.nan)), float)
        n = np.asarray(run["wait_n" + ("_gated" if gated else "")], float)
    elif estimator != "exposure":
        raise ValueError("estimator must be 'exposure' or 'interval'")

    if band is None:
        band = waiting_band(run, pad_decades)
    band = (min(band), max(band))
    mask = (c >= band[0]) & (c <= band[1]) & (n >= float(min_events)) \
        & np.isfinite(y) & (y > 0)

    fit = wls_powerlaw(c, y, sy=(sy if weighted else None), mask=mask)
    p, sp = fit["p"], fit["sigma_p"]

    trials = []
    for dlo in (0.0, 0.25, 0.5):
        for dhi in (0.0, 0.25):
            bb = (band[0] * 10.0 ** dlo, band[1] / 10.0 ** dhi)
            mk = (c >= bb[0]) & (c <= bb[1]) & (n >= float(min_events)) \
                & np.isfinite(y) & (y > 0)
            if mk.sum() >= 5:
                q = wls_powerlaw(c, y, sy=(sy if weighted else None), mask=mk)["p"]
                if np.isfinite(q):
                    trials.append(q)
    p_spread = float(np.max(trials) - np.min(trials)) if len(trials) > 1 else np.nan
    b = 1.0 / p if (np.isfinite(p) and p != 0.0) else np.nan
    sb = (b ** 2) * sp if np.isfinite(b) and np.isfinite(sp) else np.nan

    other = "interval" if estimator == "exposure" else "exposure"
    p_other = np.nan
    if _cross:
        try:
            p_other = waiting_law(run, estimator=other, gated=gated, band=band,
                                  pad_decades=pad_decades, min_events=min_events,
                                  weighted=weighted, _cross=False)["p"]
        except Exception:
            p_other = np.nan

    out = dict(fit)
    out.update(p=p, sigma_p=sp, b=b, sigma_b=sb, mask=mask, m=c, dt=y, sem=sy,
               n_events=n, band=tuple(band), gated=bool(gated), estimator=estimator,
               p_other=p_other, other=other, p_spread=p_spread, weighted=bool(weighted),
               sigma_b_sys=(b ** 2) * p_spread if np.isfinite(b) and np.isfinite(p_spread)
               else np.nan,
               decades=(np.log10(c[mask].max() / c[mask].min()) if mask.sum() > 1
                        else np.nan))
    if verbose:
        print(waiting_report(run, out))
    return out

def waiting_report(run, wl=None, alpha=None, sigma_alpha=None):
    """Text block: 1/b and b from the waiting law, next to b from the closure, with the censoring check and the honest fraction."""
    wl = wl if wl is not None else waiting_law(run)
    lam = run["meta"].get("lambda")
    L = ["-" * 74,
         "WAITING LAW   <dt>(m) ~ m^(1/b)      [%s estimator, %s]"
         % (wl["estimator"], "gated" if wl["gated"] else "ungated, from t=0"),
         "  window        %.3g .. %.3g   (%d bins, %.2f decades)"
         % (wl["band"][0], wl["band"][1], wl["n"], wl["decades"]),
         "  1/b           %+.5f +- %.5f (stat) +- %.5f (window)   chi2/dof = %s"
         % (wl["p"], wl["sigma_p"], wl["p_spread"],
            "n/a" if not np.isfinite(wl["chi2_dof"]) else "%.2f" % wl["chi2_dof"]),
         "  b             %.3f +- %.3f (stat) +- %.3f (window)"
         % (wl["b"], wl["sigma_b"], wl["sigma_b_sys"]),
         "                (sigma_b = b^2 sigma_(1/b): the amplification is physics,"
         " not a defect)"]
    if np.isfinite(wl["p_other"]):
        d = wl["p_other"] - wl["p"]
        L.append("  cross-check   %s estimator gives 1/b = %+.5f  (%+.5f, %s)"
                 % (wl["other"], wl["p_other"], d,
                    "consistent" if abs(d) <= 2 * wl["sigma_p"]
                    else "DIFFERENT -> censoring, run longer"))
    if lam is not None and np.isfinite(lam):
        _invb_th = 0.5 * (1.0 - lam)
        _b_th = (2.0 / (1.0 - lam)) if abs(1.0 - lam) > 1e-12 else np.inf
        L.append("  theory        1/b = (1-lambda)/2 = %+.5f   ->  b = %s"
                 % (_invb_th, ("%.3f" % _b_th) if np.isfinite(_b_th)
                    else "inf  (lambda = 1: exponential drift, no growth exponent)"))
    if alpha is not None and np.isfinite(alpha):
        L.append("  from alpha    alpha = %+.4f%s" %
                 (alpha, "" if sigma_alpha is None else " +- %.4f" % sigma_alpha))
        L.append("                closure  1/b = 2+alpha        = %+.5f  ->  b = %.3f"
                 % (2 + alpha, 1.0 / (2 + alpha) if (2 + alpha) != 0 else np.nan))
        if lam is not None and np.isfinite(lam):
            k = -(1.0 + lam + alpha)
            L.append("                kinetic  1/b = -(1+lam+alpha) = %+.5f  ->  b = %.3f"
                     % (k, 1.0 / k if k != 0 else np.nan))
            L.append("                measured minus kinetic     = %+.5f"
                     % (wl["p"] - k))
    hn = run.get("honest_num")
    if hn is not None and np.size(hn):
        L.append("  honest        %.3f of the population has a real history"
                 % float(np.asarray(hn)[-1]))
    if run["meta"].get("seeded"):
        s = run["meta"].get("seed") or {}
        L.append("  SEEDED        alpha_seed = %+.4f on [%.3g, %.3g], N = %s"
                 % (s.get("alpha", np.nan), s.get("m_lo", np.nan),
                    s.get("m_hi", np.nan), s.get("N", "?")))
        L.append("                the measured alpha is only meaningful if it has"
                 " MOVED off this value")
    mi, mo = float(np.asarray(run["M_in"])[-1]), float(np.asarray(run["M_out"])[-1])
    L.append("  mass balance  M_out/M_in = %.3f   (1.0 = the box is flushed)"
             % (mo / mi if mi > 0 else np.nan))
    L.append("-" * 74)
    return "\n".join(L)

def time_avg_spectrum(run, gated=None):
    """dN/dm averaged over the run rather than sampled at snapshots."""
    if "dndm_time_avg" not in run:
        return None, None
    if gated is None:
        gated = bool(np.isfinite(run.get("wait_t_span_gated", np.nan))
                     and float(run["wait_t_span_gated"]) > 0)
    key = "dndm_time_avg_gated" if gated else "dndm_time_avg"
    return np.asarray(run["centers"], float), np.asarray(run[key], float)

def seed_spectrum(alpha, m_lo, m_hi, N, rng=None):
    """N masses from n(m) ~ m^alpha on [m_lo, m_hi]."""
    if sample_powerlaw is None:
        raise ValueError("the engine on the path has no seeding (need v5)")
    return sample_powerlaw(alpha, m_lo, m_hi, N, rng)

def honesty(run):
    """The seeded fraction as a time series: how much of the run is usable for anything that depends on ancestry (generations, isochrones, ages)."""
    if "honest_num" not in run:
        return None
    return {"t": np.asarray(run["t"], float),
            "n_out": np.asarray(run["n_out"], float),
            "num": np.asarray(run["honest_num"], float),
            "mass": np.asarray(run["honest_mass"], float),
            "seeded": bool(run["meta"].get("seeded", False))}

def stationarity(run, tail=0.5, tol_mbar=0.03, tol_alpha=0.010,
                 tol_msys=0.03, band_pad=1.0, verbose=False):
    """Tolerances are stated per flush, not per comparison window."""
    """Whether the run is stationary enough to take a checkpoint from it.

    The TAIL of the run is taken (the last fraction `tail` of the snapshots),
    split in half, and the means of the two halves are compared.  The two halves
    of the tail rather than the beginning against the end: the question is whether
    the system is moving NOW, not how far it has travelled from its initial
    condition.

    What is checked:
      mbar   <m> = M_sys/N        -- the sharpest indicator of shape
      alpha  the slope over the guard band, snapshot by snapshot
      msys   the total mass       -- a derivative, not a cumulative ratio
      dnout  dn_out/devents       -- weak, kept for completeness

    Returns a dict with the numbers, the verdict `ok` and the text `report`.
    Default tolerances: 3% on <m> and M_sys, 0.010 on alpha -- the last being
    about four times smaller than the typical scatter of the plateau, i.e. a
    demand stricter than the accuracy with which alpha can be measured at all.
    """

    e = np.asarray(run["events"], float)
    if e.size < 8:
        return {"ok": False, "report": "fewer than eight snapshots -- nothing to judge",
                "n": int(e.size)}
    lv = np.asarray(run["live"], float)
    Ms = np.asarray(run["M_sys"], float)
    no = np.asarray(run["n_out"], float)
    c = np.asarray(run["centers"], float)
    D = np.asarray(run["dndm"], float)
    meta = run["meta"]
    m_inj = float(meta.get("injection_mass") or meta["ic"]["m"])
    m_sink = float(meta.get("sink_mass") or 0.0)
    lo, hi = (m_sink, m_inj) if m_sink < m_inj else (m_inj, m_sink)
    band = (c >= lo * 10.0 ** band_pad) & (c <= hi / 10.0 ** band_pad)

    mbar = np.where(lv > 0, Ms / np.maximum(lv, 1), np.nan)
    al = np.full(e.size, np.nan)
    for i in range(e.size):
        k = band & (D[i] > 0)
        if k.sum() >= 6:
            al[i] = np.polyfit(np.log(c[k]), np.log(D[i][k]), 1)[0]

    i0 = max(int(e.size * (1.0 - tail)), 0)
    idx = np.arange(i0, e.size)
    half = i0 + (e.size - i0) // 2
    A, B = np.arange(i0, half), np.arange(half, e.size)
    if A.size < 2 or B.size < 2:
        return {"ok": False, "report": "the tail is too short for two halves",
                "n": int(e.size)}

    def _cmp(x, rel):
        a, b = np.nanmean(x[A]), np.nanmean(x[B])
        sa = np.nanstd(x[A]) / np.sqrt(max(effective_snapshots(x[A]), 1.0))
        sb = np.nanstd(x[B]) / np.sqrt(max(effective_snapshots(x[B]), 1.0))
        sd = float(np.hypot(sa, sb))
        if rel:
            d = (b - a) / a if np.isfinite(a) and a != 0 else np.nan
            sd = sd / abs(a) if np.isfinite(a) and a != 0 else np.nan
        else:
            d = b - a
        return a, b, d, sd

    mb_a, mb_b, d_mb, s_mb = _cmp(mbar, True)
    ms_a, ms_b, d_ms, s_ms = _cmp(Ms, True)
    al_a, al_b, d_al, s_al = _cmp(al, False)
    dn = (no[-1] - no[i0]) / max(e[-1] - e[i0], 1.0)

    flush0 = Ms[0] / m_sink if m_sink > 0 else np.nan
    span = (float(e[B].mean()) - float(e[A].mean())) / flush0 if np.isfinite(flush0) else np.nan
    if not np.isfinite(span) or span <= 0:
        span = np.nan
    d_mb, s_mb = d_mb / span, s_mb / span
    d_ms, s_ms = d_ms / span, s_ms / span
    d_al, s_al = d_al / span, s_al / span

    flush = flush0
    checks = [("<m>", d_mb, s_mb, tol_mbar), ("alpha", d_al, s_al, tol_alpha),
              ("M_sys", d_ms, s_ms, tol_msys)]

    def _verdict(v, sd, t):
        if not np.isfinite(v):
            return "?"
        if abs(v) <= t:
            return "ok"
        return "MOVING" if (np.isfinite(sd) and abs(v) > 2.0 * sd) else "noise?"

    verd = [_verdict(v, sd, t) for _, v, sd, t in checks]
    ok = all(x == "ok" for x in verd)
    noisy = any(x == "noise?" for x in verd)

    L = ["-" * 70,
         "STATIONARITY  (two halves of the last %.0f%% of the snapshots)" % (100 * tail),
         "  run length      %.3g events = %.2f flushes of the box" % (e[-1], e[-1] / flush)
         if np.isfinite(flush) else "  run length      %.3g events" % e[-1],
         "  half separation %.4f flushes -- every shift below is divided by it" % span,
         "  %-8s %11s %11s %18s  %-7s %s"
         % ("", "first", "second", "shift per flush", "tol", "verdict")]
    for (name, v, sd, t), vd in zip(checks, verd):
        a, b = ((mb_a, mb_b) if name == "<m>" else
                (al_a, al_b) if name == "alpha" else (ms_a, ms_b))
        L.append("  %-8s %11.4g %11.4g %18s  %-7.3g %s"
                 % (name, a, b, "%+.4f +- %.4f" % (v, sd), t, vd))
    L.append("  %-8s %11.3f %11s %11s  %-8s %s"
             % ("dn/dev", dn, "", "", "1.00",
                "(weak criterion: settles on one long before the shape does)"))
    mi = float(np.asarray(run["M_in"])[-1]); mo = float(np.asarray(run["M_out"])[-1])
    L.append("  %-8s %11.3f %11s %11s  %-8s %s"
             % ("M_o/M_i", mo / mi if mi > 0 else np.nan, "", "", "1.00",
                "(weak: -> 1 automatically, see the note above)"))
    L.append("  VERDICT: %s"
             % ("STATIONARY -- usable as the start of a measurement" if ok else
                ("NOT STATIONARY -- keep running" if not noisy else
                 "NOT STATIONARY, but some shifts are within their own error -- "
                 "what is needed is length, not a verdict")))
    if noisy:
        L.append("  ('noise?': the shift exceeds the tolerance but is smaller than")
        L.append("   twice its own error -- movement cannot be told from fluctuation")
        L.append("   at this length.)")
    else:
        L.append("  (the error on the shift is computed from INDEPENDENT snapshots:")
        L.append("   adjacent ones share the population, and the naive error from")
        L.append("   their number is too small by a factor of several.)")
    est = np.nan
    if not ok and np.isfinite(d_mb) and abs(d_mb) > tol_mbar and np.isfinite(flush):
        t0 = max(int(e.size * (1.0 - min(tail * 1.5, 0.9))), 0)
        cut = np.array_split(np.arange(t0, e.size), 3)
        if all(c_.size >= 2 for c_ in cut):
            mm = [float(np.nanmean(mbar[c_])) for c_ in cut]
            d1 = (mm[1] - mm[0]) / mm[0]
            d2 = (mm[2] - mm[1]) / mm[1]
            dn_fl = float(e[cut[2]].mean() - e[cut[1]].mean()) / flush
            if np.isfinite(d1) and np.isfinite(d2) and d1 * d2 > 0 and abs(d2) < abs(d1):
                n0 = dn_fl / np.log(abs(d1) / abs(d2))
                est = n0 * np.log(abs(d_mb) / tol_mbar)
                L.append("  estimate: the drift of <m> decays with a scale of %.2f flushes," % n0)
                L.append("            to the %.0f%% tolerance about %.1f flushes more = %.3g events"
                         % (100 * tol_mbar, est, est * flush))
            else:
                L.append("  estimate: the drift of <m> does NOT decay along the tail (%+.3f -> %+.3f) --"
                         % (d1, d2))
                L.append("            the run is far from stationary and no term can be named")
    L.append("-" * 70)
    rep = "\n".join(L)
    if verbose:
        print(rep)
    return {"ok": bool(ok), "report": rep, "flushes": float(e[-1] / flush) if np.isfinite(flush) else np.nan,
            "mbar": (mb_a, mb_b, d_mb, s_mb), "alpha": (al_a, al_b, d_al, s_al),
            "msys": (ms_a, ms_b, d_ms, s_ms), "dn_dev": float(dn),
            "verdicts": dict(zip(("mbar", "alpha", "msys"), verd)),
            "noisy": bool(noisy),
            "flushes_left": float(est),
            "mbar_series": mbar, "alpha_series": al, "events": e}

def generations(run, weight="mass"):
    """Unpack the generation histogram."""
    if "gen_counts" not in run:
        raise ValueError("this run carries no generation histogram (engine older than v4)")
    H = np.asarray(run["gen_mass" if weight == "mass" else "gen_counts"], float)
    c = np.asarray(run["centers"], float)
    minj = float(run["meta"].get("injection_mass") or run["meta"]["ic"]["m"])
    return {"g": np.arange(H.shape[0]), "H": H, "centers": c,
            "widths": np.asarray(run["widths"], float), "x": np.log(c / minj),
            "num": np.asarray(run["gen_num"], float),
            "tau": np.asarray(run["gen_tau"], float),
            "tau_live": np.asarray(run.get("gen_tau_live", run["gen_tau"]), float),
            "reach_n": np.asarray(run.get("gen_reach_n", run["gen_num"]), float),
            "m_inj": minj, "m_sink": float(run["meta"].get("sink_mass") or 0.0),
            "weight": weight, "snapshots": float(run.get("gen_snapshots", 0.0))}

def gen_moments(gen, min_count=500.0, clear_sink=2.0):
    """<x> and Var(x) per generation, with a VALIDITY MASK."""
    H, x = gen["H"], gen["x"]
    xs = np.log(gen["m_sink"] / gen["m_inj"]) if gen["m_sink"] > 0 else -np.inf
    n = H.sum(axis=1)
    mu = np.full(H.shape[0], np.nan); va = np.full(H.shape[0], np.nan)
    for g in range(H.shape[0]):
        if n[g] < min_count:
            continue
        p = H[g] / n[g]
        mu[g] = float((p * x).sum())
        va[g] = float((p * (x - mu[g]) ** 2).sum())
    ok = np.isfinite(mu) & (mu - clear_sink * np.sqrt(np.maximum(va, 0)) > xs)
    ok[0] = False

    rn = np.asarray(gen.get("reach_n", n), float)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.full(rn.size, np.nan)
        ratio[1:] = rn[1:] / np.where(rn[:-1] > 0, rn[:-1], np.nan)
    ok_flux = np.nan_to_num(ratio, nan=0.0) > 1.8
    return {"g": gen["g"], "mu": mu, "var": va, "n": n, "ok": ok,
            "ok_flux": ok & ok_flux, "reach_n": rn, "reach_ratio": ratio,
            "x_sink": xs}

def gen_growth(gen, mom=None, min_count=500.0, s_max=120.0):
    """b from <tau>(g): the mean time to reach generation g, over the whole population."""
    if mom is None:
        mom = gen_moments(gen, min_count=min_count)
    k = mom.get("ok_flux", mom["ok"]) & np.isfinite(gen["tau"])
    if k.sum() < 4:
        return {"b": np.nan, "T": np.nan, "g": gen["g"][k], "tau": gen["tau"][k],
                "resid": np.nan, "resid_linear": np.nan, "scale": np.nan}
    gf, tf = gen["g"][k].astype(float), gen["tau"][k]
    sg = np.linspace(0.5, s_max, int(4 * s_max) + 1)
    rs = []
    for sv in sg:
        h = 1.0 - np.exp(-gf / sv)
        T = np.sum(tf * h) / np.sum(h * h)
        rs.append(np.std(tf - T * h))
    rs = np.array(rs); sb = float(sg[np.argmin(rs)])
    h = 1.0 - np.exp(-gf / sb); T = float(np.sum(tf * h) / np.sum(h * h))
    lin = np.polyfit(gf, tf, 1)
    mu_step = 0.5 if gen["weight"] == "mass" else 1.0
    return {"b": sb * mu_step, "scale": sb, "T": T, "g": gf, "tau": tf,
            "resid": float(rs.min() / tf.std()),
            "resid_linear": float(np.std(tf - np.polyval(lin, gf)) / tf.std())}

def step_stats(frag_split_width, weight="mass"):
    """(<ln xi>, Var(ln xi)) for ONE split, exactly, by quadrature."""
    xi = np.linspace(0.5 - frag_split_width, 0.5 + frag_split_width, 200001)[1:-1]
    tz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    p = (xi if weight == "mass" else np.ones_like(xi))
    p = p / tz(p, xi)
    mu = float(tz(p * np.log(xi), xi))
    return mu, float(tz(p * np.log(xi) ** 2, xi) - mu ** 2)

def add_last_run(run, name, analysis=None, drop_particles=True, runs_dir="runs"):
    """Write one finished run to runs/<name>.npz, replacing the previous one."""
    import os
    payload = {k: v for k, v in run.items()
               if not (drop_particles and k in ("final_mass", "final_inj_time",
                                                "final_age", "final_gen",
                                                "final_gen_time"))}
    meta = dict(run["meta"])
    meta["stop_reason"] = run.get("stop_reason", meta.get("stop_reason"))
    meta["saved_as"] = name
    meta["particles_dropped"] = bool(drop_particles)
    if analysis:
        meta["analysis"] = {k: _jsonable(v) for k, v in analysis.items()}
    payload["meta"] = meta

    os.makedirs(runs_dir, exist_ok=True)
    path = os.path.join(runs_dir, name + ".npz")
    BF.save_run(path, payload)
    a = meta.get("analysis", {})
    if a and not np.isfinite(a.get("alpha_plateau", 0.0)):
        print("   !! plateau is nan -- saved anyway, but this run has no inertial range")
    print("saved -> %s   (%.2f MB, stop = %s)"
          % (path, os.path.getsize(path) / 1e6, meta.get("stop_reason")))
    return path
