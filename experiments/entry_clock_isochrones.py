"""
entry_clock_isochrones.py -- FRAGMENTATION ISOCHRONES ON THE ENTRY CLOCK.

WHAT THIS IS FOR.  An isochrone is the mass distribution of the bodies of a
given age, and in fragmentation the word "age" is ambiguous: `frag_age_rule`
decides which piece of a split keeps the parent's clock.  The production runs
use 'heavier', which resets the clock of the lighter piece at every split.  That
is the right tracer for the FRONT -- the always-heavy chain is a characteristic
of the transport equation -- but it makes the age of a particle the time since
its own birth, not the time it has spent in the system.  Two consequences, both
measured on a six-decade geometric run:

  *  the young age classes are not young cascades.  A body enters at the rate
     q/m_inj, so an age bin of width dtau holds q dtau/m_inj entering bodies and
     of order N_ss dtau/<dt> reborn fragments; below a few per cent of the
     residence time the second number wins by six orders of magnitude and the
     class is made entirely of fresh light fragments, at every mass.

  *  the birth spectrum is itself a power law of index -2 -- B(m) ~ F(m) nu(m)
     ~ m^(2 alpha + lambda + 1) = m^-2 when alpha = -(3+lambda)/2 -- so a young
     isochrone is FLAT in m^2 dN/dm and looks as if it spanned the whole
     inertial range.  By number it does not: on that run 99% of it sat within
     half a decade of the sink.

So the early cascade is invisible on that clock, and the question it was meant
to answer -- what does a body look like after ONE split, after two, after three
-- has to be asked on the clock that starts when the body ENTERS.  That is
`frag_age_rule='inherit'`: both fragments keep the parent's stamp, so tau is the
time since the material entered and an age class is one cohort of injected
bodies together with everything it has broken into.

THREE DIFFERENCES FROM run_local.py, and nothing else:

  1.  frag_age_rule = 'inherit'.  It is an argument of the engine, not a patch:
      masscascade.py is not touched by this file.

  2.  A COLD start, not a seeded power law.  The seed enters with gen = -1 and
      inj_time = 0, and under 'inherit' so does its whole descendant tree: they
      would all pile into the oldest age bin and drown the cohorts.  A cold
      start has the same defect for its own fifty bodies, which is why

  3.  the isochrones are gated by TIME: accumulation starts at
      --gate-tau residence times, by which the cold bodies and their
      descendants have left through the sink and every live body entered as an
      injection.

THE AGE GRID IS HYBRID, and that is deliberate.  The interesting structure is
not where a linear grid puts its resolution.  One split takes a time of order
tau_1 = tau_res / <number of splits to the sink>, i.e. a few per cent of the
residence time, and the first few cohorts live there; the late cascade instead
needs a grid that does not stretch, because merging bins is exact and splitting
them is impossible.  So the stored grid is logarithmic (uniform in log tau,
--per-decade edges per decade) up to --knee residence times and linear beyond
it, with the linear step equal to the last logarithmic one.  Both readings are
then available at analysis time: the drawn groups are placed uniformly in log
tau, which on this grid is an exact merge, and a linear reading is one too.

WHAT IT PRINTS AND DRAWS.  The isochrones compensated by m^(11/6), coloured by
age, with their sum against the stationary spectrum as a closure check; the
generation histogram beside them, which is the same motion labelled by the
number of splits instead of by the clock; <m>(tau) with the generation points
overlaid, which is the conversion between the two labels; m^(1-beta) against
tau, which is a STRAIGHT LINE if the drift is dm/dtau ~ -D m^beta; and the
width of the packet in decades, which is the quantity the generation histogram
cannot give -- it stores one mean age per generation and no spread.

USAGE.  The gate costs one residence time before anything is binned, so the run
has to be long enough to reach it.  Passes accumulate, so this can be done an
invocation at a time:

    python experiments/entry_clock_isochrones.py --events 3e7
    python experiments/entry_clock_isochrones.py --events 3e7      # adds more
    python experiments/entry_clock_isochrones.py --analyse-only    # redraw

A cheap smoke test, seconds rather than hours, on a three-decade box with a
thousand bodies:

    python experiments/entry_clock_isochrones.py --n-ss 2e3 --m-top 1e3 \
        --events 2e6 --fresh

Outputs go to runs/, which is in .gitignore.
"""

import argparse
import os
import sys
import time

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import masscascade as BF          # noqa: E402
import analysis as AN             # noqa: E402
import run_local as RL            # noqa: E402


#  Everything is quoted against these, and they come from the kernel alone:
#  beta = (1+lambda)/2, alpha = -(3+lambda)/2, b = 2/(1-lambda) at lambda = 2/3.
LAMBDA = RL.LAMBDA
BETA_TH = RL.BETA_TH
ALPHA_TH = RL.ALPHA_TH
B_TH = RL.B_TH
COMP = RL.COMP                    # spectra are drawn compensated by m^(11/6)


# ======================================================================
#  THE BOX AND THE CLOCKS
# ======================================================================

def box(m_top=None, n_ss=None):
    """The fragmentation geometry, with the two knobs a smoke test needs.

    RL.geometry() and everything downstream of it read the module-level
    constants of run_local, so a test box is set by rebinding them rather than
    by duplicating the formulas here.  Single process, no workers, so there is
    nobody to disagree with.
    """
    if m_top is not None:
        RL.M_TOP = float(m_top)
    if n_ss is not None:
        RL.N_SS = float(n_ss)
    return RL.geometry("fragmentation")


def clocks(g):
    """t_c at the populated end, and the residence time of one injected body.

    The drift is a power law here, dm/dtau ~ -D m^beta with beta = 5/6, so the
    time to cross R decades is b t_c (R^(1-beta) - 1) and NOT t_c ln R, which
    is the lambda = 1 answer and is six times too small over six decades.
    """
    tc = RL._tc_of(BF, g, "fragmentation")
    R = max(g["m_inj"], g["m_sink"]) / min(g["m_inj"], g["m_sink"])
    return tc, B_TH * (R ** (1.0 - BETA_TH) - 1.0) * tc


def age_edges(tau_res, kind="hybrid", span=1.5, knee=0.05, per_decade=40,
              lo_frac=2e-4, n_linear=800):
    """The stored age grid: fine, and finer still where the first splits are.

    'log'     uniform in log tau from lo_frac*tau_res to span*tau_res, with a
              leading edge at zero so that nothing is dropped below the grid.
    'linear'  uniform in tau, the shape run_local uses for coagulation.
    'hybrid'  logarithmic up to knee*tau_res, linear beyond it with the step of
              the last logarithmic bin.  The default, and the one to use: the
              early cohorts need resolution in log tau, the late ones need a
              grid that does not stretch, and merging either way afterwards is
              exact because the engine accumulates counts.
    """
    t_max = float(span) * float(tau_res)
    if kind == "linear":
        return np.linspace(0.0, t_max, int(n_linear) + 1)
    lo = float(lo_frac) * float(tau_res)
    if kind == "log":
        n = max(int(round(per_decade * np.log10(t_max / lo))), 8)
        return np.concatenate([[0.0], np.logspace(np.log10(lo),
                                                  np.log10(t_max), n + 1)])
    if kind != "hybrid":
        raise ValueError("kind must be 'hybrid', 'log' or 'linear'")
    t_knee = float(knee) * float(tau_res)
    n = max(int(round(per_decade * np.log10(t_knee / lo))), 8)
    lg = np.logspace(np.log10(lo), np.log10(t_knee), n + 1)
    step = float(lg[-1] - lg[-2])
    lin = np.arange(lg[-1] + step, t_max + 0.5 * step, step)
    return np.concatenate([[0.0], lg, lin])


# ======================================================================
#  THE RUN
# ======================================================================

def sim_kwargs(g, args, ic, edges_age, events, q):
    """Exactly run_local's fragmentation box, plus the entry clock and the age
    grid.  No engine argument here is new; 'inherit' is the documented default
    of frag_age_rule and the age grid is the same one coagulation already uses.
    """
    return dict(
        process="fragmentation", system="open",
        kernel=getattr(BF, RL.KERNEL_NAME), edges=RL.mass_edges(g),
        ic=ic, injection_rate=q, injection_mass=g["m_inj"],
        sink_mass=g["m_sink"],
        frag_min_ratio=float(args.f), frag_split_width=float(args.w),
        #  THE WHOLE POINT OF THIS FILE.
        frag_age_rule="inherit",
        iso_age_edges=edges_age,
        #  The gate is on TIME, and it is computed from the box rather than
        #  typed: the clock runs as dt ~ 1/N^2, so a fixed number tuned at one
        #  population is meaningless at another.
        iso_t_start=float(args.gate_tau) * float(args.tau_res),
        iso_start_sink=0,
        track_generations=True, gen_max=RL.GEN_MAX, track_waiting=False,
        snapshot_mode="events",
        snapshot_stride=max(int(events) // int(args.snapshots), 1),
        max_events=int(events),
        rng=np.random.default_rng(args.seed), verbose=True)


def accumulate(path, r, edges_age, fresh):
    """Add this pass to whatever is already on disk.

    iso_counts and gen_counts are SUMS over snapshots, so passes add with no
    weights at all; gen_tau is a mean and is carried as its two parts.  The
    spectrum is a density and is kept per pass, weighted by events, exactly as
    run_local does it.  A pass whose age grid differs is refused rather than
    interpolated: a different grid means a different run.
    """
    new = dict(
        iso_counts=np.asarray(r["iso_counts"], float),
        iso_snapshots=float(r["iso_snapshots"]),
        gen_counts=np.asarray(r["gen_counts"], float),
        gen_mass=np.asarray(r["gen_mass"], float),
        gen_num=np.asarray(r["gen_num"], float),
        gen_tau_sum=np.nan_to_num(np.asarray(r["gen_tau"], float), nan=0.0)
        * np.asarray(r["gen_reach_n"], float),
        gen_reach_n=np.asarray(r["gen_reach_n"], float),
        gen_snapshots=float(r["gen_snapshots"]),
        events=float(r["events"][-1]),
        n_out=float(r["n_out"][-1]))
    F = AN.spectrum(r)["F"]
    if (not fresh) and os.path.exists(path):
        old = np.load(path)
        if np.array_equal(np.asarray(old["iso_age_edges"], float), edges_age):
            for k in ("iso_counts", "iso_snapshots", "gen_counts", "gen_mass",
                      "gen_num", "gen_tau_sum", "gen_reach_n", "gen_snapshots",
                      "events", "n_out"):
                new[k] = new[k] + np.asarray(old[k], float)
            F_p = np.vstack([np.asarray(old["F_passes"], float), F[None, :]])
            ev_p = np.concatenate([np.asarray(old["ev_passes"], float),
                                   [float(r["events"][-1])]])
        else:
            print("  !! the stored age grid differs -- starting a new file")
            F_p, ev_p = F[None, :], np.array([float(r["events"][-1])])
    else:
        F_p, ev_p = F[None, :], np.array([float(r["events"][-1])])
    np.savez_compressed(
        path, iso_age_edges=edges_age, centers=np.asarray(r["centers"], float),
        widths=np.asarray(r["widths"], float),
        edges=np.asarray(r["edges"], float),
        F_passes=F_p, ev_passes=ev_p,
        m_inj=float(r["meta"]["injection_mass"]),
        m_sink=float(r["meta"]["sink_mass"]),
        t_c=float(r["meta"]["analysis"]["t_c"]),
        tau_res=float(r["meta"]["analysis"]["tau_res"]),
        frag_split_width=float(r["meta"]["frag_split_width"]),
        **new)
    return path


# ======================================================================
#  ANALYSIS
# ======================================================================

def regroup(edges, counts, per_decade=4.0, min_counts=1e3):
    """Merge the fine age bins into groups placed uniformly in LOG age.

    Merging is exact -- the engine accumulates counts and counts add -- so a
    group is precisely the histogram a coarser grid would have produced.  The
    placement is uniform in log tau because that is the axis the early cascade
    lives on: the first cohorts double their number of splits in a fixed
    MULTIPLE of the time, not in a fixed amount of it.  Groups holding fewer
    than min_counts particles are absorbed into the next one, so the bins widen
    where the population has run out and nowhere else.
    """
    e = np.asarray(edges, float)
    C = np.asarray(counts, float)
    s = C.sum(axis=1)
    live = np.flatnonzero(s > 0)
    if live.size < 2:
        return None
    #  Bins are labelled by their upper edge: the first fine bin starts at zero
    #  and has no logarithm.
    hi = e[1:]
    lo_t = hi[live[0]]
    n_dec = np.log10(hi[live[-1]] / lo_t)
    n_grp = max(int(round(per_decade * n_dec)), 2)
    ladder = np.logspace(np.log10(lo_t), np.log10(hi[live[-1]]), n_grp + 1)
    idx = np.clip(np.searchsorted(ladder, hi[live], side="left") - 1,
                  0, n_grp - 1)
    groups = []
    for k in range(n_grp):
        members = live[idx == k]
        if members.size:
            groups.append((int(members[0]), int(members[-1])))
    if not groups:
        return None
    #  The count floor, applied forward and then folded back for a starved tail.
    merged, k = [], 0
    while k < len(groups):
        a, b = groups[k]
        while C[a:b + 1].sum() < min_counts and k + 1 < len(groups):
            k += 1
            b = groups[k][1]
        merged.append((a, b))
        k += 1
    if len(merged) > 1 and C[merged[-1][0]:merged[-1][1] + 1].sum() < min_counts:
        merged[-2] = (merged[-2][0], merged[-1][1])
        merged.pop()
    rows = np.asarray([C[a:b + 1].sum(axis=0) for a, b in merged], float)
    #  THE AGE OF A GROUP IS THE AGE OF THE PARTICLES IN IT, weighted by how
    #  many sit in each fine bin, not the midpoint of its edges.  The first
    #  group starts at tau = 0 and is heavily skewed towards its upper end, so
    #  a geometric or arithmetic midpoint there is off by orders of magnitude
    #  and drags the whole colour scale with it.
    mid = 0.5 * (e[:-1] + e[1:])
    tau = np.array([float((C[a:b + 1].sum(axis=1) * mid[a:b + 1]).sum()
                          / max(C[a:b + 1].sum(), 1e-300))
                    for a, b in merged])
    return dict(rows=rows,
                tau_lo=e[[a for a, _ in merged]],
                tau_hi=e[[b + 1 for _, b in merged]],
                tau=tau, counts=rows.sum(axis=1))


def packet(rows, centers):
    """Mean mass and width of each cohort, by number and by mass.

    The width is in DECADES of mass and it is the quantity the generation
    histogram cannot supply: that one stores a single mean age per generation
    and no spread at all.
    """
    c = np.asarray(centers, float)
    L = np.log10(c)
    n = rows.sum(axis=1)
    ok = n > 0
    mu = np.full(rows.shape[0], np.nan)
    sd = np.full(rows.shape[0], np.nan)
    mb = np.full(rows.shape[0], np.nan)
    mm = np.full(rows.shape[0], np.nan)
    mu[ok] = (rows[ok] * L[None, :]).sum(axis=1) / n[ok]
    sd[ok] = np.sqrt(np.maximum(
        (rows[ok] * (L[None, :] - mu[ok][:, None]) ** 2).sum(axis=1) / n[ok],
        0.0))
    mb[ok] = (rows[ok] * c[None, :]).sum(axis=1) / n[ok]
    mass = (rows * c[None, :]).sum(axis=1)
    mm[ok] = (rows[ok] * c[None, :] ** 2).sum(axis=1) / np.maximum(mass[ok],
                                                                   1e-300)
    return dict(n=n, logm=mu, width=sd, m_number=mb, m_mass=mm)


def gen_table(d):
    """Mean mass, width and mean age of each generation, from the stored sums.

    gen_tau is rebuilt from its parts so that a long pass carries the weight it
    earned; a generation with no arrivals gives nan rather than a zero that
    would plot.
    """
    C = np.asarray(d["gen_counts"], float)
    c = np.asarray(d["centers"], float)
    L = np.log10(c)
    rn = np.asarray(d["gen_reach_n"], float)
    tau = np.where(rn > 0, np.asarray(d["gen_tau_sum"], float)
                   / np.maximum(rn, 1e-300), np.nan)
    n = C.sum(axis=1)
    ok = n > 0
    mu = np.full(n.size, np.nan)
    sd = np.full(n.size, np.nan)
    mb = np.full(n.size, np.nan)
    mu[ok] = (C[ok] * L[None, :]).sum(axis=1) / n[ok]
    sd[ok] = np.sqrt(np.maximum(
        (C[ok] * (L[None, :] - mu[ok][:, None]) ** 2).sum(axis=1) / n[ok], 0.0))
    mb[ok] = (C[ok] * c[None, :]).sum(axis=1) / n[ok]
    return dict(g=np.arange(n.size), n=n, logm=mu, width=sd, m=mb, tau=tau)


def front_fit(tau, m, m_inj):
    """dm/dtau ~ -D m^beta means m^(1-beta) falls LINEARLY in tau.

    One free constant, and the test is the straightness rather than the value:
    a cohort that decays exponentially, or one whose mean is set by the sink
    rather than by the drift, does not lie on a line here.  The intercept is
    NOT forced to m_inj^(1-beta) -- it is fitted and then compared with it,
    which is the only way the comparison says anything.
    """
    t = np.asarray(tau, float)
    y = np.asarray(m, float) ** (1.0 - BETA_TH)
    k = np.isfinite(t) & np.isfinite(y) & (y > 0)
    if k.sum() < 3:
        return None
    sl, ic = np.polyfit(t[k], y[k], 1)
    res = y[k] - (sl * t[k] + ic)
    return dict(slope=float(sl), intercept=float(ic),
                rms=float(np.std(res) / max(np.std(y[k]), 1e-300)),
                m0=float(max(ic, 0.0) ** (1.0 / (1.0 - BETA_TH))),
                m_inj=float(m_inj), n=int(k.sum()))


# ======================================================================
#  FIGURES
# ======================================================================

def figure_isochrones(d, I, P, G, path, tau_res, min_counts=3.0):
    c = np.asarray(d["centers"], float)
    w = np.asarray(d["widths"], float)
    S = max(float(d["iso_snapshots"]), 1.0)
    fig, ax = plt.subplots(1, 2, figsize=(11.8, 4.5))

    norm = LogNorm(vmin=max(I["tau"].min(), 1e-300) / tau_res,
                   vmax=I["tau"].max() / tau_res)
    cm = plt.get_cmap("viridis")
    for i in range(I["rows"].shape[0]):
        y = I["rows"][i] / w / S
        k = I["rows"][i] >= min_counts
        if k.sum() < 2:
            continue
        ax[0].loglog(c[k], y[k] * c[k] ** COMP, "-", lw=1.2,
                     color=cm(norm(I["tau"][i] / tau_res)))
    tot = I["rows"].sum(axis=0) / w / S
    kt = tot > 0
    ax[0].loglog(c[kt], tot[kt] * c[kt] ** COMP, "--", lw=1.3, color="C3",
                 label="sum of the isochrones")
    F = np.asarray(d["F_passes"], float)
    ev = np.asarray(d["ev_passes"], float)
    Fm = (F * ev[:, None]).sum(axis=0) / max(ev.sum(), 1e-300)
    ok = Fm > 0
    ax[0].loglog(c[ok], Fm[ok] * c[ok] ** COMP, "-", lw=2.4, color="0.45",
                 alpha=0.8, zorder=0, label="stationary spectrum")
    ax[0].legend(fontsize=7, loc="lower left")
    ax[0].set_title("(a) isochrones on the ENTRY clock", fontsize=10)
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cm), ax=ax[0])
    cb.set_label(r"age $\tau/\tau_{\rm res}$", fontsize=8)

    gmax = int(min(np.flatnonzero(G["n"] > 0)[-1] if np.any(G["n"] > 0) else 0,
                   RL.GEN_MAX))
    Cg = np.asarray(d["gen_counts"], float) / max(float(d["gen_snapshots"]), 1.0)
    cols = plt.get_cmap("plasma")(np.linspace(0.0, 0.85, 6))
    for gg in range(min(6, gmax + 1)):
        y = Cg[gg] / w
        k = np.asarray(d["gen_counts"], float)[gg] >= min_counts
        if k.sum() < 2:
            continue
        ax[1].loglog(c[k], y[k] * c[k] ** COMP, "-", lw=1.8, color=cols[gg],
                     label="g = %d" % gg)
    for gg in range(6, gmax + 1, max((gmax - 6) // 12, 1)):
        y = Cg[gg] / w
        k = np.asarray(d["gen_counts"], float)[gg] >= min_counts
        if k.sum() >= 2:
            ax[1].loglog(c[k], y[k] * c[k] ** COMP, "-", lw=0.7, color="0.6")
    ax[1].legend(fontsize=7, ncol=2, loc="lower left")
    ax[1].set_title("(b) the same motion labelled by splits, not by the clock",
                    fontsize=10)

    lo, hi = float(d["m_sink"]), float(d["m_inj"])
    for a in ax:
        a.set_xlim(lo / 3.0, hi * 3.0)
        a.set_xlabel("m")
        a.set_ylabel(r"$m^{11/6}\,dN/dm$")
    fig.suptitle(r"fragmentation, geometric kernel $\lambda=2/3$, "
                 r"entry clock (frag_age_rule = inherit)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_growth(d, I, P, G, fit, path, tau_res):
    m_inj = float(d["m_inj"])
    fig, ax = plt.subplots(1, 3, figsize=(13.2, 4.0))
    t = I["tau"] / tau_res

    ax[0].loglog(t, P["m_number"], "o-", ms=3.4, lw=0.8, color="C0",
                 label=r"$\langle m\rangle$ by number")
    ax[0].loglog(t, P["m_mass"], "s-", ms=3.0, lw=0.8, color="C1",
                 label=r"$\langle m\rangle$ by mass")
    k = np.isfinite(G["tau"]) & (G["n"] > 0) & (G["tau"] > 0)
    ax[0].loglog(G["tau"][k] / tau_res, G["m"][k], "^", ms=4.0, color="C3",
                 label=r"generations, $\langle\tau\rangle(g)$")
    ax[0].axhline(m_inj, ls=":", lw=1.0, color="0.5")
    ax[0].set_ylabel(r"$\langle m\rangle$")
    ax[0].legend(fontsize=7)
    ax[0].set_title("(a) the two labels against each other", fontsize=10)

    #  A LINEAR age axis here, and only here: the statement being tested is
    #  that m^(1-beta) falls linearly in tau, and on a logarithmic axis a
    #  straight line is not straight.
    y = P["m_mass"] ** (1.0 - BETA_TH)
    ax[1].plot(t, y, "s", ms=3.4, color="C1")
    if fit is not None:
        ax[1].plot(t, fit["slope"] * I["tau"] + fit["intercept"], "-", lw=1.6,
                   color="C3",
                   label=r"$m^{1/6}$ linear in $\tau$, rms %.3f" % fit["rms"])
        ax[1].axhline(m_inj ** (1.0 - BETA_TH), ls=":", lw=1.0, color="0.5",
                      label=r"$m_{\rm inj}^{1/6}$")
        ax[1].legend(fontsize=7)
    ax[1].set_ylabel(r"$\langle m\rangle_{\rm mass}^{\,1-\beta}"
                     r"=\langle m\rangle^{1/6}$")
    ax[1].set_title(r"(b) a straight line here is $dm/d\tau\sim-Dm^{\beta}$",
                    fontsize=10)

    ax[2].plot(t, P["width"], "o-", ms=3.4, lw=0.8, color="C0",
               label="isochrone")
    kk = np.isfinite(G["width"]) & (G["tau"] > 0)
    ax[2].plot(G["tau"][kk] / tau_res, G["width"][kk], "^", ms=4.0,
               color="C3", label="generation")
    ax[2].set_ylabel("packet width, decades of mass")
    ax[2].legend(fontsize=7)
    ax[2].set_title("(c) how fast a cohort spreads", fontsize=10)

    for a in ax:
        a.set_xlabel(r"age  $\tau/\tau_{\rm res}$")
        a.grid(alpha=0.25)
    ax[0].set_xscale("log")
    ax[2].set_xscale("log")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ======================================================================
#  MAIN
# ======================================================================

def analyse(path, args, tau_res):
    d = np.load(path)
    I = regroup(np.asarray(d["iso_age_edges"], float),
                np.asarray(d["iso_counts"], float),
                per_decade=args.per_decade_draw, min_counts=args.min_counts)
    if I is None:
        print("  isochrones: nothing binned yet -- the time gate at %.2f "
              "tau_res was never reached.  Run more events." % args.gate_tau)
        return
    P = packet(I["rows"], np.asarray(d["centers"], float))
    G = gen_table(d)
    #  THE FRONT IS THE MASS-WEIGHTED MEAN, not the number-weighted one.  A
    #  cohort doubles its membership at every split, so by number its mean is
    #  dominated by the many small pieces and falls faster than any single
    #  body does; weighting by mass follows the material instead.
    fit = front_fit(I["tau"], P["m_mass"], float(d["m_inj"]))

    print("-" * 78)
    print("  isochrones : %d groups, %.0f snapshots, %.4g counts"
          % (I["rows"].shape[0], float(d["iso_snapshots"]),
             float(np.sum(I["counts"]))))
    print("")
    print("    tau/tau_res      <m>_num      <m>_mass    width(dex)     counts")
    for i in range(I["rows"].shape[0]):
        print("  %6.4f-%-8.4f %11.4g %13.4g %11.3f %10.4g"
              % (I["tau_lo"][i] / tau_res, I["tau_hi"][i] / tau_res,
                 P["m_number"][i], P["m_mass"][i], P["width"][i],
                 I["counts"][i]))
    print("")
    k = np.isfinite(G["tau"]) & (G["n"] > 0)
    print("    g      <tau>/tau_res        <m>    width(dex)      counts")
    for g in G["g"][k][:8]:
        print("  %3d %15.4f %12.4g %10.3f %12.4g"
              % (g, G["tau"][g] / tau_res, G["m"][g], G["width"][g], G["n"][g]))
    if fit is not None:
        print("")
        print("  front      : m^(1/6) linear in tau, rms of residual %.4f "
              "over %d points" % (fit["rms"], fit["n"]))
        print("               intercept -> m = %.4g against m_inj = %.4g"
              % (fit["m0"], fit["m_inj"]))
    f1 = os.path.splitext(path)[0] + "_iso.png"
    f2 = os.path.splitext(path)[0] + "_growth.png"
    figure_isochrones(d, I, P, G, f1, tau_res, args.draw_min_counts)
    figure_growth(d, I, P, G, fit, f2, tau_res)
    print("  -> %s" % f1)
    print("  -> %s" % f2)


def main():
    ap = argparse.ArgumentParser(
        description="Fragmentation isochrones measured on the clock that "
                    "starts when a body ENTERS the system, so that an age "
                    "class is one cohort of injected bodies rather than a "
                    "crop of freshly born fragments.  The engine is used "
                    "unmodified: frag_age_rule='inherit' is an argument.")
    ap.add_argument("--events", type=float, default=3e7,
                    help="events added by this invocation.  Passes accumulate, "
                         "so the command can simply be repeated.")
    ap.add_argument("--f", type=float, default=0.30,
                    help="disruption threshold m_small >= f m_large.  "
                         "Fragmentation needs one at any lambda because "
                         "alpha < -1 (engine note [8]).")
    ap.add_argument("--w", type=float, default=0.5,
                    help="half-width of the split: xi uniform on (0.5-w, "
                         "0.5+w).  Below about 0.25 the spectrum combs.")
    ap.add_argument("--q", type=float, default=0.0,
                    help="injection rate.  0 uses run_local's estimate, which "
                         "is a starting guess and not a calibration: the live "
                         "count is printed so it can be corrected by hand.")
    ap.add_argument("--gate-tau", type=float, default=1.5,
                    help="start binning ages only after this many residence "
                         "times, by which the cold start and its descendants "
                         "have left through the sink.")
    ap.add_argument("--age-grid", choices=["hybrid", "log", "linear"],
                    default="hybrid")
    ap.add_argument("--span", type=float, default=1.5,
                    help="the grid covers this multiple of the residence time.")
    ap.add_argument("--knee", type=float, default=0.05,
                    help="hybrid grid: where the logarithmic part ends, in "
                         "residence times.")
    ap.add_argument("--per-decade", type=float, default=40.0,
                    help="stored age edges per decade in the logarithmic part.")
    ap.add_argument("--per-decade-draw", type=float, default=4.0,
                    help="drawn isochrones per decade of age.  Analysis time: "
                         "merging stored bins is exact, so this can be changed "
                         "without re-running.")
    ap.add_argument("--min-counts", type=float, default=1e3,
                    help="a drawn isochrone rests on at least this many "
                         "particles; thinner groups are merged into the next.")
    ap.add_argument("--draw-min-counts", type=float, default=3.0,
                    help="mass bins below this count are left out of the "
                         "curves rather than drawn as shot noise.")
    ap.add_argument("--snapshots", type=int, default=400)
    ap.add_argument("--n-ss", type=float, default=RL.N_SS,
                    help="target live population.  Lower it for a smoke test.")
    ap.add_argument("--m-top", type=float, default=RL.M_TOP,
                    help="injection mass; the sink stays at %g.  Lower it for "
                         "a smoke test." % RL.M_FLOOR_FRAG)
    ap.add_argument("--seed", type=int, default=20260918)
    ap.add_argument("--outdir", default="runs")
    ap.add_argument("--tag", default="entry_clock")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore what is on disk and start the accumulation "
                         "again.")
    ap.add_argument("--analyse-only", action="store_true",
                    help="redraw from the stored file without running.")
    args = ap.parse_args()

    g = box(args.m_top, args.n_ss)
    tc, tau_res = clocks(g)
    args.tau_res = tau_res
    edges_age = age_edges(tau_res, args.age_grid, args.span, args.knee,
                          args.per_decade)
    q = float(args.q) if args.q > 0 else RL._q_of(BF, g, "fragmentation")

    outdir = (args.outdir if os.path.isabs(args.outdir)
              else os.path.join(ROOT, args.outdir))
    os.makedirs(outdir, exist_ok=True)
    acc = os.path.join(outdir, "%s_f%.2f_w%.2f.npz" % (args.tag, args.f, args.w))
    state = os.path.join(outdir, "%s_f%.2f_w%.2f_state.npz"
                         % (args.tag, args.f, args.w))

    print("=" * 78)
    print("  ENTRY-CLOCK ISOCHRONES, fragmentation, geometric kernel "
          "lambda = %.4f" % LAMBDA)
    print("  theory   : alpha = %+.4f, beta = %.4f, b = %.3f"
          % (ALPHA_TH, BETA_TH, B_TH))
    print("  box      : injection %g -> sink %g  (%.1f decades), N_ss = %.3g"
          % (g["m_inj"], g["m_sink"],
             np.log10(g["m_inj"] / g["m_sink"]), RL.N_SS))
    print("  clocks   : t_c = %.4g at the sink, tau_res = %.4g = %.1f t_c"
          % (tc, tau_res, tau_res / tc))
    print("  age grid : %s, %d edges, %.3g .. %.3g  (gate at %.2f tau_res)"
          % (args.age_grid, edges_age.size, edges_age[1], edges_age[-1],
             args.gate_tau))
    print("  rule     : frag_age_rule = inherit -- tau is time IN the system")
    print("  file     : %s" % acc)
    print("=" * 78)

    if not args.analyse_only:
        ic = {"m": g["m_inj"], "N": RL.frag_n0(g)}
        if (not args.fresh) and os.path.exists(state):
            prev = AN.load(state)
            ic = {"state": BF.state_of(prev, source=state)}
            print("resuming from %s" % state)
            del prev
        t0 = time.time()
        r = BF.simulate(**sim_kwargs(g, args, ic, edges_age,
                                     int(args.events), q))
        md = abs(float(r["mass_drift"]))
        if md > 1e-4:
            raise SystemExit("mass_drift = %.3e -- bookkeeping failure" % md)
        an = dict(t_c=tc, tau_res=tau_res, gate_tau=float(args.gate_tau),
                  f=float(args.f), w=float(args.w), q=float(q),
                  entry_clock=True)
        r["meta"]["analysis"] = an
        RL._save_atomic(AN, r, state, False, dict(an, checkpoint=True))
        accumulate(acc, r, edges_age, args.fresh)
        print("\n  %.4g events in %.1f min | live %d | absorptions %.4g"
              % (float(r["events"][-1]), (time.time() - t0) / 60.0,
                 int(r["live"][-1]), float(r["n_out"][-1])))
        print("  t reached %.4g = %.2f tau_res | isochrone snapshots %d"
              % (float(r["final_t_phys"]),
                 float(r["final_t_phys"]) / tau_res,
                 int(r["iso_snapshots"])))
        if float(r["live"][-1]) < 0.3 * RL.N_SS or \
                float(r["live"][-1]) > 3.0 * RL.N_SS:
            print("  !! the live count is far from N_ss: the injection rate is "
                  "a guess.  Re-run with --q %.4g"
                  % (q * RL.N_SS / max(float(r["live"][-1]), 1.0)))
        if int(r["iso_snapshots"]) == 0:
            print("  !! nothing binned: the run has not reached the gate at "
                  "%.2f tau_res yet.  Repeat the command to add events."
                  % args.gate_tau)

    if os.path.exists(acc):
        analyse(acc, args, tau_res)
    print("=" * 78)


if __name__ == "__main__":
    main()
