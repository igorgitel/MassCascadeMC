"""
ex3 -- open fragmentation, geometric kernel.

Whole bodies enter at m = 1, shatter downward, and leave at m = 1e-4.  The
constant-flux argument does not care which way the cascade runs, so the
prediction is the same -11/6 as in ex2, quoted against -2 in the same way.
Two processes, opposite directions, one exponent -- that is the claim these
two examples test.

The second figure says where the power law comes from.  Every particle carries
the number of splits behind it, and each generation is a packet in ln m that
marches at a constant speed.  The lines drawn through them are not fitted: one
uniform split gives <ln xi> = -1/2 and a variance of 1/4 exactly.  The run
starts cold, so every body is generation zero and the ancestry is real; the
pilot that sets the injection rate is seeded instead, which costs the
generations nothing because it is a separate run.

Runtime: about nine minutes.  This is the slow example, deliberately: it is the
one where the spectrum, the generations and the stationarity test all have to
hold at once.
"""

import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt

import masscascade as mc
import analysis as an

FIGURES = pathlib.Path(__file__).resolve().parent / "figures"

M_INJ = 1.0                                   # bodies enter here, whole
M_SINK = 1.0e-4                               # fragments leave here: four decades
N0 = 100                                      # cold start, monodisperse at M_INJ
N_TARGET = 50_000                             # the amplitude the pilot calibrates on
MAX_EVENTS = 15_000_000                       # the slow one: about fifteen flushes
PILOT_EVENTS = 2_000                          # calibrates the injection rate
EDGES = np.logspace(-5.5, 0.5, 121)           # 20 bins per decade
F_LOC = 0.30                                  # locality: impactor >= f * target
SPLIT_W = 0.5                                 # half-width: the uniform split
PAD = 1.0                                     # decades stripped off each end
GEN_MAX = 64
G_SHOW = 14                                   # highest generation drawn
MIN_MASS_FRAC = 2e-3                          # a generation needs this much mass
SEED = 20260913

LAM = 2.0 / 3.0
PRED = mc.predict("open", LAM)

#  PRL FIGURE STYLE.  One column is 3.375 in (8.6 cm), and two stacked panels
#  of a spectrum is exactly that shape; the generation figure is two panels
#  side by side and gets the full 7.0 in width.  The journal puts into the
#  caption everything a title would say, so what would be a title is set as a
#  small line INSIDE the panel: it reads in the paper, and it reads in the
#  repository, where nobody has the caption.
COL1, COL2 = 3.375, 7.0
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "legend.frameon": True, "legend.framealpha": 0.9,
    "legend.facecolor": "white", "legend.edgecolor": "0.6",
    "legend.fancybox": False, "legend.borderpad": 0.35,
    "axes.linewidth": 0.6, "lines.linewidth": 1.0,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "axes.grid": False, "figure.facecolor": "white",
    "savefig.dpi": 600,
})


def save_figure(fig, path, tries=6, pause=1.5):
    """Write the PNG.

    OneDrive and its kind keep a freshly written figure open while they upload
    it, and Windows then refuses the next write with OSError 22 on a path that
    is perfectly valid.  Writing under a temporary name and replacing is
    atomic, and retrying costs nothing.
    """
    tmp = path.with_name(path.stem + ".tmp.png")
    for k in range(tries):
        try:
            fig.savefig(tmp, bbox_inches="tight")
            os.replace(tmp, path)
            print("  figure -> %s" % path)
            return
        except OSError as exc:
            if k == tries - 1:
                print("  ** could not write %s: %s" % (path, exc))
                return
            time.sleep(pause)


def _common(rng, ic, gens):
    """Everything both the pilot and the measurement share."""
    return dict(
        process="fragmentation", system="open",
        kernel=mc.kernel_geometric, edges=EDGES,
        ic=ic,
        injection_mass=M_INJ, sink_mass=M_SINK,
        frag_min_ratio=F_LOC,
        frag_split_width=SPLIT_W,
        frag_age_rule="inherit",
        track_generations=gens, gen_max=GEN_MAX, track_waiting=False,
        rng=rng, verbose=False,
    )


IC_COLD = {"m": M_INJ, "N": N0}
IC_TARGET = {"steady": {"alpha": PRED["alpha"], "m_lo": M_SINK,
                        "m_hi": M_INJ, "N": N_TARGET}}


def _injection_rate():
    """Measure the split rate on a short pilot and turn it into q.

    One injected body of mass m_inj is taken apart by about m_inj/m_sink
    splits, so the injection rate is the split rate divided by that ratio.
    Coagulation is the same statement read backwards, which is why ex2 uses
    the same three lines.

    THE PILOT IS SEEDED AND THE MEASUREMENT IS NOT, and that is deliberate.
    The split rate depends on the SHAPE of the population, not only on its
    mass: a hundred bodies all sitting at m_inj collide far less often than
    the same mass spread over a developed spectrum.  Calibrating on the cold
    state therefore asks for a flux the developed cascade carries at a much
    smaller amplitude, and the box settles an order of magnitude lighter than
    it started -- which is exactly what happened before this was fixed: mass
    100 in, mass 6.6 at the end, five thousand live bodies and error bars an
    order of magnitude tall.  The pilot is a separate run, so seeding it costs
    the generations nothing: the measurement still starts cold, every body
    generation zero.
    """
    pilot = mc.simulate(injection_rate=1.0, max_events=PILOT_EVENTS,
                        snapshot_mode="events", snapshot_stride=PILOT_EVENTS,
                        **_common(np.random.default_rng(SEED + 1), IC_TARGET, False))
    rate = float(pilot["events"][-1]) / float(pilot["final_t_phys"])
    ratio = M_INJ / M_SINK
    q = rate if ratio < 1.0 else rate / ratio
    print("  pilot: %.0f events in t = %.3e  ->  injection rate q = %.4e"
          % (pilot["events"][-1], pilot["final_t_phys"], q))
    return q


def spectrum_figure(run, path):
    """The steady spectrum of the downward cascade, against -11/6 and -2."""
    c = np.asarray(run["centers"], float)
    spec = an.spectrum(run)
    plateau = an.spectrum_plateau(run, spec=spec)
    band = an.guard_band(M_SINK, M_INJ, pad_decades=PAD)
    inside = (c >= band[0]) & (c <= band[1])
    fit = an.wls_powerlaw(c, spec["F"], spec["sigma"], mask=inside)

    print("\n  prediction        alpha = %+.4f   (beta = %.4f)"
          % (PRED["alpha"], PRED["beta"]))
    print("  guard band        %.3g .. %.3g  (%.2f decades, %d bins)"
          % (band[0], band[1], np.log10(band[1] / band[0]), int(inside.sum())))
    print("  PLATEAU           alpha = %+.4f +- %.4f over %.2f decades"
          % (plateau["alpha"], plateau["scatter"], plateau["decades"]))
    print("  cross-check fit   alpha = %+.4f +- %.4f   chi2/dof = %.1f"
          % (fit["p"], fit["sigma_p"], fit["chi2_dof"]))
    #  WHICH ERROR IS THE ERROR.  chi2/dof near one says the window really is a
    #  single power law, and then the weighted fit's own error IS the
    #  uncertainty.  Far above one says it is not: the fit error then measures
    #  how many bins were used, not how well the index is known, and the
    #  scatter of the local slope across the plateau is what to quote.  Picking
    #  one of the two unconditionally is wrong in one of the two cases -- this
    #  run is the case that showed it, with chi2/dof = 0.6 and a fit error
    #  twenty times smaller than the plateau scatter.
    if fit["chi2_dof"] <= 3.0 and np.isfinite(fit["sigma_p"]):
        a_q, s_q, src = fit["p"], fit["sigma_p"], "weighted fit"
    else:
        a_q, s_q, src = plateau["alpha"], plateau["scatter"], "plateau scatter"
        print("                    chi2/dof >> 1: the window is not one power law,")
        print("                    so +-%.4f is not an uncertainty" % fit["sigma_p"])
    print("  QUOTED            alpha = %+.4f +- %.4f   (%s)" % (a_q, s_q, src))
    print("  distance to -2    %+.4f = %.0f sigma"
          % (a_q + 2.0, abs(a_q + 2.0) / max(s_q, 1e-12)))

    fig, axs = plt.subplots(2, 1, figsize=(COL1, 1.55 * COL1), sharex=True,
                            gridspec_kw={"height_ratios": [2.1, 1.0]})

    an.plot_spectrum(axs[0], run, spec=spec, plateau=plateau, compensate=True)
    xr = np.logspace(np.log10(band[0]) - 0.35, np.log10(band[1]) + 0.35, 40)
    for a_ref, ls, col, lab in ((PRED["alpha"], "--", "C0", r"$-11/6$, the prediction"),
                                (-2.0, ":", "0.35", r"$-2$, the artifact")):
        A = an.anchor_amplitude(c, spec["F"], band[0], band[1], a_ref)
        axs[0].plot(xr, A * xr ** (a_ref + 2.0), ls, lw=1.0, color=col, label=lab)
    an.compensated_ylim(axs[0], c, spec["F"])
    axs[0].set_ylabel(r"$m^{2}\,dN/dm$")
    axs[0].text(0.03, 0.96,
                r"open fragmentation, $\lambda=2/3$, $f=%.2f$" "\n"
                r"$m_{\rm inj}=%g$, $m_{\rm sink}=%g$, %.1e events"
                % (F_LOC, M_INJ, M_SINK, float(run["events"][-1])),
                transform=axs[0].transAxes, va="top", fontsize=6.5)
    axs[0].legend(loc="lower left")

    _, gamma = an.local_slope(c, spec["F"], half=3)
    axs[1].semilogx(c, gamma, "o-", ms=2.0, lw=0.7, color="k")
    axs[1].axhline(PRED["alpha"], ls="--", lw=1.0, color="C0")
    axs[1].axhline(-2.0, ls=":", lw=1.0, color="0.35")
    axs[1].text(band[0] * 1.1, PRED["alpha"] + 0.05, r"$-11/6$", color="C0", fontsize=6.5)
    axs[1].text(band[0] * 1.1, -1.97, r"$-2$", color="0.35", fontsize=6.5)
    axs[1].set_ylim(-2.45, -1.35)
    axs[1].set_xlabel(r"mass $m$")
    axs[1].set_ylabel(r"$d\log F/d\log m$")

    for ax in axs:
        ax.axvspan(band[0], band[1], color="C1", alpha=0.08)
        ax.axvspan(EDGES[0], band[0], color="0.5", alpha=0.13)
        ax.axvspan(band[1], EDGES[-1], color="0.5", alpha=0.13)
    axs[1].text(EDGES[0] * 1.3, -1.45, "sink", color="0.35", fontsize=6)
    axs[1].text(band[1] * 1.3, -1.45, "source", color="0.35", fontsize=6)

    fig.tight_layout()
    save_figure(fig, path)


def generation_figure(run, path):
    """The packets, and their exact step statistics."""
    gen = an.generations(run, weight="mass")
    mom = an.gen_moments(gen, min_count=MIN_MASS_FRAC * float(gen["H"].sum()))
    mu_step, var_step = an.step_stats(SPLIT_W, weight="mass")

    x = gen["x"]
    H = gen["H"]
    g = np.arange(H.shape[0])
    x_sink = np.log(max(gen["m_sink"], 1e-300) / gen["m_inj"])

    ok = (np.isfinite(mom["mu"])
          & (mom["mu"] - 2.0 * np.sqrt(np.maximum(mom["var"], 0.0)) > x_sink))
    ok[0] = False                                  # generation 0 has not moved yet
    ok &= g <= G_SHOW
    use = np.flatnonzero(ok)
    if use.size < 3:
        print("\n  too few usable generations to draw")
        return

    slope = np.polyfit(g[use], mom["mu"][use], 1)[0]
    spread = np.polyfit(g[use], mom["var"][use], 1)[0]
    print("\n  one split, exactly     <ln xi> = %+.4f,  Var = %.4f"
          % (mu_step, var_step))
    print("  measured across %2d gen  drift   = %+.4f,  spreading = %.4f"
          % (use.size, slope, spread))
    print("  generations used       %s" % (", ".join(str(int(k)) for k in use)))

    fig, axs = plt.subplots(1, 2, figsize=(COL2, 0.42 * COL2))

    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=float(use.min()), vmax=float(use.max()))
    for k in use:
        w = H[k]
        s = w.sum()
        if s <= 0:
            continue
        axs[0].plot(x, w / s, lw=1.0, color=cmap(norm(k)))
    axs[0].axvline(x_sink, ls=":", lw=1.0, color="C3")
    axs[0].text(x_sink + 0.2, axs[0].get_ylim()[1] * 0.92, "sink", color="C3",
                fontsize=6.5)
    axs[0].set_xlim(x_sink - 0.5, 0.8)
    axs[0].set_xlabel(r"$x=\ln\,(m/m_{\rm inj})$")
    axs[0].set_ylabel("mass fraction per bin")
    axs[0].text(0.03, 0.96, "packets march at a constant speed",
                transform=axs[0].transAxes, va="top", fontsize=6.5)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axs[0], pad=0.02, fraction=0.05)
    cb.set_label("generation $g$", fontsize=7)
    cb.ax.tick_params(labelsize=6)

    a = axs[1]
    a.plot(g[use], mom["mu"][use], "o", ms=3.5, color="C0", label=r"$\langle x\rangle$")
    a.plot(g[use], mu_step * g[use], "-", lw=1.0, color="C0",
           label=r"$g\,\langle\ln\xi\rangle=-g/2$  (exact)")
    a.set_xlabel("generation $g$")
    a.set_ylabel(r"$\langle x\rangle$", color="C0")
    a.tick_params(axis="y", labelcolor="C0")
    a.legend(loc="lower left")

    a2 = a.twinx()
    a2.plot(g[use], mom["var"][use], "s", ms=3.5, color="C3", label=r"${\rm Var}(x)$")
    a2.plot(g[use], var_step * g[use], "-", lw=1.0, color="C3",
            label=r"$g\,{\rm Var}(\ln\xi)=g/4$  (exact)")
    a2.set_ylabel(r"${\rm Var}(x)$", color="C3")
    a2.tick_params(axis="y", labelcolor="C3")
    a2.legend(loc="upper left")

    fig.tight_layout()
    save_figure(fig, path)


def ripple_report(run, path_npz):
    """Is the wave in the local slope real, or an artifact of the smoothing?

    `local_slope` averages over 2*half+1 bins, and smoothed white noise looks
    like a wave of period about twice the window whatever the data are.  The
    test is therefore to vary the window: if the period follows `half`, there
    is no wave.  If it stands still, the wave is in the spectrum, and the
    candidate scale is the generation step -- the source is monochromatic, so
    the spectrum near it is a sum of generation packets not yet washed out by
    the convolution, spaced by |<ln xi>| and spreading as sqrt(g Var(ln xi)).
    """
    c = np.asarray(run["centers"], float)
    spec = an.spectrum(run)
    band = an.guard_band(M_SINK, M_INJ, pad_decades=PAD)
    np.savez(path_npz, centers=c, F=spec["F"], sigma=spec["sigma"],
             band=np.asarray(band))
    print("\n  spectrum saved -> %s" % path_npz)

    mu_m, var_m = an.step_stats(SPLIT_W, weight="mass")
    mu_n, var_n = an.step_stats(SPLIT_W, weight="number")
    print("  one generation    %.3f decades (mass-weighted), %.3f (number-weighted)"
          % (abs(mu_m) / np.log(10.0), abs(mu_n) / np.log(10.0)))
    print("  locality window   %.3f decades wide  (ln(1/f) = %.2f)"
          % (np.log10(1.0 / F_LOC), np.log(1.0 / F_LOC)))
    print("  %-6s %8s %10s %12s" % ("half", "window", "rms(slope)", "peak spacing"))
    for half in (1, 2, 3, 6, 10):
        _, g = an.local_slope(c, spec["F"], half=half)
        k = (c >= band[0]) & (c <= band[1]) & np.isfinite(g)
        if k.sum() < 6:
            continue
        x = np.log10(c[k])
        y = g[k]
        peaks = [i for i in range(1, y.size - 1)
                 if y[i] > y[i - 1] and y[i] > y[i + 1]]
        spacing = (np.mean(np.diff(x[peaks])) if len(peaks) >= 2 else np.nan)
        print("  %-6d %8.3f %10.4f %12s"
              % (half, (2 * half + 1) * (x[1] - x[0]), np.std(y - y.mean()),
                 ("%.3f" % spacing) if np.isfinite(spacing) else "--"))
    print("  read it this way: peak spacing that tracks the window is smoothing;")
    print("  peak spacing that stands still is the generation imprint.")


def main():
    t_wall = time.perf_counter()
    an.check_grid(EDGES, M_INJ, M_SINK)
    per_flush = N0 * M_INJ / M_SINK
    print("  box               %.3g .. %.3g  (%.1f decades, downwards)"
          % (M_SINK, M_INJ, np.log10(M_INJ / M_SINK)))
    print("  locality f = %.2f" % F_LOC)
    print("  the initial mass flushes in %.2e events -> this run is %.1f of those"
          % (per_flush, MAX_EVENTS / per_flush))

    q = _injection_rate()
    run = mc.simulate(injection_rate=q, max_events=MAX_EVENTS,
                      snapshot_mode="events", snapshot_stride=MAX_EVENTS / 200,
                      **_common(np.random.default_rng(SEED), IC_COLD, True))
    print(an.describe(run, "ex3  open fragmentation / geometric kernel"))

    live = np.asarray(run["live"], float)
    print("\n  population        %.0f -> %.0f  (x%.1f: the box filling, "
          "not a leak)" % (live[0], live[-1], live[-1] / max(live[0], 1.0)))
    an.stationarity(run, verbose=True)

    FIGURES.mkdir(parents=True, exist_ok=True)
    spectrum_figure(run, FIGURES / "ex3_fragmentation_spectrum.png")
    generation_figure(run, FIGURES / "ex3_fragmentation_generations.png")
    ripple_report(run, FIGURES / "ex3_fragmentation_spectrum.npz")
    print("\n  wall time         %.1f min" % ((time.perf_counter() - t_wall) / 60.0))


if __name__ == "__main__":
    main()
