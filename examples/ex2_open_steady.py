"""
ex2 -- open coagulation, geometric kernel.

Bodies enter at m = 1, merge upward, and are absorbed at m = 1e4.  The steady
spectrum should be alpha = -11/6.  The figure quotes it against -2 as well,
because a clock advanced per event instead of per trial, or a merging rule
that is not local, gives -2 for every kernel -- so -2 here means the run
failed, not that universality held.

Three things to know before changing anything.  The injection rate is not
guessed: a short pilot run measures how fast merges happen and the injector is
set to match, which is what keeps the box from draining.  The locality window
f = 0.30 stops a heavy body from growing by eating the pile of fresh
injections, which is a boundary effect and not a cascade; the exponent itself
does not depend on f, only the speed does.  And the fitting window is the one
run_local.py uses -- a decade above the source, where the spectrum is still a
picket fence of whole multiples, and a decade and a half below the wall.

Runtime: about a minute and a half.  MAX_EVENTS is the knob; raise N_SEED with
it so the number of flushes stays put.
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

M_INJ = 1.0                                      # injection mass
M_SINK = 1.0e4                                   # absorbing wall: four decades
N_SEED = 30_000                                  # bodies in the seeded population
MAX_EVENTS = 2_000_000                           # about three and a half flushes
PILOT_EVENTS = max(500, N_SEED // 20)            # MUST stay a small fraction of
                                                 # N_SEED: the pilot runs without
                                                 # injection and every merge removes
                                                 # one body, so a long pilot measures
                                                 # a collapsing population
EDGES = np.logspace(np.log10(0.3), 4.5, 91)      # 20 bins per decade
COAG_F = 0.30                                    # locality: partner >= f * target
PAD_INJ = 1.0                                    # decades stripped above the source
PAD_SINK = 1.5                                   # decades stripped below the wall
SEED = 20260913

LAM = 2.0 / 3.0
PRED = mc.predict("open", LAM)

#  PRL FIGURE STYLE.  One column is 3.375 in (8.6 cm), and two stacked panels
#  of a spectrum is exactly that shape, so most of these are single-column.
#  The journal puts into the caption everything a title would say, so what
#  would be a title is set as a small line INSIDE the panel: it reads in the
#  paper, and it reads in the repository, where nobody has the caption.
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


def _common(rng):
    """Everything both the pilot and the measurement share."""
    return dict(
        process="coagulation", system="open",
        kernel=mc.kernel_geometric, edges=EDGES,
        ic={"steady": {"alpha": PRED["alpha"], "m_lo": M_INJ,
                       "m_hi": M_SINK, "N": N_SEED}},
        injection_mass=M_INJ, sink_mass=M_SINK,
        coag_min_ratio=COAG_F,
        track_generations=False, track_waiting=False,
        rng=rng, verbose=False,
    )


def _injection_rate():
    """Measure the merge rate on a short pilot and use it as the injection rate.

    In a steady coagulation cascade one body of mass m_sink is built out of
    m_sink/m_inj injected monomers by about that many merges, so injections and
    merges happen at the same rate.  The ratio test below carries the mirrored
    statement for fragmentation, where one injected body costs m_inj/m_sink
    splits: the same three lines serve ex3 and ex4.
    """
    pilot = mc.simulate(injection_rate=1.0, max_events=PILOT_EVENTS,
                        snapshot_mode="events", snapshot_stride=PILOT_EVENTS,
                        **_common(np.random.default_rng(SEED + 1)))
    rate = float(pilot["events"][-1]) / float(pilot["final_t_phys"])
    ratio = M_INJ / M_SINK
    q = rate if ratio < 1.0 else rate / ratio
    print("  pilot: %.0f events in t = %.3e  ->  injection rate q = %.4e"
          % (pilot["events"][-1], pilot["final_t_phys"], q))
    return q


def main():
    t_wall = time.perf_counter()
    an.check_grid(EDGES, M_INJ, M_SINK)
    mbar_seed = an.steady_mean_mass(PRED["alpha"], M_INJ, M_SINK)
    per_flush = N_SEED * mbar_seed / M_INJ
    print("  box               %.3g .. %.3g  (%.1f decades)"
          % (M_INJ, M_SINK, np.log10(M_SINK / M_INJ)))
    print("  locality f = %.2f   (0 = unrestricted kernel)" % COAG_F)
    print("  one flush of the box costs %.2e events -> this run is %.1f flushes"
          % (per_flush, MAX_EVENTS / per_flush))

    q = _injection_rate()
    run = mc.simulate(injection_rate=q, max_events=MAX_EVENTS,
                      snapshot_mode="events", snapshot_stride=MAX_EVENTS / 80,
                      **_common(np.random.default_rng(SEED)))
    print(an.describe(run, "ex2  open / geometric kernel"))

    live = np.asarray(run["live"], float)
    print("\n  population        %.0f -> %.0f  (x%.2f)"
          % (live[0], live[-1], live[-1] / live[0]))
    if not (0.6 < live[-1] / live[0] < 1.7):
        print("  ** the box is leaking: the injection rate is off, and what follows")
        print("  ** is a transient.  Shorten PILOT_EVENTS or lengthen the run.")

    c = np.asarray(run["centers"], float)
    spec = an.spectrum(run)                       # averaged over the steady snapshots
    plateau = an.spectrum_plateau(run, spec=spec)
    #  THE SAME WINDOW run_local.py QUOTES ON.  Not symmetric, and measured
    #  rather than chosen: the wall reaches a decade and a half down, the
    #  picket fence at the source a decade up.  guard_band takes one number for
    #  both ends, so the two are written out here instead.
    band = (M_INJ * 10.0 ** PAD_INJ, M_SINK / 10.0 ** PAD_SINK)
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
    #  one of the two unconditionally is wrong in one of the two cases.
    if fit["chi2_dof"] <= 3.0 and np.isfinite(fit["sigma_p"]):
        a_q, s_q, src = fit["p"], fit["sigma_p"], "weighted fit"
    else:
        a_q, s_q, src = plateau["alpha"], plateau["scatter"], "plateau scatter"
        print("                    chi2/dof >> 1: the window is not one power law,")
        print("                    so +-%.4f is not an uncertainty" % fit["sigma_p"])
    print("  QUOTED            alpha = %+.4f +- %.4f   (%s)" % (a_q, s_q, src))
    print("  distance to -2    %+.4f = %.0f sigma"
          % (a_q + 2.0, abs(a_q + 2.0) / max(s_q, 1e-12)))
    an.stationarity(run, verbose=True)

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
                r"open coagulation, $\lambda=2/3$, $f=%.2f$" "\n"
                r"$m_{\rm inj}=%g$, $m_{\rm sink}=%g$, %.1e events"
                % (COAG_F, M_INJ, M_SINK, float(run["events"][-1])),
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
    axs[1].text(EDGES[0] * 1.2, -1.45, "source", color="0.35", fontsize=6)
    axs[1].text(band[1] * 1.2, -1.45, "wall", color="0.35", fontsize=6)

    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    save_figure(fig, FIGURES / "ex2_open_steady.png")
    print("  wall time         %.1f min" % ((time.perf_counter() - t_wall) / 60.0))


if __name__ == "__main__":
    main()
