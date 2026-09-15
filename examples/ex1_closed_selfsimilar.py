"""
ex1 -- closed coagulation, constant kernel.

A million bodies of mass 1 merge until a thousand are left.  Nothing enters
and nothing leaves, so the spectrum never settles -- it just walks to larger
masses.  Three figures: one snapshot per decade of mean mass, the clock
checked against Smoluchowski's exact solution, and the time-integrated
spectrum.

The last one is the measurement.  A closed box has no steady spectrum, but
the age integral of it is a power law with alpha = -1, and for this kernel
that is known exactly rather than argued.  Below m = 4 the histogram reads
-2 instead, for a reason that is in the binning rather than the physics:
masses are whole multiples of the initial one, so a narrow log bin holds one
line or none, and dividing that line by a width that grows with m turns 1/k
into 1/m^2.  The fit starts above it.

Runtime: about forty seconds.  N0 and MAX_EVENTS are the knobs.
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

N0 = 1_000_000                                  # bodies of mass 1 at t = 0
M_INIT = 1.0
MAX_EVENTS = 999_000                            # leaves 1e3 bodies, <m> = 1000
EDGES = np.logspace(np.log10(0.5), 4.5, 48)     # 10 bins per decade -- see the
                                                # picket fence note above: fine
                                                # bins make it WORSE, not better
PAD = 0.3                                       # decades stripped off each end
N_SNAPSHOTS = 10_000                            # NOT a cosmetic number.  The
                                                # superposition is a quadrature in
                                                # PHYSICAL time, and time here is
                                                # wildly non-uniform in events:
                                                # dt = 2/R with R ~ N^2, and events
                                                # are uniform in N, so
                                                # t(N) = 2(1/N - 1/N0).  Mass m is
                                                # built near N* = N0/m, which at
                                                # the top of the band (m = 500) is
                                                # N* = 2e3.  The step in N is
                                                # MAX_EVENTS/N_SNAPSHOTS = 100, so
                                                # that end is sampled twenty times
                                                # over.  A hundred snapshots would
                                                # leave one sample per factor of
                                                # ten in N and sag the middle of
                                                # the band by 0.4 in slope.
SEED = 20260913
MIN_COUNT = 5.0                                 # blank bins thinner than this

PRED = mc.predict("closed", 0.0)                # beta = 0, alpha = -1

#  PRL FIGURE STYLE.  One column is 3.375 in (8.6 cm), and two stacked panels
#  of a spectrum is exactly that shape, so most of these are single-column.
#  The journal puts into the caption everything a title would say, so what
#  would be a title is set as a small line INSIDE the panel: it reads in the
#  paper, and it reads in the repository, where nobody has the caption.
#  Times if the system has it, DejaVu Serif otherwise; stix mathtext ships
#  with matplotlib and matches either.
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


def picket_fence_mass():
    """Below this mass a log bin cannot hold two integer masses.

    Mass in a closed box is always an integer multiple of the initial one, so
    the spectrum at small m is a set of LINES spaced by M_INIT.  A logarithmic
    bin of `dex` decades at mass m is (10^dex - 1)*m wide and holds more than
    one line only above M_INIT/(10^dex - 1).  Below that, a bin holds one line
    or none, and the density divides the line by a width that grows with m:
    n_k ~ 1/k becomes F ~ 1/m^2.  That is where the -2 at the bottom of the
    figure comes from, manufactured by binning alone.
    Finer bins push the threshold UP, which is the opposite of the usual
    instinct.
    """
    dex = np.log10(EDGES[1] / EDGES[0])
    return M_INIT / (10.0 ** dex - 1.0)


def run_cascade():
    an.check_grid(EDGES, M_INIT)
    return mc.simulate(
        process="coagulation", system="closed",
        kernel=mc.kernel_constant, edges=EDGES,
        ic={"m": M_INIT, "N": N0},
        max_events=MAX_EVENTS,
        snapshot_mode="events", snapshot_stride=MAX_EVENTS / N_SNAPSHOTS,
        track_generations=False, track_waiting=False,
        rng=np.random.default_rng(SEED), verbose=False,
    )


def collapse_figure(run, path_shape, path_clock):
    """The shape that does not move, and the clock that says the run is right."""
    c = np.asarray(run["centers"], float)
    D = np.asarray(run["dndm"], float)
    counts = an.raw_counts(run)
    t = np.asarray(run["t"], float)
    live = np.asarray(run["live"], float)
    M_sys = np.asarray(run["M_sys"], float)
    mbar = M_sys / np.maximum(live, 1.0)

    mbar_exact = mc.exact_closed_constant(t, m_init=M_INIT, n_init=float(N0),
                                          K1=1.0, process="coagulation")
    err = np.nanmax(np.abs(mbar / mbar_exact - 1.0))
    print("\n  <m> reached       %.1f  (%.2f decades of growth)"
          % (mbar[-1], np.log10(mbar[-1] / M_INIT)))
    print("  <m>(t) against the exact solution: worst relative error %.3f%%"
          % (100 * err))

    #  ONE SNAPSHOT PER DECADE OF <m>, and nothing else.  The run stores ten
    #  thousand of them because the time integral needs that resolution, but a
    #  figure does not: the snapshot whose mean mass sits closest to 1, 10, 100,
    #  ... is a fair representative of its decade, and three or four such curves
    #  say what the whole fan says without hiding any of them behind the rest.
    #  A target with no snapshot within a fifth of a decade is dropped rather
    #  than drawn from whatever happened to be nearest -- which is why <m> = 1
    #  does not appear: at t = 0 a single bin holds everything.
    targets = 10.0 ** np.arange(0.0, 6.0)
    cmap = plt.cm.viridis
    cnorm = plt.matplotlib.colors.LogNorm(vmin=1.0, vmax=max(mbar[-1], 10.0))

    fig, ax = plt.subplots(1, 1, figsize=(COL1, 0.80 * COL1))
    drawn = 0
    for target in targets:
        k = int(np.argmin(np.abs(np.log10(np.maximum(mbar, 1e-300) / target))))
        if abs(np.log10(mbar[k] / target)) > 0.2:
            continue
        ok = (D[k] > 0) & (counts[k] >= MIN_COUNT)
        if ok.sum() < 2:
            continue
        ax.loglog(c[ok], D[k][ok], lw=1.1, color=cmap(cnorm(target)),
                  label=r"$\langle m\rangle=%g$" % target)
        drawn += 1

    ax.set_xlabel(r"mass $m$")
    ax.set_ylabel(r"$dN/dm$")
    ax.text(0.03, 0.05,
            r"closed coagulation, $\lambda=0$" "\n"
            r"%.0e bodies, %.1e events" % (N0, float(run["events"][-1])),
            transform=ax.transAxes, va="bottom", fontsize=6.5)
    ax.legend(loc="upper right")
    print("  snapshots drawn   %d, one per decade of <m>" % drawn)
    fig.tight_layout()
    save_figure(fig, path_shape)

    #  THE CLOCK, on its own axes because it is a different statement: the
    #  figure above is about shape, this one is about whether the time step was
    #  built correctly at all.
    figc, ax = plt.subplots(1, 1, figsize=(COL1, 0.80 * COL1))
    ax.plot(t, mbar, "o", ms=2.5, color="C0", label=r"measured $\langle m\rangle$")
    ax.plot(t, mbar_exact, "-", lw=1.0, color="C3", label=r"Smoluchowski, $K=1$")
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    ax.set_xlabel(r"time $t$")
    ax.set_ylabel(r"$\langle m\rangle$")
    ax.legend(loc="upper left")
    ax.text(0.97, 0.06, "worst error %.2f%%" % (100 * err), fontsize=6.5,
            ha="right", transform=ax.transAxes)
    figc.tight_layout()
    save_figure(figc, path_clock)
    return float(mbar[-1])


def spectrum_figure(run, m_top, path):
    """The age-integrated spectrum of the closed cascade, against -1 and -2."""
    c = np.asarray(run["centers"], float)
    spec = an.spectrum(run)                       # closed -> int (dN/dm) dt
    plateau = an.spectrum_plateau(run, spec=spec)
    m_fence = picket_fence_mass()
    band = an.guard_band(max(m_fence, M_INIT), m_top, pad_decades=PAD)
    inside = (c >= band[0]) & (c <= band[1])
    fit = an.wls_powerlaw(c, spec["F"], spec["sigma"], mask=inside)

    print("\n  prediction        alpha = %+.4f   (exact for K = 1: int n dt ~ 1/m)"
          % PRED["alpha"])
    print("  picket fence      below m = %.2f a bin cannot hold two integer masses,"
          % m_fence)
    print("                    and 1/k divided by a growing bin width reads as -2")
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
    print("  offset from -2    %+.4f = %.0f sigma"
          % (a_q + 2.0, abs(a_q + 2.0) / max(s_q, 1e-12)))

    fig, axs = plt.subplots(2, 1, figsize=(COL1, 1.55 * COL1), sharex=True,
                            gridspec_kw={"height_ratios": [2.1, 1.0]})

    an.plot_spectrum(axs[0], run, spec=spec, plateau=plateau, compensate=True)
    xr = np.logspace(np.log10(band[0]) - 0.35, np.log10(band[1]) + 0.35, 40)
    A = an.anchor_amplitude(c, spec["F"], band[0], band[1], PRED["alpha"])
    axs[0].plot(xr, A * xr ** (PRED["alpha"] + 2.0), "--", lw=1.0, color="C0",
                label=r"$-1$, the prediction")
    A2 = an.anchor_amplitude(c, spec["F"], band[0], band[1], -2.0)
    axs[0].plot(xr, A2 * np.ones_like(xr), ":", lw=1.0, color="0.35")
    an.compensated_ylim(axs[0], c, spec["F"])
    axs[0].set_ylabel(r"$m^{2}\int (dN/dm)\,dt$")
    axs[0].text(0.03, 0.96,
                r"closed coagulation, $\lambda=0$" "\n"
                r"age-integrated spectrum",
                transform=axs[0].transAxes, va="top", fontsize=6.5)
    axs[0].legend(loc="lower right")

    _, gamma = an.local_slope(c, spec["F"], half=3)
    axs[1].semilogx(c, gamma, "o-", ms=2.0, lw=0.7, color="k")
    axs[1].axhline(PRED["alpha"], ls="--", lw=1.0, color="C0")
    axs[1].axhline(-2.0, ls=":", lw=1.0, color="0.35")
    axs[1].text(band[0] * 1.1, PRED["alpha"] + 0.06, r"$-1$", color="C0", fontsize=6.5)
    axs[1].set_ylim(-2.6, -0.4)
    axs[1].set_xlabel(r"mass $m$")
    axs[1].set_ylabel(r"$d\log F/d\log m$")

    for ax in axs:
        ax.axvspan(band[0], band[1], color="C1", alpha=0.08)
        ax.axvspan(EDGES[0], band[0], color="0.5", alpha=0.13)
        ax.axvspan(band[1], EDGES[-1], color="0.5", alpha=0.13)
        ax.set_xlim(EDGES[0], m_top * 10.0)
    axs[1].text(EDGES[0] * 1.15, -0.6, "picket fence", color="0.35", fontsize=6)
    axs[1].text(band[1] * 1.3, -0.6, r"$\langle m\rangle$ at the end",
                color="0.35", fontsize=6)

    fig.tight_layout()
    save_figure(fig, path)


def main():
    t_wall = time.perf_counter()
    run = run_cascade()
    print(an.describe(run, "ex1  closed / constant kernel"))

    FIGURES.mkdir(parents=True, exist_ok=True)
    m_top = collapse_figure(run, FIGURES / "ex1_closed_selfsimilar.png",
                            FIGURES / "ex1_closed_clock.png")
    spectrum_figure(run, m_top, FIGURES / "ex1_closed_spectrum.png")
    print("\n  wall time         %.1f min" % ((time.perf_counter() - t_wall) / 60.0))


if __name__ == "__main__":
    main()
