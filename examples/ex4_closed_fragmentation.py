"""
ex4 -- closed fragmentation, geometric kernel, bounded splits.

A hundred bodies of mass 1 shatter until the mean mass reaches the floor at
m = 1e-4.  Nothing enters and nothing leaves, so this is the mirror of ex1:
no steady state, and the observable is the time-integrated spectrum, which
for a closed box is a power law with alpha = -(1+lambda) = -5/3.

The split is uniform on (0.15, 0.85) rather than on (0, 1), and that is the
whole reason this example can be closed at all.  With unbounded splits a body
just above the floor can throw off a fragment decades below it, which then
freezes there for ever: dust that carries almost none of the mass and almost
all of the NUMBER, and dN/dm counts number.  Bounded splits keep the fragment
above 0.15 of its parent, so the floor only thins the decade above it and the
window is clean.  A sink, as in ex3, is the other way to solve the same
problem.  The price is that narrow splits keep the generations from
overlapping and comb the spectrum at the mean step; 0.35 is where that is
small and the dust is already gone.

ONE THING TO KNOW BEFORE WIDENING THE WINDOW.  This cascade ENDS -- everything
settles on the floor and nothing can fragment again -- and the clock keeps
running after it does, because the time step is charged per trial and the
trials continue.  So int (dN/dm) dt does not converge here the way it does in
ex1, where the cascade never terminates: the frozen final state accumulates
time without limit.  That divergent part sits in the bins at and below the
floor, which is why the window starts 0.85 decades above it.  Remove the
padding and the number stops meaning anything.

Runtime: under a minute.  N0 and MAX_EVENTS are the knobs, and they scale
together.
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

M0 = 1.0                                      # every body starts here
MIN_FRAG = 1.0e-4                             # the floor: lighter bodies rest
N0 = 100                                      # independent cascades
MAX_EVENTS = 550_000                          # SET BY THE ACCEPTANCE, not by a
                                              # generation count.  Bodies under
                                              # the floor never fragment again
                                              # but are still drawn as trial
                                              # partners, so once the mean mass
                                              # reaches the floor the acceptance
                                              # falls off a cliff: measured at
                                              # N0 = 20 it was 0.90 up to 90k
                                              # events, 0.50 at 105k and 0.09 at
                                              # 150k, with the cpu time growing
                                              # four times faster than the event
                                              # count.  This budget is the same
                                              # fraction of the cascade, scaled
                                              # to N0.
EDGES = np.logspace(-6.0, 0.5, 131)           # 20 bins per decade
F_LOC = 0.30                                  # locality: impactor >= f * target
SPLIT_W = 0.35                                # xi uniform on (0.15, 0.85).  The
                                              # engine warns against going below
                                              # 0.25: narrow splits stop the
                                              # generations from overlapping and
                                              # comb the spectrum at the mean
                                              # step -- 13% at w = 0.25.  This is
                                              # the compromise: no dust, and the
                                              # comb about half of that.
PAD = 0.85                                    # decades stripped off each end.
                                              # A fragment can be as light as
                                              # 0.15 of its parent, so the region
                                              # thinned by freezing reaches 0.82
                                              # decades above the floor.  That
                                              # leaves 2.3 decades of window.
GEN_MAX = 64
G_SHOW = 16                                   # highest generation drawn
N_SNAPSHOTS = 2_000                           # the superposition is a quadrature
                                              # in PHYSICAL time, and dt ~ 1/N^2
                                              # with N growing, so most of the
                                              # time sits in the first events.
                                              # Too few snapshots there and the
                                              # top of the spectrum is a guess.
MIN_MASS_FRAC = 2e-3                          # a generation needs this much mass
CLEAR_FLOOR = 2.0                             # stay this many sigma above it
VERBOSE = False                               # the engine's own progress line:
                                              # events, live bodies, <m>, m_max
                                              # and the acceptance.  Turn it on
                                              # when changing N0 or MAX_EVENTS
                                              # and watch the acceptance: when it
                                              # falls the box is settling on the
                                              # floor and the rest of the run is
                                              # trials thrown away.
VERBOSE_EVERY = 30                            # seconds between progress lines
SEED = 20260913

LAM = 2.0 / 3.0
PRED = mc.predict("closed", LAM)               # beta = 2/3, alpha = -5/3

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


def run_cascade():
    an.check_grid(EDGES, M0, MIN_FRAG)
    return mc.simulate(
        process="fragmentation", system="closed",
        kernel=mc.kernel_geometric, edges=EDGES,
        ic={"m": M0, "N": N0},
        min_frag_mass=MIN_FRAG,
        frag_min_ratio=F_LOC,
        frag_split_width=SPLIT_W,
        frag_age_rule="inherit",
        track_generations=True, gen_max=GEN_MAX, track_waiting=False,
        max_events=MAX_EVENTS,
        snapshot_mode="events", snapshot_stride=MAX_EVENTS / N_SNAPSHOTS,
        rng=np.random.default_rng(SEED), verbose=VERBOSE,
        verbose_every=VERBOSE_EVERY,
    )


def spectrum_figure(run, path):
    """The time-integrated spectrum, against -5/3 and -2."""
    c = np.asarray(run["centers"], float)
    spec = an.spectrum(run)                        # closed -> the superposition
    plateau = an.spectrum_plateau(run, spec=spec)
    band = an.guard_band(MIN_FRAG, M0, pad_decades=PAD)
    inside = (c >= band[0]) & (c <= band[1])
    fit = an.wls_powerlaw(c, spec["F"], spec["sigma"], mask=inside)

    print("\n  prediction        alpha = %+.4f   (beta = lambda = %.4f)"
          % (PRED["alpha"], PRED["beta"]))
    print("  guard band        %.3g .. %.3g  (%.2f decades, %d bins)"
          % (band[0], band[1], np.log10(band[1] / band[0]), int(inside.sum())))
    print("  PLATEAU           alpha = %+.4f +- %.4f over %.2f decades"
          % (plateau["alpha"], plateau["scatter"], plateau["decades"]))
    print("  cross-check fit   alpha = %+.4f +- %.4f   chi2/dof = %.1f"
          % (fit["p"], fit["sigma_p"], fit["chi2_dof"]))
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
                label=r"$-5/3$, the prediction")
    A2 = an.anchor_amplitude(c, spec["F"], band[0], band[1], -2.0)
    axs[0].plot(xr, A2 * np.ones_like(xr), ":", lw=1.0, color="0.35")
    an.compensated_ylim(axs[0], c, spec["F"])
    axs[0].set_ylabel(r"$m^{2}\int (dN/dm)\,dt$")
    axs[0].text(0.03, 0.96,
                r"closed fragmentation, $\lambda=2/3$, $f=%.2f$" "\n"
                r"$\xi\in(%.2f,%.2f)$, %.1e splits"
                % (F_LOC, 0.5 - SPLIT_W, 0.5 + SPLIT_W, float(run["events"][-1])),
                transform=axs[0].transAxes, va="top", fontsize=6.5)
    axs[0].legend(loc="lower left")

    _, gamma = an.local_slope(c, spec["F"], half=3)
    axs[1].semilogx(c, gamma, "o-", ms=2.0, lw=0.7, color="k")
    axs[1].axhline(PRED["alpha"], ls="--", lw=1.0, color="C0")
    axs[1].axhline(-2.0, ls=":", lw=1.0, color="0.35")
    axs[1].text(band[0] * 1.1, PRED["alpha"] + 0.05, r"$-5/3$", color="C0", fontsize=6.5)
    axs[1].set_ylim(-2.45, -1.25)
    axs[1].set_xlabel(r"mass $m$")
    axs[1].set_ylabel(r"$d\log F/d\log m$")

    for ax in axs:
        ax.axvspan(band[0], band[1], color="C1", alpha=0.08)
        ax.axvspan(EDGES[0], band[0], color="0.5", alpha=0.13)
        ax.axvspan(band[1], EDGES[-1], color="0.5", alpha=0.13)
        #  The two ends of the cascade, drawn rather than implied: bodies below
        #  MIN_FRAG never fragment again and pile up there, and M0 is where
        #  every body started.  The padding exists to keep both out of the fit.
        ax.axvline(MIN_FRAG, ls="-.", lw=1.0, color="C3")
        ax.axvline(M0, ls="-.", lw=1.0, color="C3")
    axs[1].text(MIN_FRAG * 1.2, -1.33, "floor", color="C3", fontsize=6)
    axs[1].text(M0 * 1.2, -1.33, "start", color="C3", fontsize=6)

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
    x_floor = np.log(MIN_FRAG / gen["m_inj"])

    #  A generation is usable while its packet is still clear of the floor.  The
    #  closed box records no sink mass, so gen_moments cannot apply this cut for
    #  us; it is the same rule, written out.
    ok = (np.isfinite(mom["mu"])
          & (mom["mu"] - CLEAR_FLOOR * np.sqrt(np.maximum(mom["var"], 0.0)) > x_floor))
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
    axs[0].axvline(x_floor, ls=":", lw=1.0, color="C3")
    axs[0].text(x_floor + 0.2, axs[0].get_ylim()[1] * 0.92, "floor", color="C3",
                fontsize=6.5)
    axs[0].set_xlim(x_floor - 0.5, 0.8)
    axs[0].set_xlabel(r"$x=\ln\,(m/m_{0})$")
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
           label=r"$g\,\langle\ln\xi\rangle$  (exact)")
    a.set_xlabel("generation $g$")
    a.set_ylabel(r"$\langle x\rangle$", color="C0")
    a.tick_params(axis="y", labelcolor="C0")
    a.legend(loc="lower left")

    a2 = a.twinx()
    a2.plot(g[use], mom["var"][use], "s", ms=3.5, color="C3", label=r"${\rm Var}(x)$")
    a2.plot(g[use], var_step * g[use], "-", lw=1.0, color="C3",
            label=r"$g\,{\rm Var}(\ln\xi)$  (exact)")
    a2.set_ylabel(r"${\rm Var}(x)$", color="C3")
    a2.tick_params(axis="y", labelcolor="C3")
    a2.legend(loc="upper left")

    fig.tight_layout()
    save_figure(fig, path)


def main():
    t_wall = time.perf_counter()
    mu_step, _ = an.step_stats(SPLIT_W, weight="mass")
    print("  box               %.3g .. %.3g  (%.1f decades, downwards)"
          % (MIN_FRAG, M0, np.log10(M0 / MIN_FRAG)))
    print("  split             xi uniform on (%.2f, %.2f), <ln xi> = %+.4f"
          % (0.5 - SPLIT_W, 0.5 + SPLIT_W, mu_step))
    print("  the floor is      %.1f generations down"
          % (np.log(MIN_FRAG / M0) / mu_step))

    run = run_cascade()
    print(an.describe(run, "ex4  closed fragmentation / geometric kernel"))

    live = np.asarray(run["live"], float)
    M_sys = np.asarray(run["M_sys"], float)
    mbar_end = float(M_sys[-1]) / max(float(live[-1]), 1.0)
    print("\n  population        %.0f -> %.0f  (every split adds one body,"
          " nothing is removed)" % (live[0], live[-1]))
    print("  <m> at the end    %.3g = %.1f x the floor"
          % (mbar_end, mbar_end / MIN_FRAG))
    if mbar_end > 5.0 * MIN_FRAG:
        print("  the cascade has NOT reached the floor: the bottom of the window")
        print("  will be empty.  Raise MAX_EVENTS.")
    else:
        print("  the cascade has reached the floor, so the window is filled.")

    FIGURES.mkdir(parents=True, exist_ok=True)
    spectrum_figure(run, FIGURES / "ex4_closed_fragmentation_spectrum.png")
    generation_figure(run, FIGURES / "ex4_closed_fragmentation_generations.png")
    print("\n  wall time         %.1f min" % ((time.perf_counter() - t_wall) / 60.0))


if __name__ == "__main__":
    main()
