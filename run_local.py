"""
run_local.py -- GEOMETRIC KERNEL.  Built from the additive lambda = 1 runner of
Majorant_v2_parallel by build_geometric_runner.py, and then curated by hand for
this repository.  THIS COPY IS EDITED DIRECTLY: the builder lives in the working
folder and does not reach here, so there is no rebuild to throw an edit away.
The engine and the analysis layer are named `masscascade` and `analysis` here,
not `BF_warm_start_v7` and `BF_analysis`; the aliases BF and AN are kept because
the whole file reads them.

WHAT CHANGED.  The kernel is BF.kernel_geometric, lambda = 2/3, so the open
predictions are

        beta  = (1+lambda)/2 = 5/6
        alpha = -(3+lambda)/2 = -11/6 = -1.8333
        b     = 2/(1-lambda) = 6

and three things follow, which are the whole difference.

  THE TARGET IS -11/6, NOT -2.  Every reference line, tolerance and axis was
  moved.  A run that returns -2 here is NOT a success: it is the fingerprint of
  engine notes [5iv] and [8] -- the clock advanced per event instead of per
  trial, or a non-local fragmentation rule -- both of which pin the index at -2
  for EVERY kernel.  At lambda = 1 that degeneracy is invisible because -2 is
  also the right answer; at lambda = 2/3 it is a decisive test, and that is the
  main reason for this folder.

  THE GROWTH IS A POWER LAW, m ~ tau^6, not an exponential.  The growth panel
  is drawn on LOG-LOG axes, the fit returns b, and the column that measures it
  is dlog(tau) per fixed step in log m: a constant there is a power law.  Note
  db/dalpha = 36 at this kernel, so quote b to two significant figures at most
  and let alpha carry the result.

  THE LOCALITY WINDOW IS NO LONGER WHAT MAKES THE POWER LAW EXIST.  At
  lambda = 1 it was.  At lambda = 2/3 the coagulation flux integral converges
  by itself (engine note [17]), so the default is f = 0 for coagulation.
  Fragmentation still needs a disruption threshold for its own reason --
  alpha < -1, engine note [8] -- so the default there stays f = 0.30, as in the
  earlier geometric runs of Majorant_v2\runs.

NO SIPHON.  `siphon` is never passed, so the engine runs its lossless branch.

EVERYTHING BELOW IS THE ORIGINAL DOCSTRING OF THE ADDITIVE RUNNER.  It is kept
because the machinery it describes -- the pass accumulation, the picket fence,
the isochrone regrouping, the half-vs-half test -- is unchanged, and that is
why this file is a patched copy rather than a rewrite.  Its MEASURED NUMBERS do
not transfer: the table of drift against f, the residence time t_c ln R, the
argument that the age grid should be linear, and every appearance of -2 as the
answer were all taken at lambda = 1.  Read them as history.
----------------------------------------------------------------------------
run_local.py -- the local cascade at lambda = 1, coagulation and fragmentation.

WHAT THIS IS.  The production runner for lambda = 1 with the locality window on.
It replaces scan_lambda1.py, which ran the same box without the window and found
no power law in it; the window is the difference, and the measurement that
justified it is in note [17] of the engine.

THE MEASUREMENT IT IS BUILT ON.  In the four-decade box the local slope changed
with f exactly as a locality argument predicts, monotonically and by a large
factor:

        f       drift of Gamma per decade      plateau reached
        0            +0.293                    0.2 decades
        0.01         +0.033                    0.2
        0.10         -0.007                    1.4
        0.30         +0.024                    1.8

A residual drift of order 0.01-0.02 per decade survives at f >= 0.1, small but
larger than its own error.  Whether that is a boundary layer or a real
dependence on f cannot be told in four decades, because the boundary layers eat
two of them.  Six decades is the test: if the residual is boundary, it falls as
the range grows.  That is what this file is for.

THE BOX, and why the injection sits a decade below the measurement.
        coagulation    injection at m = 0.1,   wall at m = 1e6
        fragmentation  injection at m = 1e6,   wall at m = 0.1
    and the measured range is m = 1 .. 3.3e4, a decade above the injection and
    1.5 decades below the wall: 4.5 decades quoted out of seven.

THE FIRST DECADE IS SPENT ON PURPOSE.  Every mass in the
run is an integer multiple of the injected one, so the spectrum is a set of
lines, not a continuum.  A logarithmic bin of 0.1 dex at mass m is 0.259 m wide,
and it can only look smooth once it is wider than the gap between consecutive
lines -- one injection mass.  The condition is

        0.259 m > m_inj      ->      m > 3.9 m_inj

With injection AT the bottom of the measured range this fails over the whole
first decade, and the histogram alternates between bins holding one line and
bins holding two.  That is a picket fence, not noise: it does not average down,
because there is nothing between the lines to average.  Adding particles cannot
touch it.

Putting the measured range a decade ABOVE the injection moves the fence out of
it: with m_inj = 0.1 the fence lives below m = 0.39, and from m = 1 upward every
bin holds ten or more lines.  The price is one extra decade of range, and it
buys a low end that is smooth for the same reason the high end always was.

The whole histogram is drawn, injection decade included, so the fence is visible
rather than hidden; the core window starts at m = 1 and the figure shades it.

A MILLION BODIES still matters, but at the other end.  The steady spectrum is
m^-2, so the number of bodies above m falls as 1/m and the top decades are
carried by the fewest particles.  That is where the population buys statistics
and where it is spent.

HOW IT RUNS.  Eight replicas, one process each, independent cold starts with
different seeds -- a shared warm start would leave them correlated through the
initial condition and the spread between them would understate the error.  Each
replica is stopped on a fixed number of EVENTS, not on the clock, because the
cores of this machine are not identical and a wall-clock brake would hand the
fast ones more statistics.

Every pass of --pass-hours writes the replica's state, its result and its
figure, atomically, so a cut costs the pass in progress and nothing more.
Re-running the same command continues from those states; --fresh ignores them.

WHAT THE RANGE ACTUALLY COSTS, measured rather than modelled.  Filling is cheap:
in the four-decade box the leading edge reached the wall after about 3e6 events,
a few per cent of one pass.  What is expensive is the WALL: one absorption
requires m_sink/m_inj merges, because that is how many injected bodies go into
one body the size of the wall.  Here that is 1e7 events per absorption,
against 1e4 in the four-decade box, so an eight-hour replica collects of order a
hundred absorptions rather than ten thousand.

That is a real limitation and it is stated rather than hidden.  It does NOT
bear on the core window, which stops 1.5 decades below the wall, but it means
the top of the range is thinly sampled and the wall layer is not the place to
read anything off.  Because absorption counts cannot certify the middle, the
run instead tests the middle directly: every pass compares the local slope
built from the first half of its snapshots against the second half, and reports
the largest disagreement inside the core window.  A middle that has stopped
moving is stationary whatever the wall is doing.

Checkpoints are hourly and the run is resumable -- look at the figures as they
appear rather than waiting for the end.

EVERYTHING ACCUMULATES ACROSS PASSES, and it has to.  simulate() rebuilds its
histograms from scratch on every call, and every pass is a fresh call from the
saved state, so a runner that simply writes what the last call returned reports
a spectrum standing on one pass while the header quotes the total events.  That
happened: a two-pass fragmentation run announced 1.8e8 events per replica above
a spectrum built from 9.0e7, and the discarded fraction grows with every extra
hour.  Here every pass is kept:

    the spectrum       is a density, so passes are averaged with weight
                       proportional to their events;
    the isochrones     are counts, so passes simply ADD, with no weights at all.

Both counts are printed -- what was computed and what the answer actually stands
on -- so they cannot silently disagree again.  Passes are stored SEPARATELY as
well as summed, so --skip-passes k drops the early ones afterwards, for free,
when the box turns out to have still been relaxing.

THE ISOCHRONES, and the age binning, which is the delicate part.

An isochrone is the mass distribution of the bodies of a given age, and at
lambda = 1 it is the growth law made visible: m = m_inj exp(tau / t_c), so
log m is LINEAR in tau.  Two consequences follow, and the second was learned the
hard way.

    THE AGE GRID MUST BE LINEAR.  With logarithmically spaced ages, fourteen of
    thirty-nine bins held nothing but the injection mass -- identical curves
    resolving a stretch in which nothing had happened -- while by tau = 3 t_c a
    single bin was 1.2 t_c wide, which is half a decade of mass smeared inside
    it.  That is the isochrone that "already fills the whole spectrum".  A
    linear grid puts consecutive ages a constant distance apart in log m, which
    is what the picture wants.

    THE GRID IS RUN FINE AND MERGED AFTERWARDS.  Merging age bins is exact --
    the engine accumulates counts and counts add, so a merged group is precisely
    the histogram a coarser grid would have produced -- while splitting them is
    impossible.  So the resolution is bought once, at 0.05 t_c, and spent at
    analysis time by a rule that can be changed without re-running: keep
    merging until the mean mass has advanced --iso-dlogm decades and the group
    holds --iso-min-counts particles.

    WHAT THAT LEAVES FREE IS A MEASUREMENT.  With the separation in log-mass
    fixed, the age STEP between consecutive isochrones is not imposed by
    anything.  A constant step means log m advances linearly in time -- growth
    is exponential, b is infinite -- which is exactly what beta = 1 requires.  A
    step that grows down the column would be a power law instead.  The column is
    printed for that reason, and the growth panel is drawn on LOG-LINEAR axes
    where an exponential is a straight line and nothing else is.

Isochrones are coagulation only.  In fragmentation the age of a piece is a
matter of which fragment inherits the clock, and it is `gen` that carries the
cascade there.

USAGE.  An hour at a time, repeated as often as you like -- each invocation
adds one more hour to every replica:

        python run_local.py --process coag --add-hours 1
        python run_local.py --process coag --add-hours 1
        python run_local.py --process coag --add-hours 1     ...

Or a fixed total in one go, where the number is what a replica will have done
when it stops, not what it will add:

        python run_local.py --process coag --hours 8
        python run_local.py --process frag --hours 8

Repeating a --hours command does nothing: the replicas already hold that much
and return at once.  That is why --add-hours exists.

        python run_local.py --process coag --stack-only     # just redraw
        python run_local.py --process coag --fresh --add-hours 1   # start over

THE ISOCHRONES ARE REDRAWN WITHOUT RE-RUNNING.  The age grid on disk is much
finer than anything worth drawing, so the three knobs below are analysis-time
and can be turned as often as you like:

        python run_local.py --process coag --stack-only --iso-dlogm 0.5
        python run_local.py --process coag --stack-only --iso-mmin 10

    --iso-dlogm       how far apart, in decades of mean mass, the drawn
                      isochrones stand.  Larger = fewer, fatter, better counted.
    --iso-min-counts  the floor that widens the groups near the top of the
                      cascade, where the particles have run out.
    --iso-mmin        the mass floor.  Default is the bottom of the core window,
                      m = 1: the injection decade is COMPUTED and STORED, and
                      left out of the measurement because it is a picket fence
                      of discrete multiples rather than a continuum.

BOTH PROCESSES IN ONE GO, unattended.  PowerShell runs these one after the
other, the second starting whether or not the first succeeded:

    python run_local.py --process coag --add-hours 2; python run_local.py --process frag --add-hours 8

Sequentially, and NOT in two windows at once.  Two commands together would put
sixteen processes on twenty cores, where the measured efficiency of this machine
falls to 0.56 against 0.87 at eight -- the extra processes contend for memory
bandwidth rather than compute, and the pair finishes with LESS total work done
than if they had waited for each other.  Sleep stays blocked for as long as
either command is alive.

One thing to settle before leaving it overnight: if `runs` sits inside a synced
folder, eight replicas rewriting their states every hour is of order a gigabyte
an hour for the sync client to re-upload, and it may hold a file open while the
run tries to replace it.  The atomic write survives a power cut, not another
program's lock.  Exclude the folder from sync, or pause it.

WHAT TO READ IN THE MORNING.  Two lines, and only two:

    drift per decade    zero within its error means a power law.  <Gamma>
                        near -2 does not: an average taken across a sloping
                        curve can land on the right answer by accident.

    half-vs-half        the MEDIAN against the scatter.  The worst single bin
                        is printed beside it with the mass it sits at, and
                        that mass is the whole reading: at the top of the
                        window it is the wall still filling, in the middle it
                        is the box still relaxing.  Those are different
                        things and only the second invalidates anything.
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


#  POPULATION.  Counts in a bin are proportional to it, and the steady spectrum
#  is m^-2, so the number of bodies per logarithmic bin falls as 1/m and the top
#  of the range is carried by the fewest.  Raising it is the lever that buys
#  statistics there; the price is the transient, which scales with the
#  population.  Memory is not the constraint at any of these values: about
#  106 bytes per live body is 106 MB per replica, 850 MB across eight.
N_SS = 1.0e6                     # bodies held in the box
#  THE MASS GRID: 0.1 dex bins, from one decade below the lower scale of the
#  box to one decade above the upper one.  It used to run 1e-4 .. 1e12 for
#  every run regardless -- a hundred and sixty bins of which two thirds were
#  permanently empty, kept only because one fixed array was simpler than one
#  per box.  With the two processes no longer sharing a box that stopped being
#  simpler and started being wrong: `rebin` and the isochrone widths read the
#  global array, so a file written under one box could be re-binned under
#  another without anything complaining.
#
#  The margin of one decade on each side is not decoration.  Below the sink it
#  holds nothing in a healthy run -- a body under the sink is removed at the
#  moment it is made, so it is never live at a snapshot -- and that emptiness
#  is the check: anything appearing there is a body that escaped the sink.
#  Above the injection it catches the pile-up at a coagulation wall.
#
#  Each run stores its own edges in its spec file, so analysis never has to
#  guess which grid a file was written on.
def mass_edges(g):
    lo = np.log10(min(g["m_inj"], g["m_sink"])) - 1.0
    hi = np.log10(max(g["m_inj"], g["m_sink"])) + 1.0
    return 10.0 ** np.round(np.arange(lo, hi + 0.05, 0.1), 6)


#  Kept for callers that have no geometry to hand; it is the widest of the two
#  boxes, so slicing it is always safe.
EDGES = 10.0 ** np.round(np.arange(-2.0, 7.05, 0.1), 6)
#  GENERATIONS.  64 was not enough: with the locality window every split
#  takes off between 0.3 and 0.7 of the parent, so reaching the sink can need
#  forty-odd of them, and the engine reported live bodies at g = 69 with
#  everything above the cap going silently into gen_overflow.  A truncated
#  generation histogram is exactly the quantity the fragmentation panel is
#  built from, so the cap is doubled and the overflow warning is left in as
#  the tripwire.
GEN_MAX = 128
PROBE_EVENTS = 200_000
SNAPSHOTS = 200
#  REBINNING is OFF by default: the histogram is reported at the resolution the
#  engine produced it, 0.1 dex.  The machinery stays because it costs nothing --
#  the fine histogram is on disk and merging counts is exact and reversible --
#  so --rebin k can be tried after the fact without re-running anything.  It is
#  not applied unasked.
REBIN_DEFAULT = 1
REACH_TOL = 0.05

#  ISOCHRONES: THE AGE GRID IS LINEAR, and that is the whole point of it.
#
#  At lambda = 1 a body grows exponentially, m = m_inj exp(tau / t_c), so
#  log m is LINEAR in tau.  A logarithmically spaced age grid is therefore the
#  wrong shape twice over: it spends most of its bins on the stretch before
#  anything has happened, and it makes each late bin so wide that one isochrone
#  covers half the cascade.  Measured on a real run with 0.15 dex age bins:
#  fourteen of the thirty-nine bins held nothing but the injection mass, and by
#  tau = 3 t_c a single bin was 1.2 t_c wide, which is 0.52 decades of mass
#  smeared inside it.  That is the isochrone that "already fills the spectrum".
#
#  A linear grid of constant step dt puts consecutive isochrones a constant
#  distance apart in log m, which is what a picture of exponential growth wants.
#  The step below is deliberately FINER than anything that will be drawn:
#  merging age bins afterwards is exact (the engine accumulates counts, and
#  counts add), splitting them is impossible, so the resolution is bought once
#  at run time and spent at analysis time by iso_regroup.
ISO_DT = 0.05                    # fine age step, in units of t_c
ISO_SPAN = 1.6                   # cover this multiple of the residence time
ISO_DLOGM = 0.25                 # target separation of drawn isochrones, decades
ISO_MIN_COUNTS = 1.0e4           # a drawn isochrone holds at least this much
ISO_START_SINK = 0               # accumulate from the first snapshot


# ============================================================================
#  THE KERNEL, AND EVERYTHING THAT FOLLOWS FROM IT.
#
#  One name is changed here and the rest of the file reads these constants, so
#  there is no second place where lambda is written down.  main() checks them
#  against the engine's own predict() before anything long starts.
# ============================================================================
KERNEL_NAME = "kernel_geometric"
LAMBDA = 2.0 / 3.0
BETA_TH = 0.5 * (1.0 + LAMBDA)            # 5/6
ALPHA_TH = -0.5 * (3.0 + LAMBDA)          # -11/6 = -1.8333
B_TH = 2.0 / (1.0 - LAMBDA)               # 6
P_TH = -ALPHA_TH                          # dN/dm ~ m^-P_TH
#  Spectra are drawn compensated by m^COMP, so a correct one is FLAT.  The
#  additive runner used m^2 because there p = 2; here that would leave a
#  m^(1/6) tilt across the range -- a factor 5.6 over 4.5 decades, which reads
#  as a real slope on a log axis.
COMP = P_TH


def _kernel(BF):
    return getattr(BF, KERNEL_NAME)


def box_mass(g):
    """The mass a steady p = 11/6 spectrum holds when N_SS bodies are alive.

    For dN/dm = A m^-p with 1 < p < 2 the two moments live at opposite ends of
    the range: the COUNT is set by the light end and the MASS by the heavy one,

        N = A lo^(1-p)/(p-1),        M = A hi^(2-p)/(2-p),

    so M = N_ss (p-1)/(2-p) lo^(p-1) hi^(2-p).  The additive runner had the
    p -> 2 form of this, M = N_ss lo ln(hi/lo), where the logarithm is what is
    left of (2-p)^-1 when the mass integral turns marginal.  Getting this wrong
    is not cosmetic: it is the line that once asked the engine for 7e10 live
    particles.
    """
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


#  THE BOX: 0.1 .. 1e6, seven decades, of which the lowest is spent buying a
#  smooth low end (see the picket-fence note above) and the measured range is
#  1 .. 3.3e4.
#
#  ONE CONSEQUENCE TO KNOW ABOUT.  Masses are multiples of 0.1, which binary
#  doubles cannot represent exactly, so the mass-conservation sum accumulates
#  round-off of about N * eps = 1e9 * 2.2e-16 = 2e-7 over a billion events.
#  That is measured, not guessed -- a run here reported 2.9e-7 on all eight
#  replicas, agreeing to three digits, which is the signature of round-off and
#  not of a bug -- and physically it is a mass leak of three ten-thousandths of
#  a per cent, far below anything being measured.
#
#  What it does cost is the sharpness of the conservation check: with integer
#  masses that check returns a hard zero and a real leak shows instantly, while
#  here it sits on a floor of 1e-7 and a genuine leak below 1e-6 would hide in
#  it.  The check is graded accordingly further down rather than being either
#  ignored or allowed to kill a run.
#  Both are module-level constants on purpose.  The worker processes are
#  spawned, not forked, so they re-import this file and read these values
#  directly; threading them through as arguments would add a way for the
#  parent and the workers to disagree about the box.  To change the range,
#  edit the two numbers here.
#  THE BOTTOM OF THE BOX IS NOT THE SAME FOR THE TWO PROCESSES, and the
#  asymmetry is bought, not sloppy.
#
#  COAGULATION is driven at the bottom: it injects a million bodies of mass
#  m_inj and they merge upward.  Its problem there is the picket fence --
#  every mass is an integer multiple of m_inj, so a 0.1 dex bin is smooth only
#  above 3.9 m_inj.  Injecting at 0.1 puts the fence below 0.39 and buys a
#  clean low end for one extra decade of range, which costs nothing: the
#  injections are astronomically many either way.
#
#  FRAGMENTATION is driven at the TOP, and there the same extra decade is
#  ruinous.  One injected body of mass m_inj must be cut into m_inj/m_sink
#  pieces before it is gone, so it costs that many events:
#
#        sink 1.0  ->  1e6 events per injected body
#        sink 0.1  ->  1e7 events per injected body
#
#  and the number of INDEPENDENT injected cascades per event is m_sink/m_inj,
#  which does not depend on the injection rate at all -- raising q buys
#  proportionally more events and a proportionally larger population, and the
#  same statistics.  A two-hour run at sink 0.1 delivered fewer than THREE
#  injected bodies per replica: the replicas' populations then differed by a
#  factor of 3.5, the mean generation climbed all pass, and nothing settled.
#  The same two hours at sink 1.0 deliver of order two hundred, and that box
#  had already produced Gamma = -2.0063 with a 3.5 decade plateau.
#
#  There is no fence to fear at the fragmentation end: fragments have
#  continuous masses.  So the decade is spent where it pays and not where it
#  does not, and the measured window below is common to both.
M_FLOOR_COAG = 0.1               # coagulation injects here
M_FLOOR_FRAG = 1.0               # fragmentation sinks here
M_TOP = 1.0e6                    # the top, shared

#  A COMMON WINDOW FOR BOTH PANELS, measured rather than chosen: 1.5 decades
#  above a fragmentation sink, 2 decades below a coagulation wall.  Not used by
#  the runner -- see core_window -- and kept here because it is where the
#  numbers live once the figure needs them.
CORE_LO, CORE_HI = 30.0, 1.0e4


def geometry(process):
    """Injection and wall, and the direction the cascade runs in."""
    if process == "coagulation":
        return dict(m_inj=M_FLOOR_COAG, m_sink=M_TOP)
    return dict(m_inj=M_TOP, m_sink=M_FLOOR_FRAG)


def core_window(g):
    """The window every number is quoted on: one decade above the lower scale
    of THIS box, up to 1.5 decades below the upper one.

    Per-box, as it has always been.  A single window common to both processes
    is the right thing for the paper -- the two panels are only comparable if
    they are quoted on the same stretch of mass -- and the boundary layers that
    set it have been measured (1.5 decades at a fragmentation sink, 2 decades
    at a coagulation wall, which would give 30 .. 1e4).  That is a decision for
    the figure, not for the runner, and it is deliberately not taken here: the
    runner reports each box on its own terms and the common window is imposed
    at analysis time, where it can be changed without re-running anything.

    Fixed before any data is looked at, and shaded on both panels of the
    figure, so a quoted drift cannot be the product of choosing where to
    measure it.
    """
    lo = min(g["m_inj"], g["m_sink"])
    hi = max(g["m_inj"], g["m_sink"])
    return lo * 10.0, hi / 30.0


def _q_of(BF, g, process):
    """The injection rate.  It is NOT the same formula for the two processes,
    and treating it as if it were is what wrecked a night of fragmentation.

    COAGULATION.  Bodies pile up at the injection scale, which is the bottom of
    the box, so N_SS is directly the live population and the old estimate --
    inject at the rate at which the population at m_inj consumes itself -- holds
    the population near N_SS.  Unchanged.

    FRAGMENTATION.  Everything is mirrored.  Bodies pile up at the SINK, and a
    single injected body of mass m_inj becomes m_inj/m_sink bodies of sink size
    before it leaves.  With a spectrum n = A m^-2 between the two scales,

        N = Int A m^-2 dm = A (1/m_sink - 1/m_inj) ~ A / m_sink
        M = Int A m^-1 dm = A ln(m_inj/m_sink)

    so N ~ M / (m_sink ln R) -- the population is set by the mass divided by the
    SINK scale, not by the injection scale.  Putting N_SS bodies of mass m_inj
    in the box therefore asks for

        N = N_SS (m_inj/m_sink) / ln R

    live bodies, which for N_SS = 1e6 over six decades is 7e10.  The engine
    duly tried to build it: eight processes reached 7.5 GB each, the machine ran
    out of memory, and the run spent four hours paging instead of computing.

    The rate below is only a STARTING guess; calibrate_q measures the real one.
    It comes from mass balance: to hold M in the box the injector must supply
    what the sink removes, and one injected body supplies m_inj of mass.
    """
    K = _kernel(BF)
    if process == "coagulation":
        return 0.5 * K(g["m_inj"], g["m_inj"]) * N_SS ** 2
    #  Seeded mass for N_SS live bodies, and a rate that would replace it once
    #  per unit of the seed's own collision time.  Deliberately rough; it is
    #  refined by measurement.  The mass is box_mass, i.e. the p = 11/6
    #  spectrum, not the p = 2 one the additive runner assumed.
    M_box = box_mass(g)
    return 0.5 * K(g["m_sink"], g["m_sink"]) * N_SS ** 2 \
        * (M_box / (N_SS * g["m_inj"]))


def _tc_of(BF, g, process):
    """The one-particle collision time at the populated end of the cascade.

    Every age in this file is quoted in units of it, so the age grid does not
    change meaning when the box or the population does.  The rate seen by one
    body among N of its own size is K(m,m) N, hence t_c = 1 / (K(m,m) N_ss),
    with K the kernel this file was built for -- here the geometric one.
    """
    m = g["m_inj"] if process == "coagulation" else g["m_sink"]
    return 1.0 / (_kernel(BF)(m, m) * N_SS)


def iso_age_grid(BF, g, process):
    """The linear age grid, sized from the box rather than guessed.

    A body has to climb from m_inj to m_sink before it leaves, and the drift is
    a POWER LAW here, dm/dtau ~ D m^beta with beta = 5/6, so

        tau_res = b t_c (R^(1-beta) - 1),      1 - beta = 1/b = 1/6,

    which is 82 t_c over the seven decades of the coagulation box against the
    13.8 t_c that t_c ln R would have given.  Using the logarithm here -- the
    exponential-growth answer inherited from lambda = 1 -- would put five sixths
    of the population past the end of the grid, where the histogram drops it
    without saying so.  The grid covers ISO_SPAN times tau_res, and the fraction
    that lands outside anyway is measured and printed per pass; if it is not
    small, this number is wrong.
    """
    tc = _tc_of(BF, g, process)
    R = max(g["m_inj"], g["m_sink"]) / min(g["m_inj"], g["m_sink"])
    t_max = ISO_SPAN * B_TH * (R ** (1.0 - BETA_TH) - 1.0) * tc
    n = max(int(round(t_max / (ISO_DT * tc))), 8)
    return np.arange(n + 1, dtype=float) * (ISO_DT * tc)


def frag_n0(g):
    """How many bodies a COLD fragmentation start needs at the injection scale.

    Not N_SS.  N_SS is the live population wanted, and in fragmentation the
    live population sits at the SINK: for dN/dm ~ A m^-p with 1 < p < 2 the
    count is set by the light end and the mass by the heavy one, so
    N = A m_sink^(1-p)/(p-1) and M = A m_inj^(2-p)/(2-p).  Inverting for the
    mass and dividing by the mass of one injected body,

        N0 = M / m_inj = N_SS (p-1)/(2-p) (m_sink/m_inj)^(p-1)

    which at p = 11/6, for a million bodies over six decades, is FIFTY.  (The
    additive runner carried the p -> 2 form of the same line and got fourteen.)
    Putting N_SS bodies there instead asks the box for orders of magnitude more
    live particles; the engine tried exactly that once, took 7.5 GB per replica,
    and spent four hours paging.  The whole failure is this one line.

    Fifty is a cheap start, not an awkward one: the population climbs to the
    target at one particle per event, so a million events -- seconds -- and the
    box is full.
    """
    return max(int(round(box_mass(g) / max(g["m_inj"], g["m_sink"]))), 2)


def frag_ic(g, alpha=ALPHA_TH):
    """Fragmentation starts from a SEEDED power law, not from a delta.

    Two reasons, and the second is the important one.

    Cheap: the delta at m_inj has to walk twenty decades of halvings before the
    first fragment reaches the sink, and every event on the way adds a particle
    with nothing removing any -- that walk is exactly the population explosion.

    Honest: seeding the answer and finding the answer proves nothing, so the
    seed is deliberately wrong.  --seed-alpha sets it, and the pair to run is
    one above and one below -11/6: if both relax onto -11/6 the result is a
    measurement, if only the one seeded there stays it is an artefact.
    Note [12] of the engine says the same thing about the warm start generally.
    """
    return {"steady": {"alpha": float(alpha), "m_lo": min(g["m_sink"], g["m_inj"]),
                       "m_hi": max(g["m_sink"], g["m_inj"]), "N": int(N_SS)}}


def _sim_kwargs(BF, g, f, process, ic, max_events, stride, seed, verbose=False,
                q=None):
    #  ONE replica talks and the rest are silent.  Eight verbose workers
    #  interleaving is unreadable, and one is enough -- they all run the same
    #  box for the same number of events.
    #
    #  This is not decoration.  Without it the file prints NOTHING between
    #  checkpoints, and an eight-hour pass is eight hours of silence: a
    #  fragmentation run once spent four hours at two thirds speed with the
    #  first checkpoint still out of reach, and there was no way to see it.
    #  The line to watch is the event counter against the pass budget, and
    #  m_min, which is the leading edge of a fragmentation cascade.
    if q is None:
        q = _q_of(BF, g, process)
    kw = dict(
        process=process, system="open", kernel=_kernel(BF),
        edges=mass_edges(g),
        ic=ic, injection_rate=q, injection_mass=g["m_inj"],
        sink_mass=g["m_sink"],
        snapshot_mode="events", snapshot_stride=stride, max_events=int(max_events),
        #  Generations stay on ONLY because a state saved without them cannot be
        #  continued -- the engine refuses, final_gen absent.  The waiting law
        #  stays off because it is the expensive tracker -- but unlike at
        #  lambda = 1, where b was infinite and there was nothing to measure,
        #  here it WOULD measure something.  Turn it on deliberately, and for
        #  FRAGMENTATION only: for coagulation off a spectrum steeper than m^-1
        #  it reads the injection scale rather than the drift and returns 1/b of
        #  the wrong sign (engine note [16]).
        track_generations=True, gen_max=GEN_MAX, track_waiting=False,
        rng=np.random.default_rng(seed), verbose=bool(verbose),
        age_rule="mass_weighted")
    if process == "coagulation":
        kw["coag_min_ratio"] = float(f)
        #  ISOCHRONES, coagulation only.  In fragmentation `gen` is what carries
        #  the cascade and the age of a fragment is a matter of which piece
        #  inherits the clock; here the age is unambiguous -- time since the
        #  body's own material entered -- and it is the growth law itself.
        kw["iso_age_edges"] = iso_age_grid(BF, g, process)
        kw["iso_start_sink"] = ISO_START_SINK
    else:
        kw["frag_min_ratio"] = float(f)
    return kw


def _gen_of(r):
    """The generation histogram of one pass, in RAW counts.

    The mirror of _iso_of, and it exists for the same reason: in fragmentation
    it is `gen` that carries the cascade -- the number of splittings a piece has
    behind it -- while an age depends on which fragment inherits the clock.  The
    counts accumulate over snapshots exactly as the isochrones do, so passes add.

    gen_tau is a MEAN, not a count, so it is carried as its two parts: the sum
    over particles and the number of them.  Accumulating the ratio instead would
    weight a short pass equally with a long one.
    """
    if "gen_counts" not in r:
        return None
    w = float(r.get("meta", {}).get("weight", 1.0)) or 1.0
    tau = np.asarray(r["gen_tau"], float)
    num = np.asarray(r["gen_num"], float)
    return dict(counts=np.asarray(r["gen_counts"], float) / w,
                num=num, tau_sum=np.nan_to_num(tau, nan=0.0) * num,
                snaps=float(r["gen_snapshots"]))


def _snap_weighted(dts, snaps):
    """Mean snapshot spacing over passes, weighted by snapshots contributed.

    Passes are not equal in length -- a checkpoint can land anywhere -- so the
    spacing of one of them is not the spacing of the accumulated histogram.
    Weighting by snapshot count gives the spacing the accumulated counts were
    actually taken at.  Entries that are not finite (a pass with a single
    snapshot has no spacing) are dropped rather than poisoning the mean.
    """
    d = np.asarray(dts, float)
    w = np.asarray(snaps, float)
    ok = np.isfinite(d) & np.isfinite(w) & (w > 0) & (d > 0)
    if not np.any(ok):
        return np.nan
    return float(np.sum(d[ok] * w[ok]) / np.sum(w[ok]))


def _iso_of(r):
    """The isochrone histogram of one pass, in RAW counts.

    The engine stores counts multiplied by the particle weight w.  Dividing it
    out here means passes add whatever w was, and it means ISO_MIN_COUNTS is a
    statement about how many particles a drawn isochrone rests on rather than
    about a rescaled number that happens to look large.
    """
    if "iso_counts" not in r:
        return None
    w = float(r.get("meta", {}).get("weight", 1.0)) or 1.0
    t = np.asarray(r["t"], float)
    dt = float(np.median(np.diff(t))) if t.size > 1 else np.nan
    return dict(counts=np.asarray(r["iso_counts"], float) / w,
                edges=np.asarray(r["iso_age_edges"], float),
                snaps=float(r["iso_snapshots"]), dt_snap=dt)


def _iso_overflow(r, iso):
    """The fraction of the population that fell OUTSIDE the age grid.

    The histogram drops anything older than its last edge without saying so, so
    the count it holds is compared against the count it should hold: the live
    population summed over exactly the snapshots the isochrones were taken on.
    Returns a fraction, or nan if it cannot be established.
    """
    try:
        live = np.asarray(r["live"], float)
        n = int(iso["snaps"])
        if n <= 0 or live.size < n:
            return np.nan
        expect = float(live[-n:].sum())
        return float(1.0 - iso["counts"].sum() / max(expect, 1e-300))
    except Exception:                                    # noqa: BLE001
        return np.nan


def _save_atomic(AN, run, path, drop_particles, analysis=None):
    """np.savez_compressed writes in place and takes tens of seconds on a large
    state; a cut inside that window destroys the new file AND the old one, which
    is the one case a checkpoint exists to prevent.  Temporary name, then
    os.replace, which is atomic."""
    d = os.path.dirname(path) or "."
    base = os.path.basename(path).replace(".npz", "")
    tmp = AN.add_last_run(run, base + ".__writing__", drop_particles=drop_particles,
                          runs_dir=d, analysis=analysis)
    os.replace(tmp, path)
    return path


def calibrate_q(process, f, rounds=3, burst=15_000_000, seed=31337):
    """Find the injection rate that HOLDS the population, by measuring it.

    No formula survives contact with this: the steady population depends on the
    drift, the drift on the kernel and on the locality window, and the whole
    point of the window is that it changes the cascade.  So the rate is not
    derived, it is tuned -- three or four short bursts, single process, a couple
    of minutes.

    THE LOOP MEASURES WHAT THE RUN WILL ACTUALLY DO.
    It started from a SEEDED spectrum and watched 1.5e6 events; over that
    stretch the population is dominated by the seed rearranging itself, and
    injection barely enters.  The evidence is unambiguous: four rounds moved q
    by a factor of 2.8 and the answer moved by 0.2 per cent -- 1.4099, 1.4088,
    1.4113, 1.4112 million -- which is a loop reading its own initial
    condition.  The run then settled at 2.37e6 bodies against a target of 1e6.

    So v2 starts each burst COLD, from the same fourteen bodies the real run
    starts from, and runs 1.5e7 events, long enough for the population to
    saturate rather than merely to move.  It costs minutes before a run of
    hours.  The population is read as the mean over the LAST HALF of the burst,
    not the final value, because it oscillates by a few per cent about its
    plateau.

    The loop is proportional control on one number.  Start the box cold, run a
    burst, and look at where the live count settled:

        grew   -> the injector is feeding faster than the sink drains: lower q
        shrank -> the reverse: raise q

    Each round scales q by (target/observed), which is exact if N were linear in
    q and close enough that it converges in three.  It stops early once the
    population sits within a few per cent of the target.

    Returns the rate, and the caller stores it so every replica and every later
    pass uses the same one -- a rate that drifted between replicas would make
    them different experiments.
    """
    BF, _ = _imports()
    g = geometry(process)
    if process == "coagulation":
        return _q_of(BF, g, process), []          # the old estimate is right there
    q = _q_of(BF, g, process)
    hist = []
    for it in range(rounds):
        r = BF.simulate(**_sim_kwargs(BF, g, f, process,
                                      {"m": g["m_inj"], "N": frag_n0(g)},
                                      burst, max(burst // 60, 1), seed + it,
                                      q=q))
        lv = np.asarray(r["live"], float)
        n0 = float(lv[0])
        n1 = float(np.mean(lv[lv.size // 2:]))   # the plateau, not the last point
        ratio = n1 / max(N_SS, 1.0)
        hist.append((q, n0, n1, float(r["n_out"][-1])))
        if 0.95 < ratio < 1.05:
            break
        #  Guard the step: a burst that emptied or exploded the box gives a
        #  ratio that is meaningless as a linear correction, so the move is
        #  capped at a factor of thirty either way.
        q = q / min(max(ratio, 1.0 / 30.0), 30.0)
    return q, hist


def probe(job):
    """Rate measured with every worker running, so it is the rate under load."""
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
    except Exception:                                   # noqa: BLE001
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
        #  COLD, the same shape of start as coagulation: a delta at the
        #  injection scale.  Only the COUNT differs, and it has to -- see
        #  frag_n0.
        ic = {"m": g["m_inj"], "N": frag_n0(g)}
    if (not cfg["fresh"]) and os.path.exists(state_file):
        prev = AN.load(state_file)
        ic = {"state": BF.state_of(prev, source=state_file)}
        done = int(prev["meta"].get("analysis", {}).get("events_done", 0))
        del prev

    #  ADDITIVE OR ABSOLUTE.  With --add-hours the budget is measured from
    #  where this replica already is, so the same command can be repeated as
    #  often as one likes and each run adds another stretch.  Without it the
    #  number is the total a replica will have done when it stops, and a replica
    #  that already holds more simply returns.
    total = (done + int(cfg["events_total"]) if cfg.get("additive")
             else int(cfg["events_total"]))
    per_pass = max(int(cfg["events_pass"]), 1)
    log, npass = [], 0
    while done < total:
        budget = int(min(per_pass, total - done))
        #  NO STUB PASSES.  Integer division of the budget leaves a remainder,
        #  and a remainder of ONE EVENT started a fifth pass whose three
        #  snapshots then overwrote n_out, live, the half-vs-half test and the
        #  snapshot spacing with values measured over no time at all: the run
        #  reported "absorptions 0", "half-vs-half nan", and a snapshot spacing
        #  of 8.5e-12 where the true one is 2.7e-6.  A pass worth less than a
        #  twentieth of the interval is not worth its own checkpoint.
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
        #  MASS CONSERVATION, checked but not weaponised.
        #
        #  This used to raise on anything above 1e-10, and that threshold threw
        #  away eight replicas after an hour of work because a change of mass
        #  UNITS had pushed round-off to 3e-7.  A check that destroys good data
        #  over the last digits of a float is worse than no check.
        #
        #  Two levels now.  Above 1e-4 the bookkeeping is genuinely broken and
        #  nothing downstream is worth keeping, so it stops.  Between the two it
        #  is reported once and the pass is written: the number is on the record
        #  and the decision is left to a person.
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

        #  IS THE MIDDLE STILL MOVING?  The absorption count cannot answer that
        #  in a range this wide, so it is asked directly: the snapshots of this
        #  pass are split in half and the local slope is rebuilt from each half.
        #  A drift between the halves is the box still relaxing; agreement is
        #  stationarity in the only place it is being measured.
        drift_half = drift_half_max = drift_half_at = np.nan
        D = np.asarray(r["dndm"], float)
        if D.ndim == 2 and D.shape[0] >= 8:
            cw = core_window(g)
            _, G1 = AN.local_slope(c, D[: D.shape[0] // 2].mean(axis=0))
            _, G2 = AN.local_slope(c, D[D.shape[0] // 2:].mean(axis=0))
            kk = (mg >= cw[0]) & (mg <= cw[1]) & np.isfinite(G1) & np.isfinite(G2)
            if kk.any():
                #  The MEDIAN, not the maximum.  A max over forty-odd bins is
                #  set by the single worst one, which is always the top of the
                #  window where the counts are thinnest -- it once read 0.61
                #  against a scatter of 0.08 and looked like a verdict, while
                #  the median of the same numbers was zero and every bin below
                #  m = 1000 agreed to 0.005.  The max is still recorded, and so
                #  is the mass it sits at, because "where" is the whole point:
                #  a disagreement confined to the last decade is the wall still
                #  filling, not the middle still moving.
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
        # ------------------------------------------------------------------
        #  ACCUMULATE ACROSS PASSES.
        #
        #  simulate() builds its histogram from scratch every call, and every
        #  pass is a fresh call starting from the saved state.  v1 wrote the
        #  histogram of the CURRENT pass and called it the run's spectrum, so a
        #  two-pass run reported "1.8e8 events" above a spectrum that stood on
        #  9e7 -- half the computation thrown away, and the fraction thrown away
        #  growing with every extra hour.
        #
        #  The fix is arithmetic, not cleverness.  F is a density, dN/dm,
        #  averaged over the snapshots of one pass; passes of equal length carry
        #  equal weight, unequal ones carry weight in proportion to their
        #  events.  So every pass is kept and the accumulated spectrum is the
        #  weighted mean.  Keeping the passes SEPARATELY rather than only their
        #  running sum costs 160 floats each and buys the ability to drop the
        #  first few afterwards, when it turns out the box was still relaxing --
        #  see --skip-passes, which needs no recomputation at all.
        spec_path = os.path.join(outdir, tag + "_spec.npz")
        F_passes, ev_passes, no_passes = [], [], []
        #  --fresh discards what is on disk ONCE, at the start of the
        #  invocation, and not at every pass inside it.  Testing cfg["fresh"]
        #  alone made a --fresh run of several passes throw away each pass as
        #  the next one finished, which is the very bug this accumulation
        #  exists to fix, reintroduced through the back door.
        if os.path.exists(spec_path) and not (cfg.get("fresh") and npass == 1):
            try:
                old = np.load(spec_path)
                if "F_passes" in old:
                    F_passes = [np.asarray(x, float) for x in old["F_passes"]]
                    ev_passes = [float(x) for x in old["ev_passes"]]
                    no_passes = ([float(x) for x in old["no_passes"]]
                                 if "no_passes" in old
                                 else [0.0] * len(ev_passes))
            except Exception:                            # noqa: BLE001
                #  A torn or v1-format file is not worth failing over: start the
                #  accumulation again rather than lose the pass just computed.
                F_passes, ev_passes, no_passes = [], [], []
        F_passes.append(np.asarray(F, float))
        ev_passes.append(float(r["events"][-1]))
        #  ABSORPTIONS ARE A COUNT AND THEY ACCUMULATE.  n_out is reset by
        #  every simulate() call, so writing the last pass's value under-
        #  reported the run by however many passes it had.
        no_passes.append(float(r["n_out"][-1]))
        Fp = np.asarray(F_passes, float)
        wp = np.asarray(ev_passes, float)
        F_acc = (Fp * wp[:, None]).sum(axis=0) / max(wp.sum(), 1e-300)
        mg_acc, G_acc = AN.local_slope(c, F_acc)

        #  ------------------------------------------------------------------
        #  THE ISOCHRONES ACCUMULATE TOO, and they do it more simply than the
        #  spectrum: iso_counts is a plain SUM of per-snapshot histograms, not a
        #  mean, so passes add with no weights at all.  The engine's own note on
        #  merging age bins says the same thing about age resolution -- summing
        #  is exactly the histogram a coarser grid would have produced -- and
        #  summing over passes is that identity applied to time instead.
        #
        #  Ages survive the seam.  On a restart every inherited timestamp is
        #  shifted back by the previous leg's end time, so a body that entered
        #  before the checkpoint carries a NEGATIVE inj_time and tau = t - t_inj
        #  returns its full age rather than restarting the clock.  Without that
        #  the accumulated isochrones would be a mixture of true ages and ages
        #  truncated at each checkpoint, which is worse than no accumulation.
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
                except Exception:                        # noqa: BLE001
                    iso_stack, iso_snap_list, iso_dt_list = [], [], []
            iso_stack.append(iso_new["counts"])
            iso_snap_list.append(iso_new["snaps"])
            iso_dt_list.append(float(iso_new["dt_snap"]))
            iso_sum = np.asarray(iso_stack, float).sum(axis=0)
            iso_snaps = float(np.sum(iso_snap_list))
            #  THE SNAPSHOT SPACING BELONGS TO ALL THE PASSES, NOT THE LAST.
            #  It is what converts a count of snapshot appearances into a count
            #  of independent bodies, so taking it from whichever pass happened
            #  to finish last is how a one-event stub once wrote 8.5e-12 in
            #  place of 2.7e-6 and inflated the statistics a million-fold.
            #  Passes differ in length, so the passes are averaged with the
            #  weight they earned: the number of snapshots each contributed.
            iso_dt_mean = _snap_weighted(iso_dt_list, iso_snap_list)
            #  DOES THE AGE GRID COVER THE POPULATION?  Everything older than
            #  the top edge is silently dropped by the histogram, so the
            #  fraction lost is measured against the live count rather than
            #  assumed to be zero.  A grid that is too short shows up here as a
            #  per cent that does not fall.
            iso_over = _iso_overflow(r, iso_new)

        #  GENERATIONS, the fragmentation half of the same idea, accumulated the
        #  same way.  Only the running sums are kept here: the generation index
        #  is already a coarse, integer label, so there is nothing to re-bin
        #  afterwards and therefore no reason to store the passes apart.
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
                except Exception:                        # noqa: BLE001
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
        #  The per-replica figure shows the ACCUMULATED spectrum, so what is on
        #  screen after each pass is what the stack will use.
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
    ax[1].axhline(-2.0, ls=":", lw=1.0, color="0.5",
                  label=r"$-2$ (the failure mode)")
    ax[1].set_ylim(ALPHA_TH - 0.45, ALPHA_TH + 0.45)
    ax[1].set_ylabel(r"$\Gamma=d\log F/d\log m$")
    ax[1].legend(fontsize=7)
    ax[1].set_title("(b) local slope", fontsize=10)
    fig.suptitle("%s | %.3g of %.3g events | live %d | absorptions %.4g"
                 % (tag, done, total, live, n_out), fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ============================================================================
#  ISOCHRONES
# ============================================================================
def iso_load(paths, skip=0):
    """Stack the accumulated age histograms of every replica.

    Two sums and nothing else.  iso_counts holds raw particle counts summed over
    snapshots, so replicas add exactly as passes do -- there is no weighting to
    get wrong here, which is the one place in this file where that is true.

    Replicas whose age grid differs are refused rather than interpolated onto
    the first one: a grid mismatch means the two were run under different
    settings, and silently stacking them would produce a curve belonging to
    neither.  The count of what was refused is returned.
    """
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
    #  Replicas are combined the same way passes are, by the snapshots each
    #  brought, not by a median that would let one short replica speak as
    #  loudly as one long one.
    return dict(edges=edges, counts=C, snaps=snaps, n=used, refused=refused,
                dt_snap=_snap_weighted(dt_snap, dt_w),
                edges_m=edges_m)


def gen_load(paths):
    """Stack the accumulated generation histograms of every replica.

    The fragmentation mirror of iso_load.  gen_counts and gen_num add; gen_tau
    is a mean and is rebuilt from its two parts, so a replica that ran longer
    carries the weight it earned rather than one vote like any other.
    """
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
    """AUTOMATIC AGE BINNING: merge the fine age bins until each isochrone is a
    step of the growth law rather than a slice of the clock.

    THE RULE.  Walk the fine grid from the young end, accumulating bins into a
    group, and close the group when BOTH hold:

        its mean mass has advanced at least `dlogm` decades beyond the mean mass
        of the group before it, and

        it holds at least `min_counts` particles.

    THE BOUNDARIES ARE PLACED GLOBALLY, not greedily.  <m> is first measured on
    the FINE grid, which costs nothing since it is already on disk; the group
    edges are then the ages at which that curve crosses m_start * 10^(k dlogm).
    A greedy walk that closed a group the moment its running mean had advanced
    far enough was tried first and is wrong: the running mean of a half-filled
    group is not the mean of the finished one, so groups closed at accidental
    moments and came out with counts differing by a factor of a hundred and age
    spans differing by thirty.  Placing the crossings first and cutting there
    gives edges that depend on the growth law and on nothing else.

    So the isochrones come out evenly spaced along the axis they are drawn on,
    which is the mass axis, and never along the clock -- the clock is the
    parameter, not the observable.  Age resolution is spent only where the mass
    was not moving, which is precisely where it was worthless: under a
    logarithmic grid fourteen of thirty-nine bins were identical curves sitting
    on the injection mass, and here they collapse into the first group.

    WHY MERGING IS ALLOWED AT ALL.  The engine accumulates counts, and counts
    add, so a merged group is exactly the histogram a coarser age grid would
    have produced.  Nothing is smoothed and nothing is modelled -- the same
    particles are resampled in age.  That is why the fine grid is run in the
    first place: it can always be coarsened afterwards and never refined.

    WHAT THE SPACING THEN MEASURES.  With the separation in log-mass held fixed,
    the age STEP between consecutive isochrones is the quantity left free.  If
    the growth is exponential, m = m_inj exp(tau/t_c), that step is a constant;
    if it is a power law it is not.  The step is returned and printed for
    exactly that reason -- the binning rule turns the growth law into something
    the eye can check on a column of numbers.

    m_floor drops mass bins below it before anything is measured.  In
    coagulation nothing born in the box can be lighter than the injection, so
    anything there is inherited from an earlier box with a different injection
    scale and is not part of this cascade at all.
    """
    C = np.asarray(C, float)
    m = np.asarray(centers, float)
    w = np.diff(edges_m)[:m.size] if edges_m is not None else np.gradient(m)
    use = np.ones(m.size, bool) if m_floor is None else (m >= float(m_floor))
    dropped = float(C[:, ~use].sum()) / max(float(C.sum()), 1e-300)
    Cu, mu, wu = C[:, use], m[use], w[use]

    #  STEP 1: the growth law on the fine grid.  Ages holding nothing above the
    #  mass floor are not part of the cascade being measured and are skipped
    #  outright -- under the old logarithmic grid they were the dozens of
    #  identical curves sitting on the injection mass.
    s_fine = Cu.sum(axis=1)
    ok = np.flatnonzero(s_fine > 0)
    if ok.size < 2:
        return None
    mb_fine = np.full(Cu.shape[0], np.nan)
    mb_fine[ok] = (Cu[ok] * mu[None, :]).sum(axis=1) / s_fine[ok]

    #  STEP 2: cut where <m> crosses a ladder of fixed steps in log mass.  The
    #  running maximum is used for the crossing test only: <m>(tau) is monotone
    #  in the physics and a dip in it is counting noise, which must not be
    #  allowed to open a spurious boundary.  The masses reported afterwards are
    #  the true group means, not the monotonised ones.
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

    #  STEP 3: the count floor.  Groups thinner than min_counts are absorbed
    #  into the next one, so the bins widen exactly where the cascade has run
    #  out of particles -- the top of the range -- and nowhere else.  A starved
    #  tail folds backwards instead, because there is no next one.
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
    #  THE REPEAT CORRECTION.  The histogram counts APPEARANCES, not particles:
    #  a body sitting inside one age bin is counted once per snapshot it spends
    #  there.  Those counts are not independent, so the error bar has to be
    #  widened by the square root of how many times the same body was seen --
    #  age-bin width over snapshot spacing, never below one.
    n_rep = np.ones_like(S0)
    if np.isfinite(dt_snap) and dt_snap > 0:
        n_rep = np.clip((tau_hi - tau_lo) / dt_snap, 1.0, None)
    sigma = np.sqrt(spread2 * n_rep) / np.maximum(S0, 1e-300)
    return dict(rows=R, centers=mu, widths=wu, counts=S0, mbar=mbar,
                sigma=sigma, tau_lo=tau_lo, tau_hi=tau_hi,
                tau=0.5 * (tau_lo + tau_hi), n_rep=n_rep, dropped=dropped,
                n_fine=int(C.shape[0]))


def iso_growth_fit(tau, mbar, sigma=None, tc=1.0):
    """Fit log <m> against log tau -- a STRAIGHT LINE on log-log axes.

    At lambda = 2/3 the growth is a power law, m ~ tau^b with b = 6, so this is
    the natural fit and its slope IS b.  An exponential would curve here, and
    the residual is reported so that "it looked straight" is not the argument.
    Read the slope against the age-step column, which measures the same thing
    without a fit: at fixed dlog m, a constant dlog tau is a power law.
    """
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
    #  A LINEAR colour scale, because the age axis is linear now.  Under the old
    #  logarithmic grid a log scale was the only readable choice, and it hid the
    #  fact that most of the colour range was spent before anything moved.
    norm = Normalize(vmin=float(tau.min()), vmax=float(tau.max()))
    cmap = plt.get_cmap("viridis")

    #  (a) the isochrones in the SAME units as the stationary spectrum: counts
    #      divided by the snapshots they were summed over.  That is not
    #      cosmetic.  Every live body sits in exactly one age bin, so the
    #      isochrones must ADD UP to the stationary spectrum, and the dashed sum
    #      drawn on top of the grey curve is that identity being checked rather
    #      than asserted.  Any gap between them is population outside the age
    #      grid or below the mass floor, both of which are reported as numbers.
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

    #  (b) THE GROWTH LAW ON LOG-LINEAR AXES.  Exponential growth is a straight
    #      line here and nothing else is, which is the whole reason for the
    #      choice: on log-log a curve bending gently reads as a power law with
    #      an inconvenient exponent, and at lambda = 1 there is no exponent to
    #      find -- b is infinite.
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
    ax[1].set_title("(b) growth law: a straight line here is a power law",
                    fontsize=10)
    fig.suptitle(r"coagulation, geometric kernel $\lambda=2/3$, locality $f=%.2f$"
                 r"   |   isochrones accumulated over %d replicas" % (f, n_rep_files),
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ============================================================================
#  THE STACK
# ============================================================================
def rebin(centers, F, k, edges=None):
    """Merge k adjacent mass bins.  Exact: counts add.

    F is a density, dN/dm, so it is turned back into counts with the bin width,
    summed, and divided by the merged width.  Bins are dropped from the END if
    the count does not divide, which is the empty tail of the grid.
    """
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
    #  The UN-rebinned spectrum is kept as well.  The isochrones live on the
    #  engine's own 0.1 dex mass grid, and drawing the stationary spectrum
    #  underneath them only means anything if the two share that grid.
    raw, raw_c = [], None
    centers = None
    for p in paths:
        d = np.load(p)
        #  DROPPING THE TRANSIENT, after the fact and for free.  Every
        #  pass is stored separately, so the accumulated spectrum can be
        #  rebuilt from pass `skip` onwards without recomputing anything.  A
        #  box that was still relaxing for its first hour does not have to
        #  contaminate the answer, and does not have to be re-run either.
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
        #  The per-replica slope is recomputed on the merged bins too, so the
        #  spread that becomes the error bar refers to the same quantity that
        #  is plotted rather than to the fine-binned one.
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
        #  Spectra are summed and divided by n -- every replica ran the same box
        #  for the same number of events, so the plain mean is the stacked
        #  estimator.  The local slope is taken FROM that mean, not averaged
        #  separately: a mean of ratios is not the ratio of the means, and the
        #  two panels have to be answerable to each other.
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
                #  The error on the spectrum is the spread BETWEEN replicas,
                #  divided by the root of their number -- the same rule as every
                #  other error here, and not the bin-to-bin scatter, which is a
                #  different quantity that happens to have the same units.
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
    """Drift of Gamma on the core window, with its error taken over replicas.

    The error on the DRIFT is what decides flatness, and it cannot be read off
    the error on <Gamma>: a curve can have a perfectly known mean and still be
    sloping.  Each replica is fitted on its own and the spread between the fits
    is the error, which is the same rule used for every other error here.
    """
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
        #  Tight limits: on a tall axis a factor-of-two sag reads as a straight
        #  line and a flatness test always passes.
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
    #  -2 is drawn as well, and faintly, because it is the ANSWER THIS RUN MUST
    #  NOT GIVE: engine notes [5iv] and [8] both drive the index there for every
    #  kernel.  At lambda = 1 the two lines coincided and the test did not
    #  exist; here they are 1/6 apart and it does.
    ax[1].axhline(-2.0, ls=":", lw=1.0, color="0.5",
                  label=r"$-2$ (the failure mode)")
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
        description="The local cascade with the GEOMETRIC kernel, lambda = "
                    "2/3: coagulation m = 0.1 .. 1e6, fragmentation "
                    "m = 1e6 .. 1, each quoted on its own core window. "
                    "Predicted alpha = -11/6; a run that returns -2 has "
                    "failed, it has not confirmed anything.")
    ap.add_argument("--process", choices=["coag", "frag"], required=True)
    ap.add_argument("--f", type=float, default=-1.0,
                    help="locality window m_small >= f m_large.  NEGATIVE means "
                         "the default for the process: 0 for coagulation, where "
                         "at lambda = 2/3 the flux integral converges on its own "
                         "(engine note [17]), and 0.30 for fragmentation, which "
                         "needs a disruption threshold at any lambda because "
                         "alpha < -1 (engine note [8]).  Set it explicitly to "
                         "test the dependence -- at this kernel alpha should NOT "
                         "move with f.")
    ap.add_argument("--nreps", type=int, default=8)
    ap.add_argument("--hours", type=float, default=8.0,
                    help="TOTAL wall hours per replica -- the amount each "
                         "replica will have done when it stops, NOT an amount "
                         "to add.  A replica that already holds more finishes "
                         "at once.  Use --add-hours to extend instead.")
    ap.add_argument("--add-hours", type=float, default=0.0,
                    help="ADD this many hours to whatever each replica has "
                         "already done, and stop there.  This is the flag for "
                         "'run another hour, then another': repeat the same "
                         "command as often as you like and each invocation "
                         "adds one more hour.  Overrides --hours.")
    ap.add_argument("--pass-hours", type=float, default=1.0)
    ap.add_argument("--events", type=float, default=0.0,
                    help="fix the per-replica TOTAL budget and skip the probe.")
    ap.add_argument("--add-events", type=float, default=0.0,
                    help="add this many events to what each replica has done.")
    ap.add_argument("--start", choices=["cold", "seed"], default="cold",
                    help="FRAGMENTATION only.  cold: a delta at the injection "
                         "scale, the same shape of start as coagulation, with "
                         "the body count set by frag_n0 -- fifty at this "
                         "kernel, not a million (fourteen was the p = 2 "
                         "answer of the additive runner).  seed: begin from a "
                         "ready power law instead, which skips the transient "
                         "and lets --seed-alpha probe the answer from a "
                         "deliberately wrong slope.")
    ap.add_argument("--seed-alpha", type=float, default=ALPHA_TH,
                    help="FRAGMENTATION only: the slope the box is seeded with. "
                         "Seeding -11/6 and finding -11/6 proves nothing, so "
                         "run the pair --seed-alpha -1.5 and --seed-alpha -2.1: "
                         "if both relax onto -11/6 it is a measurement, if only "
                         "the one seeded there stays it is an artefact.  Seeding "
                         "-2 is worth doing for its own sake: if the box CANNOT "
                         "leave -2, the index is being set by a boundary and not "
                         "by the kernel.  The value goes into the file names, so "
                         "the runs do not collide.")
    ap.add_argument("--events-per-pass", type=float, default=0.0,
                    help="checkpoint interval in EVENTS, set directly.  "
                         "Overrides the hours/pass-hours ratio, which is only "
                         "as good as the rate probe -- and the probe measures "
                         "the cold start, where the cascade sits on one mass "
                         "and is at its fastest.  If the real rate is three "
                         "times lower, the first 'hour' takes three, and "
                         "nothing is on disk until it does.")
    ap.add_argument("--skip-passes", type=int, default=0,
                    help="leave out the first k passes of every replica "
                         "when building the spectrum.  Costs nothing and needs "
                         "no re-running: the passes are stored separately.  Use "
                         "it when the half-vs-half number says the box was "
                         "still relaxing early on.")
    ap.add_argument("--rebin", type=int, default=REBIN_DEFAULT,
                    help="merge this many 0.1 dex bins into one before "
                         "measuring: %d gives %.1f dex bins and %d times the "
                         "counts, for nothing.  Applied at analysis time, so it "
                         "can be changed without re-running.  1 = raw."
                         % (REBIN_DEFAULT, 0.1 * REBIN_DEFAULT, REBIN_DEFAULT))
    ap.add_argument("--iso-dlogm", type=float, default=ISO_DLOGM,
                    help="COAGULATION.  How far apart, in decades of mean mass, "
                         "consecutive isochrones are placed.  The age bins are "
                         "merged automatically until each step is this wide, so "
                         "the curves come out evenly spaced along the mass axis "
                         "instead of along the clock.  Applied at analysis time "
                         "on a much finer stored grid, so it can be changed "
                         "without re-running anything.  Default %.2f."
                         % ISO_DLOGM)
    ap.add_argument("--iso-min-counts", type=float, default=ISO_MIN_COUNTS,
                    help="COAGULATION.  Smallest number of particles a drawn "
                         "isochrone may rest on.  Groups widen on their own "
                         "near the top of the cascade, where the counts are "
                         "thinnest, rather than being drawn as shot noise.")
    ap.add_argument("--iso-mmin", type=float, default=0.0,
                    help="COAGULATION.  Mass floor for the isochrones and the "
                         "growth law.  0 means the bottom of the core window, "
                         "which is where the picket fence ends and the "
                         "histogram becomes a continuum.  Everything below it "
                         "is still computed and still stored -- this only "
                         "decides what is measured and drawn.")
    ap.add_argument("--q", type=float, default=0.0,
                    help="FRAGMENTATION: set the injection rate directly and "
                         "skip the calibration.  The calibration tunes q by "
                         "watching the population over a short burst, and it "
                         "has now been wrong twice for the same reason: the "
                         "burst is shorter than the time the box needs to "
                         "settle, so it measures a transient and reports it as "
                         "steady.  Its own last output gave it away -- q fell "
                         "by a factor of nine while the 'settled' population "
                         "fell by two.  When you already know the rate, or "
                         "when you want to watch the live count yourself for "
                         "an hour and adjust, this is the honest way in.")
    ap.add_argument("--outdir", default="runs")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--stack-only", action="store_true")
    args = ap.parse_args()
    #  The locality window is not the same decision for the two processes here;
    #  see the help above.  A value given on the command line always wins.
    if args.f < 0.0:
        args.f = 0.0 if args.process == "coag" else 0.30

    process = "coagulation" if args.process == "coag" else "fragmentation"
    short = args.process
    BF, AN = _imports()
    g = geometry(process)
    cw = core_window(g)
    pr = BF.predict("open", BF.KERNEL_LAMBDA[KERNEL_NAME])
    #  The engine owns the predictions and this file owns the axes; if they ever
    #  disagree, the axes are wrong and every figure below is mislabelled.  Stop
    #  here rather than after eight hours.
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
    print("           Gamma = -2 here is a FAILURE, not a success: engine notes")
    print("           [5iv] and [8] both pin -2 for every kernel.")
    print("  box    : injection %g -> wall %g   (%.1f decades)  N_ss = %g"
          % (g["m_inj"], g["m_sink"],
             np.log10(max(g["m_inj"], g["m_sink"])
                      / min(g["m_inj"], g["m_sink"])), N_SS))
    print("  core   : %.4g .. %.4g  (%.1f decades) -- where every number is quoted"
          % (cw[0], cw[1], np.log10(cw[1] / cw[0])))
    print("  binning: %d x 0.1 dex merged -> %.1f dex, %dx the counts per bin"
          % (args.rebin, 0.1 * args.rebin, args.rebin))
    print("  note   : masses are integer multiples of m_inj, so a 0.1 dex bin")
    print("           holds one or two possible masses below m = %.3g and the"
          % (3.9 * min(g["m_inj"], g["m_sink"])))
    print("           histogram is a picket fence there.  Not noise; no number")
    print("           of particles smooths it.  The core window starts above it.")
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
        except Exception:                               # noqa: BLE001
            pass
        print("disk      : %.0f GB free" % (shutil.disk_usage(outdir).free / 1e9))

        #  CALIBRATION, before anything long is started.  For coagulation this
        #  returns the old estimate untouched.  For fragmentation it measures.
        if process == "fragmentation":
            print("calibrate : tuning the injection rate to hold %.3g bodies"
                  % N_SS)
            if args.q > 0:
                q_use, hist = float(args.q), []
                print("            given by hand: q = %.6g (calibration "
                      "skipped)" % q_use)
            else:
                q_use, hist = calibrate_q(process, args.f)
            for i, (qq, n0, n1, nout) in enumerate(hist):
                print("            round %d: q = %.4g -> live %d, settled at %d "
                      "(target %.3g), absorbed %.4g"
                      % (i + 1, qq, n0, n1, N_SS, nout))
            print("            chosen q = %.6g" % q_use)
            print("start     : %s"
                  % ("cold, %d bodies at m_inj = %.3g (NOT N_ss: see frag_n0)"
                     % (frag_n0(geometry(process)), geometry(process)["m_inj"])
                     if args.start == "cold"
                     else "seeded power law, alpha = %+.2f" % args.seed_alpha))
            if not hist:
                print("            !! calibration produced nothing -- check it")
        else:
            q_use = _q_of(BF, geometry(process), process)
            print("injection : q = %.6g (coagulation estimate, unchanged)" % q_use)

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
        if args.events_per_pass > 0:
            ev_pass = max(int(args.events_per_pass), 1)
            print("checkpoint: every %.4g events (given)\n" % ev_pass)
        else:
            _h = args.add_hours if args.add_hours > 0 else args.hours
            ev_pass = max(int(ev * args.pass_hours / max(_h, 1e-9)), 1)
            print("checkpoint: every %.4g events (about %.2f h IF the probe "
                  "was right)\n" % (ev_pass, args.pass_hours))

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
        args.rebin, args.skip_passes)
    if R is None:
        raise SystemExit("no results in %s" % outdir)
    png = os.path.join(outdir, "%s_f%.2f%s_stacked.png" % (short, args.f, _st))
    stacked_figure(R, g, cw, args.f, process, png)

    st = report(R, cw)
    print("\n" + "=" * 78)
    print("STACKED over %d replicas | %d passes each" % (R["n"], R["npass"]))
    #  TWO event counts, not one.  v1 printed only the first and stood a
    #  spectrum built from a fraction of it underneath -- the number flattered
    #  the result by however many passes had been run.  They are equal now, and
    #  printing both is what keeps them so.
    print("  events computed  : %.5g per replica" % R["events"])
    if not np.isfinite(R["ev_spec"]):
        #  A v1 file carries no such field, and that absence IS the diagnosis:
        #  v1 stored the last pass and nothing else, so the count behind the
        #  spectrum is one pass, whatever the header above says.
        print("  events IN the spectrum: UNKNOWN -- these files were written by")
        print("     v1, which kept only the LAST pass.  The spectrum therefore")
        print("     stands on one pass, not on the %.5g above.  One more pass"
              % R["events"])
        print("     under v2 starts the accumulation; the passes already")
        print("     computed under v1 cannot be recovered.")
    else:
        print("  events IN the spectrum: %.5g per replica%s"
              % (R["ev_spec"],
                 "   (first %d pass(es) skipped)" % args.skip_passes
                 if args.skip_passes else ""))
        if not args.skip_passes and R["ev_spec"] < 0.99 * R["events"]:
            print("  !! the spectrum stands on less than what was computed.")
    print("  live %.4g | absorptions %.4g" % (R["live"], R["n_out"]))
    #  EQUAL EVENT COUNTS ARE WHAT MAKES THE STACK HONEST.  Summing spectra of
    #  unequal length weights the longer replicas more, and the error bar taken
    #  over replicas then understates the truth.  They can fall out of step if
    #  one replica died and was restarted, or if an interrupt landed between
    #  passes for some and not others; --add-hours preserves such a gap, while
    #  --hours levels everyone to the same total.  So the gap is reported rather
    #  than assumed away.
    if R["ev_spread"] > 0.01 * max(R["events"], 1.0):
        print("  !! replicas are NOT in step: %.5g .. %.5g events (spread %.3g)"
              % (R["ev_all"].min(), R["ev_all"].max(), R["ev_spread"]))
        print("     the stack weights the longer ones more.  To level them, run")
        print("     once with  --hours H  (a TOTAL) instead of --add-hours.")
    elif R["ev_spread"] > 0:
        print("  replicas in step to %.2f%% -- close enough to stack."
              % (100.0 * R["ev_spread"] / max(R["events"], 1.0)))
    else:
        print("  replicas in step: every one did exactly the same events.")
    print("-" * 78)
    if st is None:
        print("too few bins in the core window to judge")
    else:
        print("  core window     : %.4g .. %.4g   (%.2f decades, %d bins)"
              % (cw[0], cw[1], np.log10(cw[1] / cw[0]), st["nbin"]))
        print("  <Gamma>         : %+.4f +- %.4f      (theory %+.4f)"
              % (st["mean"], st["se"], pr["alpha"]))
        print("  drift per decade: %+.4f +- %.4f      (%.1f sigma from zero)"
              % (st["drift"], st["drift_se"],
                 abs(st["drift"]) / max(st["drift_se"], 1e-12)))
        print("  scatter rms     :  %.4f" % st["rms"])
        print("  still moving?   :  half-vs-half |dGamma| in the core,")
        print("                     median over replicas   %.4f" % R["half"])
        print("                     worst single bin       %.4f  at m = %.3g"
              % (R["half_max"], R["half_at"]))
        print("                     Read the MEDIAN against the scatter above.")
        print("                     A worst bin sitting at the top of the window")
        print("                     is the wall still filling, not the middle.")
        print("  plateau reached : %s"
              % ("%.3g .. %.3g   = %.2f decades"
                 % (st["reach_lo"], st["reach_hi"], st["reach_dec"])
                 if st["reach_dec"] > 0 else "none"))
        print("-" * 78)
        print("  The number that decides a power law is the DRIFT, not <Gamma>:")
        print("  an average across a sloping curve can land on -11/6 by chance.")
        print("  There is no f-scan for THIS kernel yet.  The numbers the")
        print("  additive runner quoted (+0.293 at f = 0, +0.024 at f = 0.30)")
        print("  were measured at lambda = 1 and say nothing here: at lambda =")
        print("  2/3 the coagulation integral converges without a window, so")
        print("  alpha is expected NOT to move with f.  If it does, the window")
        print("  is doing something other than restoring locality.")
    print("  -> %s" % png)

    # ------------------------------------------------------------------------
    #  THE ISOCHRONES.  Coagulation only, and stacked the same way the spectrum
    #  is -- except that here the sum is unweighted, because iso_counts is a sum
    #  of counts rather than a mean of densities.
    if process == "coagulation":
        S = iso_load(sorted(glob.glob(os.path.join(
            outdir, "%s_f%.2f_rep*_spec.npz" % (short, args.f)))),
            args.skip_passes)
        print("-" * 78)
        if S is None:
            print("  isochrones      : none stored.  Files written before the age")
            print("                    grid existed carry no iso_counts; one more")
            print("                    pass writes them.")
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
                #  THE COLUMN THAT IS A MEASUREMENT.  d(tau) is what the binning
                #  rule left free: the separation in log-mass was FIXED, so a
                #  constant age step means log m advances linearly in time, i.e.
                #  the growth is exponential and b is infinite -- which is what
                #  beta = 1 says it must be.  A step that grows down the column
                #  is a power law instead.
                #  THE FIRST AND LAST GROUPS ARE BOUNDARIES, not cascade.  The
                #  first straddles the mass floor -- it is where the cohort
                #  crosses into the measured range, so its mean is set by the
                #  cut and not by the growth -- and the last is the starved tail
                #  folded back, which reaches the wall.  Both are printed, and
                #  neither is allowed into the number.
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
