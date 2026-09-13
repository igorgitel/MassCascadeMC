"""
MassCascadeMC
=============

Event-driven Monte Carlo engine for scale-free mass cascades.

The engine covers coagulation and fragmentation, in closed and open
systems, with a common majorant-collision framework.  The physics that
differs between the configurations enters only through what an accepted
event does to the two selected particles and through the boundary
conditions; pair selection, clock evolution, majorant construction and
bookkeeping share the same implementation.

Configurations
--------------
    process = "coagulation" | "fragmentation"
    system  = "closed"      | "open"

An open system may carry three boundary channels at once: deterministic
injection at a fixed mass and rate, an absorbing wall at `sink_mass`,
and an optional distributed first-order sink xi(m) = xi0 (m/m_ref)^omega.

Particle weight
---------------
One simulated particle is one physical particle: w = 1 is fixed for the
whole run and nothing in this file changes it.  There is no population
resampling and no weight thinning, so Sum(w) and Sum(w*m) are conserved
to machine precision and `mass_drift` is identically zero.  The cost is
dynamic range.

Measurements
------------
The spectrum dndm on the mass grid; the isochrone histogram in
(age, mass); the generation histogram in (gen, mass) with <tau>(g); and
the waiting law <dt>(m) ~ m^(1/b).

Documentation
-------------
Numerical methods, implementation choices and safeguards are documented
in NUMERICS.txt as numbered notes [1]-[21].  Comments in this file refer
to those numbers.
"""

import numpy as np
import json
import inspect
import warnings
import re
from pathlib import Path
import time as _time


# ======================================================================
#  Sentinel for "this came from the signature, not from the caller"
# ======================================================================
#
# iso_start_sink is armed by default in v2.  An int subclass is used rather
# than None so that the value behaves as a plain 10 EVERYWHERE -- comparisons,
# arithmetic, json, inspect.signature all see 10 -- while `isinstance(x,
# _Default)` still tells the validator whether the caller typed it.

class _Default(int):
    """Marks a value that came from a signature default.  Behaves as int."""
    __slots__ = ()


_ISO_START_SINK_DEFAULT = _Default(10)


# ======================================================================
#  KERNELS  --  passed in as a parameter; lambda recorded for the theory
# ======================================================================

def kernel_constant(m1, m2):
    """lambda = 0."""
    return 1.0


def kernel_geometric(m1, m2):
    """Geometric cross section, sigma ~ (r1+r2)^2 with r ~ m^(1/3).  lambda = 2/3."""
    return (m1 ** (1.0 / 3.0) + m2 ** (1.0 / 3.0)) ** 2


def kernel_sum(m1, m2):
    """lambda = 1/3."""
    return m1 ** (1.0 / 3.0) + m2 ** (1.0 / 3.0)


def kernel_product(m1, m2):
    """lambda = 2  --  ABOVE the gelation threshold; the stationary cascade does not exist."""
    return m1 * m2


def kernel_additive(m1, m2):
    """lambda = 1  --  exactly the marginal / gelation-threshold kernel; drift is exponential."""
    return m1 + m2


#: homogeneity degree of each kernel, used only to state the prediction
KERNEL_LAMBDA = {
    "kernel_constant":  0.0,
    "kernel_geometric": 2.0 / 3.0,
    "kernel_sum":       1.0 / 3.0,
    "kernel_additive":  1.0,
    "kernel_product":   2.0,
}


# ======================================================================
# SEEDING A POWER-LAW POPULATION                          note [12]
# ======================================================================

def sample_powerlaw(alpha, m_lo, m_hi, N, rng=None):
    """N masses drawn from n(m) ~ m^alpha on [m_lo, m_hi], by inverse CDF.

    For alpha != -1 the normalised CDF is

        F(m) = (m^(a) - m_lo^(a)) / (m_hi^(a) - m_lo^(a)),   a = alpha + 1

    so m = [m_lo^a + u (m_hi^a - m_lo^a)]^(1/a).  The a -> 0 case
    (alpha = -1) is log-uniform and is handled separately, because the
    general formula is 0/0 there and loses all its digits well before it.

    Done in logs rather than in powers: with alpha = -1.83 and six
    decades, m_lo^a spans 10^(-5) of m_hi^a, and the naive form throws
    away half the mantissa on the light end -- exactly the end that
    carries the statistics.
    """
    a = float(alpha) + 1.0
    N = int(N)
    if N <= 0:
        return np.zeros(0, dtype=float)
    if not (0.0 < m_lo < m_hi):
        raise ValueError("need 0 < m_lo < m_hi")
    if rng is None:
        rng = np.random.default_rng()
    u = rng.random(N)
    L_lo = np.log(float(m_lo))
    L_hi = np.log(float(m_hi))
    if abs(a) < 1e-9:
        return np.exp(L_lo + u * (L_hi - L_lo))
    # work with the LOG of the CDF endpoints; factor out the larger one
    if a > 0.0:
        # m_hi^a dominates: m = m_hi * [ (1-u) e^{a(L_lo-L_hi)} + u ]^(1/a)
        z = (1.0 - u) * np.exp(a * (L_lo - L_hi)) + u
        return np.exp(L_hi + np.log(z) / a)
    else:
        # m_lo^a dominates: m = m_lo * [ (1-u) + u e^{a(L_hi-L_lo)} ]^(1/a)
        z = (1.0 - u) + u * np.exp(a * (L_hi - L_lo))
        return np.exp(L_lo + np.log(z) / a)


def steady_N_for_mass(alpha, m_lo, m_hi, M_target):
    """How many bodies a power-law seed needs to carry mass M_target.

    <m> = Int m^(alpha+1) dm / Int m^(alpha) dm over [m_lo, m_hi], so
    N = M_target / <m>.  Handy when the cold run's steady state is known
    by its MASS (which is set by the injection rate and the residence
    time) rather than by its number.
    """
    a = float(alpha)
    lo, hi = float(m_lo), float(m_hi)

    def _mom(p):
        q = a + p + 1.0
        if abs(q) < 1e-12:
            return np.log(hi / lo)
        return (hi ** q - lo ** q) / q

    mbar = _mom(1) / _mom(0)
    return max(int(round(float(M_target) / mbar)), 1), float(mbar)


# ======================================================================
#  ANALYTIC PREDICTION
# ======================================================================

def predict(system, lam):
    """
    Return the analytic prediction for a given configuration.

    Parameters
    ----------
    system : {'closed','open'}
    lam    : kernel homogeneity degree lambda

    Returns
    -------
    dict with
        beta  : drift exponent,      |dm0/dtau| ~ m0^beta
        b     : growth exponent,     m0 ~ tau^b   (or (tau_*-tau)^b)
        alpha : spectrum exponent,   F(m) ~ m^alpha
    Identical for coagulation and fragmentation -- only the sign of the
    drift differs, not the exponents.
    """
    if system == "closed":
        beta = lam                      # partner density ~ m0^-1  (mass conserving packet)
    elif system == "open":
        beta = (1.0 + lam) / 2.0        # partner density ~ m0^alpha (stationary background)
    else:
        raise ValueError("system must be 'closed' or 'open'")

    alpha = -(1.0 + beta)
    b = np.inf if abs(1.0 - beta) < 1e-12 else 1.0 / (1.0 - beta)
    return {"beta": beta, "b": b, "alpha": alpha, "lambda": lam, "system": system}


# ======================================================================
#  THE ENGINE
# ======================================================================

def simulate(
    *,
    process,                    # 'coagulation' | 'fragmentation'
    system,                     # 'closed'      | 'open'
    kernel,                     # callable K(m1,m2) >= 0, non-decreasing in both arguments
    edges,                      # 1D strictly increasing mass-bin edges
    ic,                         # THREE accepted forms:
                                #   {'m': m0, 'N': N0}
                                #       monodisperse, exactly as in v2..v4.
                                #   {'mass': array}
                                #       an explicit list of masses.
                                #   {'steady': {'alpha':a, 'm_lo':.., 'm_hi':..,
                                #               'N':..}}                 [v5]
                                #       N bodies drawn from n(m) ~ m^a on
                                #       [m_lo, m_hi] -- the WARM START, note [12].
                                #       These bodies carry gen = -1 and so do all
                                #       of their descendants, note [13].
                                #       Optionally 'M' instead of 'N' to fix the
                                #       seeded MASS and let N follow.
    # ---- open-system boundary conditions (ignored when system='closed') ----
    injection_rate=0.0,         # particles per unit physical time
    injection_mass=None,        # mass of an injected particle; default ic['m']
    sink_mass=None,             # ABSORBING boundary.  coagulation: remove products >= sink
                                #                     fragmentation: remove fragments <= sink
    siphon=None,                # DISTRIBUTED SINK, note [16].  A first-order removal that
                                # acts on every particle at once, at the rate
                                #        xi(m) = xi0 * (m/m_ref)^omega     [per particle]
                                # as opposed to `sink_mass`, which is a wall at one end of
                                # the range.  Two accepted forms:
                                #   {'xi0':.., 'omega':.., 'm_ref':1.0}
                                #       the rate itself.
                                #   {'delta0':.., 'nu':0.0, 'tau_drift':(A, e), 'dlnm':c}
                                #       the DIMENSIONLESS loss per e-fold of the drift, which
                                #       is what the analytic prediction is written in.
                                #       tau_drift is the DRIFT TIME of a pilot run with
                                #       siphon=None, as amplitude and exponent:
                                #            tau_drift(m) = (A/c) m^e,   e = 1 - beta,
                                #       whence omega = nu - e and xi0 = delta0 * c / A.
                                #       The divisor c defaults to 1, i.e. A is the drift
                                #       time itself.  Measure it from the constant flux,
                                #            tau_drift = m^2 F(m)/J,
                                #       which needs no locality assumption and works for
                                #       both processes.  Passing the waiting law instead --
                                #       A, 1/b from <dt> = A m^(1/b) with c = dlnm_mean --
                                #       is valid only where the collision integral is
                                #       LOCAL, i.e. fragmentation with a threshold; in
                                #       coagulation off a steep spectrum <dt> is set by the
                                #       injection scale and even its SIGN is not 1 - beta.
                                # None disables the channel completely -- not one random
                                # number is drawn -- so the RNG stream is the one v5 consumed
                                # and a siphon=None run is bit-identical to v5.
    gate_counts_siphon=None,    # Do the two sink GATES (iso_start_sink, stop_sink_events)
                                # count siphon removals as well as boundary absorptions?
                                # None means "yes when the siphon is on".  With a strong
                                # siphon only (m_sink/m_inj)^delta0 of the flux ever reaches
                                # the boundary -- 1e-3 at delta0 = 0.5 over six decades -- so
                                # a gate counting the boundary alone opens a thousand times
                                # later and makes runs at different delta0 incomparable.
                                # Removal by ANY route is delta0-independent in the steady
                                # state, which is what makes the sum the right gate.  The
                                # counters themselves stay separate everywhere else.
    # ---- passive limiter (closed fragmentation needs one to terminate) ----
    min_frag_mass=None,         # particles below this never fragment (mass stays conserved)
    frag_min_ratio=0.0,         # LOCALITY CONTROL, fragmentation.  The heavier particle
                                # breaks only if the impactor satisfies  m_small >= f * m_large.
                                # f = 0 reproduces "any impactor shatters any target", which is
                                # NON-LOCAL against a power-law background (see note [8]) and
                                # drives the drift from the sink scale instead of from m0.
                                # Use f of order 0.1-1 for a scale-free (local) cascade.
    coag_min_ratio=0.0,         # LOCALITY CONTROL, coagulation.  [v7, note [17]]  The SAME
                                # rule, on the same quantity, for the other process: two
                                # bodies merge only if  m_small >= f * m_large, so a body
                                # grows by finding a partner of comparable mass rather than
                                # by eating monomers.  f = 0 is the unrestricted kernel and
                                # leaves v7 bit-identical to v6.
                                #
                                # WHY IT EXISTS.  The two processes were not being treated
                                # alike: fragmentation has been run at f = 0.30 since v4 --
                                # the number is in the file names -- while coagulation had no
                                # such control at all.  For a kernel of homogeneity lambda the
                                # constant-flux integral converges only in a window of lambda,
                                # and at the edge of that window it diverges logarithmically
                                # from the m' << m region: the cascade is then NON-LOCAL, its
                                # shape is set by the boundaries rather than by the kernel,
                                # and no power law appears at any box size.  Measured on the
                                # additive kernel, lambda = 1: the local slope was found to
                                # collapse on log m / log m_sink rather than on m, i.e. to
                                # depend on POSITION IN THE BOX and not on mass, and Gamma
                                # crossed -(1+beta) at 0.51 and 0.52 of the range in two boxes
                                # differing by a hundred in size.  Cutting the m' << m region
                                # is what removes that divergence.
                                #
                                # WHY IT DOES NOT CHANGE THE EXPONENTS.  The condition is a
                                # function of the RATIO alone, so the effective kernel
                                #     K_eff = K(m1,m2) * Theta[min/max - f]
                                # has the same homogeneity degree lambda as K.  beta, b and
                                # alpha are unchanged; only locality is.  The AMPLITUDE of the
                                # flux does change, and downwards: the interactions removed
                                # are the ones that carried most of it, so the cascade is
                                # slower in physical time at the same spectrum.
    frag_split_width=0.5,       # HALF-WIDTH of the fragment split, fragmentation only.
                                # xi is drawn uniformly on (0.5 - w, 0.5 + w) and the pieces
                                # are xi*m and (1-xi)*m.  w = 0.5 is the old uniform split on
                                # (0,1) exactly, so this default changes nothing.  Smaller w
                                # narrows the isochrone -- the mass-age relation sharpens --
                                # but the generations stop overlapping and the spectrum
                                # develops a picket fence at the mean step.  Generations
                                # merge only after ~(mu/2sigma)^2 splits while the cascade
                                # takes 6/mu of them, so w below about 0.25 combs the
                                # spectrum: at w = 0.05 the modulation is a factor 2.3 per
                                # 0.3 dex, at w = 0.25 it is 13%.  Do not go below 0.25
                                # without looking at the raw histogram first.
    frag_age_rule="inherit",    # AGE INHERITANCE, fragmentation only.  Which fragment keeps
                                # the parent's clock and which is born now, at t_phys.
                                #   'inherit'  both keep the parent's t_born (v2 behaviour).
                                #              The age class is then a FAMILY -- the whole
                                #              descendant tree of one injected body -- and it
                                #              spans the entire inertial range by construction.
                                #   'heavier'  the heavier fragment keeps the clock, the lighter
                                #              is reborn.  The tracer is the always-heavy chain:
                                #              a characteristic of the transport equation, i.e.
                                #              the front.  This is the isochrone one wants.
                                #   'lighter'  mirror of the above; kept as a sensitivity test,
                                #              not as physics.
                                #   'both_new' both reborn.  Null test: destroys the age axis
                                #              on purpose, so <m>(tau) must collapse to noise.
    # ---- stopping ----
    max_time=np.inf,
    max_events=np.inf,
    stop_max_mass=None,         # stop when m_max >= this (closed coagulation)
    stop_min_mass=None,         # stop when m_mean <= this (closed fragmentation)
    stop_sink_events=None,      # [v2] stop once this many PHYSICAL particles have been absorbed
                                # at the sink.  Unlike max_events, which is a statement about
                                # RESOURCES, this is a statement about the PHYSICS: the cascade
                                # has demonstrably delivered that many particles across the whole
                                # inertial range.  It therefore needs no retuning when N_ss or
                                # the mass range changes.  None disables it (the default).
                                # Open systems only; silently ignored when system='closed'.
    max_stall_tries=2_000_000,  # stop if this many majorant attempts yield no event
                                # (happens once everything sits below min_frag_mass)
    # ---- snapshots ----
    snapshot_mode="events",     # 'events' | 't_phys' | 'log_m0'   (see note [10])
                                #   'log_m0' is the one to use beyond ~2 decades
    snapshot_stride=1e4,        # events / time units / SNAPSHOTS PER DECADE of m0
    snapshot_first_at_start=True,
    # ---- isochrones ----
    iso_age_edges=None,         # 1D age-bin edges; None disables
    iso_t_start=0.0,            # start accumulating isochrones only after this ABSOLUTE time.
                                # Beware: the clock runs as dt = 2V/(wR) with R ~ N^2, so the
                                # time reached after E attempts is t = 2E/N^2.  A fixed value
                                # tuned at one N is 100x too large at 10N and the isochrone
                                # histogram then stays empty (b comes back as nan).  Prefer
                                # iso_start_sink, or scale this with 1/N_SS.
    iso_start_sink=_ISO_START_SINK_DEFAULT,
                                # start accumulating isochrones only after this many PHYSICAL
                                # particles have been absorbed at the sink.  This is the
                                # N-independent steady-state gate: mass has demonstrably
                                # crossed the whole inertial range before any age is binned.
                                # Both gates apply (AND).
                                # [v2] The default is 10 instead of 0, and it applies to
                                # COAGULATION and FRAGMENTATION alike -- the sink is the sink,
                                # whichever end of the cascade it sits at.  In a CLOSED system
                                # there is no sink, so the gate is silently disarmed instead of
                                # raising; only an explicitly passed value on a closed run is an
                                # error (see the validation block below).
    track_generations=True,     # [v4] see note [7].  Carry `gen` on every particle -- the
                                # number of splits between it and the body that ENTERED --
                                # and bin the live population by (gen, mass) at each
                                # snapshot, exactly as the isochrones bin it by (age, mass).
                                # This is what the tracer machinery of v2/v3 was for, done
                                # on the whole population rather than on a few thousand
                                # tagged paths: same quantity, four orders of magnitude more
                                # of it, 16 MB at N = 4e6.
    gen_max=64,                 # Highest generation resolved.  Everything above is counted
                                # in `gen_overflow` and NOT binned, so a clipped tail is a
                                # number rather than a spurious pile in the last bin.  A
                                # body needs ~log(m_inj/m_sink)/|<ln xi>| splits to reach
                                # the sink -- 28 over six decades at w = 1/2.
    age_rule="min",             # coagulation age inheritance: 'min'|'mass_weighted'|'heavier'
    track_waiting=True,         # [v5] see note [14].  Accumulate <dt>(m), the mean time
                                # between successive MASS-CHANGING events for a particle
                                # sitting in a given mass bin.  <dt> ~ m^(1/b), so this
                                # measures b DIRECTLY, without going through alpha and
                                # its db/dalpha = b^2 = 36 amplification.  Memoryless,
                                # therefore valid on a seeded population from t = 0.
                                # Costs one float64 per particle and two adds per event.
    # ---- misc ----
    weight=1.0,                 # CONSTANT weight per simulated particle (see notes [5ii], [9])
    volume=1.0,
    rng=None,
    verbose=True,
    verbose_every=1,
):
    """
    Event-driven majorant-frequency Monte Carlo for coagulation or
    fragmentation, in a closed or an open system.

    Returns a dict of arrays; see the bottom of this function.
    """

    # ------------------------------------------------------------------
    # 0.  Validate and set up
    # ------------------------------------------------------------------

    if process not in ("coagulation", "fragmentation"):
        raise ValueError("process must be 'coagulation' or 'fragmentation'")
    if system not in ("closed", "open"):
        raise ValueError("system must be 'closed' or 'open'")
    if frag_age_rule not in ("inherit", "heavier", "lighter", "both_new"):
        raise ValueError("frag_age_rule must be 'inherit', 'heavier', "
                         "'lighter' or 'both_new'")
    if age_rule not in ("min", "max", "mass_weighted", "heavier"):
        raise ValueError("age_rule must be 'min', 'max', 'mass_weighted' or 'heavier'")
    if int(gen_max) < 1:
        raise ValueError("gen_max must be >= 1")
    if not (0.0 <= frag_split_width <= 0.5):
        raise ValueError("frag_split_width is a half-width: 0 <= w <= 0.5 "
                         "(0.5 = uniform split, 0 = exact halving)")
    # The sink gate is now armed BY DEFAULT (10 particles), so a bare
    # `raise` here would make every CLOSED run fail on an argument the caller
    # never passed.  Distinguish the two cases with the _Default marker: a value
    # that came from the signature is silently disarmed, a value the caller typed
    # is still an error, because asking for a sink gate in a system that has no
    # sink is a mistake worth hearing about.
    if system != "open":
        if not isinstance(iso_start_sink, _Default) and iso_start_sink > 0:
            raise ValueError("iso_start_sink needs an absorbing sink, i.e. system='open'")
        iso_start_sink = 0
        if stop_sink_events is not None:
            raise ValueError("stop_sink_events needs an absorbing sink, i.e. system='open'")

    if system == "closed":
        # A closed system has no source and no absorbing boundary: total mass
        # is exactly conserved.  min_frag_mass is a PASSIVE floor -- particles
        # below it simply stop fragmenting -- so mass conservation is untouched.
        injection_rate = 0.0
        sink_mass = None
    else:
        if injection_rate <= 0.0:
            raise ValueError("an open system needs injection_rate > 0")
        if sink_mass is None:
            raise ValueError("an open system needs an absorbing sink_mass")

    edges = np.asarray(edges, dtype=float)
    if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("edges must be strictly increasing, len >= 2")
    widths = np.diff(edges)
    centers = np.sqrt(edges[:-1] * edges[1:])       # geometric (log) centres
    B = edges.size - 1

    w = float(weight)                               # physical particles per sim particle
    V = float(volume)

    # The rng is built BEFORE the initial condition is parsed, because a
    # seeded IC draws from it.  Nothing else draws in between, so a run with
    # ic={'m','N'} consumes the stream in exactly the order v4 did -- which is
    # what makes the bit-identity test meaningful.
    if rng is None:
        rng = np.random.default_rng()
    rnd = rng.random
    rint = rng.integers

    # ------------------------------------------------------------------
    # 0b.  THE INITIAL CONDITION -- three forms, one of them warm
    # ------------------------------------------------------------------
    seeded = False              # was the population planted rather than grown?
    seed_info = None
    restart_from = None         # [v5] name of the state file this leg grew from
    t_origin = 0.0              # [v5] physical time accumulated in EARLIER legs
    _gen0 = _gt0 = _it0 = None  # [v5] inherited per-particle state, if any
    if "steady" in ic:
        # ---- WARM START, note [12] ----
        _s = dict(ic["steady"])
        alpha_seed = float(_s.get("alpha", -1.8333333333333333))
        m_lo_seed = float(_s["m_lo"])
        m_hi_seed = float(_s["m_hi"])
        if "N" in _s and _s["N"] is not None:
            N0 = int(_s["N"])
            _mbar = None
        elif "M" in _s and _s["M"] is not None:
            N0, _mbar = steady_N_for_mass(alpha_seed, m_lo_seed, m_hi_seed,
                                          float(_s["M"]) / w)
        else:
            raise ValueError("ic['steady'] needs either 'N' (bodies) or 'M' (mass)")
        if N0 < 2:
            raise ValueError("a seeded population needs at least 2 bodies")
        m_init_arr = sample_powerlaw(alpha_seed, m_lo_seed, m_hi_seed, N0, rng)
        seeded = True
        seed_info = {"alpha": alpha_seed, "m_lo": m_lo_seed, "m_hi": m_hi_seed,
                     "N": int(N0), "mean_mass": float(m_init_arr.mean())}
        # `m_init` is now only a REPORTING number and must not be used as a
        # default for the injection mass -- the seed has no single mass.  Which
        # end injection sits at is fixed by the process, not by the IC.
        m_init = float(m_init_arr.mean())
        if injection_mass is None:
            injection_mass = m_hi_seed if process == "fragmentation" else m_lo_seed
    elif "state" in ic:
        # ---- RESTART FROM A SAVED STATE, note [15] ----
        _st = ic["state"]
        m_init_arr = np.asarray(_st["mass"], dtype=float).ravel().copy()
        N0 = int(m_init_arr.size)
        if N0 < 2:
            raise ValueError("ic['state'] needs at least 2 particles")
        for _k in ("gen", "gen_time", "inj_time"):
            if _k not in _st:
                raise ValueError("ic['state'] needs 'mass', 'gen', 'gen_time' and "
                                 "'inj_time'; '%s' is missing.  A run saved with "
                                 "drop_particles=True does not carry them." % _k)
            if np.asarray(_st[_k]).ravel().size != N0:
                raise ValueError("ic['state']['%s'] has %d entries against %d masses"
                                 % (_k, np.asarray(_st[_k]).ravel().size, N0))
        _gen0 = np.asarray(_st["gen"], dtype=np.int64).ravel().copy()
        # THE TIME OFFSET, and it is the whole trick.  The clock of this branch
        # starts at zero, so every inherited timestamp is shifted back by the
        # previous branch's end time and lands NEGATIVE.  That is not a hack: those
        # particles genuinely entered before t = 0, and tau = t_phys - t_anc then
        # returns the FULL age across the seam instead of restarting it.  This is
        # exactly what was censored in v4, where tau(g) saturated at t_end and
        # gave b = 1.74 instead of 6.  Restarts compose: each branch subtracts its
        # own end time, so an age carried through five branches is still the true one.
        _t_prev = float(_st.get("t_end", 0.0))
        _gt0 = np.asarray(_st["gen_time"], dtype=float).ravel() - _t_prev
        _it0 = np.asarray(_st["inj_time"], dtype=float).ravel() - _t_prev
        restart_from = _st.get("source")
        t_origin = float(_st.get("t_origin", 0.0)) + _t_prev   # time before this branch
        m_init = float(m_init_arr.mean())
        if injection_mass is None:
            raise ValueError("ic={'state': ...} carries no single mass; "
                             "pass injection_mass explicitly")
    elif "mass" in ic:
        m_init_arr = np.asarray(ic["mass"], dtype=float).ravel().copy()
        N0 = int(m_init_arr.size)
        if N0 < 1:
            raise ValueError("ic['mass'] is empty")
        m_init = float(m_init_arr.mean())
        if injection_mass is None:
            raise ValueError("ic={'mass': ...} carries no single mass; "
                             "pass injection_mass explicitly")
    else:
        # ---- COLD START: v2/v3/v4 behaviour, unchanged ----
        m_init = float(ic["m"])
        N0 = int(ic["N"])
        m_init_arr = np.full(N0, m_init, dtype=float)
        if injection_mass is None:
            injection_mass = m_init

    if np.any(m_init_arr <= 0.0):
        raise ValueError("initial masses must be strictly positive")
    if _gen0 is not None:
        _gmax_in = int(_gen0.max()) if _gen0.size else 0
        if _gmax_in > int(gen_max):
            warnings.warn(
                "the restored state carries generations up to %d while gen_max = %d: "
                "everything above the cap goes straight into gen_overflow and is "
                "never binned.  Raise gen_max to at least %d."
                % (_gmax_in, int(gen_max), _gmax_in + 16),
                RuntimeWarning, stacklevel=2)
    # A seeded mass outside `edges` is silently CLAMPED into the end bin by
    # mass_to_bin, which turns a mis-specified seed into a spurious spike at
    # one end of the spectrum rather than into an error.  Say so instead.
    _n_out_of_grid = int(np.count_nonzero((m_init_arr < edges[0])
                                          | (m_init_arr >= edges[-1])))
    if _n_out_of_grid:
        warnings.warn(
            "%d of %d initial masses lie outside the bin grid [%.3g, %.3g) and "
            "will be clamped into the end bins, which shows up as a spike there. "
            "Widen `edges` or narrow the seed range."
            % (_n_out_of_grid, N0, edges[0], edges[-1]),
            RuntimeWarning, stacklevel=2)

    # ------------------------------------------------------------------
    # 0c. THE DISTRIBUTED SINK -- the siphon, note [16]
    #
    #     xi(m) = xi0 * (m/m_ref)^omega is a removal rate PER PARTICLE, so the
    #     removal is linear in n(m) and the whole analytic family of note [16]
    #     applies: nu = omega + 1 - beta decides whether the spectral index
    #     moves at all, and the direction of the cascade decides which way.
    # ------------------------------------------------------------------
    siphon_on = siphon is not None
    xi0 = 0.0
    omega = 0.0
    m_ref_siphon = 1.0
    nu_siphon = None
    delta0_siphon = None
    siphon_spec = None
    if siphon_on:
        if system != "open":
            raise ValueError("siphon needs system='open': the removed mass has to leave "
                             "the box, and a closed run is defined by conserving it")
        _sp = dict(siphon)
        siphon_spec = {k: (list(v) if isinstance(v, (tuple, list)) else v)
                       for k, v in _sp.items()}
        if "delta0" in _sp:
            # ---------------- CALIBRATED FORM ----------------
            # delta(m) = xi(m) * tau_drift(m) = delta0 * m^nu is the fraction of
            # a cohort lost while the drift carries its scale up by a factor e.
            # The caller supplies the drift time as (A, e) with
            #        tau_drift(m) = (A/c) m^e,        e = 1 - beta,
            # so that   delta0 = xi0 * A / c   =>   xi0 = delta0 * c / A,
            # and   nu = omega + 1 - beta = omega + e   =>   omega = nu - e.
            # The divisor c defaults to 1: A is then the drift time itself, which
            # is what the flux route  tau_drift = m^2 F(m)/J  returns.  c exists
            # for the waiting-law route, where A is the amplitude of <dt> and a
            # particle needs 1/c events to move one e-fold, c = out["dlnm_mean"].
            delta0_siphon = float(_sp["delta0"])
            nu_siphon = float(_sp.get("nu", 0.0))
            _A, _invb = _sp["tau_drift"]
            _A = float(_A); _invb = float(_invb)
            _c = float(_sp.get("dlnm", 1.0))
            if _A <= 0.0 or _c <= 0.0:
                raise ValueError("siphon: the tau_drift amplitude A and dlnm c must be > 0")
            omega = nu_siphon - _invb
            xi0 = delta0_siphon * _c / _A
            m_ref_siphon = 1.0            # A was fitted in absolute mass units
        else:
            # ---------------- RAW FORM ----------------
            xi0 = float(_sp["xi0"])
            omega = float(_sp["omega"])
            m_ref_siphon = float(_sp.get("m_ref", 1.0))
            if m_ref_siphon <= 0.0:
                raise ValueError("siphon: m_ref must be > 0")
        # A NaN here would silently switch the channel off -- nan > 0 is False --
        # and the run would come back looking like the lossless one.  That happens
        # for real: the calibrated form divides by a waiting-law amplitude, and a
        # pilot too short to fit returns nan.  Refuse it loudly instead.
        if not (np.isfinite(xi0) and np.isfinite(omega)):
            raise ValueError(
                "siphon: xi0 = %r, omega = %r -- not finite.  With the calibrated "
                "form this normally means the pilot run had no waiting law to fit "
                "(A or 1/b came back nan), so there is nothing to calibrate against."
                % (xi0, omega))
        if xi0 < 0.0:
            raise ValueError("siphon: xi0 must be >= 0")
        siphon_on = xi0 > 0.0
    if gate_counts_siphon is None:
        gate_counts_siphon = bool(siphon_on)
    gate_counts_siphon = bool(gate_counts_siphon)

    # Fold m_ref into the amplitude once and for all: the loop then works with
    # the bare power m^omega and never divides.
    xi_amp = xi0 * m_ref_siphon ** (-omega) if siphon_on else 0.0

    # THE REJECTION BOUND, note [16].  Selecting a victim with probability
    # proportional to xi(m) is done by a uniform draw thinned against a bound on
    # m^omega over the LIVE population.  That bound is known before the run
    # starts and never moves, in every one of the four configurations:
    #
    #   fragmentation, omega < 0 : nothing alive is lighter than the sink
    #   fragmentation, omega > 0 : nothing ever gets heavier than what entered
    #   coagulation,   omega > 0 : nothing alive is heavier than the sink
    #   coagulation,   omega < 0 : nothing ever gets lighter than what entered
    #
    # Taking the extremum over {sink, injection, seed} covers all four at once
    # and stays valid when the seed sticks out past the sink.  Do NOT use the
    # running records m_min / m_max instead: in fragmentation m_min is written
    # on the fragment BEFORE the sink deletes it, so the record walks far below
    # sink_mass and the acceptance is spoiled for the rest of the run; m_max is
    # its mirror image in coagulation.
    if siphon_on and omega != 0.0:
        _cand = [float(injection_mass)]
        if sink_mass is not None:
            _cand.append(float(sink_mass))
        _cand.append(float(m_init_arr.max()) if omega > 0.0 else float(m_init_arr.min()))
        m_bound = max(_cand) if omega > 0.0 else min(_cand)
    else:
        m_bound = 1.0
    if m_bound <= 0.0:
        raise ValueError("siphon: the rejection bound came out non-positive")
    # Cheap forecast of the acceptance, so that a bound sitting far outside the
    # population is caught here rather than a million draws later inside the
    # loop.  The usual cause is a nominal `sink_mass` set decades away from the
    # range actually in use, which makes the bound valid but useless.
    if siphon_on and omega != 0.0:
        _acc0 = float(np.mean((m_init_arr / m_bound) ** omega))
        if _acc0 < 1e-3:
            warnings.warn(
                "the siphon's rejection bound m_bound = %.3g gives a forecast "
                "acceptance of %.2e against the initial population: selection will "
                "cost ~%.0f draws per removal.  The bound is the extremum over "
                "{sink_mass, injection_mass, seed}, so this means one of them sits "
                "decades outside the range actually in use.  See note [16]."
                % (m_bound, _acc0, 1.0 / max(_acc0, 1e-300)),
                RuntimeWarning, stacklevel=2)

    # ------------------------------------------------------------------
    # 1.  Particle arrays.  The live prefix [0:live] is the active set.
    # ------------------------------------------------------------------
    cap = max(N0, 1024)
    mass = np.empty(cap, dtype=np.float64)
    inj_time = np.empty(cap, dtype=np.float64)
    bin_of = np.empty(cap, dtype=np.int32)
    pos_in_bin = np.empty(cap, dtype=np.int32)

    gen = np.zeros(cap, dtype=np.int32)             # splits since the ancestor entered
    gen_time = np.zeros(cap, dtype=np.float64)      # when that ancestor entered
    # [v5] when this particle's MASS last changed.  Not the same as inj_time
    # (which is a family clock and survives fragmentation) and not the same as
    # gen_time (which points at the ancestor).  This one is strictly local: it
    # is reset every time the particle takes part in an event.       note [14]
    t_change = np.zeros(cap, dtype=np.float64)

    mass[:N0] = m_init_arr
    #  On a RESTART the ages and the counters are INHERITED, note [15];
    #  otherwise everything starts from zero, as in v2..v4.
    inj_time[:N0] = _it0 if _it0 is not None else 0.0
    #  t_change is ALWAYS zeroed: it is the purely local clock of the waiting
    #  law, and the interval in progress was closed in the previous leg.  This
    #  does not touch the `exposure` estimator at all -- occupancy accrues
    #  independently of it -- and biases only the `interval` estimator, and
    #  only on the first interval of each particle.
    t_change[:N0] = 0.0
    # [v5] A SEEDED body did not enter the system -- it was planted, and its
    # split count is unknowable.  gen = -1 marks that, and note [13] makes the
    # mark contagious.  A COLD monodisperse IC at m_inj is indistinguishable
    # from an injected body, so it stays honest at gen = 0, as in v4.
    if _gen0 is not None:
        gen[:N0] = np.clip(_gen0, -1, np.iinfo(np.int32).max).astype(np.int32)
        gen_time[:N0] = _gt0
    else:
        gen[:N0] = -1 if seeded else 0
        gen_time[:N0] = -np.inf if seeded else 0.0
    live = N0

    bins = [[] for _ in range(B)]                   # bin -> list of particle indices

    _PARR = ("mass", "inj_time", "gen", "gen_time", "t_change", "bin_of", "pos_in_bin")

    def ensure_capacity(need):
        """Grow all per-particle arrays together, keeping the live prefix."""
        nonlocal mass, inj_time, gen, gen_time, t_change, bin_of, pos_in_bin
        if need <= mass.size:
            return
        new_cap = max(need, int(mass.size * 1.6) + 1)
        for name in _PARR:
            old = locals_ref[name]
            new = np.empty(new_cap, dtype=old.dtype)
            new[:live] = old[:live]
            locals_ref[name] = new
        mass = locals_ref["mass"]
        inj_time = locals_ref["inj_time"]
        gen = locals_ref["gen"]
        gen_time = locals_ref["gen_time"]
        t_change = locals_ref["t_change"]
        bin_of = locals_ref["bin_of"]
        pos_in_bin = locals_ref["pos_in_bin"]

    locals_ref = {"mass": mass, "inj_time": inj_time, "gen": gen, "gen_time": gen_time,
                  "t_change": t_change, "bin_of": bin_of, "pos_in_bin": pos_in_bin}

    # ------------------------------------------------------------------
    # 2.  Bin index, and O(1) bin membership  [trick 3]
    # ------------------------------------------------------------------
    def mass_to_bin(m):
        b = int(np.searchsorted(edges, m, side="right") - 1)
        return 0 if b < 0 else (B - 1 if b >= B else b)

    def bin_add(i, b):
        bin_of[i] = b
        pos_in_bin[i] = len(bins[b])
        bins[b].append(i)

    def bin_remove(i):
        b = int(bin_of[i]); p = int(pos_in_bin[i])
        last = bins[b][-1]
        bins[b][p] = last
        pos_in_bin[last] = p
        bins[b].pop()

    for i in range(live):
        bin_add(i, mass_to_bin(mass[i]))

    N_bin = np.array([len(bins[b]) for b in range(B)], dtype=np.float64)

    # ------------------------------------------------------------------
    # 3.  Static bin-pair majorant table  [trick 2]
    #     fmaj[b,c] = K(upper_b, upper_c) bounds K on the whole bin pair,
    #     PROVIDED K is non-decreasing in both arguments.  Violations are
    #     checked at run time rather than assumed.
    # ------------------------------------------------------------------
    upper = edges[1:]
    fmaj = np.empty((B, B), dtype=np.float64)
    for b in range(B):
        for c in range(B):
            val = float(kernel(float(upper[b]), float(upper[c])))
            if val < 0.0:
                raise ValueError("kernel must be non-negative")
            fmaj[b, c] = val

    # ------------------------------------------------------------------
    # 3b.  THE LOCALITY BAND  [v7, note [17]]
    #      One rule, one place, both processes.  loc_ratio is the f of
    #      whichever process is running; the pair-level test further down
    #      is what enforces it exactly, and this block is what stops the
    #      enforcement from being ruinously expensive.
    #
    #      A bin pair (b, c) is entirely forbidden when even its most
    #      favourable member fails the test.  For b below c the best case
    #      puts b at its TOP edge and c at its BOTTOM edge, so
    #
    #          best ratio(b, c) = upper[b] / edges[c]        (b < c)
    #
    #      and the pair is dead when that is already below f.  Those
    #      entries are zeroed, which removes them from the majorant
    #      process altogether.
    #
    #      THIS CHANGES NOTHING PHYSICAL, and the reason is the identity
    #      the whole scheme rests on: a trial picks the pair with
    #      probability fmaj_ij/R and accepts it with K_ij/fmaj_ij, so the
    #      realised rate is proportional to K_ij and fmaj cancels.  Pairs
    #      with K_eff = 0 contributed nothing but rejected trials.  Their
    #      removal lowers R, which lengthens dt = 2V/(wR) per trial by
    #      exactly the factor by which the number of trials falls, and the
    #      physical clock is untouched.
    #
    #      What it DOES change is the sequence of random draws.  A run
    #      with f > 0 under v7 is therefore not bit-comparable with the
    #      same f under v6 -- only statistically comparable.  With f = 0
    #      the block is skipped entirely and v7 is bit-identical to v6.
    loc_ratio = float(coag_min_ratio if process == "coagulation" else frag_min_ratio)
    if not np.isfinite(loc_ratio) or loc_ratio < 0.0 or loc_ratio > 1.0:
        raise ValueError("min_ratio for %s must lie in [0, 1]; got %r"
                         % (process, loc_ratio))
    n_band = -1
    if loc_ratio > 0.0:
        lower = edges[:-1]
        best = np.minimum.outer(upper, upper) / np.maximum.outer(lower, lower)
        dead = best < loc_ratio
        np.fill_diagonal(dead, False)      # a bin can always pair with itself
        fmaj[dead] = 0.0
        n_band = int((~dead).sum(axis=1).max()) if B else 0
        if not np.isfinite(fmaj).all():
            raise RuntimeError("locality band produced a non-finite majorant")
        if fmaj.sum() <= 0.0:
            raise ValueError(
                "the locality band killed every bin pair: f = %.4g is tighter "
                "than one bin of the grid (%.4g dex).  Widen f or refine the grid."
                % (loc_ratio, np.log10(edges[1] / edges[0])))
        if verbose:
            print("   locality: %s f = %.4g -> band of %d bins, %.1f%% of the "
                  "pair table live"
                  % (process, loc_ratio, n_band, 100.0 * (~dead).mean()))

    diag = np.diag(fmaj).copy()

    # ------------------------------------------------------------------
    # 4.  Rate bookkeeping  [trick 4]
    #     T[b]        = sum_c fmaj[b,c] N_c
    #     rate_row[b] = N_b (T[b] - fmaj[b,b])       (ordered pairs, i != j)
    #     R           = sum_b rate_row[b]  = sum_{i != j} fmaj_ij
    #                                      = 2 * sum_{i<j} fmaj_ij      [note 5i]
    # ------------------------------------------------------------------
    T = fmaj @ N_bin
    rate_row = N_bin * (T - diag)
    R = float(rate_row.sum())

    # [v5] Occupancy hook, note [14].  apply_delta is the ONE choke point every
    # change of N_bin passes through -- injection, coagulation, fragmentation,
    # the sink -- so hanging the Int N_b dt flush here is what makes that
    # integral exact rather than sampled.  A one-element list rather than a
    # forward reference to a function defined further down: no definition-order
    # hazard, and the hook is a no-op until the accumulators exist.
    _occ_hook = [None]

    def apply_delta(b, dN):
        """One bin count changed by dN: update T, rate_row and R in O(B), vectorised."""
        nonlocal R
        if dN == 0:
            return
        if _occ_hook[0] is not None:
            _occ_hook[0]()                       # flush BEFORE N_bin moves
        N_bin[b] += dN
        T[:] += dN * fmaj[:, b]
        np.multiply(N_bin, T - diag, out=rate_row)
        R = float(rate_row.sum())

    # --- vectorised selection: cumsum + searchsorted instead of a python loop ---
    def sample_row():
        cum = np.cumsum(rate_row)
        return int(np.searchsorted(cum, rnd() * cum[-1], side="right"))

    def sample_col(b):
        q = N_bin * fmaj[b]
        q[b] = (N_bin[b] - 1.0) * fmaj[b, b]        # exclude the self-pair
        s = q.sum()
        if s <= 0.0:
            return None
        cum = np.cumsum(q)
        return int(np.searchsorted(cum, rnd() * s, side="right"))

    # ------------------------------------------------------------------
    # 5.  Particle removal / relocation
    # ------------------------------------------------------------------
    def delete_particle(j):
        """Remove particle j: drop from its bin, then swap-with-last in the arrays."""
        nonlocal live, S_pow
        if siphon_on:
            S_pow -= float(mass[j]) ** omega              # note [16]
        b_j = int(bin_of[j])
        bin_remove(j)
        apply_delta(b_j, -1)
        last = live - 1
        if j != last:
            mass[j] = mass[last]
            inj_time[j] = inj_time[last]
            gen[j] = gen[last]
            gen_time[j] = gen_time[last]
            t_change[j] = t_change[last]
            b_last = int(bin_of[last]); p_last = int(pos_in_bin[last])
            bins[b_last][p_last] = j
            bin_of[j] = b_last
            pos_in_bin[j] = p_last
        live -= 1

    def move_bin(i, b_new):
        b_old = int(bin_of[i])
        if b_new == b_old:
            return
        bin_remove(i); apply_delta(b_old, -1)
        bin_add(i, b_new); apply_delta(b_new, +1)

    def add_particle(m, t_born, g=0, t_anc=None):
        """Append one particle; returns its index.  `g` is its generation and `t_anc` the
        moment its ancestor entered -- both default to a newly ENTERED body."""
        nonlocal live, S_pow
        if siphon_on:
            S_pow += float(m) ** omega                    # note [16]
        ensure_capacity(live + 1)
        i = live
        mass[i] = m
        inj_time[i] = t_born
        gen[i] = g
        gen_time[i] = t_born if t_anc is None else t_anc
        t_change[i] = t_phys          # [v5] its mass is new as of NOW, note [14]
        b = mass_to_bin(m)
        bin_add(i, b)
        apply_delta(b, +1)
        live += 1
        return i


    # ------------------------------------------------------------------
    # 6.  Age inheritance for coagulation  [trick 7]
    # ------------------------------------------------------------------
    def inherit_age(t1, t2, m1, m2):
        if age_rule == "min":
            return min(t1, t2)                       # oldest ancestor wins
        if age_rule == "max":
            return max(t1, t2)                       # youngest ancestor wins
        if age_rule == "heavier":
            return t1 if m1 >= m2 else t2
        return (m1 * t1 + m2 * t2) / (m1 + m2)       # mass-weighted
    # ------------------------------------------------------------------
    # 7.  State, counters, snapshot storage
    # ------------------------------------------------------------------
    t_phys = 0.0
    events = 0                 # accepted events
    tries = 0                  # majorant attempts
    tries_since_event = 0      # stall guard: see stop condition below
    inj_residual = 0.0         # [trick 6]
    # [v5] summed, not N0*m_init: with a seeded population m_init is only the
    # mean, and N0*mean differs from the true sum in the last few bits -- which
    # would show up as a non-zero mass_drift, the one number in this engine
    # that is supposed to be identically zero.
    M_sys = w * float(m_init_arr.sum())   # physical mass currently in the system
    M_in = 0.0
    M_out = 0.0
    m_max = float(m_init_arr.max())
    m_min = float(m_init_arr.min())
    sink_events = 0
    M_ref = M_sys                        # reference mass; drift must stay identically zero

    # ---- the distributed sink, note [16] ----
    # S_pow = sum over live particles of m^omega.  It is the ONLY population
    # quantity the siphon needs, it is maintained exactly by O(1) updates at
    # every mass change, and it cancels between the quota and the acceptance --
    # which is why the channel needs no histogram of any kind.
    S_pow = float(np.sum(m_init_arr ** omega)) if siphon_on else 0.0
    spow_drift = [0.0]                   # worst relative drift of that running sum
    siphon_residual = 0.0                # deterministic residual, exactly as [trick 6]
    siphon_events = 0                    # ACCEPTED removals, kept separate from sink_events
    siphon_tries = 0                     # draws spent on selection: acceptance diagnostic
    M_siphon = 0.0                       # physical mass eaten by the siphon
    siphon_counts = np.zeros(B, dtype=np.float64)   # where it ate, per mass bin
    # <|dln m|> per accepted event.  One multiplication per event, and it is what
    # turns the waiting law <dt>(m) into the drift time tau_drift = <dt>/c, i.e.
    # what makes delta0 = xi0 * tau_drift a number rather than a proportionality.
    dlnm_sum = 0.0
    dlnm_n = 0

    def removed_count():
        """Physical particles removed so far, by whichever route the two sink
        GATES have been told to count -- note [16].  The counters themselves
        stay separate everywhere else; only the gates ever add them, because
        only the gates ask "has the box turned over", a question to which the
        route of departure is irrelevant."""
        return w * (sink_events + siphon_events) if gate_counts_siphon \
            else w * sink_events

    # [v2] "n_out" is the sink counter as a TIME SERIES.  Deliberately NOT named
    # "sink_events": the packing step builds `out` from `rows` first and then
    # assigns the final scalar out["sink_events"], which would silently clobber
    # the array.  Two names, no collision.
    # "n_siphon" / "M_siphon" are the siphon's time series and are deliberately
    # kept APART from "n_out" / "M_out", which stay what they have always been:
    # what crossed the absorbing boundary.  Events answer "how many times", and
    # they are route-specific; mass answers "has it settled", and there the
    # routes add -- see "M_removed" and note [16].
    rows = {k: [] for k in ("t", "events", "live", "N_phys", "m_mean", "m_max",
                            "m_min", "M_sys", "M_in", "M_out", "n_out", "weight",
                            "honest_num", "honest_mass", "gen_mean", "gen_max_live",
                            "n_siphon", "M_siphon", "M_removed", "dndm")}

    # [v4] Generation histogram, gated exactly like the isochrones -- the same
    # statement that the steady state is up.  gen_num and gen_tau_sum give <tau>(g)
    # for free, which is the clock and therefore b.
    use_gen = bool(track_generations)
    G_MAX = int(gen_max)
    gen_counts = np.zeros((G_MAX + 1, B), dtype=np.float64) if use_gen else None
    gen_num = np.zeros(G_MAX + 1, dtype=np.float64) if use_gen else None
    gen_tau_sum = np.zeros(G_MAX + 1, dtype=np.float64) if use_gen else None
    gen_mass_sum = np.zeros((G_MAX + 1, B), dtype=np.float64) if use_gen else None
    gen_overflow = 0.0
    gen_snaps = 0
    # [v4] TWO different times per generation, and they are not interchangeable.
    #   gen_tau_live  -- mean age of the particles CURRENTLY at g, taken at snapshots.
    #                    It saturates at the residence time of the box: after one
    #                    turnover everything alive is young whatever its generation.
    #   gen_tau_reach -- age AT THE MOMENT the particle reached g, accumulated at the
    #                    event.  This is the "time to reach generation g", the direct
    #                    analogue of the tracer's tau(k), and the one that carries b.
    gen_reach_sum = np.zeros(G_MAX + 1, dtype=np.float64) if use_gen else None
    gen_reach_n = np.zeros(G_MAX + 1, dtype=np.float64) if use_gen else None

    def gen_reached(g, t_anc):
        """A particle has just arrived at generation g; record how long that took.

        [v5] g < 0 is the seeded-ancestor mark of note [13] and is silently
        skipped -- t_anc is -inf there, so the guard is load-bearing, not
        cosmetic."""
        if use_gen and 0 <= g <= G_MAX and removed_count() >= iso_start_sink \
                and t_phys >= iso_t_start:
            gen_reach_sum[g] += w * (t_phys - t_anc)
            gen_reach_n[g] += w

    # [v5] THE WAITING LAW, note [14].  <dt>(m) per mass bin, accumulated at
    # the moment each interval ENDS, which is what makes it unbiased.  Two
    # copies: ungated (from t = 0, the one a warm start is for) and gated by
    # the same steady-state gate as the isochrones, so the size of the
    # transient contamination is a measurement rather than a hope.
    # TWO estimators of the same <dt>, and the difference between them is
    # itself the diagnostic:
    #
    #   INTERVAL   sum of completed intervals / their number.  Simple, but
    #              CENSORED: an interval still running when the run stops is
    #              never recorded, and long intervals are preferentially the
    #              ones still running.  Since <dt> grows with m, the loss is
    #              concentrated at large m and it FLATTENS the slope -- i.e.
    #              it inflates b.  Same disease as tau(g) in v4, milder.
    #
    #   EXPOSURE   total particle-time spent in the bin / number of events
    #              that ended a stay there.  This is the standard
    #              occupancy/exposure ratio, and it is unbiased WHATEVER the
    #              censoring, because a partly-elapsed interval contributes
    #              its elapsed part to the numerator and nothing to the
    #              denominator, which is exactly right.  Use this one.
    #
    # The occupancy integral Int N_b(t) dt is exact, not sampled: N_bin is
    # piecewise constant in time and every change to it goes through
    # apply_delta, so flushing there costs one length-B add per EVENT (not
    # per trial -- within one event dt is zero and the add is skipped).
    use_wait = bool(track_waiting)
    wait_sum = np.zeros(B, dtype=np.float64) if use_wait else None
    wait_sq = np.zeros(B, dtype=np.float64) if use_wait else None
    wait_n = np.zeros(B, dtype=np.float64) if use_wait else None
    wait_sum_g = np.zeros(B, dtype=np.float64) if use_wait else None
    wait_sq_g = np.zeros(B, dtype=np.float64) if use_wait else None
    wait_n_g = np.zeros(B, dtype=np.float64) if use_wait else None
    occ = np.zeros(B, dtype=np.float64) if use_wait else None      # Int N_b dt * w
    occ_g = np.zeros(B, dtype=np.float64) if use_wait else None
    wait_ev = np.zeros(B, dtype=np.float64) if use_wait else None  # stays ended in b
    wait_ev_g = np.zeros(B, dtype=np.float64) if use_wait else None
    t_occ_last = 0.0
    t_occ_gate = np.nan          # when the gated occupancy started

    def occ_flush():
        """Add the particle-time accrued since the last flush.  Must run BEFORE
        any change to N_bin, which is why it lives at the top of apply_delta --
        the single choke point every count change passes through."""
        nonlocal t_occ_last, t_occ_gate
        if not use_wait:
            return
        dtl = t_phys - t_occ_last
        if dtl > 0.0:
            # `occ[:] +=`, not `occ +=`: the bare form would rebind the name and
            # make it a local of this closure.
            _inc = N_bin * (w * dtl)
            occ[:] += _inc
            if removed_count() >= iso_start_sink and t_phys >= iso_t_start:
                if not np.isfinite(t_occ_gate):
                    t_occ_gate = t_occ_last
                occ_g[:] += _inc
        t_occ_last = t_phys

    def wait_close(i):
        """Particle i is about to have its mass changed: close its stay.

        The particle has sat at a CONSTANT mass since t_change[i], so the whole
        interval belongs to bin_of[i] with no ambiguity -- that is the reason
        this observable is local while tau(g) is not."""
        if not use_wait:
            return
        b_i = int(bin_of[i])
        gated = (removed_count() >= iso_start_sink and t_phys >= iso_t_start)
        wait_ev[b_i] += w                      # exposure estimator: denominator
        if gated:
            wait_ev_g[b_i] += w
        dt_i = t_phys - t_change[i]
        if dt_i > 0.0:                         # interval estimator: cross-check
            wait_sum[b_i] += w * dt_i
            wait_sq[b_i] += w * dt_i * dt_i
            wait_n[b_i] += w
            if gated:
                wait_sum_g[b_i] += w * dt_i
                wait_sq_g[b_i] += w * dt_i * dt_i
                wait_n_g[b_i] += w
        t_change[i] = t_phys

    if use_wait:
        _occ_hook[0] = occ_flush          # arm it; see apply_delta

    use_iso = iso_age_edges is not None
    if use_iso:
        iso_age_edges = np.asarray(iso_age_edges, dtype=float)
        iso_counts = np.zeros((iso_age_edges.size - 1, B), dtype=np.float64)
        iso_snaps = 0
    else:
        iso_counts = None
        iso_snaps = 0
    iso_t_begin = np.nan       # physical time at which accumulation actually started

    def snapshot():
        nonlocal iso_snaps, iso_t_begin, gen_snaps, gen_overflow, S_pow
        N_phys = w * live
        rows["t"].append(t_phys)
        rows["events"].append(events)
        rows["live"].append(live)
        rows["N_phys"].append(N_phys)
        rows["m_mean"].append(M_sys / N_phys if N_phys > 0 else np.nan)
        rows["m_max"].append(m_max)
        rows["m_min"].append(m_min)
        rows["M_sys"].append(M_sys)
        rows["M_in"].append(M_in)
        rows["M_out"].append(M_out)
        rows["n_out"].append(w * sink_events)        # [v2] physical particles absorbed so far
        rows["n_siphon"].append(w * siphon_events)   # note [16]: a separate counter, always
        rows["M_siphon"].append(M_siphon)
        # THE STATIONARITY NUMERATOR.  Siphon EVENTS say nothing about whether
        # the cascade has settled -- a strong siphon eats next to the injection
        # scale and its counter ticks while nothing has crossed the range.  Mass
        # does say it, because mass is conserved whichever way it left.  So the
        # criterion is M_removed/M_in -> 1, not M_out/M_in.
        rows["M_removed"].append(M_out + M_siphon)
        rows["weight"].append(w)
        # [v5] What fraction of the population still has an honest history,
        # note [13].  On a cold run this is 1.0 at every snapshot by
        # construction; on a warm one it is the direct readout of how far the
        # seeded bodies have been flushed, i.e. of how much of the run is
        # actually usable for anything that depends on ancestry.
        if live > 0:
            _hm = gen[:live] >= 0
            rows["honest_num"].append(float(np.count_nonzero(_hm)) / live)
            _mm = mass[:live]
            _tot = float(_mm.sum())
            rows["honest_mass"].append(float(_mm[_hm].sum()) / _tot if _tot > 0 else np.nan)
        else:
            rows["honest_num"].append(np.nan)
            rows["honest_mass"].append(np.nan)
        # The depth of the cascade in SPLITS.  In fragmentation this is the half
        # of the front that m_max cannot report, m_max being identically constant
        # there.  The MEAN over honest bodies is the one to watch -- it rises and
        # then flattens when the steady state is up -- while the maximum is a
        # single particle, i.e. noise, and is kept only as an extremum.
        if use_gen and live > 0:
            _gl = gen[:live]
            _gh = _gl[_gl >= 0]
            rows["gen_mean"].append(float(_gh.mean()) if _gh.size else np.nan)
            rows["gen_max_live"].append(float(_gh.max()) if _gh.size else np.nan)
        else:
            rows["gen_mean"].append(np.nan)
            rows["gen_max_live"].append(np.nan)
        # The running S_pow is a sum of some 10^6 signed increments, so it is
        # re-derived exactly here, once per snapshot, and the worst relative
        # discrepancy is reported as out["siphon_spow_drift"].  A few 1e-9 over
        # a hundred thousand events is ordinary cancellation in float64 and is
        # harmless -- the rate is reset to the exact value right here.  Orders
        # of magnitude above that mean a mass change somewhere does not update
        # S_pow, which is an engine bug.
        if siphon_on:
            _sp_exact = float(np.sum(mass[:live] ** omega)) if live > 0 else 0.0
            _den = abs(_sp_exact) if _sp_exact != 0.0 else 1.0
            spow_drift[0] = max(spow_drift[0], abs(S_pow - _sp_exact) / _den)
            S_pow = _sp_exact
        # dN/dm in PHYSICAL units: counts * w / bin width / volume
        rows["dndm"].append(N_bin * (w / V) / widths)

        # Two gates, both permissive by default.  iso_start_sink is the physical one:
        # it waits until the cascade has actually delivered mass to the sink, which is
        # the only N-independent statement of "the steady state is up".
        if (use_gen and live > 0 and t_phys >= iso_t_start
                and removed_count() >= iso_start_sink):
            _g = gen[:live]
            # [v5] THREE classes now, not two: honest and resolved, honest and
            # overflowing, and dishonest (gen < 0, note [13]).  The dishonest
            # ones must not land in gen_overflow either -- that counter is a
            # statement about gen_max being too small, and mixing the seeded
            # population into it would make it read as a truncation warning
            # for the whole first residence time.
            _hon = _g >= 0
            _ok = _hon & (_g <= G_MAX)
            gen_overflow += w * float(np.count_nonzero(_hon & (_g > G_MAX)))
            _gg = _g[_ok].astype(np.int64)
            _bb = bin_of[:live][_ok].astype(np.int64)
            np.add.at(gen_counts, (_gg, _bb), w)
            np.add.at(gen_mass_sum, (_gg, _bb), w * mass[:live][_ok])
            np.add.at(gen_num, _gg, w)
            np.add.at(gen_tau_sum, _gg, w * (t_phys - gen_time[:live][_ok]))
            gen_snaps += 1

        if (use_iso and live > 0 and t_phys >= iso_t_start
                and removed_count() >= iso_start_sink):
            if iso_snaps == 0:
                iso_t_begin = t_phys
            ages = t_phys - inj_time[:live]
            H, _, _ = np.histogram2d(ages, mass[:live], bins=[iso_age_edges, edges])
            iso_counts[:] += H * w
            iso_snaps += 1

    if snapshot_first_at_start:
        snapshot()

    if snapshot_mode not in ("events", "t_phys", "log_m0"):
        raise ValueError("snapshot_mode must be 'events', 't_phys' or 'log_m0'")
    # log_m0: fire whenever m0 has moved by a factor 10^(1/snapshot_stride)
    log_step = 1.0 / float(snapshot_stride) if snapshot_mode == "log_m0" else None
    m0_last  = (M_sys / (w * live)) if live > 0 else m_init
    next_thr = (events if snapshot_mode == "events" else t_phys) + float(snapshot_stride)

    t_wall0 = _time.perf_counter()
    n_snap_printed = 0

    # ==================================================================
    # 8.  MAIN LOOP
    # ==================================================================
    while True:
        # ---- stopping conditions -------------------------------------
        if live < 2 or R <= 0.0:
            stop_reason = "population exhausted"
            break
        if t_phys >= max_time:
            stop_reason = "max_time"
            break
        if events >= max_events:
            stop_reason = "max_events"
            break
        if stop_max_mass is not None and m_max >= stop_max_mass:
            stop_reason = "stop_max_mass"
            break
        if stop_min_mass is not None and rows["m_mean"] and rows["m_mean"][-1] <= stop_min_mass:
            stop_reason = "stop_min_mass"
            break
        # [v2] The PHYSICAL brake.  w * sink_events, not the bare counter, so the
        # criterion stays in physical particles and survives the move to BF.py,
        # where w != 1.  Same convention as the iso_start_sink gate below.
        if stop_sink_events is not None and removed_count() >= stop_sink_events:
            stop_reason = "stop_sink_events"
            break
        # Stall guard.  A passive floor (min_frag_mass) or a collapsed acceptance can
        # leave the majorant firing while no event is ever accepted; without this the
        # loop spins forever, since max_events is then never reached.
        if tries_since_event > max_stall_tries:
            stop_reason = "acceptance collapsed (stall guard)"
            break

        # ---- THE CLOCK  [note 5] -------------------------------------
        # R counts ORDERED pairs, hence the factor 2; w converts the
        # simulated sub-volume to the physical one.  Quadratic in N by
        # construction, because R ~ N^2.
        dt = 2.0 * V / (w * R)
        t_phys += dt
        tries += 1
        tries_since_event += 1

        # ---- deterministic injection  [trick 6] ----------------------
        if injection_rate > 0.0:
            inj_residual += injection_rate * dt / w      # in SIMULATED particles
            k = int(inj_residual)
            if k > 0:
                inj_residual -= k
                ensure_capacity(live + k)
                for _ in range(k):
                    i_inj = add_particle(float(injection_mass), t_phys, 0, t_phys)
                M_sys += w * k * injection_mass
                M_in += w * k * injection_mass
                if injection_mass > m_max:
                    m_max = float(injection_mass)
                if injection_mass < m_min:
                    m_min = float(injection_mass)

        # ---- the distributed sink  [note 16] -------------------------
        # The same deterministic-residual device as the injector above.  The
        # quota is the exact integral of the total removal rate over the step,
        # and only its integer part fires, so the channel carries no shot noise
        # of its own -- which matters more here than at the injection scale,
        # because this sink sits INSIDE the inertial range and its noise would
        # land straight on the measured slope.
        #
        # The total rate is xi_amp * w * S_pow physical particles per unit time;
        # dividing by w to get SIMULATED particles cancels the weight exactly.
        # The injector keeps a 1/w because its rate is imposed from outside,
        # while this one is assembled from the population itself.
        if siphon_on:
            siphon_residual += xi_amp * S_pow * dt
            k_s = int(siphon_residual)
            if k_s > 0:
                k_cap = min(k_s, live - 1)     # never empty the box in a single step
                siphon_residual -= k_cap       # the remainder stays owed, not lost
                for _ in range(k_cap):
                    # THE SELECTION.  Uniform draw, then rejection against the
                    # static bound of section 0c.  The normalisation S_pow
                    # cancels between the quota and the acceptance, so what this
                    # implements is exactly P(i) = xi(m_i) dt for every particle
                    # independently -- a loss linear in n, which is what the
                    # analytic family of note [16] is built on.
                    n_try = 0
                    while True:
                        js = int(rint(0, live))
                        siphon_tries += 1
                        n_try += 1
                        if omega == 0.0:
                            break                      # flat rate: no thinning at all
                        acc = (float(mass[js]) / m_bound) ** omega
                        if acc > 1.0 + 1e-12:
                            raise RuntimeError(
                                f"Siphon bound violated: (m/m_bound)^omega = {acc:.6g} "
                                f"for m = {float(mass[js]):.4g}, m_bound = {m_bound:.4g}, "
                                f"omega = {omega:.4g}.  See note [16]."
                            )
                        if rnd() < acc:
                            break
                        if n_try > 1_000_000:
                            raise RuntimeError(
                                "Siphon selection is not converging: a million draws "
                                "without an acceptance.  m_bound = %.4g is far outside "
                                "the live population, see note [16]." % m_bound
                            )
                    m_s = float(mass[js]); b_s = int(bin_of[js])
                    # NO wait_close(js) here, and that is deliberate.  Removal by
                    # the siphon is not a mass-changing event but a CENSORING of
                    # the interval in progress, note [14].  The occupancy this
                    # particle has accrued is already in `occ` -- delete_particle
                    # goes through apply_delta, which flushes it before N_bin
                    # moves -- while the unfinished stay must not enter the
                    # denominator.  That is what keeps the `exposure` estimator
                    # unbiased under the siphon while `interval` is pulled down;
                    # the gap between the two then measures the strength of the
                    # sink instead of hiding it.
                    delete_particle(js)
                    M_sys -= w * m_s
                    M_siphon += w * m_s
                    siphon_counts[b_s] += w
                    siphon_events += 1
                if live < 2:
                    continue                   # the stopping block at the top will catch it

        # ---- pick a bin pair, then two particles ---------------------
        b = sample_row()
        c = sample_col(b)
        if c is None:
            continue
        if b != c:
            ib, jc = bins[b], bins[c]
            if not ib or not jc:
                continue
            i = ib[rint(0, len(ib))]
            j = jc[rint(0, len(jc))]
        else:
            ib = bins[b]
            if len(ib) < 2:
                continue
            p1 = rint(0, len(ib))
            p2 = rint(0, len(ib) - 1)
            if p2 >= p1:
                p2 += 1
            i, j = ib[p1], ib[p2]

        # ---- thin against the bin-pair majorant  [trick 1] -----------
        m1 = float(mass[i]); m2 = float(mass[j])
        # ---- THE LOCALITY WINDOW, exactly  [v7, note [17]] -----------
        #  The band above removes the bin pairs that are dead as a whole;
        #  bins have width, so the pairs on the edge of the band are only
        #  PARTLY allowed and have to be tested one by one.  Both tests
        #  are the same statement -- m_small >= f * m_large -- and this is
        #  the one that makes it exact.
        #
        #  It sits BEFORE the majorant check on purpose.  A banded-out
        #  pair has fmaj = 0 while K > 0, which would trip the violation
        #  guard below and report a majorant bug that is not one.  Tested
        #  here, such a pair is simply a null event, and the guard keeps
        #  its meaning for the pairs that are genuinely live.
        if loc_ratio > 0.0:
            if m1 <= m2:
                if m1 < loc_ratio * m2:
                    continue                             # outside the window
            elif m2 < loc_ratio * m1:
                continue
        f_now = float(kernel(m1, m2))
        f_cap = float(fmaj[b, c])
        if f_now > f_cap * (1.0 + 1e-12):
            raise RuntimeError(
                f"Majorant violated: K({m1:.4g},{m2:.4g})={f_now:.4g} > "
                f"fmaj[{b},{c}]={f_cap:.4g}.  The corner bound assumes a kernel "
                f"non-decreasing in both arguments."
            )
        if f_cap <= 0.0 or rnd() >= f_now / f_cap:
            continue                                     # null (thinned) event

        # ==============================================================
        #  ACCEPTED EVENT -- the only place the four cases differ
        # ==============================================================
        if process == "coagulation":
            mn = m1 + m2
            # [v5] Close BOTH intervals before either mass changes, note [14].
            # j is closed too even though it is about to be deleted: the time
            # it spent at m2 is data, and throwing it away would bias <dt>
            # against exactly the bins that merge most often.
            wait_close(i); wait_close(j)
            inj_time[i] = inherit_age(inj_time[i], inj_time[j], m1, m2)
            # [v4] one split deeper than the deeper parent; inherits the OLDER ancestor
            # [v5] unless either parent is seeded, in which case the product is
            # seeded too and the count is meaningless, note [13].
            if gen[i] < 0 or gen[j] < 0:
                gen[i] = -1
                gen_time[i] = -np.inf
            else:
                gen[i] = (gen[i] if gen[i] >= gen[j] else gen[j]) + 1
                if gen_time[j] < gen_time[i]:
                    gen_time[i] = gen_time[j]
            if siphon_on:
                # m2 leaves the sum inside delete_particle(j) just below
                S_pow += mn ** omega - m1 ** omega            # note [16]
            dlnm_sum += abs(np.log(mn / m1)); dlnm_n += 1     # calibration constant c
            mass[i] = mn
            move_bin(i, mass_to_bin(mn))
            delete_particle(j)
            events += 1; tries_since_event = 0
            if mn > m_max:
                m_max = mn
            # absorbing sink at LARGE mass (open systems only)
            if sink_mass is not None and mn >= sink_mass:
                # index i may have been relocated by delete_particle(j)
                i_now = i if i < live else j
                delete_particle(i_now)
                M_sys -= w * mn
                M_out += w * mn
                sink_events += 1

        else:  # ---------------- fragmentation ----------------------
            # the more massive of the pair breaks; the other is untouched
            ip = i if m1 >= m2 else j
            mp = max(m1, m2)
            ms = min(m1, m2)
            if min_frag_mass is not None and mp < min_frag_mass:
                continue                                 # passive floor: no event
            #  [v7] The impactor-too-small test USED TO BE HERE.  It now lives
            #  with the coagulation one, above the majorant thinning, because
            #  one rule enforced in two places is a rule that eventually
            #  disagrees with itself.  Nothing about fragmentation changed
            #  except WHEN the test is applied, and that moves it to before a
            #  random draw rather than after -- so a fragmentation run with
            #  frag_min_ratio > 0 is statistically identical to v6 but not
            #  bit-identical.  With frag_min_ratio = 0 nothing moves at all.
            xi = 0.5 + frag_split_width * (2.0 * rnd() - 1.0)
            ma, mb = xi * mp, (1.0 - xi) * mp
            t_par = float(inj_time[ip])                  # the parent's clock
            # [v5] note [14].  ONLY the parent is closed: in this process the
            # impactor is untouched, its mass does not change, and its own
            # interval is still running.  Closing it here would count a
            # non-event and pull <dt> down in every bin that acts as an
            # impactor -- which, with a power law, is all of them.
            wait_close(ip)

            # Which piece carries the parent's clock and which starts a new one.
            # 'inherit' reproduces v2 exactly; everything else resets one of the two
            # to t_phys, which is ALREADY the current time here -- dt was added at the
            # top of the trial loop, the same t_phys the injector stamps on
            # new monomers.  Do not recompute it.
            if frag_age_rule == "inherit":
                t_a = t_b = t_par
            elif frag_age_rule == "both_new":
                t_a = t_b = t_phys
            else:
                keep_a = (ma >= mb) if frag_age_rule == "heavier" else (ma < mb)
                t_a, t_b = (t_par, t_phys) if keep_a else (t_phys, t_par)

            if siphon_on:
                # mb enters the sum inside add_particle(mb, ...) just below
                S_pow += ma ** omega - mp ** omega            # note [16]
            dlnm_sum += abs(np.log(ma / mp)); dlnm_n += 1     # calibration constant c
            mass[ip] = ma
            inj_time[ip] = t_a          # <-- MUST be written now: the slot is REUSED and
                                        #     under v2 it was silently already correct.
            move_bin(ip, mass_to_bin(ma))
            # [v4] BOTH fragments go one generation deeper and keep the ancestor's
            # clock, whatever frag_age_rule did to inj_time -- gen_time is a separate
            # field precisely so the two cannot interfere.
            # [v5] -1 is contagious and never heals, note [13]: a fragment of a
            # seeded body is as historyless as the body was.
            _g0 = int(gen[ip])
            _g1 = -1 if _g0 < 0 else _g0 + 1
            _tanc = float(gen_time[ip])
            gen[ip] = _g1
            i_new = add_particle(mb, t_b, _g1, _tanc)
            gen_reached(_g1, _tanc)      # both fragments arrived at the same time
            gen_reached(_g1, _tanc)      # (no-op when _g1 < 0)
            events += 1; tries_since_event = 0
            if ma < m_min: m_min = ma
            if mb < m_min: m_min = mb
            # absorbing sink at SMALL mass (open systems only).
            # Delete the higher index first so the swap-with-last cannot
            # invalidate the other index.
            if sink_mass is not None:
                doomed = []
                if mb <= sink_mass: doomed.append((i_new, mb))
                if ma <= sink_mass: doomed.append((ip, ma))
                for idx, mm in sorted(doomed, key=lambda z: -z[0]):
                    delete_particle(idx)
                    M_sys -= w * mm
                    M_out += w * mm
                    sink_events += 1

        # ---- snapshots ----------------------------------------------
        if snapshot_mode == "log_m0":
            # PLACEMENT: uniform in log m0, so every decade is resolved [note 10].
            # The WEIGHT used later is still the actual dt between snapshots.
            m0_now = (M_sys / (w * live)) if live > 0 else np.nan
            fire = (np.isfinite(m0_now) and m0_now > 0
                    and abs(np.log10(m0_now / m0_last)) >= log_step)
            if fire:
                m0_last = m0_now
            progress = 1.0 if fire else 0.0
            next_thr = 1.0
        else:
            progress = events if snapshot_mode == "events" else t_phys
        while progress >= next_thr:
            snapshot()
            if snapshot_mode == "log_m0":
                n_snap_printed += 1
                break
            next_thr += float(snapshot_stride)
            n_snap_printed += 1
            if verbose and (n_snap_printed % verbose_every == 0):
                # Print the LEADING EDGE of the process at hand.  A fragmentation
                # cascade advances downward in mass and downward through the
                # generations, and its m_max is identically constant -- nothing
                # ever gets heavier than what entered -- so printing m_max there
                # would be printing a constant.  In coagulation m_min is the dead
                # one, by the same argument mirrored.
                if process == "fragmentation":
                    _edge = (f"m_min={m_min:.3g} | "
                             f"<g>={rows['gen_mean'][-1]:.2f}"
                             f"(max {rows['gen_max_live'][-1]:.0f})")
                else:
                    _edge = f"m_max={m_max:.3g}"
                _siph = f" | siphon={siphon_events}" if siphon_on else ""
                print(f"[{_time.strftime('%H:%M:%S')}] "
                      f"t={t_phys:.6g} | events={events:.3e} | live={live} | "
                      f"<m>={rows['m_mean'][-1]:.4g} | {_edge} | "
                      f"sink={sink_events}{_siph} | acc={events/max(tries,1):.3f} | "
                      f"cpu={_time.perf_counter()-t_wall0:.1f}s")
            progress = events if snapshot_mode == "events" else t_phys
    else:
        stop_reason = "loop exit"

    if use_wait:
        occ_flush()   # [v5] close the occupancy integral at t_end, note [14]

    snapshot()   # always record the final state

    # ------------------------------------------------------------------
    # 9.  Pack results
    # ------------------------------------------------------------------
    out = {k: np.asarray(v, dtype=float) for k, v in rows.items() if k != "dndm"}
    out["dndm"] = np.vstack(rows["dndm"]) if rows["dndm"] else np.zeros((0, B))
    out["centers"] = centers
    out["edges"] = edges
    out["widths"] = widths
    out["final_mass"] = mass[:live].copy()
    out["final_inj_time"] = inj_time[:live].copy()
    out["final_t_phys"] = float(t_phys)
    out["tries"] = float(tries)
    out["acceptance"] = float(events / max(tries, 1))
    out["stop_reason"] = stop_reason
    out["final_weight"] = float(w)          # constant by construction
    # with no resampling this is exactly zero; anything else is a bug
    # THE CONSERVATION IDENTITY of note [9], with both removal routes on the
    # outgoing side:  M_sys = M_ref + M_in - (M_out + M_siphon).  It is still
    # identically zero, so a non-zero value here is still a bug and never a
    # fluctuation -- that property is the reason the baseline exists and the
    # siphon was not allowed to cost it.
    out["mass_drift"] = float((M_sys - M_in + M_out + M_siphon - M_ref)
                              / max(M_ref, 1e-300))
    out["sink_events"] = float(sink_events)
    # ---- the distributed sink, note [16].  Scalars, deliberately named apart
    # from the rows arrays "n_siphon"/"M_siphon"/"M_removed" for the same reason
    # "sink_events" is named apart from "n_out": the packing step writes the
    # arrays first, and a name collision would silently clobber one of them.
    out["siphon_events"] = float(siphon_events)
    out["M_siphon_total"] = float(M_siphon)
    out["M_removed_total"] = float(M_out + M_siphon)
    out["siphon_counts"] = siphon_counts
    out["siphon_acceptance"] = float(siphon_events / max(siphon_tries, 1))
    out["siphon_spow_drift"] = float(spow_drift[0])
    out["siphon_xi0"] = float(xi0)
    out["siphon_omega"] = float(omega)
    out["siphon_m_bound"] = float(m_bound)
    # <|dln m|> per accepted event: the constant c that turns the waiting law
    # <dt>(m) into the drift time tau_drift = <dt>/c.  Measure it on a pilot run
    # with siphon=None and feed it back as siphon={'dlnm': c, ...}; without it
    # delta0 is defined only up to a factor of order unity.
    out["dlnm_mean"] = float(dlnm_sum / dlnm_n) if dlnm_n else np.nan
    if use_iso:
        out["iso_age_edges"] = iso_age_edges
        out["iso_counts"] = iso_counts
        out["iso_dndm"] = iso_counts / widths[None, :]
        out["iso_snapshots"] = float(iso_snaps)
        out["iso_t_begin"] = float(iso_t_begin)
        # An empty isochrone histogram is the silent failure mode: every downstream
        # quantity comes back as nan and nothing says why.  Say why.
        if iso_snaps == 0:
            warnings.warn(
                "isochrones requested but NEVER accumulated: the run ended at "
                "t = %.3g with %d sink absorptions, while the gates ask for "
                "t >= %.3g and >= %g absorptions.  iso_counts is all zeros, so "
                "<m>(tau) and b will be nan.  The clock is t = 2*attempts/N^2, so "
                "raising N at fixed max_events SHORTENS the run."
                % (t_phys, sink_events, iso_t_start, iso_start_sink),
                RuntimeWarning, stacklevel=2)

    # ---- generations -----------------------------------------------------------
    if use_gen:
        out["gen_counts"] = gen_counts
        out["gen_dndm"] = gen_counts / widths[None, :] / max(gen_snaps, 1)
        out["gen_mass"] = gen_mass_sum / max(gen_snaps, 1)
        out["gen_num"] = gen_num
        with np.errstate(invalid="ignore", divide="ignore"):
            out["gen_tau_live"] = np.where(gen_num > 0, gen_tau_sum / gen_num, np.nan)
            out["gen_tau"] = np.where(gen_reach_n > 0, gen_reach_sum / gen_reach_n, np.nan)
        out["gen_reach_n"] = gen_reach_n
        out["gen_snapshots"] = float(gen_snaps)
        out["gen_overflow"] = float(gen_overflow)
        out["gen_index"] = np.arange(G_MAX + 1, dtype=float)
        out["final_gen"] = gen[:live].astype(float)
        out["final_gen_time"] = gen_time[:live].copy()
        # [v5] how much of the seeded population is left at the end, note [13]
        _fh = gen[:live] >= 0
        out["final_honest_num"] = (float(np.count_nonzero(_fh)) / live) if live else np.nan
        if seeded and live and np.count_nonzero(_fh) == 0:
            warnings.warn(
                "the run ended with NO honest bodies at all: every particle still "
                "descends from the seed, so gen_counts / gen_tau are empty and the "
                "only usable estimate of b is the waiting law.  The run is shorter "
                "than one residence time.", RuntimeWarning, stacklevel=2)
        if gen_snaps == 0:
            warnings.warn(
                "generations requested but NEVER accumulated: the run ended at t = %.3g "
                "with %d sink absorptions, while the gate asks for t >= %.3g and >= %g "
                "absorptions.  gen_counts is all zeros."
                % (t_phys, sink_events, iso_t_start, iso_start_sink),
                RuntimeWarning, stacklevel=2)
        elif gen_overflow > 0:
            warnings.warn(
                "%.3g particle-snapshots had gen > gen_max = %d and were NOT binned "
                "(%.2f%% of the total).  Raise gen_max, or read the histogram as "
                "truncated." % (gen_overflow, G_MAX,
                                100 * gen_overflow / max(gen_num.sum() + gen_overflow, 1.0)),
                RuntimeWarning, stacklevel=2)

    # ---- [v5] the waiting law, note [14] ---------------------------------------
    if use_wait:
        with np.errstate(invalid="ignore", divide="ignore"):
            # --- EXPOSURE estimator: particle-time per event.  The primary one.
            out["wait_mean"] = np.where(wait_ev > 0, occ / wait_ev, np.nan)
            out["wait_mean_gated"] = np.where(wait_ev_g > 0, occ_g / wait_ev_g, np.nan)
            # Poisson error on the COUNT propagates straight through, since the
            # exposure is a measured time and not a random variable here:
            # sd(1/nu)/(1/nu) = 1/sqrt(n_events).
            out["wait_sem"] = out["wait_mean"] / np.sqrt(np.maximum(wait_ev, 1.0))
            out["wait_sem_gated"] = (out["wait_mean_gated"]
                                     / np.sqrt(np.maximum(wait_ev_g, 1.0)))
            # --- INTERVAL estimator: the censored cross-check.
            _mi = np.where(wait_n > 0, wait_sum / wait_n, np.nan)
            _mig = np.where(wait_n_g > 0, wait_sum_g / wait_n_g, np.nan)
            _var = np.where(wait_n > 0, wait_sq / wait_n - _mi ** 2, np.nan)
            out["wait_mean_interval"] = _mi
            out["wait_mean_interval_gated"] = _mig
            out["wait_sem_interval"] = np.sqrt(np.maximum(_var, 0.0)
                                               / np.maximum(wait_n, 1.0))
        out["wait_events"] = wait_ev
        out["wait_events_gated"] = wait_ev_g
        out["wait_occupancy"] = occ
        out["wait_occupancy_gated"] = occ_g
        out["wait_n"] = wait_n
        out["wait_n_gated"] = wait_n_g
        out["wait_t_gate"] = float(t_occ_gate)
        # FREE BONUS, and a better spectrum than any single snapshot: the
        # occupancy integral IS the time-averaged number per bin.  Dividing by
        # the elapsed time and the bin width gives a dN/dm averaged over the
        # whole run rather than sampled at one instant, so it is smooth where
        # the snapshots are ragged.  Two versions, ungated and gated; on a warm
        # start the gated one is the physical spectrum.
        _T = float(t_phys)
        _Tg = (float(t_phys) - t_occ_gate) if np.isfinite(t_occ_gate) else np.nan
        with np.errstate(invalid="ignore", divide="ignore"):
            out["dndm_time_avg"] = occ / max(_T, 1e-300) / widths / V
            out["dndm_time_avg_gated"] = occ_g / _Tg / widths / V if _Tg and _Tg > 0 \
                else np.full(B, np.nan)
        out["wait_t_span"] = _T
        out["wait_t_span_gated"] = float(_Tg)
        if float(wait_ev.sum()) <= 0.0:
            warnings.warn(
                "the waiting law was requested but no stay ever ended: "
                "wait_mean is all nan.  Either the run accepted no events, or "
                "track_waiting was toggled after the fact.",
                RuntimeWarning, stacklevel=2)

    out["meta"] = {
        "process": process,
        "system": system,
        "kernel": getattr(kernel, "__name__", "kernel"),
        "lambda": KERNEL_LAMBDA.get(getattr(kernel, "__name__", ""), None),
        "ic": {"m": m_init, "N": N0},
        # [v5] the warm start.  `seeded` is the one flag every downstream plot
        # should read before it believes anything about ancestry.
        "engine": "masscascade.py",
        "seeded": bool(seeded),
        "seed": seed_info,
        # Provenance of the run.  Restarts compose, and t_origin is the
        # physical time accumulated in ALL previous legs, so that
        # t_origin + final_t_phys is the full age of the box.
        "restarted_from": restart_from,
        "t_origin": float(t_origin),
        "track_waiting": bool(use_wait),
        "injection_rate": float(injection_rate),
        "injection_mass": float(injection_mass),
        "sink_mass": None if sink_mass is None else float(sink_mass),
        # ---- the distributed sink, note [16] ----
        "siphon": siphon_spec,                 # as the caller wrote it
        "siphon_on": bool(siphon_on),
        "siphon_xi0": float(xi0),              # and as the engine resolved it
        "siphon_omega": float(omega),
        "siphon_m_ref": float(m_ref_siphon),
        "siphon_nu": None if nu_siphon is None else float(nu_siphon),
        "siphon_delta0": None if delta0_siphon is None else float(delta0_siphon),
        "siphon_m_bound": float(m_bound),
        "siphon_scheme": "deterministic residual quota, population-level rejection",
        "gate_counts_siphon": bool(gate_counts_siphon),
        "min_frag_mass": None if min_frag_mass is None else float(min_frag_mass),
        "frag_min_ratio": float(frag_min_ratio),
        "coag_min_ratio": float(coag_min_ratio),
        "loc_ratio": float(loc_ratio),          # the f actually in force [v7]
        "loc_band_bins": int(n_band),           # -1 when the band is off
        "frag_split_width": float(frag_split_width),   
        "frag_age_rule": frag_age_rule,     
        "age_rule": age_rule,
        "track_generations": bool(use_gen),
        "gen_max": int(G_MAX),
        "weight": w,
        "volume": V,
        "clock": "dt = 2*V/(w*R), R over ordered pairs",
        "resampling": "none - baseline variant, w constant",
        "injection_scheme": "deterministic residual (no Poisson noise)",
        "stop_reason": stop_reason,
        "max_stall_tries": int(max_stall_tries),
        # [v2] both sink criteria recorded, so a saved run says what gated it
        "iso_start_sink": int(iso_start_sink),
        "stop_sink_events": None if stop_sink_events is None else float(stop_sink_events),
    }
    return out


# ======================================================================
#  ANALYSIS HELPERS
# ======================================================================

def superpose(dndm, times):
    """
    Age-integrated spectrum  F(m) = sum_t (dN/dm)(m,t) * dt.

    This is the correct estimator for a CLOSED system, where no
    stationary state exists and the observable spectrum is the
    superposition of independently evolving parcels with a stationary
    age distribution.  For an OPEN system the steady state is the
    instantaneous spectrum and this is NOT what you want -- use the
    last snapshot instead.
    """
    D = np.asarray(dndm, float)
    t = np.asarray(times, float).ravel()
    dt = np.diff(t, prepend=t[0])
    return (D * dt[:, None]).sum(axis=0)


def fit_powerlaw(x, y, xmin, xmax):
    """Least-squares fit y ~ A x^alpha over [xmin,xmax]; returns dict."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = (x >= xmin) & (x <= xmax) & np.isfinite(x) & np.isfinite(y) & (y > 0)
    if m.sum() < 3:
        return {"alpha": np.nan, "A": np.nan, "r2": np.nan, "n": int(m.sum())}
    lx, ly = np.log10(x[m]), np.log10(y[m])
    s, c = np.polyfit(lx, ly, 1)
    pred = c + s * lx
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    return {"alpha": float(s), "A": float(10 ** c),
            "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
            "n": int(m.sum())}


def local_slope(m, F, half=3, min_val=0.0):
    """
    Local logarithmic slope  Gamma(m) = d log F / d log m,  computed by least
    squares on a sliding window of (2*half+1) bins.

    This is the honest way to look at a measured spectrum: a genuine inertial
    range shows up as a PLATEAU in Gamma, and the contaminated ends show up as
    the places where Gamma bends.  Fitting a single power law across the whole
    array averages the plateau together with the bends and returns a number
    that describes neither.  (The same diagnostic is used as the lower subpanel
    of every spectrum figure in Laor & Gitelman, Phys. Rev. E 113, 044135.)

    Returns (m_valid, Gamma) with NaN where the window is not usable.
    """
    m = np.asarray(m, float); F = np.asarray(F, float)
    ok = np.isfinite(m) & np.isfinite(F) & (F > min_val) & (m > 0)
    lm, lF = np.log10(m), np.log10(np.where(ok, F, np.nan))
    G = np.full(m.size, np.nan)
    for i in range(m.size):
        a, b = max(0, i - half), min(m.size, i + half + 1)
        w = ok[a:b]
        if w.sum() >= max(4, half + 1):
            G[i] = np.polyfit(lm[a:b][w], lF[a:b][w], 1)[0]
    return m, G


def find_inertial_range(m, F, half=4, tol=0.30, min_decades=0.8, min_val=0.0):
    """
    Locate the inertial range automatically as the LONGEST contiguous run of
    bins over which the local slope stays within `tol` of that run's own mean.

    Parameters
    ----------
    tol          : allowed scatter of Gamma inside the plateau (dex per dex)
    min_decades  : reject a plateau narrower than this (a two-bin "plateau" is
                   not a power law)

    Returns dict(m_lo, m_hi, alpha, scatter, decades, n_bins) -- or NaNs if no
    plateau qualifies, which is itself the correct answer when the run has no
    inertial range yet.
    """
    m, G = local_slope(m, F, half=half, min_val=min_val)
    good = np.isfinite(G)
    best = None
    i = 0
    n = m.size
    while i < n:
        if not good[i]:
            i += 1; continue
        j = i
        while j + 1 < n and good[j + 1]:
            seg = G[i:j + 2]
            if seg.max() - seg.min() > 2 * tol:
                break
            j += 1
        seg = G[i:j + 1]
        if seg.size >= 3:
            dec = np.log10(m[j] / m[i])
            if dec >= min_decades and (best is None or dec > best[0]):
                best = (dec, i, j)
        i = j + 1
    if best is None:
        return {"m_lo": np.nan, "m_hi": np.nan, "alpha": np.nan,
                "scatter": np.nan, "decades": 0.0, "n_bins": 0}
    dec, i, j = best
    seg = G[i:j + 1]
    return {"m_lo": float(m[i]), "m_hi": float(m[j]),
            "alpha": float(np.mean(seg)), "scatter": float(np.std(seg)),
            "decades": float(dec), "n_bins": int(j - i + 1)}


def guard_band(m_low_scale, m_high_scale, pad_decades=0.5):
    """
    The a-priori inertial range: strip `pad_decades` from each end of the
    interval between the two characteristic masses of the problem.

        closed system : (m_inj , max m0 reached)
        open system   : (m_inj , m_sink)

    Use it as an independent cross-check on `find_inertial_range`: the two
    should agree.  If they do not, the run has not developed a cascade over
    the range you assumed.
    """
    lo = m_low_scale * 10.0 ** pad_decades
    hi = m_high_scale / 10.0 ** pad_decades
    return (lo, hi) if hi > lo else (np.nan, np.nan)


def iso_mean_mass(iso_counts, centers, age_edges):
    """
    Mean mass of each isochrone,  <m>(tau), from the accumulated 2D
    (age, mass) histogram.  This is the direct measurement of the
    growth law  m0(tau)  in an OPEN system.
    """
    C = np.asarray(iso_counts, float)
    n = C.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mbar = (C * centers[None, :]).sum(axis=1) / n
    tau = np.sqrt(np.asarray(age_edges)[:-1] * np.asarray(age_edges)[1:])
    return tau, mbar, n


def growth_fit(t, m0, t_star=None, mask=None):
    """
    Fit the growth (or decay) law.

    Coagulation:    m0 ~ tau^b            -> pass t_star=None
    Fragmentation:  m0 ~ (t_star - t)^b   -> pass t_star (e.g. the final time)

    Returns dict with b, r2, and the R^2 of an exponential fit for
    comparison -- if the exponential wins, the clock is wrong.
    """
    t = np.asarray(t, float); m0 = np.asarray(m0, float)
    if mask is None:
        mask = np.isfinite(t) & np.isfinite(m0) & (m0 > 0)
    x = (t_star - t[mask]) if t_star is not None else t[mask]
    y = m0[mask]
    good = x > 0
    x, y = x[good], y[good]
    if x.size < 4:
        return {"b": np.nan, "r2_power": np.nan, "r2_exp": np.nan, "n": int(x.size)}
    b, c = np.polyfit(np.log(x), np.log(y), 1)
    r2p = float(np.corrcoef(np.log(x), np.log(y))[0, 1] ** 2)
    r2e = float(np.corrcoef(x, np.log(y))[0, 1] ** 2)
    return {"b": float(b), "A": float(np.exp(c)),
            "r2_power": r2p, "r2_exp": r2e, "n": int(x.size)}


# ---------------------------------------------------------------------
#  Exact reference solutions for the constant kernel (lambda = 0),
#  closed system.  Used to validate the clock -- see note [5].
# ---------------------------------------------------------------------

def exact_closed_constant(t, m_init, n_init, K1=1.0, process="coagulation"):
    """
    Smoluchowski with K = K1 = const in a closed box.

        coagulation:    n(t) = n0 / (1 + n0 K1 t / 2),  m0 = m_init (1 + n0 K1 t/2)
        fragmentation:  n(t) = n0 / (1 - n0 K1 t / 2),  m0 = m_init (1 - n0 K1 t/2)

    In BOTH cases m0(t) is a straight line on LINEAR axes.  If a run
    instead gives a straight line for ln m0 versus t, the particle
    weights are missing from the time step.
    """
    t = np.asarray(t, float)
    s = 0.5 * n_init * K1 * t
    return m_init * (1.0 + s) if process == "coagulation" else m_init * (1.0 - s)


# ======================================================================
#  I/O
# ======================================================================

def _kernel_tag(f):
    name = getattr(f, "__name__", "kernel")
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def state_of(run, source=None):
    """Build the ic={'state': ...} dict from a finished (or loaded) run.

    Everything needed to continue the run exactly where it stopped: the live
    masses, their generation counters, the entry time of each one's ancestor,
    and the family clock.  `t_end` and `t_origin` travel with it so the time
    offset of note [15] composes across arbitrarily many legs.

    Raises rather than guesses if the particle arrays are absent -- a run saved
    with drop_particles=True cannot be continued, and finding that out here is
    much better than finding it out from a silently wrong age axis.
    """
    need = ("final_mass", "final_gen", "final_gen_time", "final_inj_time")
    missing = [k for k in need if k not in run]
    if missing:
        raise KeyError(
            "this run cannot be continued: %s absent.  Save it with "
            "add_last_run(..., drop_particles=False), which is not the default "
            "because the arrays are one float per live particle." % ", ".join(missing))
    return {
        "mass": np.asarray(run["final_mass"], float),
        "gen": np.asarray(run["final_gen"]).astype(np.int64),
        "gen_time": np.asarray(run["final_gen_time"], float),
        "inj_time": np.asarray(run["final_inj_time"], float),
        "t_end": float(run["final_t_phys"]),
        "t_origin": float(run["meta"].get("t_origin", 0.0)) if "meta" in run else 0.0,
        "source": source if source is not None else (
            run["meta"].get("saved_as") if "meta" in run else None),
    }


def save_run(path, out):
    """Save a run to .npz; metadata goes in as a JSON blob."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    d = {k: np.asarray(v) for k, v in out.items()
         if k != "meta" and not isinstance(v, str)}
    d["meta_json"] = np.frombuffer(
        json.dumps(out["meta"], ensure_ascii=False).encode("utf-8"), dtype=np.uint8)
    np.savez_compressed(path, **d)
    return str(path)


def load_run(path):
    out = {}
    with np.load(path, allow_pickle=False) as z:
        for k in z.files:
            if k == "meta_json":
                out["meta"] = json.loads(bytes(z[k]).decode("utf-8"))
            else:
                out[k] = z[k]
    return out