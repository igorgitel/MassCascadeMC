"""
masscascade.py -- Monte Carlo engine for mass cascades:
coagulation and fragmentation, closed or open system.
"""

import numpy as np
import json
import inspect
import warnings
import re
from pathlib import Path
import time as _time

class _Default(int):
    """Marks a signature default; behaves as int."""
    __slots__ = ()

_ISO_START_SINK_DEFAULT = _Default(10)

def kernel_constant(m1, m2):
    """Constant kernel, lambda = 0."""
    return 1.0

def kernel_geometric(m1, m2):
    """Geometric kernel, lambda = 2/3."""
    return (m1 ** (1.0 / 3.0) + m2 ** (1.0 / 3.0)) ** 2

def kernel_sum(m1, m2):
    """Sum kernel, lambda = 1/3."""
    return m1 ** (1.0 / 3.0) + m2 ** (1.0 / 3.0)

def kernel_product(m1, m2):
    """Product kernel, lambda = 2."""
    return m1 * m2

def kernel_additive(m1, m2):
    """Additive kernel, lambda = 1."""
    return m1 + m2

KERNEL_LAMBDA = {
    "kernel_constant":  0.0,
    "kernel_geometric": 2.0 / 3.0,
    "kernel_sum":       1.0 / 3.0,
    "kernel_additive":  1.0,
    "kernel_product":   2.0,
}

def sample_powerlaw(alpha, m_lo, m_hi, N, rng=None):
    """Masses from n(m) ~ m^alpha, inverse CDF."""
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
    if a > 0.0:
        z = (1.0 - u) * np.exp(a * (L_lo - L_hi)) + u
        return np.exp(L_hi + np.log(z) / a)
    else:
        z = (1.0 - u) + u * np.exp(a * (L_hi - L_lo))
        return np.exp(L_lo + np.log(z) / a)

def steady_N_for_mass(alpha, m_lo, m_hi, M_target):
    """Seed size N carrying mass M_target."""
    a = float(alpha)
    lo, hi = float(m_lo), float(m_hi)

    def _mom(p):
        q = a + p + 1.0
        if abs(q) < 1e-12:
            return np.log(hi / lo)
        return (hi ** q - lo ** q) / q

    mbar = _mom(1) / _mom(0)
    return max(int(round(float(M_target) / mbar)), 1), float(mbar)

def predict(system, lam):
    """Analytic exponents, closed or open system."""
    if system == "closed":
        beta = lam                      # closed: mass-conserving packet
    elif system == "open":
        beta = (1.0 + lam) / 2.0        # open: stationary background
    else:
        raise ValueError("system must be 'closed' or 'open'")

    alpha = -(1.0 + beta)
    b = np.inf if abs(1.0 - beta) < 1e-12 else 1.0 / (1.0 - beta)
    return {"beta": beta, "b": b, "alpha": alpha, "lambda": lam, "system": system}

def simulate(
    *,
    process,                    # 'coagulation' | 'fragmentation'
    system,                     # 'closed'      | 'open'
    kernel,                     # K(m1,m2) >= 0, non-decreasing
    edges,                      # increasing mass-bin edges
    ic,                         # {'m','N'} | {'mass'} | {'steady'} | {'state'}
    injection_rate=0.0,         # particles per unit physical time
    injection_mass=None,        # injected mass; default ic['m']
    sink_mass=None,             # absorbing wall mass
    siphon=None,                # distributed sink; unused here
    gate_counts_siphon=None,    # gates count distributed-sink removals
    min_frag_mass=None,         # no fragmentation below this mass
    frag_min_ratio=0.0,         # fragmentation locality f, note [15]
    coag_min_ratio=0.0,         # coagulation locality f, note [15]
    frag_split_width=0.5,       # split half-width, note [10]
    frag_age_rule="inherit",    # 'inherit'|'heavier'|'lighter'|'both_new', note [7]
    max_time=np.inf,
    max_events=np.inf,
    stop_max_mass=None,         # stop when m_max >= this (closed coagulation)
    stop_min_mass=None,         # stop when m_mean <= this (closed fragmentation)
    stop_sink_events=None,      # stop after this many absorptions
    max_stall_tries=2_000_000,  # stall guard: attempts without event
    snapshot_mode="events",     # 'events' | 't_phys' | 'log_m0', note [9]
    snapshot_stride=1e4,        # events | time | snapshots per decade
    snapshot_first_at_start=True,
    iso_age_edges=None,         # 1D age-bin edges; None disables
    iso_t_start=0.0,            # isochrones start after this time
    iso_start_sink=_ISO_START_SINK_DEFAULT,
    track_generations=True,     # carry generation counters, note [7]
    gen_max=64,                 # highest resolved generation
    age_rule="min",             # 'min'|'max'|'mass_weighted'|'heavier'
    track_waiting=True,         # waiting-law accumulators, note [13]
    weight=1.0,                 # constant particle weight, note [8]
    volume=1.0,
    rng=None,
    verbose=True,
    verbose_every=1,
):
    """Run one coagulation or fragmentation cascade."""

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
    if system != "open":
        if not isinstance(iso_start_sink, _Default) and iso_start_sink > 0:
            raise ValueError("iso_start_sink needs an absorbing sink, i.e. system='open'")
        iso_start_sink = 0
        if stop_sink_events is not None:
            raise ValueError("stop_sink_events needs an absorbing sink, i.e. system='open'")

    if system == "closed":
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

    if rng is None:
        rng = np.random.default_rng()
    rnd = rng.random
    rint = rng.integers

    seeded = False              # warm-started population?
    seed_info = None
    restart_from = None         # state file this leg started from
    t_origin = 0.0              # time accumulated in previous legs
    _gen0 = _gt0 = _it0 = None  # inherited per-particle state
    if "steady" in ic:
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
        m_init = float(m_init_arr.mean())
        if injection_mass is None:
            injection_mass = m_hi_seed if process == "fragmentation" else m_lo_seed
    elif "state" in ic:
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
        _t_prev = float(_st.get("t_end", 0.0))
        _gt0 = np.asarray(_st["gen_time"], dtype=float).ravel() - _t_prev
        _it0 = np.asarray(_st["inj_time"], dtype=float).ravel() - _t_prev
        restart_from = _st.get("source")
        t_origin = float(_st.get("t_origin", 0.0)) + _t_prev   # time before this leg
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
    _n_out_of_grid = int(np.count_nonzero((m_init_arr < edges[0])
                                          | (m_init_arr >= edges[-1])))
    if _n_out_of_grid:
        warnings.warn(
            "%d of %d initial masses lie outside the bin grid [%.3g, %.3g) and "
            "will be clamped into the end bins, which shows up as a spike there. "
            "Widen `edges` or narrow the seed range."
            % (_n_out_of_grid, N0, edges[0], edges[-1]),
            RuntimeWarning, stacklevel=2)

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
            delta0_siphon = float(_sp["delta0"])
            nu_siphon = float(_sp.get("nu", 0.0))
            _A, _invb = _sp["tau_drift"]
            _A = float(_A); _invb = float(_invb)
            _c = float(_sp.get("dlnm", 1.0))
            if _A <= 0.0 or _c <= 0.0:
                raise ValueError("siphon: the tau_drift amplitude A and dlnm c must be > 0")
            omega = nu_siphon - _invb
            xi0 = delta0_siphon * _c / _A
            m_ref_siphon = 1.0            # distributed-sink reference mass
        else:
            xi0 = float(_sp["xi0"])
            omega = float(_sp["omega"])
            m_ref_siphon = float(_sp.get("m_ref", 1.0))
            if m_ref_siphon <= 0.0:
                raise ValueError("siphon: m_ref must be > 0")
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

    xi_amp = xi0 * m_ref_siphon ** (-omega) if siphon_on else 0.0

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

    cap = max(N0, 1024)
    mass = np.empty(cap, dtype=np.float64)
    inj_time = np.empty(cap, dtype=np.float64)
    bin_of = np.empty(cap, dtype=np.int32)
    pos_in_bin = np.empty(cap, dtype=np.int32)

    gen = np.zeros(cap, dtype=np.int32)             # splits since the ancestor entered
    gen_time = np.zeros(cap, dtype=np.float64)      # when that ancestor entered
    t_change = np.zeros(cap, dtype=np.float64)

    mass[:N0] = m_init_arr
    inj_time[:N0] = _it0 if _it0 is not None else 0.0
    t_change[:N0] = 0.0
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
        """Grow per-particle arrays."""
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

    upper = edges[1:]
    fmaj = np.empty((B, B), dtype=np.float64)
    for b in range(B):
        for c in range(B):
            val = float(kernel(float(upper[b]), float(upper[c])))
            if val < 0.0:
                raise ValueError("kernel must be non-negative")
            fmaj[b, c] = val

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

    T = fmaj @ N_bin
    rate_row = N_bin * (T - diag)
    R = float(rate_row.sum())

    _occ_hook = [None]

    def apply_delta(b, dN):
        """Update rates after a bin count change."""
        nonlocal R
        if dN == 0:
            return
        if _occ_hook[0] is not None:
            _occ_hook[0]()                       # flush before N_bin changes
        N_bin[b] += dN
        T[:] += dN * fmaj[:, b]
        np.multiply(N_bin, T - diag, out=rate_row)
        R = float(rate_row.sum())

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

    def delete_particle(j):
        """Remove particle j, swap-with-last."""
        nonlocal live, S_pow
        if siphon_on:
            S_pow -= float(mass[j]) ** omega              # distributed-sink sum
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
        """Append one particle; return its index."""
        nonlocal live, S_pow
        if siphon_on:
            S_pow += float(m) ** omega                    # distributed-sink sum
        ensure_capacity(live + 1)
        i = live
        mass[i] = m
        inj_time[i] = t_born
        gen[i] = g
        gen_time[i] = t_born if t_anc is None else t_anc
        t_change[i] = t_phys          # mass changed now, note [13]
        b = mass_to_bin(m)
        bin_add(i, b)
        apply_delta(b, +1)
        live += 1
        return i

    def inherit_age(t1, t2, m1, m2):
        if age_rule == "min":
            return min(t1, t2)                       # oldest ancestor wins
        if age_rule == "max":
            return max(t1, t2)                       # youngest ancestor wins
        if age_rule == "heavier":
            return t1 if m1 >= m2 else t2
        return (m1 * t1 + m2 * t2) / (m1 + m2)       # mass-weighted
    t_phys = 0.0
    events = 0                 # accepted events
    tries = 0                  # majorant attempts
    tries_since_event = 0      # stall guard counter
    inj_residual = 0.0         # injection residual, note [6]
    M_sys = w * float(m_init_arr.sum())   # physical mass currently in the system
    M_in = 0.0
    M_out = 0.0
    m_max = float(m_init_arr.max())
    m_min = float(m_init_arr.min())
    sink_events = 0
    M_ref = M_sys                        # reference; drift must stay zero

    S_pow = float(np.sum(m_init_arr ** omega)) if siphon_on else 0.0
    spow_drift = [0.0]                   # worst drift of that sum
    siphon_residual = 0.0                # distributed-sink residual
    siphon_events = 0                    # distributed-sink removals
    siphon_tries = 0                     # distributed-sink draws
    M_siphon = 0.0                       # mass removed by distributed sink
    siphon_counts = np.zeros(B, dtype=np.float64)   # distributed-sink removals per bin
    dlnm_sum = 0.0
    dlnm_n = 0

    def removed_count():
        """Particles removed, as counted by gates."""
        return w * (sink_events + siphon_events) if gate_counts_siphon \
            else w * sink_events

    rows = {k: [] for k in ("t", "events", "live", "N_phys", "m_mean", "m_max",
                            "m_min", "M_sys", "M_in", "M_out", "n_out", "weight",
                            "honest_num", "honest_mass", "gen_mean", "gen_max_live",
                            "n_siphon", "M_siphon", "M_removed", "dndm")}

    use_gen = bool(track_generations)
    G_MAX = int(gen_max)
    gen_counts = np.zeros((G_MAX + 1, B), dtype=np.float64) if use_gen else None
    gen_num = np.zeros(G_MAX + 1, dtype=np.float64) if use_gen else None
    gen_tau_sum = np.zeros(G_MAX + 1, dtype=np.float64) if use_gen else None
    gen_mass_sum = np.zeros((G_MAX + 1, B), dtype=np.float64) if use_gen else None
    gen_overflow = 0.0
    gen_snaps = 0
    gen_reach_sum = np.zeros(G_MAX + 1, dtype=np.float64) if use_gen else None
    gen_reach_n = np.zeros(G_MAX + 1, dtype=np.float64) if use_gen else None

    def gen_reached(g, t_anc):
        """Record time to reach generation g."""
        if use_gen and 0 <= g <= G_MAX and removed_count() >= iso_start_sink \
                and t_phys >= iso_t_start:
            gen_reach_sum[g] += w * (t_phys - t_anc)
            gen_reach_n[g] += w

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
        """Accumulate occupancy since last flush."""
        nonlocal t_occ_last, t_occ_gate
        if not use_wait:
            return
        dtl = t_phys - t_occ_last
        if dtl > 0.0:
            _inc = N_bin * (w * dtl)
            occ[:] += _inc
            if removed_count() >= iso_start_sink and t_phys >= iso_t_start:
                if not np.isfinite(t_occ_gate):
                    t_occ_gate = t_occ_last
                occ_g[:] += _inc
        t_occ_last = t_phys

    def wait_close(i):
        """Close particle i's stay at constant mass."""
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
        _occ_hook[0] = occ_flush          # arm occupancy hook

    use_iso = iso_age_edges is not None
    if use_iso:
        iso_age_edges = np.asarray(iso_age_edges, dtype=float)
        iso_counts = np.zeros((iso_age_edges.size - 1, B), dtype=np.float64)
        iso_snaps = 0
    else:
        iso_counts = None
        iso_snaps = 0
    iso_t_begin = np.nan       # isochrone accumulation start time

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
        rows["n_out"].append(w * sink_events)        # physical particles absorbed
        rows["n_siphon"].append(w * siphon_events)   # distributed-sink counter
        rows["M_siphon"].append(M_siphon)
        rows["M_removed"].append(M_out + M_siphon)
        rows["weight"].append(w)
        if live > 0:
            _hm = gen[:live] >= 0
            rows["honest_num"].append(float(np.count_nonzero(_hm)) / live)
            _mm = mass[:live]
            _tot = float(_mm.sum())
            rows["honest_mass"].append(float(_mm[_hm].sum()) / _tot if _tot > 0 else np.nan)
        else:
            rows["honest_num"].append(np.nan)
            rows["honest_mass"].append(np.nan)
        if use_gen and live > 0:
            _gl = gen[:live]
            _gh = _gl[_gl >= 0]
            rows["gen_mean"].append(float(_gh.mean()) if _gh.size else np.nan)
            rows["gen_max_live"].append(float(_gh.max()) if _gh.size else np.nan)
        else:
            rows["gen_mean"].append(np.nan)
            rows["gen_max_live"].append(np.nan)
        if siphon_on:
            _sp_exact = float(np.sum(mass[:live] ** omega)) if live > 0 else 0.0
            _den = abs(_sp_exact) if _sp_exact != 0.0 else 1.0
            spow_drift[0] = max(spow_drift[0], abs(S_pow - _sp_exact) / _den)
            S_pow = _sp_exact
        rows["dndm"].append(N_bin * (w / V) / widths)

        if (use_gen and live > 0 and t_phys >= iso_t_start
                and removed_count() >= iso_start_sink):
            _g = gen[:live]
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
    log_step = 1.0 / float(snapshot_stride) if snapshot_mode == "log_m0" else None
    m0_last  = (M_sys / (w * live)) if live > 0 else m_init
    next_thr = (events if snapshot_mode == "events" else t_phys) + float(snapshot_stride)

    t_wall0 = _time.perf_counter()
    n_snap_printed = 0

    while True:
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
        if stop_sink_events is not None and removed_count() >= stop_sink_events:
            stop_reason = "stop_sink_events"
            break
        if tries_since_event > max_stall_tries:
            stop_reason = "acceptance collapsed (stall guard)"
            break

        dt = 2.0 * V / (w * R)
        t_phys += dt
        tries += 1
        tries_since_event += 1

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

        if siphon_on:
            siphon_residual += xi_amp * S_pow * dt
            k_s = int(siphon_residual)
            if k_s > 0:
                k_cap = min(k_s, live - 1)     # never empty the box in a single step
                siphon_residual -= k_cap       # remainder carried forward
                for _ in range(k_cap):
                    n_try = 0
                    while True:
                        js = int(rint(0, live))
                        siphon_tries += 1
                        n_try += 1
                        if omega == 0.0:
                            break                      # flat rate: no thinning
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
                    delete_particle(js)
                    M_sys -= w * m_s
                    M_siphon += w * m_s
                    siphon_counts[b_s] += w
                    siphon_events += 1
                if live < 2:
                    continue                   # caught by stopping block

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

        m1 = float(mass[i]); m2 = float(mass[j])
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

        if process == "coagulation":
            mn = m1 + m2
            wait_close(i); wait_close(j)
            inj_time[i] = inherit_age(inj_time[i], inj_time[j], m1, m2)
            if gen[i] < 0 or gen[j] < 0:
                gen[i] = -1
                gen_time[i] = -np.inf
            else:
                gen[i] = (gen[i] if gen[i] >= gen[j] else gen[j]) + 1
                if gen_time[j] < gen_time[i]:
                    gen_time[i] = gen_time[j]
            if siphon_on:
                S_pow += mn ** omega - m1 ** omega            # distributed-sink sum
            dlnm_sum += abs(np.log(mn / m1)); dlnm_n += 1     # calibration constant c
            mass[i] = mn
            move_bin(i, mass_to_bin(mn))
            delete_particle(j)
            events += 1; tries_since_event = 0
            if mn > m_max:
                m_max = mn
            if sink_mass is not None and mn >= sink_mass:
                i_now = i if i < live else j
                delete_particle(i_now)
                M_sys -= w * mn
                M_out += w * mn
                sink_events += 1

        else:  # fragmentation
            ip = i if m1 >= m2 else j
            mp = max(m1, m2)
            ms = min(m1, m2)
            if min_frag_mass is not None and mp < min_frag_mass:
                continue                                 # passive floor: no event
            xi = 0.5 + frag_split_width * (2.0 * rnd() - 1.0)
            ma, mb = xi * mp, (1.0 - xi) * mp
            t_par = float(inj_time[ip])                  # the parent's clock
            wait_close(ip)

            if frag_age_rule == "inherit":
                t_a = t_b = t_par
            elif frag_age_rule == "both_new":
                t_a = t_b = t_phys
            else:
                keep_a = (ma >= mb) if frag_age_rule == "heavier" else (ma < mb)
                t_a, t_b = (t_par, t_phys) if keep_a else (t_phys, t_par)

            if siphon_on:
                S_pow += ma ** omega - mp ** omega            # distributed-sink sum
            dlnm_sum += abs(np.log(ma / mp)); dlnm_n += 1     # calibration constant c
            mass[ip] = ma
            inj_time[ip] = t_a          # write now: slot is reused
            move_bin(ip, mass_to_bin(ma))
            _g0 = int(gen[ip])
            _g1 = -1 if _g0 < 0 else _g0 + 1
            _tanc = float(gen_time[ip])
            gen[ip] = _g1
            i_new = add_particle(mb, t_b, _g1, _tanc)
            gen_reached(_g1, _tanc)      # both fragments arrive together
            gen_reached(_g1, _tanc)      # no-op for seeded bodies
            events += 1; tries_since_event = 0
            if ma < m_min: m_min = ma
            if mb < m_min: m_min = mb
            if sink_mass is not None:
                doomed = []
                if mb <= sink_mass: doomed.append((i_new, mb))
                if ma <= sink_mass: doomed.append((ip, ma))
                for idx, mm in sorted(doomed, key=lambda z: -z[0]):
                    delete_particle(idx)
                    M_sys -= w * mm
                    M_out += w * mm
                    sink_events += 1

        if snapshot_mode == "log_m0":
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
        occ_flush()   # close occupancy at t_end, note [13]

    snapshot()   # always record the final state

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
    out["mass_drift"] = float((M_sys - M_in + M_out + M_siphon - M_ref)
                              / max(M_ref, 1e-300))
    out["sink_events"] = float(sink_events)
    out["siphon_events"] = float(siphon_events)
    out["M_siphon_total"] = float(M_siphon)
    out["M_removed_total"] = float(M_out + M_siphon)
    out["siphon_counts"] = siphon_counts
    out["siphon_acceptance"] = float(siphon_events / max(siphon_tries, 1))
    out["siphon_spow_drift"] = float(spow_drift[0])
    out["siphon_xi0"] = float(xi0)
    out["siphon_omega"] = float(omega)
    out["siphon_m_bound"] = float(m_bound)
    out["dlnm_mean"] = float(dlnm_sum / dlnm_n) if dlnm_n else np.nan
    if use_iso:
        out["iso_age_edges"] = iso_age_edges
        out["iso_counts"] = iso_counts
        out["iso_dndm"] = iso_counts / widths[None, :]
        out["iso_snapshots"] = float(iso_snaps)
        out["iso_t_begin"] = float(iso_t_begin)
        if iso_snaps == 0:
            warnings.warn(
                "isochrones requested but NEVER accumulated: the run ended at "
                "t = %.3g with %d sink absorptions, while the gates ask for "
                "t >= %.3g and >= %g absorptions.  iso_counts is all zeros, so "
                "<m>(tau) and b will be nan.  The clock is t = 2*attempts/N^2, so "
                "raising N at fixed max_events SHORTENS the run."
                % (t_phys, sink_events, iso_t_start, iso_start_sink),
                RuntimeWarning, stacklevel=2)

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

    if use_wait:
        with np.errstate(invalid="ignore", divide="ignore"):
            out["wait_mean"] = np.where(wait_ev > 0, occ / wait_ev, np.nan)
            out["wait_mean_gated"] = np.where(wait_ev_g > 0, occ_g / wait_ev_g, np.nan)
            out["wait_sem"] = out["wait_mean"] / np.sqrt(np.maximum(wait_ev, 1.0))
            out["wait_sem_gated"] = (out["wait_mean_gated"]
                                     / np.sqrt(np.maximum(wait_ev_g, 1.0)))
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
        "engine": "masscascade.py",
        "seeded": bool(seeded),
        "seed": seed_info,
        "restarted_from": restart_from,
        "t_origin": float(t_origin),
        "track_waiting": bool(use_wait),
        "injection_rate": float(injection_rate),
        "injection_mass": float(injection_mass),
        "sink_mass": None if sink_mass is None else float(sink_mass),
        "siphon": siphon_spec,                 # as given by caller
        "siphon_on": bool(siphon_on),
        "siphon_xi0": float(xi0),              # as resolved by engine
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
        "loc_ratio": float(loc_ratio),          # f in force
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
        "iso_start_sink": int(iso_start_sink),
        "stop_sink_events": None if stop_sink_events is None else float(stop_sink_events),
    }
    return out

def superpose(dndm, times):
    """Age-integrated spectrum, sum of dN/dm dt."""
    D = np.asarray(dndm, float)
    t = np.asarray(times, float).ravel()
    dt = np.diff(t, prepend=t[0])
    return (D * dt[:, None]).sum(axis=0)

def fit_powerlaw(x, y, xmin, xmax):
    """Least-squares fit y ~ A x^alpha."""
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
    """Local slope d log F / d log m."""
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
    """Longest run of constant local slope."""
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
    """A-priori inertial range between characteristic masses."""
    lo = m_low_scale * 10.0 ** pad_decades
    hi = m_high_scale / 10.0 ** pad_decades
    return (lo, hi) if hi > lo else (np.nan, np.nan)

def iso_mean_mass(iso_counts, centers, age_edges):
    """Mean mass of each isochrone."""
    C = np.asarray(iso_counts, float)
    n = C.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mbar = (C * centers[None, :]).sum(axis=1) / n
    tau = np.sqrt(np.asarray(age_edges)[:-1] * np.asarray(age_edges)[1:])
    return tau, mbar, n

def growth_fit(t, m0, t_star=None, mask=None):
    """Fit the growth or decay law."""
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

def exact_closed_constant(t, m_init, n_init, K1=1.0, process="coagulation"):
    """Exact closed-box solution, constant kernel."""
    t = np.asarray(t, float)
    s = 0.5 * n_init * K1 * t
    return m_init * (1.0 + s) if process == "coagulation" else m_init * (1.0 - s)

def _kernel_tag(f):
    name = getattr(f, "__name__", "kernel")
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)

def state_of(run, source=None):
    """Build ic={'state': ...} from a run."""
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
    """Save a run to .npz."""
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