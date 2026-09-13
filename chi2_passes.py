"""
chi2_passes.py -- has the box stopped moving?  A counting test, not an average.

    python chi2_passes.py --process coag
    python chi2_passes.py --process frag --split 1

WHAT IT ASKS.  Split the accumulated passes into an EARLY group and a LATE
group, and ask whether the two spectra differ by more than the number of
particles behind them allows.  Per mass bin

    z_i = (N_i^early - r N_i^late) / sqrt(N_i^early + r^2 N_i^late),

with r the ratio of exposures, refined by one overall scale so that a pure
difference in normalisation is not counted as a difference in shape.  Then
chi2 = sum z_i^2 over the core window, dof = bins - 1.  A box in equilibrium
gives chi2/dof near 1.  A box still relaxing gives a large chi2 AND a run of
same-sign z, which the printout shows bin by bin.

WHY THIS AND NOT THE ERROR BARS WE HAVE BEEN QUOTING.  The +-0.0003 on <Gamma>
is the spread over the eight replicas divided by root eight.  The replicas
differ only in their seed: same box, same q, same number of events.  Whatever
is systematic -- a transient, the choice of window, curvature of the spectrum
-- is IDENTICAL in all eight and invisible to that error.  It measures the
random part and nothing else.

WHERE THIS TEST IS STILL OPTIMISTIC, and it matters.  Snapshots inside one
pass are correlated: over 0.05 t_c the box barely changes, yet each snapshot
enters the histogram as an independent count.  So sqrt(N) UNDERSTATES the true
uncertainty and chi2/dof will read high even for a box that has stopped.  The
second block of the output avoids this by using the scatter ACROSS REPLICAS as
the error, which has honest degrees of freedom -- eight independent seeds.
Read the two together: the Poisson chi2 localises WHERE the two halves differ,
the replica chi2 says whether the difference is real.
"""

import argparse
import glob
import os

import numpy as np


def load(tag, runs):
    files = sorted(glob.glob(os.path.join(runs, tag + "_rep*_spec.npz")))
    if not files:
        raise SystemExit("no files for %s in %s" % (tag, runs))
    out = []
    for fn in files:
        d = np.load(fn)
        Fp = np.array(d["F_passes"], dtype=float)          # (passes, bins)
        w = np.diff(np.array(d["edges"], dtype=float))
        c = np.array(d["centers"], dtype=float)
        #  Recover raw counts.  F is stored as counts / width / snapshots, so
        #  the snapshot count is whichever of the stored totals makes
        #  F * w * S integral.  Checked rather than assumed.
        S_tot = None
        for key in ("iso_snapshots", "gen_snapshots"):
            if key not in d.files:
                continue
            S = float(d[key])
            x = Fp.mean(axis=0) * w * S
            k = x > 0
            if k.sum() and np.mean(np.abs(x[k] - np.round(x[k])) < 0.05) > 0.98:
                S_tot = S
                break
        if S_tot is None:
            raise SystemExit("cannot recover raw counts from %s" % fn)
        S_pass = S_tot / Fp.shape[0]
        out.append(dict(counts=Fp * w[None, :] * S_pass, centers=c,
                        m_inj=float(d["m_inj"]), m_sink=float(d["m_sink"]),
                        file=os.path.basename(fn)))
    return out


def core_window(m_inj, m_sink):
    lo, hi = min(m_inj, m_sink), max(m_inj, m_sink)
    return 10.0 * lo, hi / 30.0


def chi2_poisson(NA, NB, expA, expB):
    """Counts in two exposures, one free overall scale."""
    r = expA / expB
    #  The scale is refined so that a difference in TOTAL population is not
    #  charged to the shape; that is the one parameter the dof accounts for.
    r *= (NA.sum() / max(r * NB.sum(), 1e-30))
    var = NA + r * r * NB
    z = np.where(var > 0, (NA - r * NB) / np.sqrt(np.maximum(var, 1e-30)), 0.0)
    return z, float(np.sum(z ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--process", choices=["coag", "frag"], required=True)
    ap.add_argument("--runs", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--split", type=int, default=None,
                    help="passes in the EARLY group (default: half)")
    a = ap.parse_args()

    runs = a.runs or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "runs")
    tag = a.tag or ("coag_f0.00" if a.process == "coag" else "frag_f0.30")
    reps = load(tag, runs)
    P = reps[0]["counts"].shape[0]
    k_split = a.split if a.split is not None else max(1, P // 2)
    if not 1 <= k_split < P:
        raise SystemExit("split must be between 1 and %d" % (P - 1))

    c = reps[0]["centers"]
    lo, hi = core_window(reps[0]["m_inj"], reps[0]["m_sink"])
    band = (c >= lo) & (c <= hi)

    print("=" * 74)
    print("  %s : %d replicas, %d passes, early = first %d, late = last %d"
          % (tag, len(reps), P, k_split, P - k_split))
    print("  core window %.3g .. %.3g  (%d bins)" % (lo, hi, int(band.sum())))
    print("=" * 74)

    #  ---- Poisson, replicas pooled -------------------------------------
    NA = sum(r["counts"][:k_split].sum(axis=0) for r in reps)[band]
    NB = sum(r["counts"][k_split:].sum(axis=0) for r in reps)[band]
    z, chi2 = chi2_poisson(NA, NB, float(k_split), float(P - k_split))
    dof = int(band.sum()) - 1
    print("  POISSON, all replicas pooled")
    print("    chi2/dof = %.2f   (chi2 = %.1f, dof = %d)" % (chi2 / dof, chi2, dof))
    big = np.nonzero(np.abs(z) > 3.0)[0]
    print("    bins past 3 sigma : %d of %d" % (big.size, dof + 1))
    if big.size:
        mm = c[band][big]
        print("      at m = " + ", ".join("%.3g(%+.1f)" % (mm[i], z[big][i])
                                          for i in range(min(big.size, 12))))
    #  A transient shows up as a RUN of one sign, not as scatter.
    pos = int(np.sum(z > 0))
    print("    sign balance      : %d positive, %d negative"
          % (pos, z.size - pos))
    print("    NOTE: snapshots inside a pass are correlated, so this chi2 is")
    print("          biased high.  Use it to see WHERE, not whether.")

    #  ---- replica scatter, the honest error ----------------------------
    lr = []
    for r in reps:
        A = r["counts"][:k_split].sum(axis=0)[band] / float(k_split)
        B = r["counts"][k_split:].sum(axis=0)[band] / float(P - k_split)
        ok = (A > 0) & (B > 0)
        v = np.full(A.shape, np.nan)
        v[ok] = np.log(A[ok] / B[ok])
        lr.append(v)
    lr = np.array(lr)
    n = np.sum(np.isfinite(lr), axis=0)
    mu = np.nanmean(lr, axis=0)
    sd = np.nanstd(lr, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1))
    use = (n >= 3) & np.isfinite(mu) & (sd > 0)
    mu2, sd2 = mu[use], sd[use]
    #  One free overall scale again.
    off = float(np.sum(mu2 / sd2 ** 2) / np.sum(1.0 / sd2 ** 2))
    zz = (mu2 - off) / sd2
    dof2 = int(use.sum()) - 1
    print()
    print("  REPLICA SCATTER, %d independent seeds" % len(reps))
    print("    chi2/dof = %.2f   (chi2 = %.1f, dof = %d)"
          % (np.sum(zz ** 2) / dof2, np.sum(zz ** 2), dof2))
    print("    mean log(early/late) = %+.4f  -> the population moved by %.2f%%"
          % (off, 100.0 * (np.exp(off) - 1.0)))
    big2 = np.nonzero(np.abs(zz) > 3.0)[0]
    print("    bins past 3 sigma : %d of %d" % (big2.size, dof2 + 1))
    if big2.size:
        mm = c[band][use][big2]
        print("      at m = " + ", ".join("%.3g(%+.1f)" % (mm[i], zz[big2][i])
                                          for i in range(min(big2.size, 12))))
    print("=" * 74)
    print("  chi2/dof near 1 in the REPLICA block means the two halves agree")
    print("  within the spread of independent seeds: the box has stopped.")


if __name__ == "__main__":
    main()
