"""
chi2_passes.py -- settling test on stored results of run_local_*.py.

Splits the passes into an early and a late group and compares the two
spectra bin by bin in the core window: chi2/dof from Poisson counts
(replicas pooled) and from the scatter between replicas.

    python chi2_passes.py --process coag --runs runs_lambda2_3
    python chi2_passes.py --process frag --runs runs_lambda2_3 --split 1
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
        #  raw counts = F * width * snapshots
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
    #  fit one overall scale
    r *= (NA.sum() / max(r * NB.sum(), 1e-30))
    var = NA + r * r * NB
    z = np.where(var > 0, (NA - r * NB) / np.sqrt(np.maximum(var, 1e-30)), 0.0)
    return z, float(np.sum(z ** 2))


def main():
    ap = argparse.ArgumentParser(
        description="Settling test on stored results of run_local_*.py.")
    ap.add_argument("--process", choices=["coag", "frag"], required=True)
    ap.add_argument("--runs", default=None,
                    help="folder with the results; default: runs.")
    ap.add_argument("--tag", default=None,
                    help="file prefix; default: coag_f0.00 or frag_f0.30.")
    ap.add_argument("--split", type=int, default=None,
                    help="passes in the early group; default: half.")
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
    pos = int(np.sum(z > 0))
    print("    sign balance      : %d positive, %d negative"
          % (pos, z.size - pos))

    #  ---- replica scatter ----------------------------------------------
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
    #  fit one overall scale
    off = float(np.sum(mu2 / sd2 ** 2) / np.sum(1.0 / sd2 ** 2))
    zz = (mu2 - off) / sd2
    dof2 = int(use.sum()) - 1
    print()
    print("  REPLICA SCATTER, %d independent seeds" % len(reps))
    print("    chi2/dof = %.2f   (chi2 = %.1f, dof = %d)"
          % (np.sum(zz ** 2) / dof2, np.sum(zz ** 2), dof2))
    print("    mean log(early/late) = %+.4f  (%+.2f%%)"
          % (off, 100.0 * (np.exp(off) - 1.0)))
    big2 = np.nonzero(np.abs(zz) > 3.0)[0]
    print("    bins past 3 sigma : %d of %d" % (big2.size, dof2 + 1))
    if big2.size:
        mm = c[band][use][big2]
        print("      at m = " + ", ".join("%.3g(%+.1f)" % (mm[i], zz[big2][i])
                                          for i in range(min(big2.size, 12))))
    print("=" * 74)


if __name__ == "__main__":
    main()
