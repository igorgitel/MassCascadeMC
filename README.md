# MassCascadeMC

Monte Carlo code that reproduces the simulation results of the paper:
coagulation and fragmentation in an open system, with injection at one end
of the mass range and an absorbing wall at the other.

## Requirements

Python 3 with numpy and matplotlib (`pip install -r requirements.txt`).

## Files

| file | what it is |
| --- | --- |
| `masscascade.py` | Monte Carlo engine |
| `analysis.py` | measurement layer |
| `run_local_lambda2_3.py` | run for the geometric kernel, $\lambda = 2/3$ |
| `run_local_lambda1.py` | run for the additive kernel, $\lambda = 1$ |
| `chi2_passes.py` | settling test on stored results |

## Run 1: $\lambda = 2/3$

Prediction $dN/dm \propto m^{-p}$ with $p = 11/6$.

```bash
python run_local_lambda2_3.py --process coag --add-hours 8 --outdir runs_lambda2_3
python run_local_lambda2_3.py --process frag --add-hours 8 --outdir runs_lambda2_3
```

| output | content |
| --- | --- |
| `runs_lambda2_3/coag_f0.00_stacked.png` | coagulation spectrum |
| `runs_lambda2_3/coag_f0.00_iso.png` | coagulation isochrones and growth law |
| `runs_lambda2_3/frag_f0.30_stacked.png` | fragmentation spectrum |

## Run 2: $\lambda = 1$

Prediction $dN/dm \propto m^{-2}$.

```bash
python run_local_lambda1.py --process coag --add-hours 8 --outdir runs_lambda1
python run_local_lambda1.py --process frag --add-hours 8 --outdir runs_lambda1
```

| output | content |
| --- | --- |
| `runs_lambda1/coag_f0.30_stacked.png` | coagulation spectrum |
| `runs_lambda1/coag_f0.30_iso.png` | coagulation isochrones and growth law |
| `runs_lambda1/frag_f0.30_stacked.png` | fragmentation spectrum |

## Running

- Run the commands one after the other.
- Each run writes to its own folder, set by `--outdir`; the two runs
  produce files with the same names.
- Each command adds the given time to every one of eight replicas; runs
  resume from disk, so time can be added in steps (`--add-hours 1`,
  repeated).
- Fragmentation first calibrates its injection rate q: up to three
  rounds of 1.5e7 events from a cold start; after each round q is divided
  by the ratio of the measured population to 1e6, and calibration stops
  when that ratio is within 5%; otherwise the q after the third round is
  used. This takes about 20 minutes. To skip it, give q directly:
  `--q 586053` for $\lambda = 2/3$, `--q 784682` for $\lambda = 1$. For coagulation q is computed directly.
- Each checkpoint writes eight state files of about 11 MB; keep the output
  folders out of OneDrive or Dropbox.
- The printed report gives the measured index, its `drift per decade`
  (zero within its error for a power law) and `half-vs-half` (whether the
  box has settled).

Redraw the figures without running:

```bash
python run_local_lambda2_3.py --process coag --stack-only --outdir runs_lambda2_3
python run_local_lambda1.py --process coag --stack-only --outdir runs_lambda1
```

Settling test on stored results; a value near one means the box has settled:

```bash
python chi2_passes.py --process coag --runs runs_lambda2_3
python chi2_passes.py --process frag --runs runs_lambda2_3
python chi2_passes.py --process coag --runs runs_lambda1 --tag coag_f0.30
python chi2_passes.py --process frag --runs runs_lambda1
```

All options: `python run_local_lambda2_3.py --help` or
`python run_local_lambda1.py --help`; both runs take the same options.

## License

MIT. See `LICENSE`.
