# MassCascadeMC

A Monte Carlo engine for mass cascades: coagulation and fragmentation, in a
closed box or with a source and a sink. One simulated particle is one real
particle, so the total mass is conserved exactly.

## Requirements

Python 3 with numpy; the analysis layer and the examples also need matplotlib.

## Files

| file | what it is |
| --- | --- |
| `masscascade.py` | the engine. `simulate(...)` runs one cascade and returns arrays plus a record of the full configuration |
| `analysis.py` | spectral index with an error bar, power-law range, generation and age statistics, settling tests |
| `run_local.py` | the long experiment: eight independent copies, run for hours |
| `chi2_passes.py` | settling test: does the first half of a long run agree with the second half? |
| `NUMERICS.txt` | why the engine is built the way it is, notes `[1]`–`[21]` |
| `examples/` | four short scripts, below |

The configuration is set by two switches:

```
process = "coagulation" | "fragmentation"
system  = "closed"      | "open"
```

## Examples

```bash
cd examples
python ex1_closed_selfsimilar.py          # closed coagulation,   alpha = -1,     < 1 min
python ex2_open_steady.py                 # open coagulation,     alpha = -11/6,  ~ 1.5 min
python ex3_fragmentation_generations.py   # open fragmentation,   alpha = -11/6,  ~ 9 min
python ex4_closed_fragmentation.py        # closed fragmentation, alpha = -5/3,   < 1 min
```

Each script prints its measured slope next to the prediction and writes its
figures to `examples/figures/`. `MAX_EVENTS` at the top of each file sets the
run length.

## Long runs

```bash
python run_local.py --help
```

Give each process an hour first, one after the other (not in parallel):

```bash
python run_local.py --process coag --add-hours 1
python run_local.py --process frag --add-hours 1
```

Then add more time; runs resume from disk:

```bash
python run_local.py --process coag --add-hours 3
python run_local.py --process frag --add-hours 3
```

Redraw figures from stored results without running:

```bash
python run_local.py --process coag --stack-only --iso-dlogm 0.5
```

Fragmentation from two wrong starting slopes (both should relax to -11/6):

```bash
python run_local.py --process frag --start seed --seed-alpha -1.5 --add-hours 2
python run_local.py --process frag --start seed --seed-alpha -2.1 --add-hours 2
```

In the output, `drift per decade` should be zero within its error, and
`half-vs-half` shows whether the box has settled.

## Settling test

Reads stored results, runs nothing:

```bash
python chi2_passes.py --process coag
python chi2_passes.py --process frag
```

The second output block (error from the spread of the eight copies) decides:
a value near one means the box has settled.

## Notes

- Each checkpoint writes eight state files of about 11 MB. Keep `runs/` out of
  OneDrive or Dropbox.
- Create the random generator with an explicit seed and store it yourself; the
  engine does not record it.
- `dndm` is count per bin width per volume; `dN/dm ~ m^alpha` corresponds to
  `dn/dln m ~ m^(alpha+1)`.

## License

MIT. See `LICENSE`.

## Reference

Laor & Gitelman, *Phys. Rev. E* **113**, 044135.
