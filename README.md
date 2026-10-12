# OptBench

OptBench compares geometry optimizers, saddle searches, and global optimization on a fixed set of problems. Chill, Stevenson, Ruehle, Shang, Xiao, Farrell, Wales, and Henkelman set the suite out in J. Chem. Theory Comput. 2014, 10, 5476-5482 (doi:10.1021/ct5008718). The 2014 tables are at <https://theory.cm.utexas.edu/benchmarks/>. This repository is a replay of that suite. The public board is <https://lode-org.github.io/optbench/>.

A problem file under `problems/` states the potential, the stopping rule, and the keys a result must carry. A result is one `benchmark.dat`: the first whitespace on a line splits the key from the value. The geometries are already in this clone, under `inputs/`. A campaign reads them from `inputs/manifest.jsonl`. Nothing else has to be downloaded to run a problem that ships here.

## Install the harness

The harness needs Python 3.11 or newer and no third-party packages. From a clone:

```bash
git clone https://github.com/lode-org/optbench
cd optbench
python -m optbench check
python -m unittest discover -s tests -v
python -m optbench templates
```

`check` exits 0 when every partial contract covers its archived rows and `oracle/archive_gaps.txt` matches the departures. `templates` prints every template id. On linux-64, `pixi run test` and `pixi run check` are the same two commands with Python 3.12.

The harness does not contain eOn, pele, or Sella. Those are installed separately, and the commands below say which release produced the files in `results/campaign/`.

## Reproduce the Lennard-Jones 38 L-BFGS mean

`results/campaign/eon-client-lbfgs-m100.dat` is:

```
force_calls 181
force_calls_min 90
force_calls_max 403
nfailed 0
code eOn
code_version 3.5.0
algorithm LBFGS
template eon-client/lbfgs-m100
```

That file is eOn 3.5.0, commit `4d8b34baf1ba8d2d980df29838bac1f969d1662c`, on all 1000 starts in `problems/minimization/lj38.toml`. The stop is the Euclidean norm of the force at 0.01, or 10000 force calls. L-BFGS memory is 100. The client default is 20, and that default is a different score (about 201).

Build `eonclient` from that commit of <https://github.com/TheochemUI/eon>. In an activated environment from that tree:

```bash
meson setup builddir --prefix="$CONDA_PREFIX" --libdir=lib --buildtype release
meson install -C builddir
command -v eonclient
```

A run prints `Eon version 3.5.0`. `pip install pyeonclient` is a different interface. It does not set L-BFGS memory, and the campaign does not call it.

From this repository, with `eonclient` on `PATH`:

```bash
python -m optbench run minimization/lj38 eon-client/lbfgs-m100 \
  --client "$(command -v eonclient)" \
  --workers 4 \
  --output benchmark.dat
```

`EONCLIENT` may replace `--client`. `pixi run campaign -- minimization/lj38 eon-client/lbfgs-m100 --client "$(command -v eonclient)"` is the same command. `--workers` is the number of client processes. The default is 1. The problem file names `website/minimization/lj38/lj38-clusters.tgz`, and the manifest stores that archive at `inputs/sha256/86bf774955b906c486d37282c48e42628e8fe4af3579222f6e07fcf5913157b1/lj38-clusters.tgz`. The command unpacks it. `--work` chooses the scratch directory. The default is next to `--output`.

`benchmark.dat` is the result. `force_calls` is the mean over starts that finished under the cap. Compare it to the file quoted above.

A directory passed as `--geometries` runs only the `.con` files in that directory. That checks the client. It is not the published mean, because the published mean uses all 1000 starts.

## Other scores you can rerun

Each command writes one `benchmark.dat`. The published copy is under `results/campaign/`.

Morse bulk is 100 starts, force norm 0.001 eV/Ang, cap 10000. The same `eonclient` and `eon-client/morse-lbfgs` produce mean 52:

```bash
python -m optbench run minimization/morse-bulk eon-client/morse-lbfgs \
  --client "$(command -v eonclient)" \
  --workers 4 \
  --output morse-lbfgs.dat
```

The Morse geometries are `inputs/sha256/db1ed9c0a17de8c1cf7dbf55f8b1d2baed263ad8686adacf5e892758d3497d8b/morse-bulk.tgz`. The other Morse templates are `eon-client/morse-cg`, `eon-client/morse-fire`, and `eon-client/morse-sd`.

pele 0.1.2 is <https://github.com/pele-python/pele>. The driver imports `pele.optimize._lbfgs_py.LBFGS` and `pele.potentials.LJ`. Install that release, unpack the same Lennard-Jones archive, and run every archived memory and step cap:

```bash
tar -xzf inputs/sha256/86bf774955b906c486d37282c48e42628e8fe4af3579222f6e07fcf5913157b1/lj38-clusters.tgz -C "$WORKDIR"
python -m optbench.pele_lj38 \
  --geometries "$WORKDIR/lj38-clusters" \
  --output-dir "$WORKDIR/pele" \
  --workers 4
```

Memory 100 at step 0.2 averages 178 calls. The 2014 file is 179. The other six files are `results/campaign/pele-0.1.2-lbfgs-*.dat`.

Sella 2.6.0 needs `sella==2.6.0` and ASE. The stop is the Euclidean norm of the force at 0.001, or 1000 steps, on 200 geometries, with cutoff 40 and the archived trust-region numbers:

```bash
python -m optbench.sella_lj38 \
  --coords inputs/sha256/bda818f7fd3c2ee47ad4d4fd74524881814544c1a805054656f612724d9d5ce6/coords-con.tar.gz \
  --output sella.dat \
  --workers 4 \
  --version 2.6.0
```

The recorded file is `results/campaign/sella-2.6.0-ts-lj38.dat`: mean 93, two starts at the step cap. Sella 0.1.1 was mean 69.55 with none failed.

Pt-heptamer NEB is 13 bands, Morse, eight images. The band method has to be the one named in the template. `eon-lbfgs` averages 40:

```bash
python -m optbench.archive_eon neb \
  --entry archive/entries/neb/Pt-heptamer/eon-lbfgs \
  --minima inputs/sha256/135c83a17e5a1769182e9a6cdb3ad98657a62a6cd20ef8ae0440e876e24c5e1b/minima.tgz \
  --client "$(command -v eonclient)" \
  --work "$WORKDIR/heptamer-lbfgs" \
  --output heptamer-lbfgs.dat \
  --workers 4 \
  --version 3.5.0
```

The same command with `eon-cg`, `eon-fire`, or `eon-qm` is the other three rows. CG is 142, FIRE is 85, QuickMin is 165.

The LJ38 climbing-image band is 50 reactant-product pairs and `potential = lammps` with `in.lammps` cutoff 40. eOn 3.5.0 loads `liblammps.so` at run time. Put a LAMMPS build on `LD_LIBRARY_PATH` before this command. Without that library the client cannot start these entries. The Morse heptamer runs do not need it.

```bash
python -m optbench.archive_eon pairs \
  --entry archive/entries/double-ended-saddle-search/lj38/eon-ci-neb \
  --coords inputs/sha256/d1c74db6a00956a4e30487fb80e6331b9515e5f93bba1c6f823e38c3b2986dfc/coords-con.tgz \
  --client "$(command -v eonclient)" \
  --work "$WORKDIR/ci-neb" \
  --output ci-neb.dat \
  --workers 4 \
  --count 50 \
  --version 3.5.0
```

`eon-ci-swdneb` is the doubly nudged row. A start at or above 2500 force calls is left out of the mean. The climbing-image mean is 809 with 2 failures. The doubly nudged mean is 708 with 5 failures.

The Henkelman client is revision 2549 of <https://theory.cm.utexas.edu/svn/eon>. A checkout of older revisions fails where historical paths are gone. Export revision 2549. That tree still names a QSC potential whose sources are not in the revision, so the QSC references have to come out of the build or the link fails. The binary should report `r2549`. Templates under `templates/eon-svn/` use the potential name `lj_cluster`. The 3.5.0 templates use `ljcluster`.

## What a template may set

The problem file owns the stop and the geometries. The template owns only the client. eOn 3.5.0 and SVN r2549 do not read the same keys. A key in the wrong section is ignored, not applied.

eOn 3.5.0, `templates/eon-client/`:

- QuickMin reads `time_step` from `[QuickMin]`. FIRE reads `time_step` and `time_step_max` from `[FIRE]`. The same keys under `[Optimizer]` do nothing.
- Steepest descent reads `sd_alpha` and `sd_twopoint` from `[SD]`.
- L-BFGS reads `lbfgs_memory` from `[LBFGS]`, or from `[Optimizer]` when `[LBFGS]` is absent.
- A nudged elastic band reads `opt_method` from `[Nudged Elastic Band]`. With that key absent the band is L-BFGS, including when `[Optimizer]` names CG, FIRE, or QuickMin. `converged_force` on `[Optimizer]` is the band tolerance when the band section does not set one.

SVN r2549, `templates/eon-svn/`:

- `sd_alpha`, `sd_twopoint`, and the QuickMin `time_step` are read from `[Optimizer]`.

`python -m optbench.archive_eon` copies an archived `[Optimizer]` timestep into `[QuickMin]` or `[FIRE]`, and copies `opt_method` into `[Nudged Elastic Band]`, then runs that entry's `report.sh`. The values stay the archived values.

The LJ38 minimization templates name `ljcluster` or `lj_cluster`. They do not pass the problem cutoff of 40. Every pair in that cluster is inside the cutoff, and the all-pairs potential matches the archived means. A geometry whose pairs reach the cutoff has to use `lammps` and the archived `in.lammps`, as the climbing-image command does.

## The two boards

`oracle/entries.jsonl` holds every `benchmark.dat` from branch `main` of <https://gitlab.com/optbench/optbench> at commit `0f26b796`. The archived LJ38 minimization mean counts every run under 10000 force calls as a success. A new run is a success only when the force norm is at or below the problem value and the call count is under the cap. Both rules are in `problems/minimization/lj38.toml`. The reference entry is `archive/entries/minimization/lj38/eon-cg`.

The front page is a result whose status is `sota` and whose commit is the present source in `codes/registry.toml`. The archive page is the 2014 scores plus later passes that do not meet that rule. A local `benchmark.dat` is a score whether or not it has been added to the front page. Adding it there means a `results/*.json` record with `optbench.result/v1`, the problem geometries, and the pinned commit.

`python -m optbench import-archive --bare ARCHIVE --dest .` rebuilds the oracle and the geometry store from a bare clone of the public GitLab project. It does not replace a problem file that already exists.

## Build the site

Org-mode under `docs/orgmode` is the prose. `python -m optbench.site` writes the rank tables. Emacs exports the org to reStructuredText, and Sphinx builds the HTML.

```bash
pixi run -e docs html
```

Without pixi, the same three steps are `python -m optbench.site`, `emacs --batch --load docs/export.el`, and `sphinx-build -d docs/doctrees docs/source docs/build`. Open `docs/build/index.html`.

The hosted test run is `pixi run test` and `pixi run check`. It does not run a campaign. A published score uses every start in the problem. SoftSaddle is a MATLAB smoke in `.github/workflows/softsaddle-smoke.yml`: it builds `libmorse`, `libdavidson`, and `liblanczos`, then runs two searches with a 600 second cap. That smoke does not replace the April 2019 score of 423491.1 force calls in `softsaddle/entry.toml`.
