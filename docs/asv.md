# asv

asv is the measurement store. A result file is keyed to the commit asv checked out, the environment that ran it, and the machine record next to it. The v2 score reads those tracks. It does not keep a second copy of the numbers.

`asv==0.6.6` and `asv-runner==0.3.1` are pinned in `pyproject.toml` and `uv.lock`. The benchmark classes use asv-runner's `track_*` and `time_*` discovery. CI uses `HaoZeke/asv-perch@v1` in `.github/workflows/asv.yml`. Gaps that would otherwise be local forks are `docs/asv-gaps.md`.

## What is committed

| Path | Role |
| --- | --- |
| `asv.conf.json` | `environment_type` `uv-pip`, `results_dir` `results/asv` |
| `uv.lock` | The Python environment asv installs, including the asv pin |
| `environments/eon-3.5.0.yml` | conda-forge pin for the eOn 3.5.0 binary |
| `environments/eon-develop.yml` | Python pin for the eOn develop source build. Not a conda eOn package |
| `environments/eon-svn.yml` | Python pin for the eOn SVN r2549 source build. Not a conda eOn package |
| `environments/vcsqnm.yml` | Python pin for the vc-SQNM v1.3 source build. Not a conda package |
| `benchmarks/` | The suite asv-runner imports |
| `results/asv/<label>/machine.json` | The machine record for one `BENCH_HOST_LABEL` value. `machine` is the label and `os`, `arch`, `cpu`, `num_cpu`, and `ram` are `unspecified` |
| `results/asv/<label>/<commit[:8]>-<env>.json` | Track values for that commit on that machine |

`results/asv` has one directory per machine label. The one directory that predates the label keeps the machine record asv wrote then. A result inside a directory has to name the same machine as its `machine.json`, or asv skips it, so archived results are not moved into a label directory.

`html_dir` stays `.asv/html` and is not committed. `asv publish` rebuilds it from `results/asv`.

`optbench.asv_uv.Uv` is the `uv-pip` environment. Upstream `asv.plugins.uv.Uv.matches` does not survive discovery, and that plugin installs with pip. Both are in the gaps file. This module is the bridge until those land. It is not a second statistics implementation. asv resolves `install_command` inside the environment's `bin`, so the environment also installs `uv==0.13.1` and `asv.conf.json` can call `uv pip`.

## Adding a code

1. Pin the code commit in `codes/registry.toml`. A published eOn score keeps commit `4d8b34baf1ba8d2d980df29838bac1f969d1662c`. A conda binary's printed token is not that commit.
2. If the code is a conda package, add `environments/<name>.yml` with the version. Do not invent a build string. A source build that is not a conda package still gets a yml so its pin is not the 3.5.0 file; leave the package out. asv has one `environment_type` for the run, so the yml is the pin and `asv run -E` is how a later unit selects it. There is no per-benchmark environment attribute. See the gaps file. `environments/eon-develop.yml` is that source pin. The develop classes call `EONCLIENT`.
3. Copy `benchmarks/quadratic.py`. `param_names` is `["seed", "start"]`. `measure` returns the `record_start` dict. Set `cap` on the class. The result row does not store the cap.
4. Inherit `optbench.asv_tracks.TrackedSearch`, or copy its methods. `track_*` returns a float. Set `.unit` on the function (`force calls`, `fraction`, `count`). A dict with `samples` and `number` is a timing result, and asv replaces the value. `time_search` is the wall clock.
5. Set `repeat = 1`, `rounds = 1`, `number = 1`, `sample_time = 0`, `warmup_time = 0`. `repeat == 0` means up to ten repeats.
6. Commit the benchmark. `asv run` installs that commit. Uncommitted edits are not the run.
7. Set `BENCH_HOST_LABEL` and run `uv run optbench asv-machine` once on the machine. That stores the label in `~/.asv-machine.json`, with `os`, `arch`, `cpu`, `num_cpu`, and `ram` set to `unspecified`. Use it in place of `asv machine --yes`, which on asv 0.6.6 fills every field from the host and names the machine after the host.
8. `uv run asv run --machine "$BENCH_HOST_LABEL" <commit>^! -b YourClass`. Without `--machine`, asv looks the machine up by the host name. Do not pass `--quick`. On asv 0.6.6 a quick run still writes `results/asv`. The help text says it does not. That is `docs/asv-gaps.md`. The committed file is the run without `--quick`.
9. Commit the new files under `results/asv`. The run writes them to `results/asv/<label>/` and updates `results/asv/benchmarks.json`.
10. Score with `asv.results.iter_results` and `Results.get_result_value`, then `score_v2`. `optbench.asv_score` does that for the quadratic sample. Parameter strings in the result JSON are wrapped in `repr` quotes; the reader unwraps them with `ast.literal_eval`. `uv run optbench statistics` refuses to write a document when those tracks are absent. It also refuses when `--machine` is not given and more than one machine directory has quadratic tracks, and the machine directory it reads has to hold one quadratic result.

`benchmarks/pending.py` is one skipped parameter for every problem id and every registry code, with `track_force_calls` and `time_search` only. `setup` raises `NotImplementedError`, which asv-runner stores as NaN. Replace that pair with a real class when the code runs. The Lennard-Jones classes in `benchmarks/bench_lj38.py` are separate: they call eonclient when `OPTBENCH_ASV_EON=1`. The default parameter list is one seed and the first geometry. `OPTBENCH_ASV_FULL=1` selects all 1000 starts, and it is read at import, so it has to be set in the process that launches asv.

`benchmarks/bench_eon_develop.py` is the eOn develop entry. Counting is `HarnessCounter` through the `ext_pot` bridge, not the client's own tally. The default parameter list is one seed and the first start. `OPTBENCH_ASV_FULL=1` selects the campaign grid at import. The Pt-island AKMC classes default to one pass. A one-start file is not a published mean.

The quadratic sample is the run that `results/validation/statistics.json` scores. A subset of Lennard-Jones 38 is `results/validation/lj38-counter.dat` and it is not a published mean.

## Commands

```bash
export BENCH_HOST_LABEL=workstation-a
uv run asv check
uv run optbench asv-machine
uv run asv run --machine "$BENCH_HOST_LABEL" --quick
uv run asv run --machine "$BENCH_HOST_LABEL" COMMIT^! -b quadratic
uv run optbench statistics --machine "$BENCH_HOST_LABEL" --output results/validation/statistics.json
uv run asv publish
uv run asv compare --machine "$BENCH_HOST_LABEL" COMMIT COMMIT
```

`asv check` imports the suite. `optbench asv-machine` prints the entry it stored:

```text
machine=workstation-a os=unspecified arch=unspecified cpu=unspecified num_cpu=unspecified ram=unspecified
```

`asv run --quick` is the gate before the saved run. Run it before the saved `asv run`, because asv 0.6.6 rewrites the result file. `asv compare` on one commit compares that commit with itself. With two machine directories, `asv compare` stops and asks for `--machine`.

With an empty home directory and `BENCH_HOST_LABEL=workstation-a`, the saved run on commit `007e5326` wrote `results/asv/workstation-a/machine.json` and `results/asv/workstation-a/007e5326-uv-pip-py3.12.json`. The `params` of that result are the label, `unspecified` for the five machine fields, and `python` `3.12`. `optbench statistics` then printed:

```text
wrote results/validation/statistics.json commit=007e53268ed39f6cdaf12010ddc0638ad345ba2b displacement_median=139.0 dimer_median=46.5 displacement_failed=0 dimer_failed=0
```

That document's `provenance.machine` is the same label, the five `unspecified` fields, and `python` `3.12`. For asv params, `collect_provenance` takes the label from `BENCH_HOST_LABEL`, so a statistics run on the older directory records the label of the machine that scored it. The record leaves out the name of that directory.

## CI

`.github/workflows/asv.yml` checks out the history, installs Python 3.12, syncs the lockfile, runs `uv run optbench asv-machine`, and runs `HaoZeke/asv-perch@v1`. CI does not set `BENCH_HOST_LABEL`, so the entry is `unspecified`. The benchmark command is `uv run asv run --machine unspecified --skip-existing --bench quadratic {sha}^!`. That is the in-process sample, not the 1000-start eonclient grid. `post-comment` is true on a pull request and false on a push. A pull request whose base commit does not contain `benchmarks/quadratic.py` fails this job, because asv runs that commit. `check.yml` is still the oracle and pytest job.

asv-spyglass `compare` exits 1 when any benchmark is worse by `--factor`. The default factor is 1.1. `HaoZeke/asv-perch@v1` does not ignore that status, so the step stops before `regression-threshold` is applied. A pull request compares two commits. A push compares a commit with itself, which is why those runs stay green. Quadratic timings move by about ten percent between two runs, and that is enough to exit 1. The workflow sets `regression-threshold` to 10 and passes `--factor 10`. A slowdown past that factor still fails the job.
