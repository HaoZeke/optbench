# OptBench

OptBench compares geometry optimizers, saddle searches, and global optimization on a fixed set of problems. Chill, Stevenson, Ruehle, Shang, Xiao, Farrell, Wales, and Henkelman set the suite out in J. Chem. Theory Comput. 2014, 10, 5476-5482 (doi:10.1021/ct5008718). The live tables are at <https://theory.cm.utexas.edu/benchmarks/>.

This tree is a replay of that suite. It is not a copy of the website archive. A problem file under `problems/` states the potential, the stopping rule, and the keys a result must carry. A result is a `benchmark.dat` file: one key, whitespace, then the rest of the line. `oracle/entries.jsonl` holds every `benchmark.dat` from branch `main` of <https://gitlab.com/optbench/optbench> at commit `0f26b796`.

The archived LJ38 minimization mean counts every run under 10000 force calls as a success. A new run is a success only when the force norm is at or below 0.01 and the call count is under that cap. Both rules are in `problems/minimization/lj38.toml`. The reference entry for that reading is `archive/entries/minimization/lj38/eon-cg`.

Geometries are stored once, by sha256, under `inputs/`. `inputs/manifest.jsonl` maps each archive path to that file. Code snapshots from `website/codes/` are not in this tree. The largest of those is `website/codes/pyamff-8037eb98.tgz`.

```
python -m optbench check
python -m unittest discover -s tests -v
```

`check` exits 0 when every partial contract covers its archived rows and `oracle/archive_gaps.txt` matches the departures. A stated contract is what a new run writes. A departure is a missing key on a stated contract, a repeated key, or a spelling alias such as `contributer`.

`python -m optbench import-archive --bare ARCHIVE --dest .` rebuilds the oracle, the geometry store, and any problem file that does not already exist. `ARCHIVE` is a bare clone of the public GitLab project.

## How a run is handled

The problem file says what a new `benchmark.dat` must contain. The oracle says what the archive recorded. Those two stay in git.

Every push to `main`, and every pull request, runs `pixi run test` and `pixi run check` on a hosted runner. That job does not start MATLAB and does not run a campaign.

A MATLAB code is a smoke, not a score. SoftSaddle is `.github/workflows/softsaddle-smoke.yml`. It builds `libmorse`, `libdavidson`, and `liblanczos`, then runs two searches with a 600 second cap. `softsaddle/entry.toml` keeps that smoke next to the April 2019 score of 423491.1 force calls. The smoke does not replace that score.

A published score uses the full problem: every start, the stopping rule, and one `benchmark.dat`. That job runs on a cluster. The hosted runner is the wrong machine for the thousand LJ38 minimizations and for the ten-repetition SoftSaddle campaign.

Code tarballs from the GitLab archive stay there. A new output that is too large for git is checkpointed with `tt`. The git tree keeps the pointer and the score file.
