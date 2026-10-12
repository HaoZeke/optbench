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

A campaign is one command. The problem file is the stop and the geometries. The template under `templates/` is the client configuration. The command writes one `benchmark.dat`.

```
pixi run campaign -- minimization/lj38 eon-svn/lbfgs-m100 --client /path/to/eonclient
```

`eon-client/*` speaks the 3.5.0 potential name `ljcluster`. `eon-svn/*` speaks revision 2549, potential name `lj_cluster`. `optbench templates` lists them. `EONCLIENT` may replace `--client`.

The problem file owns the stop and the geometries. The template owns only the client. eOn 3.5.0 and SVN r2549 do not read the same keys, and a key in the wrong section is not a different setting. It is ignored.

eOn 3.5.0, `templates/eon-client/`:

- QuickMin reads `time_step` from `[QuickMin]`. FIRE reads `time_step` and `time_step_max` from `[FIRE]`. The same keys under `[Optimizer]` do nothing.
- Steepest descent reads `sd_alpha` and `sd_twopoint` from `[SD]`.
- L-BFGS reads `lbfgs_memory` from `[LBFGS]`, or from `[Optimizer]` when `[LBFGS]` is absent. Memory 100 is the 2014 mean of 181. The client default is 20, and that default scores about 201.
- A nudged elastic band reads `opt_method` from `[Nudged Elastic Band]`. With that key absent the band is L-BFGS, including when `[Optimizer]` names CG, FIRE, or QuickMin. `converged_force` on `[Optimizer]` is the band tolerance when the band section does not set one.

SVN r2549, `templates/eon-svn/`:

- `sd_alpha`, `sd_twopoint`, and the QuickMin `time_step` are read from `[Optimizer]`.

`python -m optbench.archive_eon` copies an archived `[Optimizer]` timestep into `[QuickMin]` or `[FIRE]`, and copies `opt_method` into `[Nudged Elastic Band]`, then runs that entry's `report.sh`. The values stay the archived values.

The LJ38 templates name `ljcluster` or `lj_cluster`. They do not pass the problem cutoff of 40. Every pair in that cluster is inside the cutoff, and the all-pairs potential matches the archived means. A geometry whose pairs reach the cutoff has to use `lammps` and the archived `in.lammps`.

The problem file says what a new `benchmark.dat` must contain. The oracle says what the archive recorded. Those two stay in git. The public site has two boards. The front page is SOTA: a result whose status is sota and whose commit is the present source in `codes/registry.toml`. The archive page is the 2014 scores plus later passes that do not meet that rule. A problem contract is `contract/problems/<suite>/<id>.json`. A submission is `optbench.result/v1`. eOn 3.5.0, rgmin, rgsaddle, ASE 3.29.0, and pARTn DEVEL are pinned in `codes/registry.toml`. A new score from those pins is entered by Rohit Goswami, and only after the run writes `benchmark.dat`. Prose is org-mode under `docs/orgmode`. `python -m optbench.site` writes the ranks. Emacs exports the org to reStructuredText, and Sphinx builds the HTML.

Every push to `main`, and every pull request, runs `pixi run test` and `pixi run check` on a hosted runner. That job does not start MATLAB and does not run a campaign.

A MATLAB code is a smoke, not a score. SoftSaddle is `.github/workflows/softsaddle-smoke.yml`. It builds `libmorse`, `libdavidson`, and `liblanczos`, then runs two searches with a 600 second cap. `softsaddle/entry.toml` keeps that smoke next to the April 2019 score of 423491.1 force calls. The smoke does not replace that score.

A published score uses the full problem: every start, the stopping rule, and one `benchmark.dat`. That job runs on a cluster. The hosted runner is the wrong machine for the thousand LJ38 minimizations and for the ten-repetition SoftSaddle campaign.

Code tarballs from the GitLab archive stay there. A new output that is too large for git is checkpointed with `tt`. The git tree keeps the pointer and the score file.
