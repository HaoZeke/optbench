# hTST/Al dimer windows

The archive mean is 432 barrier windows and 11 rate windows (`archive/entries/hTST/Al/eon-dimer-cg2/benchmark.dat`): 46 barrier jobs and 21070 saddle force calls, 1671 rate jobs and 766300 saddle force calls. Each window here is one fresh AKMC from state 0 with its own `random_seed`. A barrier seed stops when the eight reference processes, four at 0.233 eV and four at 0.372 eV, have been seen. A rate seed stops when the unique-process rate sum, prefactor `1.2e12` at 300 K, reaches `0.999 * 592041917`. The harness counts every force evaluation. The archive column sums saddle force calls only (search result field 4). Method settings stay the shipped ini.

The archive walks one `search_results.txt` and resets the window after each hit. These windows do not share a process table: each seed starts from state 0 and stops at the first hit. Finished counts are 17 of 432 barrier windows and 1 of 11 rate windows for EON (SVN r2549), and 40 of 432 barrier windows and 0 of 11 rate windows for eOn (develop 500350d9). A rate seed that has not reached the rate still contributes its barrier window. One seed is not a ranked row. Nothing is written under `results/campaign/` for these two runs. The table is `results/htst-al-dimer-cg2/windows/aggregate.json`.

## Counts

Harness force calls are the count. Saddle force calls are the client `sad-fcs` field, the same field the archive sums. Every finished barrier is 0.23285 eV, which rounds to the archive class 0.233 eV.

| | archive | EON mean | EON median | EON min | EON max | eOn mean | eOn median | eOn min | eOn max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| barrier windows | 432 | 17 | | | | 40 | | | |
| barrier jobs | 46 | 44.35 | 41 | 17 | 76 | 34.85 | 32 | 11 | 69 |
| barrier harness calls | | 34832.5 | 26513 | 11146 | 127441 | 33553 | 20103 | 5909 | 92929 |
| barrier sad-fcs | 21070 | 27998.7 | 20014 | 8363 | 116258 | 28136.6 | 15242 | 4234 | 84464 |
| barrier mins-fcs | | 6833.8 | 6499 | 2783 | 11954 | 5416.3 | 4973.5 | 1675 | 10793 |
| rate windows | 11 | 1 | | | | 0 | | | |
| rate jobs | 1671 | 3282 | 3282 | 3282 | 3282 | | | | |
| rate harness calls | | 2692466 | 2692466 | 2692466 | 2692466 | | | | |
| rate sad-fcs | 766300 | 2187062 | 2187062 | 2187062 | 2187062 | | | | |
| rate mins-fcs | | 505404 | 505404 | 505404 | 505404 | | | | |

The one EON rate window is seed 42. Its rate sum is 591458408.07, against the archive reference 592041917 and the stop threshold 591449875.08.

## Client counters

EON (SVN r2549) `Potential::fcalls` counts both dimer images. On all 5222 recorded searches, sad-fcs + mins-fcs equals the harness.

eOn is 500350d9 on `fix/develop-regressions-vs-350`. `eonclient -v` prints `3.6.0 (500350d9)`. Commit 44931a78 counts saddle calls through every image potential. On 2853 of 2868 searches, sad-fcs + mins-fcs equals the harness. The other 15 are 14 `Iterations` results and 1 `Energy` result, and on each of those the client total is one call lower than the harness. The harness total stays the count.

## Per-window cost

Barrier wall clock below is for seeds that stopped at the barrier, on this eight-core machine while several clients ran at once.

| | n | mean | median | min | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| EON barrier wall (s) | 14 | 250.6 | 133.1 | 63.4 | 1014 |
| EON seconds per search | 14 | 5.92 | 2.61 | 2.19 | 36.2 |
| eOn barrier wall (s) | 38 | 188.2 | 77.4 | 22.9 | 1048 |
| eOn seconds per search | 38 | 5.83 | 2.38 | 1.76 | 49.9 |

EON seed 42, the finished rate window, took 18526 s for 3282 searches (5.64 s/search) while one other client shared the machine. At the median barrier pace, about 2.4–2.6 s/search on one core, a barrier window of the median length is one to two minutes and a rate window of seed 42's length is a few hours. That is why 432 and 11 are sharded.

Four rate seeds are open. Each already has its barrier window. The rate gap is the threshold minus the unique-process sum.

| code | seed | searches | elapsed (s) | barrier jobs | barrier harness | rate gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| eOn | 0 | 697 | 4152 | 30 | 19814 | 189511 |
| eOn | 1 | 838 | 3953 | 31 | 21513 | 212290 |
| EON | 0 | 696 | 4103 | 36 | 21012 | 279796 |
| EON | 1 | 584 | 4074 | 41 | 26513 | 242650 |

Their work directories are committed. Run the same command to continue them.

## Records

Records land in `results/htst-al-dimer-cg2/windows/<code>/seed-<seed>.jsonl`. `scripts/htst_al_windows.py report` rewrites each summary from its jsonl and writes `aggregate.json`.

A stop aborted one in-flight search on six seeds. Those calls are stored on the next line as `interrupted_calls` and are not part of `harness_delta`. EON seed 24 was in that set and had not closed a barrier window, so it is not in the record. Start it again with the shard command below.

The work directory `results/htst-al-dimer-cg2/work/<code>/seed-<seed>/` is the resume point. Finished barrier seeds do not need it, and those directories are not in git. A seed whose jsonl is present and whose work directory is absent is left as recorded: the run exits without starting that seed over.

## Clients

EON (SVN r2549):

```
export OPTBENCH_EON_SVN_CLIENT=/tmp/eon-svn-r2549-git/client/eonclient
export OPTBENCH_EON_SVN_ROOT=/tmp/eon-svn-r2549-git
```

eOn (develop 500350d9):

```
export OPTBENCH_EON_CLIENT=/tmp/eon-500350d9/bbdir/client/eonclient
export OPTBENCH_EON_ROOT=/tmp/eon-500350d9
```

Use the eOn pixi interpreter. It has `vesin`, which the 500350d9 AKMC import needs. The shard command uses that same interpreter for every seed.

```
PY=/tmp/eon-develop-src/.pixi/envs/default/bin/python
```

## Commands

`--seeds START:STOP` is a half-open range. Give each machine a disjoint slice. `--jobs` is the number of seeds run at once on that machine. A seed whose window is already closed is skipped.

```
$PY scripts/htst_al_windows.py run --code eon --seed 0 --until rate
$PY scripts/htst_al_windows.py run --code eon --seed 1 --until rate
$PY scripts/htst_al_windows.py shard --code eon --seeds 2:11 --until rate --jobs 4
$PY scripts/htst_al_windows.py shard --code eon --seeds 49:216 --until barrier --jobs 4
$PY scripts/htst_al_windows.py shard --code eon --seeds 216:432 --until barrier --jobs 4

$PY scripts/htst_al_windows.py run --code svn --seed 0 --until rate
$PY scripts/htst_al_windows.py run --code svn --seed 1 --until rate
$PY scripts/htst_al_windows.py shard --code svn --seeds 2:10 --until rate --jobs 4
$PY scripts/htst_al_windows.py shard --code svn --seeds 24:216 --until barrier --jobs 4
$PY scripts/htst_al_windows.py shard --code svn --seeds 216:432 --until barrier --jobs 4

$PY scripts/htst_al_windows.py report
```

EON seeds 10–23 and 42, and eOn seeds 11–48, are already closed barrier windows. EON seed 42 is also the closed rate window.
