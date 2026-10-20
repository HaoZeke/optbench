# hTST/Al dimer windows

The archive entry `archive/entries/hTST/Al/eon-dimer-cg2` is one AKMC. `config.ini` sets `random_seed = 42`, `number_of_cpus = 1`, and `num_jobs = 10000`. `benchmark.dat` averages 432 barrier windows and 11 rate windows: 46 barrier jobs and 21070 saddle force calls, 1671 rate jobs and 766300 saddle force calls.

`report.py` builds the reference process ids from `states/0/processtable`, then walks that same state's `search_results.txt`. It does not start a fresh AKMC per window. Jobs and saddle force calls accumulate, and both counters reset after a hit.

Reference ids, lines 19-48. The barrier multiset is four 0.233 eV and four 0.372 eV. The first processtable row that matches a remaining slot is kept, the slot is removed, and later rows are not added once the multiset is empty:

```
f = open(join(path, 'processtable'))
...
    if barrier in barrier_list:
        barrier_ids_ref.append(number)
        i = barrier_list.index(barrier)
        del barrier_list[i]
```

The walk and both resets, lines 74-127. Two header lines are skipped. Every later line adds one job and field 4 (`sad-fcs`), including lines that are not `good-*` or `repeat-*`. A barrier window closes when every reference id has been seen, then `search_bnum` and `fcs_bsum` go back to zero. A rate window closes when the unique-process sum, prefactor `1.2e12` at 300 K, reaches `0.999 * 592041917`, then the rate counters go back to zero:

```
f = open(join(path, 'search_results.txt'))
...
for line in f:
    search_bnum +=1
    search_rnum +=1
    ...
    fcs_bsum += saddle_forcecalls
    fcs_rsum += saddle_forcecalls
    ...
    if len(barrier_ids) == len(barrier_ids_ref):
        barrier_ids = []
        searchbarrier_num.append(search_bnum)
        fcsbarrier_sum.append(fcs_bsum)
        search_bnum = 0
        fcs_bsum    = 0
    ...
    if rate_sumt >= rate_ref * rate_accuracy:
        rate_ids    = []
        searchrate_num.append(search_rnum)
        fcsrate_sum.append(fcs_rsum)
        search_rnum = 0
        fcs_rsum    = 0
        rate_sumt   = 0.0
```

The means, lines 139-153, are `np.mean` over those windows. `benchmark.dat` records the lengths:

```
barrier_jobs 46
barrier_force_calls 2.107e+04
average over 432
rate_jobs 1671
rate_force_calls  7.663e+05
average over 11
```

Barrier windows stay open until the processtable has consumed all eight slots. A short reference list would close on the wrong ids. Rate windows do not use that list. The ranked count is the harness sum on each window. sad-fcs is the archive column, summed from field 4 on the same lines.

The reactant is `website/hTST/Al/eon-dimer-cg2/pos.con`. Archive `displace_atomlist = -1` is the last atom in the r2549 client (`free[-1]`, file index 384). eOn at 500350d9 treats a lone `-1` as every free atom, so both written inis list 384. The harness server runs one search per pass (`num_jobs = 1`, one CPU) so each `search_results.txt` line has its own harness count. Seed 42, temperature 300, confidence 1.95, prefactor `1.2e12`, dimer CG, `rotations_max` 10, `converged_angle` 1, radius 3.3, magnitude 0.5. The problem file states no force-call cap. The server counter does not stop the AKMC.

One run per client. There is no seed shard. `scripts/htst_al_continuous.py` resumes from `results/htst-al-dimer-cg2/continuous/<code>/akmc` and `searches.jsonl`. A checkpoint pauses both runs between searches, commits that directory, and pushes `cells/htst/al-dimer-cg2`. The first pause is after 20 minutes and later pauses are 90 minutes apart.

The per-seed state-0 sample remains on branch `repro/htst-al-eon-dimer-cg2` at `0ae1515d`. This branch does not rank it.

## Counts

<!-- counts:start -->
Harness force calls are the count. sad-fcs is archive field 4, shown beside it.
Mean, median, min, and max are over the closed windows.

| | archive | EON (SVN r2549) | eOn (develop 500350d9) |
| --- | ---: | ---: | ---: |
| barrier windows | 432 | 4 / 432 | 11 / 432 |
| rate windows | 11 | 0 / 11 | 0 / 11 |
| barrier jobs | 46 | 63 / 72 / 17 / 91 | 28.8182 / 28 / 14 / 55 |
| barrier harness calls |  | 65092 / 76462 / 11146 / 96298 | 27075.1 / 17072 / 7567 / 80742 |
| barrier sad-fcs | 21070 | 55441.5 / 64104.5 / 8363 / 85194 | 22655.4 / 12803 / 5399 / 74849 |
| barrier mins-fcs |  | 9650.5 / 11034 / 2783 / 13751 | 4419.55 / 4377 / 2168 / 8583 |
| rate jobs | 1671 |  |  |
| rate harness calls |  |  |  |
| rate sad-fcs | 766300 |  |  |
| rate mins-fcs |  |  |  |

EON (SVN r2549): 318 searches, reference ids 8/8, open barrier jobs 66, open rate jobs 318, client_matches_harness True.
eOn (develop 500350d9): 326 searches, reference ids 8/8, open barrier jobs 9, open rate jobs 326, client_matches_harness False.
<!-- counts:end -->

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

Use the eOn pixi interpreter. It has `vesin`, which the 500350d9 AKMC import needs.

```
PY=/tmp/eon-develop-src/.pixi/envs/default/bin/python
$PY scripts/htst_al_continuous.py selftest
$PY -u scripts/htst_al_continuous.py run --code svn
$PY -u scripts/htst_al_continuous.py run --code eon
$PY -u scripts/htst_al_continuous.py supervise --initial 1200 --interval 5400
$PY scripts/htst_al_continuous.py report
```
