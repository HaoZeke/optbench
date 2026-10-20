# Alternative A. PAR-2 penalised cost

Status: selected. The reasons are in `decision.md`.

## What is wrong with the current mean

`summarize_capped` averages force calls that sit strictly below the cap and rounds with half-even. A start at the cap is counted in `nfailed` and then left out of the mean. On the published Lennard-Jones 38 board that changes the number. The rounded legacy means below are taken from `results/sota-lj38.json` (1000 starts, cap 10000).

| Entry | Legacy mean | Failed | Illustration PAR-2 mean |
| --- | ---: | ---: | ---: |
| eon-3.5.0-lbfgs-m100 | 181 | 0 | 181 |
| rgmin-quickmin-0.3.0 | 1849 | 1 | 1867.151 |
| eon-3.5.0-qm | 3514 | 25 | 3926.15 |
| eon-3.5.0-sd | 4901 | 96 | 6350.504 |

The illustration rebuilds a PAR-2 mean as `(successes * legacy mean + failures * 2 * 10000) / 1000`. It uses the rounded legacy mean as if every success had that cost. It is not a v2 score. It only shows that dropping failures moves the published number.

## Score

A start is a success only when all of the following hold:

- the harness force-call count is strictly below the cap
- the force norm is at or below the problem stop
- a minimum has Hessian index 0, a saddle has Hessian index 1
- the energy is within the problem tolerance of the reference stationary point
- the Kabsch RMSD to that reference is within the problem tolerance

A start that stops cheaply on the wrong stationary point is a failure. Its search cost is not the number that enters the mean.

The penalised cost of a success is its harness force-call count. The penalised cost of a failure is `2 * cap`. The ranking number is the mean of those penalised costs. Factor 2 is the PAR-2 penalty used for unsolved runs in solver competitions. Factor 10 would make the mean mostly a failure count whenever any starts fail, and would hide the cost of the starts that finished.

Beside that mean the record carries:

- successes `k` out of `n`, and the Wilson 95% interval for `k/n`
- the median and the interquartile range of the same penalised costs
- a percentile bootstrap 95% interval for the median and for the interquartile range, with 2000 resamples and seed `20261015`

The success-rate interval uses `z = 1.959963984540054`. For the published 1000/1000 successes the interval is 0.996173 to 1. For the published 975/1000 successes it is 0.963355 to 0.983010. For 904/1000 it is 0.884165 to 0.920743. Those three intervals use only `k` and `n` from `results/sota-lj38.json`. They do not need the per-start vector.

## Pt islands

`problems/saddle-search/pt-island.toml` states two numbers, and they are not the same budget.

`search_budget` is 5064. That is the budget of one displacement search: climb 1000, 64 calls of overshoot, and two endpoint minimizations of 2000. The initial minimization is outside that count. A search at or above 5064 fails that search, and the campaign continues. 5064 is not the budget of the 27-saddle campaign.

The ranked cost is the harness force-call total from the start through the search that records the 27th good saddle below 1.5 eV. A campaign that never records that saddle is scored by the harness force calls it spent. There is no campaign cap, so that spent total is not doubled.

`results/campaign/eon-3.5.0-saddle-pt-dimer.dat` stores ten campaign totals. Their mean is 4982564.3 force calls. Dividing each total by `total_searches` gives a per-search figure. Those ten ratios have mean 263.590450 and median 263.284133, both to 6 decimals. The classic file's ten ratios have mean 470.691074 and median 469.759056, both to 6 decimals. That per-search figure is not the ranked cost. A result that stores one row per displacement search, and does not store the campaign total through the 27th good saddle, stays off the ranked table.

A file that records one campaign total, sets `goal` to 27, and keeps a longer per-search log is scored as that campaign total. The rgsaddle rotation-skip row is that shape. A row whose template or entry name ends in `-tuned` is left off the board.

## Tuning and provenance

`tuning` is `default` or `tuned`. `default` means every optimizer parameter is the value the code ships. `tuned` means a parameter was changed, including a change that copies the 2014 archive. The eOn 3.5.0 client default L-BFGS memory is 20, so `eon-client/lbfgs-m20` is `default` and `eon-client/lbfgs-m100` is `tuned`. Rows without that fact stay unlabelled.

Provenance is the code name, the version, the commit, the parameter object, and the machine. A score without those fields is not a v2 score.

## What the 2014 mean keeps

The 2014 `benchmark.dat` mean stays the half-even mean of starts under the cap. Archive rows remain on the site. They are not recomputed.
