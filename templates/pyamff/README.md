# PyAMFF rows

Two PyAMFF commits are on these problems. The paper baseline is commit `8037eb98e2288c132c112aa6c2032f00cd5c5277` (6 December 2022), which is the commit named in the archive `benchmark.dat` files. The extra rows are `torch.optim` at commit `240f51d01ca22969369f9bc21760c61cbb3b6665`. Both use Fortran Behler-Parrinello fingerprints, the archived `fpParas.dat`, a 10-7 tanh network, NoScaler, and one batch. The Python fingerprint path is not used.

## Paper baseline

Templates `pyamff/*-paper` run the 8037eb98 `pyamff` entry point on the archived `config.ini`. The archive's `run.sh` did the same thing: `python mlrf0_optb.py 50`, then `pyamff` in each directory, then a parse of `pyamff.log`. The twelve entries are Rprop, ADAM, and LBFGS, with force coefficient 0 or 0.1, on EMT Pt38 and C2H4.

In that commit, `training.py` builds the optimizers:

- `Rprop` is `torch.optim.Rprop(parameters, lr=learning_rate)`.
- `ADAM` is `torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)`.
- `LBFGS` is `torch.optim.LBFGS` with `max_iter=1`, `history_size=10`, `lr=learning_rate`, both tolerances `1e-10`, and `line_search_fn=None`.

There is a Fortran `adam.f90` in the tree. The archived configs set `model_type = neural_network`, and `mlrf0_optb.py` invokes the Python entry point. The log lines that became `benchmark.dat` are the Python trainer's lines. The paper rows run that trainer.

`config.yaml` at this commit defaults `fp_engine` to Fortran. Every C2H4 ini sets it. The EMT ini files omit it and keep the default. The runner copies the ini and changes only `master_port`, so parallel starts do not share gloo port 12355.

The network is `.double()`. Fingerprints and targets are float64. The loss type in the ini is `SE`.

## Extra torch.optim rows

Templates without the `-paper` suffix train with `torch.optim` on the 240f51d network. Their `tuning` is `torch.optim` and the result files set `ranked false`. Adam, Rprop, LBFGS, AdamW, NAdam, RAdam, and SGD use learning rate 0.01 and weight decay 0. Initial weights are uniform on `[-1, 1)` from a generator seeded with `20261015` plus the start index. The stop is the maximum L2 norm of the parameter gradients: 0.02 within 10000 evaluations for force coefficient 0, and 0.1 within 30000 evaluations for force coefficient 0.1.

A central difference with step `1e-5` checks the 240f51d Fortran derivative on the four-atom periodic Pt cell. The largest absolute gap is `1.253631032316349e-09`.

## Why the epochs differ

EMT Pt38 Rprop with force coefficient 0.1 is the clearest gap: the extra torch.optim harness success mean is 3464.5625 epochs and the archive mean is 1022.86. The same problem on the 8037eb98 trainer, with that commit's Fortran fingerprints and the archived ini, has an all-parsed mean of 1022.92 epochs, energy RMSE 0.0008367074, and force RMSE 0.10000784, with no failed start. The archive row is 1022.86 epochs, energy RMSE 0.0008239920000000001, and force RMSE 0.09958615799999998. The fingerprints are not the cause of the torch gap. The harness rows already use Fortran fingerprints, and their energy RMSE is in the same range as the archive (often lower). The training loop is different in the five places below.

Epochs. For Rprop and ADAM, 8037eb98 increments `nIter` once per loop. The loop body is one `optimizer.step()` and then one `getLoss`, and `getLoss` is one full-batch SE loss plus the parameter gradient. The log line is written after the increment, so the integer in the line equals the number of loss-gradient evaluations. The first `step` happens before any gradient exists. `mlrf0_optb.py` reads that integer from `tail -2 pyamff.log` and averages every parsed run, including a run whose next line is `Max Epoch Reached`. The harness counts the same kind of evaluation, including that first empty step, but `mean_epoch` leaves failures out. LBFGS in 8037eb98 logs `nIter` inside the closure before the increment, so the logged epoch is one behind the closure count. `torch.optim.LBFGS` is built with `max_iter=1`, so a step is a short closure sequence rather than a 20-iteration solve. A harness call is every closure evaluation.

Batch size. Both sides use `process_num = 1` and `batch_num_per_proc = 1`. The one batch is the whole trajectory (500 EMT images, 570 C2H4 images). There is no mini-batch.

Learning-rate schedule. Neither loop decays the learning rate. The 8037eb98 default in `config.yaml` is 0.01. EMT ini files omit `learning_rate` and keep that default. C2H4 ini files set 0.01. Rprop then uses `torch.optim.Rprop` step bounds `(1e-6, 50)` and etas `(0.5, 1.2)`. ADAM uses betas `(0.9, 0.999)`, `eps` `1e-8`, and weight decay 0. The library default Adam step of 0.001 is not the one this trainer passes. The harness uses the same nominal 0.01 and weight decay 0 on torch 2.14.1+cpu. The archive was run in December 2022, so the torch build is not the same binary. That can move a mean. It is not a schedule difference.

Stop rule. This is the large one. 8037eb98 stops when `max(abs(gradient))` of every parameter tensor is strictly below `loss_grad_tol` (0.02 or 0.1). The source comments out an L2 check and does not use it. It can also stop when the energy RMSE is below `1e-4` and the force RMSE is below `0.01`, which are the `config.yaml` defaults for `energy_tol` and `force_tol`. The ini files do not set those keys. The harness stops only when the maximum per-tensor L2 norm is at or below the same tolerance, and it has no RMSE stop. L2 is at least the max-abs component, so an L2 test is stricter on every tensor with more than one entry. On the 48 successful EMT Rprop force starts, the final L2 is a median 1.84 times the final max-abs component, and 46 of those 48 already have max-abs below 0.08 at the L2 stop of 0.1. The archive stop would have fired earlier. A start that reaches the cap is left out of the harness mean and enters PAR-2 at twice the cap. The archive mean keeps that start at its logged epoch.

Dtype. Both trains are float64. 8037eb98 calls `.double()` on the layers, casts fingerprints with `.double()`, and stores energies and forces as `torch.double`. The harness does the same. Dtype does not explain 3464 against 1022.

The remaining sample difference is the initial weights. Both draw uniform `[-1, 1)`. 8037eb98 uses `2 * torch.rand(shape) - 1` and calls `torch.manual_seed(1234)` only after the model already exists, inside `parallelFit`, so the fifty archive processes were fifty unseeded draws. The harness seeds each start. A different sample moves the mean. It does not by itself turn 1022 into 3464 once the stop rule is the same. The paper rows are that measurement: the same Fortran fingerprints, the archived configs, and 8037eb98's own loop.

## Archive against the extra torch.optim rows

`mean_epoch` here is the harness success mean. The archive mean includes every parsed log. Energy RMSE is the success mean. These torch rows are not the paper baseline.

| Problem | Optimizer | Force coefficient | Archive mean epoch | Archive energy RMSE | Harness mean epoch | Harness energy RMSE |
| --- | --- | --- | --- | --- | --- | --- |
| mlopt/emt-pt38 | Rprop | 0 | 97.58 | 0.0039067848 | 156.06 | 0.003494228 |
| mlopt/emt-pt38 | Adam | 0 | 1103.24 | 0.004438631199999999 | 1511.88 | 0.003890362 |
| mlopt/emt-pt38 | LBFGS | 0 | 1791.8 | 0.0038950800000000004 | 2297.7174 | 0.0032690299 |
| mlopt/emt-pt38 | Rprop | 0.1 | 1022.86 | 0.0008239920000000001 | 3464.5625 | 0.00068267232 |
| mlopt/emt-pt38 | Adam | 0.1 | 1708.8 | 0.0008996777999999999 | 2942.62 | 0.00079943096 |
| mlopt/C2H4 | Rprop | 0 | 184.56 | 0.008634755400000001 | 311.06 | 0.0059752931 |
| mlopt/C2H4 | Adam | 0 | 4395.3 | 0.011765715000000001 | 6073.92 | 0.0091099981 |
| mlopt/C2H4 | LBFGS | 0 | 2157.14 | 0.008125548399999998 | 3193.4082 | 0.0061878103 |

The archive force RMSE on EMT Pt38 is 0.09958615799999998 for Rprop and 0.1053114 for Adam. The harness force RMSE is 0.086180573 for Rprop and 0.094044504 for Adam. The archive LBFGS energy row has `nfailed 1`. The harness LBFGS energy row has `nfailed 4`, and the harness Rprop force row has `nfailed 2`. The archive energy files store force RMSE as `-1`, including when the trainer still evaluated forces, because `mlrf0_optb.py` writes that sentinel. The harness energy rows do the same.

The extra EMT Pt38 force rows that are not in that table are LBFGS, success mean 4522.06 epochs, energy RMSE 0.00076352096, force RMSE 0.092042391, and `nfailed 0`. SGD is 50 divergences, PAR-2 60000. NAdam has one success at 27771 epochs and 49 caps, PAR-2 59355.42. RAdam is 50 caps, PAR-2 60000. AdamW with weight decay 0 matches Adam. On C2H4 forces the extra rows are cap-heavy: Rprop success mean 6566 with `nfailed 46`, Adam and AdamW success mean 9281.6429 with `nfailed 36`, LBFGS success mean 24598.667 with `nfailed 47`, NAdam and RAdam 50 caps, and SGD 50 divergences. Those success means leave the caps out. The archive C2H4 force means keep the capped epochs.

## Archive against the paper baseline

Paper epochs and RMSE below are the all-parsed means, the same average `mlrf0_optb.py` wrote into `benchmark.dat`. When `nfailed` is 0 that mean equals the harness success mean. Energy rows keep the archive sentinel `-1` for force RMSE. A capped LBFGS start can make the all-parsed energy RMSE much larger than the success-only energy RMSE.

| Problem | Optimizer | Force coefficient | Archive epoch | Archive energy RMSE | Archive force RMSE | Archive nfailed | Paper epoch | Paper energy RMSE | Paper force RMSE | Paper nfailed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mlopt/emt-pt38 | Rprop | 0 | 97.58 | 0.0039067848 | -1 | 0 | 101.82 | 0.003879705 | -1 | 0 |
| mlopt/emt-pt38 | Adam | 0 | 1103.24 | 0.004438631199999999 | -1 | 0 | 1119.24 | 0.004420459 | -1 | 0 |
| mlopt/emt-pt38 | LBFGS | 0 | 1791.8 | 0.0038950800000000004 | -1 | 1 | 1929.58 | 0.14879437 | -1 | 2 |
| mlopt/emt-pt38 | Rprop | 0.1 | 1022.86 | 0.0008239920000000001 | 0.09958615799999998 | 0 | 1022.92 | 0.0008367074 | 0.10000784 | 0 |
| mlopt/emt-pt38 | Adam | 0.1 | 1708.8 | 0.0008996777999999999 | 0.1053114 | 0 | 1876.82 | 0.0008860064 | 0.10441512 | 0 |
| mlopt/emt-pt38 | LBFGS | 0.1 | 1943.34 | 0.0008561802000000001 | 0.1003695 | 0 | 2027.32 | 0.0008492646 | 0.10150263 | 0 |
| mlopt/C2H4 | Rprop | 0 | 184.56 | 0.008634755400000001 | -1 | 0 | 202.12 | 0.0080378594 | -1 | 0 |
| mlopt/C2H4 | Adam | 0 | 4395.3 | 0.011765715000000001 | -1 | 0 | 4547.52 | 0.011330292 | -1 | 0 |
| mlopt/C2H4 | LBFGS | 0 | 2157.14 | 0.008125548399999998 | -1 | 0 | 2994.12 | 0.0080290046 | -1 | 2 |
| mlopt/C2H4 | Rprop | 0.1 | 12926.88 | 0.018268401 | 0.16424474439999998 | 18 | 11892.38 | 0.01910448 | 0.16751768 | 15 |
| mlopt/C2H4 | Adam | 0.1 | 13876.7 | 0.021102213999999998 | 0.15999126959999999 | 10 | 14637.68 | 0.021640955 | 0.16185791 | 15 |
| mlopt/C2H4 | LBFGS | 0.1 | 23314.46 | 0.011536104600000001 | 0.11970228499999998 | 19 | 25634.96 | 0.011180646 | 0.095277654 | 26 |

Where the paper row has failures, the success-only epoch and RMSE are separate from the table. EMT Pt38 LBFGS energy: success mean 1593.3542 epochs and energy RMSE 0.0035440294. C2H4 LBFGS energy: success mean 2702.25 epochs and energy RMSE 0.008107385. C2H4 Rprop force: success mean 4131.9714 epochs, energy RMSE 0.021225425, force RMSE 0.17982375. C2H4 Adam force: success mean 8053.8286, energy RMSE 0.02542363, force RMSE 0.18019143. C2H4 LBFGS force: success mean 20907.25, energy RMSE 0.010606372, force RMSE 0.091643817. The EMT LBFGS energy all-parsed energy RMSE is large because a capped start finished with a large energy RMSE. The C2H4 LBFGS force all-parsed force RMSE is lower than the archive while the epoch is higher, and this row has 26 caps against the archive's 19.
