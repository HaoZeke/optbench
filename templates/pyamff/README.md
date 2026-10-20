# PyAMFF rows

The archive numbers are the paper rows. These templates do not rebuild PyAMFF `8037eb98`. The ranked code is the latest release: package version `0.1.0-beta` at master `240f51d01ca22969369f9bc21760c61cbb3b6665` (4 July 2026, tag `v3.0.0` plus 475 commits).

The twelve cells are EMT Pt38 and C2H4, with Rprop, Adam, and LBFGS, at force coefficient 0 and 0.1. Each cell is one ranked row. NAdam, RAdam, AdamW, and SGD are not archive rows, so they are not on this board.

## Setup

The network, Fortran Behler-Parrinello fingerprints, NoScaler, and SSE loss are that release. The archived loss name is SE. Rprop is `torch.optim.Rprop`, Adam is `torch.optim.Adam` with weight decay 0, and LBFGS is `torch.optim.LBFGS` with `max_iter` 1, `history_size` 10, both solver tolerances `1e-10`, and no line search. The learning rate is the release default, 0.01. One batch is the whole trajectory. Weights are uniform on `[-1, 1)`.

The stop is the problem criterion, not a tuned threshold. A start succeeds when `max(abs(parameter gradient))` is strictly below 0.02 for force coefficient 0, or strictly below 0.1 for force coefficient 0.1, and the evaluation count is under the cap (10000 or 30000). The problem files do not state an RMSE tolerance, so the harness does not stop on RMSE. Matching that stop is the setup.

One harness call is one full-batch SSE loss and its parameter gradient. A start that reaches the cap fails. PAR-2 charges that start twice the cap. `mean_epoch` is the success mean, which is the problem's legacy epoch column. The ranked score is PAR-2.

The EMT Pt38 Rprop force row is the latest-release measurement already produced for that cell: 50 unseeded uniform initializations, stock Rprop, loss-gradient tolerance 0.1, cap 30000, mean epoch 947.86. It is the only latest row for that cell. The other eleven cells use the same stop on this harness, with each start seeded from `20261015` plus the start index.

A central difference with step `1e-5` checks the Fortran derivative on the four-atom periodic Pt cell. The largest absolute gap is `1.253631032316349e-09`.
