# PyAMFF rows

The ranked rows train `torch.optim` on the PyAMFF 10-7 tanh network. Fingerprints come from the Fortran engine `fmodules.fpcalc.calcfps` at commit `240f51d01ca22969369f9bc21760c61cbb3b6665`. The parameter file is the archived `fpParas.dat`. The scaler is NoScaler and the loss is SSE. `config.yaml` defaults `fp_engine` to Fortran, and each template config sets that value.

A central difference with step `1e-5` checks the Fortran derivative on the four-atom periodic Pt cell from the Optax note. The cell is a 10 Å cube. The atoms sit at `[8.6964, 7.8873, 1.5741]`, `[8.8873, 4.1428, 8.8971]`, `[1.0862, 9.5476, 1.2094]`, and `[7.3070, 7.0155, 7.4174]`. The largest absolute gap between `calcfps` and `(G(x+h) - G(x-h)) / (2h)` is `1.253631032316349e-09`. The stored key is the moved atom, then the fingerprint center, then the Cartesian direction.

The 2022 archive at PyAMFF `8037eb98` used that code's own optimizers. These harness rows use `torch.optim` and count every loss-gradient evaluation. Energy RMSE is the mean over successful starts.

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

The archive force RMSE on EMT Pt38 is 0.09958615799999998 for Rprop and 0.1053114 for Adam. The harness force RMSE is 0.086180573 for Rprop and 0.094044504 for Adam. The archive LBFGS energy row has `nfailed 1`. The harness LBFGS energy row has `nfailed 4`, and the harness Rprop force row has `nfailed 2`.
