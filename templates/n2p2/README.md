# n2p2 force rows

The force rows use the pyamff squared error, energy coefficient 1 and force coefficient 0.1. One force call is one evaluation of that loss and its gradient. The cap is 30000 evaluations. The stop is the pyamff rule: every parameter tensor has max-abs gradient below 0.1.

Force updates are on. EMT Pt38 start 0000 records 29999 epochs and 59998000 Kalman updates, 500 energy updates plus 1500 force-component updates in each epoch. C2H4 start 0000 records 68397720 updates, 570 energy plus 1710 force. The reference forces are `Atoms.get_forces()` on the problem trajectory. Their root-mean-square is 0.352 on EMT Pt38 and 1.260 on C2H4.

A central difference of the summed loss, step 1e-5, matches the analytic gradient at the initial weights of start 0000. On EMT Pt38 the largest component is 1174.41 analytic and 1174.41 numeric. On C2H4 it is 4594.13 and 4594.13. The C++ evaluation matches that Python gradient, to 4.5e-13 on EMT Pt38 and 1.5e-12 on C2H4.

Force RMSE falls on that same start. The batch gradient does not settle under 0.1.

EMT Pt38 start 0000, stock structure shuffle:

- epoch 0: force RMSE 1.193, max-abs gradient 1174
- epoch 1: force RMSE 0.108, max-abs gradient 15.0
- epoch 10: force RMSE 0.0636, max-abs gradient 5.96
- epoch 80: force RMSE 0.0457, max-abs gradient 17.4
- call 30000: force RMSE 0.0229, max-abs gradient 16.8

C2H4 start 0000:

- epoch 0: force RMSE 1.271, max-abs gradient 4594
- epoch 1: force RMSE 1.114, max-abs gradient 7.40
- epoch 10: force RMSE 0.261, max-abs gradient 53.2
- epoch 40: force RMSE 0.212, max-abs gradient 286
- call 30000: force RMSE 0.0753, max-abs gradient 39.8

All 50 starts on each force row end at 30000 calls with the gradient still above 0.1. The energy-only rows use the same filter. EMT Pt38 meets the 0.02 gradient stop on all 50 starts. C2H4 meets it on 10 of 50 starts. Each force update is one structure or one force component. The stop reads the gradient of the loss summed over the whole trajectory.
