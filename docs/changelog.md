# Changelog

## 2026-10-21

The hTST/Pt TSASE dimer row quotes every closed window in the search log: 4 barrier windows and 81 rate windows. Barrier jobs are 1980 and barrier force calls are 8.188e6. Rate jobs are 94 and rate force calls are 3.929e5. The 28 Jan 2013 archive file stays the reference column: 1170 jobs, 3.239e6 force calls, 95 jobs, 2.627e5 force calls.

The archive script counts every search line, including lines that are not good or repeat. A client that writes no results file is omitted. sad-fcs is the dimer forceCalls tally, covering the translation image and the finite-difference rotation images. Minimizer steps stay in mins-fcs. On one converged search the harness counted 1196 ASE calculate calls and forceCalls was 1195.

The dimer file dated 2013-01-03, commit 01068b8, uses finite-difference separation 0.005 and stops on the maximum atom force. The 28 Jan 2013 client also passes alpha 2.5 and alpha2 2.5, which sets the translation weight gamma to 1. TSASE 1.0 at 98aac60 uses separation 0.001, projects rigid modes, hard-codes gamma 1, and also requires non-positive curvature. The client passes rotation maximum 50, angle tolerance 5, and minimum force 1e-4, and leaves the other arguments at the 98aac60 defaults. 107 searches reached the 100000-call stop and account for 10700169 of 32755229 saddle calls.

The row is on the ranked table.
