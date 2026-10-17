# Integration

Branch `integration/v2-board`, started from `design/v2-hardening` `b64538628ce060894f7ccb4c41e6f353be3fa5ca`.
Merges are `git merge` with no rebase and no force-push.
`entries/rgmin-fixes` was not on origin when `entries/rgmin` and `entries/rgsaddle-minmode` were merged.
It was merged when it appeared, after those two, without rewriting them.

| Order | Branch | Tip | Result |
| --- | --- | --- | --- |
| 1 | `entries/ase-sella-v2` | `ade7c5132e47bb4008b9d4c8ddb26492e9e7287d` | already in `design/v2-hardening` |
| 2 | `cursor/eon-develop-asv-dee4` | `c4dde48c2d8171991a931ebab95fa59a68308017` | already in `design/v2-hardening` |
| 3 | `entries/eon-full` | `f00867a5eb6a050c46e282143f8ab88f5646af9b` | already in `design/v2-hardening` |
| 4 | `entries/rgmin` | `bddbe2ddbac6606357b9043b41612ec337539a00` | merged |
| 5 | `entries/rgmin-fixes` | `5960e693f7d543f9498c81e62d467961b858ca0f` | merged after `entries/rgsaddle-minmode` |
| 6 | `entries/rgsaddle-minmode` | `cd6151590c09545fc158b9f46e2317047ec4c600` | merged |
| 7 | `entries/rgsaddle-neb` | `20a9361de70db2d444d6747431e66ae1aa7f217f` | already in `design/v2-hardening` |
| 8 | `entries/ase-sella-full` | absent | not on origin |
| 9 | `entries/eon-develop-full` | absent | not on origin |

No campaign was run. `uv run optbench board` scored the committed `results/campaign` files and `results/asv` tracks.
The ranked cell is the v2 PAR-2 mean. `mean`, `median`, `min`, and `max` are the raw force-call counts, failures included.
A row below the problem's stated full set is partial and is not ranked.
The Pt-island saddle problem states no full set, so the 16-start ASE dimer files stay off the ranked table because those result files set `ranked false`.
Two files with the same problem, template, and code version are both kept. `current` is the registry commit, and the campaign file wins a tie.
Pages were not deployed.

## Top 3

| Problem | Rank | Template | Version | Source | Commit | Current | PAR-2 | Mean | Median | Min | Max | nfailed | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `double-ended-saddle-search/lj38` | 1 | `rgsaddle/ew-ci-neb-lbfgs` | 0.2.2 | asv | `c0db2b8d333c` | yes | 853.56 | 673.94 | 529.0 | 304.0 | 2500.0 | 3 | 50 |
| `double-ended-saddle-search/lj38` | 2 | `rgsaddle/ci-neb-lbfgs` | 0.2.2 | asv | `c0db2b8d333c` | yes | 1208.52 | 767.34 | 522.0 | 322.0 | 2500.0 | 7 | 50 |
| `double-ended-saddle-search/lj38` | 3 | `rgsaddle/oci-neb-lbfgs` | 0.2.2 | asv | `c0db2b8d333c` | yes | 1230.8 | 880.8 | 553.0 | 348.0 | 2500.0 | 7 | 50 |
| `double-ended-saddle-search/lj38_2ts_separation` | 1 | `rgsaddle/ew-ci-neb-lbfgs` | 0.2.2 | asv | `c0db2b8d333c` | yes | 8024.431372549019 | 4639.803921568628 | 2414.0 | 584.0 | 10000.0 | 17 | 51 |
| `double-ended-saddle-search/lj38_2ts_separation` | 2 | `rgsaddle/ew-ci-neb` | 0.2.2 | asv | `c0db2b8d333c` | yes | 12098.058823529413 | 6803.941176470588 | 10000.0 | 1044.0 | 10000.0 | 27 | 51 |
| `double-ended-saddle-search/lj38_2ts_separation` | 3 | `rgsaddle/sidpp-ci-neb-lbfgs` | 0.2.2 | asv | `c0db2b8d333c` | yes | 15922.333333333334 | 8154.745098039216 | 10000.0 | 932.0 | 10000.0 | 39 | 51 |
| `double-ended-saddle-search/pt-island` | 1 | `rgsaddle/ew-ci-neb-lbfgs` | 0.2.2 | asv | `c0db2b8d333c` | yes | 422.135593220339 | 224.01694915254237 | 226.0 | 100.0 | 460.0 | 2 | 59 |
| `double-ended-saddle-search/pt-island` | 2 | `rgsaddle/oci-neb-lbfgs` | 0.2.2 | asv | `c0db2b8d333c` | yes | 677.8305084745763 | 474.4406779661017 | 238.0 | 127.0 | 3000.0 | 4 | 59 |
| `double-ended-saddle-search/pt-island` | 3 | `rgsaddle/idpp-ci-neb-lbfgs` | 0.2.2 | asv | `c0db2b8d333c` | yes | 707.8474576271186 | 213.3728813559322 | 206.0 | 95.0 | 863.0 | 5 | 59 |
| `lowest-eigenvector/lj38` | 1 | `rgsaddle/mode-lanczos` | 0.2.2 | campaign | `c0db2b8d333c` | yes | 26.12 | 26.12 | 26.0 | 14.0 | 98.0 | 0 | 200 |
| `lowest-eigenvector/lj38` | 2 | `rgsaddle/mode-lanczos` | 0.2.2 | asv | `c0db2b8d333c` | no | 26.12 | 26.12 | 26.0 | 14.0 | 98.0 | 0 | 200 |
| `lowest-eigenvector/lj38` | 3 | `rgsaddle/mode-dimer` | 0.2.2 | campaign | `c0db2b8d333c` | yes | 31.4 | 31.4 | 23.0 | 23.0 | 65.0 | 0 | 200 |
| `minimization/lj38` | 1 | `rgmin/lj38-dogleg` | 0.4.0 | campaign | `24236559f381` | yes | 81.881 | 81.881 | 79.0 | 45.0 | 162.0 | 0 | 1000 |
| `minimization/lj38` | 2 | `rgmin/lj38-dogleg` | 0.4.0 | asv | `24236559f381` | no | 81.881 | 81.881 | 79.0 | 45.0 | 162.0 | 0 | 1000 |
| `minimization/lj38` | 3 | `rgmin/lj38-dogleg-tuned` | 0.4.0 | campaign | `24236559f381` | yes | 81.881 | 81.881 | 79.0 | 45.0 | 162.0 | 0 | 1000 |
| `minimization/morse-bulk` | 1 | `rgmin/morse-newton-tuned` | 0.4.0 | campaign | `24236559f381` | yes | 8.35 | 8.35 | 9.0 | 4.0 | 13.0 | 0 | 100 |
| `minimization/morse-bulk` | 2 | `rgmin/morse-newton-tuned` | 0.4.0 | asv | `24236559f381` | no | 8.35 | 8.35 | 9.0 | 4.0 | 13.0 | 0 | 100 |
| `minimization/morse-bulk` | 3 | `rgmin/morse-dogleg` | 0.4.0 | campaign | `24236559f381` | yes | 10.11 | 10.11 | 10.0 | 6.0 | 17.0 | 0 | 100 |
| `neb/Pt-heptamer` | 1 | `ase/neb-pt-heptamer-tuned` | 3.29.0 | campaign | `f27c0005ae6a` | yes | 55.32692307692308 | 55.32692307692308 | 56.25 | 31.25 | 74.25 | 0 | 13 |
| `neb/Pt-heptamer` | 2 | `ase/neb-pt-heptamer-default` | 3.29.0 | campaign | `f27c0005ae6a` | yes | 75.86538461538461 | 75.86538461538461 | 77.25 | 47.25 | 99.25 | 0 | 13 |
| `neb/Pt-heptamer` | 3 | `rgsaddle/idpp-ci-neb-lbfgs` | 0.2.2 | asv | `c0db2b8d333c` | yes | 318.3076923076923 | 318.3076923076923 | 314.0 | 210.0 | 458.0 | 0 | 13 |
| `saddle-search/pt-island` | | | | | | | | | | | | | 0 ranked |
| `tsopt/LJ38` | 1 | `sella/lj38-ts-default` | 2.6.0 | campaign | `6cfb60cdb100` | yes | 68.74 | 68.74 | 59.5 | 25.0 | 257.0 | 0 | 200 |
| `tsopt/LJ38` | 2 | `rgsaddle/prfo-tuned` | 0.2.2 | campaign | `c0db2b8d333c` | yes | 192.6 | 192.6 | 173.0 | 135.0 | 1691.0 | 0 | 200 |
| `tsopt/LJ38` | 3 | `rgsaddle/prfo-tuned` | 0.2.2 | asv | `c0db2b8d333c` | no | 192.6 | 192.6 | 173.0 | 135.0 | 1691.0 | 0 | 200 |
| `tsopt/pt-island` | 1 | `sella/pt-island-ts-default` | 2.6.0 | campaign | `6cfb60cdb100` | yes | 74.28571428571429 | 74.28571428571429 | 71.0 | 41.0 | 128.0 | 0 | 49 |
| `tsopt/pt-island` | 2 | `sella/pt-island-ts-tuned` | 2.6.0 | campaign | `6cfb60cdb100` | yes | 87.55102040816327 | 87.55102040816327 | 88.0 | 67.0 | 141.0 | 0 | 49 |
| `tsopt/pt-island` | 3 | `rgsaddle/pt-dimer-rotskip` | 0.2.2 | campaign | `c0db2b8d333c` | yes | 223.89795918367346 | 223.89795918367346 | 145.0 | 105.0 | 3962.0 | 0 | 49 |

## Every ranked score (276)

| Problem | Template | Version | Source | Commit | Current | PAR-2 | Mean | Median | Min | Max | nfailed | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `double-ended-saddle-search/lj38` | `rgsaddle/ew-ci-neb-lbfgs` | 0.2.2 | asv | `c0db2b8d333c` | yes | 853.56 | 673.94 | 529.0 | 304.0 | 2500.0 | 3 | 50 |
| `double-ended-saddle-search/lj38` | `rgsaddle/ci-neb-lbfgs` | 0.2.2 | asv | `c0db2b8d333c` | yes | 1208.52 | 767.34 | 522.0 | 322.0 | 2500.0 | 7 | 50 |
| `double-ended-saddle-search/lj38` | `rgsaddle/oci-neb-lbfgs` | 0.2.2 | asv | `c0db2b8d333c` | yes | 1230.8 | 880.8 | 553.0 | 348.0 | 2500.0 | 7 | 50 |
| `double-ended-saddle-search/lj38` | `rgsaddle/idpp-ci-neb-lbfgs` | 0.2.2 | asv | `c0db2b8d333c` | yes | 1253.92 | 811.24 | 522.0 | 307.0 | 2500.0 | 7 | 50 |
| `double-ended-saddle-search/lj38` | `rgsaddle/sidpp-ci-neb-lbfgs` | 0.2.2 | asv | `c0db2b8d333c` | yes | 1282.98 | 793.8 | 534.5 | 312.0 | 2500.0 | 8 | 50 |
| `double-ended-saddle-search/lj38` | `rgsaddle/ew-ci-neb` | 0.2.2 | asv | `c0db2b8d333c` | yes | 2145.72 | 1595.72 | 1461.5 | 674.0 | 2500.0 | 11 | 50 |
| `double-ended-saddle-search/lj38` | `rgsaddle/ci-neb` | 0.2.2 | asv | `c0db2b8d333c` | yes | 2272.18 | 1617.82 | 1484.5 | 707.0 | 2500.0 | 13 | 50 |
| `double-ended-saddle-search/lj38` | `rgsaddle/oci-neb` | 0.2.2 | asv | `c0db2b8d333c` | yes | 2289.86 | 1639.86 | 1508.5 | 731.0 | 2500.0 | 13 | 50 |
| `double-ended-saddle-search/lj38` | `rgsaddle/idpp-ci-neb` | 0.2.2 | asv | `c0db2b8d333c` | yes | 2374.04 | 1670.18 | 1614.5 | 742.0 | 2500.0 | 14 | 50 |
| `double-ended-saddle-search/lj38` | `rgsaddle/sidpp-ci-neb` | 0.2.2 | asv | `c0db2b8d333c` | yes | 2473.46 | 1664.2 | 1454.5 | 602.0 | 2500.0 | 16 | 50 |
| `double-ended-saddle-search/lj38` | `ase/de-lj38-default` | 3.29.0 | campaign | `f27c0005ae6a` | yes | 4912.84 | 2462.84 | 2500.0 | 642.0 | 2500.0 | 49 | 50 |
| `double-ended-saddle-search/lj38` | `ase/de-lj38-tuned` | 3.29.0 | campaign | `f27c0005ae6a` | yes | 5000.0 | 2500.0 | 2500.0 | 2500.0 | 2500.0 | 50 | 50 |
| `double-ended-saddle-search/lj38` | `rgsaddle/neb` | 0.2.2 | asv | `c0db2b8d333c` | yes | 5000.0 | 2126.6 | 2478.5 | 987.0 | 2500.0 | 50 | 50 |
| `double-ended-saddle-search/lj38` | `rgsaddle/neb-lbfgs` | 0.2.2 | asv | `c0db2b8d333c` | yes | 5000.0 | 951.72 | 649.5 | 267.0 | 2500.0 | 50 | 50 |
| `double-ended-saddle-search/lj38_2ts_separation` | `rgsaddle/ew-ci-neb-lbfgs` | 0.2.2 | asv | `c0db2b8d333c` | yes | 8024.431372549019 | 4639.803921568628 | 2414.0 | 584.0 | 10000.0 | 17 | 51 |
| `double-ended-saddle-search/lj38_2ts_separation` | `rgsaddle/ew-ci-neb` | 0.2.2 | asv | `c0db2b8d333c` | yes | 12098.058823529413 | 6803.941176470588 | 10000.0 | 1044.0 | 10000.0 | 27 | 51 |
| `double-ended-saddle-search/lj38_2ts_separation` | `rgsaddle/sidpp-ci-neb-lbfgs` | 0.2.2 | asv | `c0db2b8d333c` | yes | 15922.333333333334 | 8154.745098039216 | 10000.0 | 932.0 | 10000.0 | 39 | 51 |
| `double-ended-saddle-search/lj38_2ts_separation` | `rgsaddle/idpp-ci-neb-lbfgs` | 0.2.2 | asv | `c0db2b8d333c` | yes | 16111.745098039215 | 8109.35294117647 | 10000.0 | 907.0 | 10000.0 | 39 | 51 |
| `double-ended-saddle-search/lj38_2ts_separation` | `rgsaddle/ci-neb` | 0.2.2 | asv | `c0db2b8d333c` | yes | 16662.745098039217 | 8309.862745098038 | 10000.0 | 1467.0 | 10000.0 | 41 | 51 |
| `double-ended-saddle-search/lj38_2ts_separation` | `rgsaddle/oci-neb` | 0.2.2 | asv | `c0db2b8d333c` | yes | 16682.13725490196 | 8470.333333333334 | 10000.0 | 1490.0 | 10000.0 | 41 | 51 |
| `double-ended-saddle-search/lj38_2ts_separation` | `rgsaddle/ci-neb-lbfgs` | 0.2.2 | asv | `c0db2b8d333c` | yes | 16919.176470588234 | 8211.921568627451 | 10000.0 | 812.0 | 10000.0 | 42 | 51 |
| `double-ended-saddle-search/lj38_2ts_separation` | `rgsaddle/oci-neb-lbfgs` | 0.2.2 | asv | `c0db2b8d333c` | yes | 16924.70588235294 | 8228.176470588236 | 10000.0 | 838.0 | 10000.0 | 42 | 51 |
| `double-ended-saddle-search/lj38_2ts_separation` | `rgsaddle/idpp-ci-neb` | 0.2.2 | asv | `c0db2b8d333c` | yes | 17645.86274509804 | 8882.941176470587 | 10000.0 | 1312.0 | 10000.0 | 44 | 51 |
| `double-ended-saddle-search/lj38_2ts_separation` | `rgsaddle/sidpp-ci-neb` | 0.2.2 | asv | `c0db2b8d333c` | yes | 17925.529411764706 | 8880.196078431372 | 10000.0 | 1297.0 | 10000.0 | 45 | 51 |
| `double-ended-saddle-search/lj38_2ts_separation` | `ase/de-lj38-2ts-default` | 3.29.0 | campaign | `f27c0005ae6a` | yes | 20000.0 | 2500.0 | 2500.0 | 2500.0 | 2500.0 | 51 | 51 |
| `double-ended-saddle-search/lj38_2ts_separation` | `ase/de-lj38-2ts-tuned` | 3.29.0 | campaign | `f27c0005ae6a` | yes | 20000.0 | 2500.0 | 2500.0 | 2500.0 | 2500.0 | 51 | 51 |
| `double-ended-saddle-search/lj38_2ts_separation` | `rgsaddle/neb` | 0.2.2 | asv | `c0db2b8d333c` | yes | 20000.0 | 9773.862745098038 | 10000.0 | 3457.0 | 10000.0 | 51 | 51 |
| `double-ended-saddle-search/lj38_2ts_separation` | `rgsaddle/neb-lbfgs` | 0.2.2 | asv | `c0db2b8d333c` | yes | 20000.0 | 8351.529411764706 | 10000.0 | 907.0 | 10000.0 | 51 | 51 |
| `double-ended-saddle-search/pt-island` | `rgsaddle/ew-ci-neb-lbfgs` | 0.2.2 | asv | `c0db2b8d333c` | yes | 422.135593220339 | 224.01694915254237 | 226.0 | 100.0 | 460.0 | 2 | 59 |
| `double-ended-saddle-search/pt-island` | `rgsaddle/oci-neb-lbfgs` | 0.2.2 | asv | `c0db2b8d333c` | yes | 677.8305084745763 | 474.4406779661017 | 238.0 | 127.0 | 3000.0 | 4 | 59 |
| `double-ended-saddle-search/pt-island` | `rgsaddle/idpp-ci-neb-lbfgs` | 0.2.2 | asv | `c0db2b8d333c` | yes | 707.8474576271186 | 213.3728813559322 | 206.0 | 95.0 | 863.0 | 5 | 59 |
| `double-ended-saddle-search/pt-island` | `rgsaddle/ci-neb-lbfgs` | 0.2.2 | asv | `c0db2b8d333c` | yes | 733.5762711864406 | 238.08474576271186 | 197.0 | 95.0 | 2336.0 | 5 | 59 |
| `double-ended-saddle-search/pt-island` | `rgsaddle/sidpp-ci-neb-lbfgs` | 0.2.2 | asv | `c0db2b8d333c` | yes | 741.457627118644 | 247.23728813559322 | 209.0 | 95.0 | 2903.0 | 5 | 59 |
| `double-ended-saddle-search/pt-island` | `ase/de-pt-island-tuned` | 3.29.0 | campaign | `f27c0005ae6a` | yes | 847.6779661016949 | 593.8644067796611 | 374.0 | 173.0 | 3005.0 | 5 | 59 |
| `double-ended-saddle-search/pt-island` | `ase/de-pt-island-default` | 3.29.0 | campaign | `f27c0005ae6a` | yes | 926.4745762711865 | 875.7118644067797 | 707.0 | 299.0 | 3005.0 | 1 | 59 |
| `double-ended-saddle-search/pt-island` | `rgsaddle/ew-ci-neb` | 0.2.2 | asv | `c0db2b8d333c` | yes | 938.1016949152543 | 561.6440677966102 | 478.0 | 334.0 | 1333.0 | 4 | 59 |
| `double-ended-saddle-search/pt-island` | `rgsaddle/oci-neb` | 0.2.2 | asv | `c0db2b8d333c` | yes | 975.9830508474577 | 823.4406779661017 | 602.0 | 335.0 | 3000.0 | 3 | 59 |
| `double-ended-saddle-search/pt-island` | `rgsaddle/idpp-ci-neb` | 0.2.2 | asv | `c0db2b8d333c` | yes | 1062.457627118644 | 585.728813559322 | 491.0 | 311.0 | 1367.0 | 5 | 59 |
| `double-ended-saddle-search/pt-island` | `rgsaddle/ci-neb` | 0.2.2 | asv | `c0db2b8d333c` | yes | 1078.4237288135594 | 601.542372881356 | 479.0 | 311.0 | 1280.0 | 5 | 59 |
| `double-ended-saddle-search/pt-island` | `rgsaddle/sidpp-ci-neb` | 0.2.2 | asv | `c0db2b8d333c` | yes | 1134.9322033898304 | 607.1525423728814 | 461.0 | 302.0 | 3000.0 | 6 | 59 |
| `double-ended-saddle-search/pt-island` | `rgsaddle/neb` | 0.2.2 | asv | `c0db2b8d333c` | yes | 6000.0 | 744.4745762711865 | 533.0 | 323.0 | 3000.0 | 59 | 59 |
| `double-ended-saddle-search/pt-island` | `rgsaddle/neb-lbfgs` | 0.2.2 | asv | `c0db2b8d333c` | yes | 6000.0 | 222.16949152542372 | 197.0 | 95.0 | 890.0 | 59 | 59 |
| `lowest-eigenvector/lj38` | `rgsaddle/mode-lanczos` | 0.2.2 | campaign | `c0db2b8d333c` | yes | 26.12 | 26.12 | 26.0 | 14.0 | 98.0 | 0 | 200 |
| `lowest-eigenvector/lj38` | `rgsaddle/mode-lanczos` | 0.2.2 | asv | `c0db2b8d333c` | no | 26.12 | 26.12 | 26.0 | 14.0 | 98.0 | 0 | 200 |
| `lowest-eigenvector/lj38` | `rgsaddle/mode-dimer` | 0.2.2 | campaign | `c0db2b8d333c` | yes | 31.4 | 31.4 | 23.0 | 23.0 | 65.0 | 0 | 200 |
| `lowest-eigenvector/lj38` | `rgsaddle/mode-dimer` | 0.2.2 | asv | `c0db2b8d333c` | no | 31.4 | 31.4 | 23.0 | 23.0 | 65.0 | 0 | 200 |
| `lowest-eigenvector/lj38` | `rgsaddle/mode-lanczos-tuned` | 0.2.2 | campaign | `c0db2b8d333c` | yes | 36.56 | 36.56 | 34.0 | 18.0 | 130.0 | 0 | 200 |
| `lowest-eigenvector/lj38` | `rgsaddle/mode-lanczos-tuned` | 0.2.2 | asv | `c0db2b8d333c` | no | 36.56 | 36.56 | 34.0 | 18.0 | 130.0 | 0 | 200 |
| `lowest-eigenvector/lj38` | `rgsaddle/mode-dimer-tuned` | 0.2.2 | campaign | `c0db2b8d333c` | yes | 45.555 | 45.555 | 33.0 | 33.0 | 126.0 | 0 | 200 |
| `lowest-eigenvector/lj38` | `rgsaddle/mode-dimer-tuned` | 0.2.2 | asv | `c0db2b8d333c` | no | 45.555 | 45.555 | 33.0 | 33.0 | 126.0 | 0 | 200 |
| `minimization/lj38` | `rgmin/lj38-dogleg` | 0.4.0 | campaign | `24236559f381` | yes | 81.881 | 81.881 | 79.0 | 45.0 | 162.0 | 0 | 1000 |
| `minimization/lj38` | `rgmin/lj38-dogleg` | 0.4.0 | asv | `24236559f381` | no | 81.881 | 81.881 | 79.0 | 45.0 | 162.0 | 0 | 1000 |
| `minimization/lj38` | `rgmin/lj38-dogleg-tuned` | 0.4.0 | campaign | `24236559f381` | yes | 81.881 | 81.881 | 79.0 | 45.0 | 162.0 | 0 | 1000 |
| `minimization/lj38` | `rgmin/lj38-dogleg-tuned` | 0.4.0 | asv | `24236559f381` | no | 81.881 | 81.881 | 79.0 | 45.0 | 162.0 | 0 | 1000 |
| `minimization/lj38` | `rgmin/lj38-newton` | 0.4.0 | campaign | `24236559f381` | yes | 166.337 | 66.649 | 64.0 | 35.0 | 128.0 | 5 | 1000 |
| `minimization/lj38` | `rgmin/lj38-newton` | 0.4.0 | asv | `24236559f381` | no | 166.337 | 66.649 | 64.0 | 35.0 | 128.0 | 5 | 1000 |
| `minimization/lj38` | `rgmin/lj38-newton-tuned` | 0.4.0 | campaign | `24236559f381` | yes | 166.337 | 66.649 | 64.0 | 35.0 | 128.0 | 5 | 1000 |
| `minimization/lj38` | `rgmin/lj38-newton-tuned` | 0.4.0 | asv | `24236559f381` | no | 166.337 | 66.649 | 64.0 | 35.0 | 128.0 | 5 | 1000 |
| `minimization/lj38` | `eon/lj38-lbfgs-m100` | 3.6.0 | campaign | `1d71c9445018` | yes | 180.137 | 180.137 | 171.0 | 90.0 | 435.0 | 0 | 1000 |
| `minimization/lj38` | `rgmin/lj38-lbfgs-m100` | 0.4.0 | campaign | `24236559f381` | yes | 197.228 | 177.367 | 170.0 | 96.0 | 468.0 | 1 | 1000 |
| `minimization/lj38` | `rgmin/lj38-lbfgs-m100` | 0.4.0 | asv | `24236559f381` | no | 197.228 | 177.367 | 170.0 | 96.0 | 468.0 | 1 | 1000 |
| `minimization/lj38` | `rgmin/lj38-lbfgs-m100-tuned` | 0.4.0 | campaign | `24236559f381` | yes | 197.228 | 177.367 | 170.0 | 96.0 | 468.0 | 1 | 1000 |
| `minimization/lj38` | `rgmin/lj38-lbfgs-m100-tuned` | 0.4.0 | asv | `24236559f381` | no | 197.228 | 177.367 | 170.0 | 96.0 | 468.0 | 1 | 1000 |
| `minimization/lj38` | `eon/lj38-lbfgs` | 3.6.0 | campaign | `1d71c9445018` | yes | 203.898 | 203.898 | 190.0 | 98.0 | 485.0 | 0 | 1000 |
| `minimization/lj38` | `sella/lj38-min-tuned` | 2.6.0 | campaign | `6cfb60cdb100` | yes | 253.921 | 253.921 | 248.0 | 150.0 | 533.0 | 0 | 1000 |
| `minimization/lj38` | `rgmin/lj38-lbfgs-m20` | 0.4.0 | campaign | `24236559f381` | yes | 282.64 | 203.366 | 192.0 | 94.0 | 464.0 | 4 | 1000 |
| `minimization/lj38` | `rgmin/lj38-lbfgs-m20` | 0.4.0 | asv | `24236559f381` | no | 282.64 | 203.366 | 192.0 | 94.0 | 464.0 | 4 | 1000 |
| `minimization/lj38` | `rgmin/lj38-lbfgs-m20-tuned` | 0.4.0 | campaign | `24236559f381` | yes | 282.64 | 203.366 | 192.0 | 94.0 | 464.0 | 4 | 1000 |
| `minimization/lj38` | `rgmin/lj38-lbfgs-m20-tuned` | 0.4.0 | asv | `24236559f381` | no | 282.64 | 203.366 | 192.0 | 94.0 | 464.0 | 4 | 1000 |
| `minimization/lj38` | `rgmin/lj38-lbfgs` | 0.4.0 | campaign | `24236559f381` | yes | 294.105 | 214.951 | 206.0 | 91.0 | 477.0 | 4 | 1000 |
| `minimization/lj38` | `rgmin/lj38-lbfgs` | 0.4.0 | asv | `24236559f381` | no | 294.105 | 214.951 | 206.0 | 91.0 | 477.0 | 4 | 1000 |
| `minimization/lj38` | `rgmin/lj38-lbfgs-tuned` | 0.4.0 | campaign | `24236559f381` | yes | 294.105 | 214.951 | 206.0 | 91.0 | 477.0 | 4 | 1000 |
| `minimization/lj38` | `rgmin/lj38-lbfgs-tuned` | 0.4.0 | asv | `24236559f381` | no | 294.105 | 214.951 | 206.0 | 91.0 | 477.0 | 4 | 1000 |
| `minimization/lj38` | `ase/lj38-lbfgs-tuned` | 3.29.0 | campaign | `f27c0005ae6a` | yes | 395.645 | 375.645 | 319.0 | 166.0 | 10000.0 | 2 | 1000 |
| `minimization/lj38` | `ase/lj38-bfgslinesearch-tuned` | 3.29.0 | campaign | `f27c0005ae6a` | yes | 506.049 | 506.049 | 490.5 | 323.0 | 889.0 | 0 | 1000 |
| `minimization/lj38` | `ase/lj38-bfgs-tuned` | 3.29.0 | campaign | `f27c0005ae6a` | yes | 519.938 | 489.938 | 414.5 | 244.0 | 10000.0 | 3 | 1000 |
| `minimization/lj38` | `ase/lj38-bfgslinesearch-default` | 3.29.0 | campaign | `f27c0005ae6a` | yes | 561.466 | 561.466 | 551.0 | 366.0 | 876.0 | 0 | 1000 |
| `minimization/lj38` | `rgmin/lj38-bb` | 0.4.0 | campaign | `24236559f381` | yes | 639.474 | 521.461 | 471.5 | 172.0 | 1596.0 | 6 | 1000 |
| `minimization/lj38` | `rgmin/lj38-bb` | 0.4.0 | asv | `24236559f381` | no | 639.474 | 521.461 | 471.5 | 172.0 | 1596.0 | 6 | 1000 |
| `minimization/lj38` | `rgmin/lj38-bb-tuned` | 0.4.0 | campaign | `24236559f381` | yes | 639.474 | 521.461 | 471.5 | 172.0 | 1596.0 | 6 | 1000 |
| `minimization/lj38` | `rgmin/lj38-bb-tuned` | 0.4.0 | asv | `24236559f381` | no | 639.474 | 521.461 | 471.5 | 172.0 | 1596.0 | 6 | 1000 |
| `minimization/lj38` | `rgmin/lj38-cg` | 0.4.0 | campaign | `24236559f381` | yes | 669.237 | 591.346 | 538.0 | 191.0 | 2329.0 | 4 | 1000 |
| `minimization/lj38` | `rgmin/lj38-cg` | 0.4.0 | asv | `24236559f381` | no | 669.237 | 591.346 | 538.0 | 191.0 | 2329.0 | 4 | 1000 |
| `minimization/lj38` | `rgmin/lj38-cg-tuned` | 0.4.0 | campaign | `24236559f381` | yes | 669.237 | 591.346 | 538.0 | 191.0 | 2329.0 | 4 | 1000 |
| `minimization/lj38` | `rgmin/lj38-cg-tuned` | 0.4.0 | asv | `24236559f381` | no | 669.237 | 591.346 | 538.0 | 191.0 | 2329.0 | 4 | 1000 |
| `minimization/lj38` | `ase/lj38-preconlbfgs-default` | 3.29.0 | campaign | `f27c0005ae6a` | yes | 690.119 | 690.119 | 652.5 | 355.0 | 1801.0 | 0 | 1000 |
| `minimization/lj38` | `ase/lj38-fire-default` | 3.29.0 | campaign | `f27c0005ae6a` | yes | 706.005 | 706.005 | 604.0 | 199.0 | 2340.0 | 0 | 1000 |
| `minimization/lj38` | `sella/lj38-min-default` | 2.6.0 | campaign | `6cfb60cdb100` | yes | 710.089 | 710.089 | 692.5 | 410.0 | 1161.0 | 0 | 1000 |
| `minimization/lj38` | `ase/lj38-fire-tuned` | 3.29.0 | campaign | `f27c0005ae6a` | yes | 719.638 | 719.638 | 610.5 | 208.0 | 2598.0 | 0 | 1000 |
| `minimization/lj38` | `rgmin/lj38-fire` | 0.4.0 | campaign | `24236559f381` | yes | 769.36 | 711.609 | 625.5 | 230.0 | 2802.0 | 3 | 1000 |
| `minimization/lj38` | `rgmin/lj38-fire` | 0.4.0 | asv | `24236559f381` | no | 769.36 | 711.609 | 625.5 | 230.0 | 2802.0 | 3 | 1000 |
| `minimization/lj38` | `rgmin/lj38-fire-tuned` | 0.4.0 | campaign | `24236559f381` | yes | 769.36 | 711.609 | 625.5 | 230.0 | 2802.0 | 3 | 1000 |
| `minimization/lj38` | `rgmin/lj38-fire-tuned` | 0.4.0 | asv | `24236559f381` | no | 769.36 | 711.609 | 625.5 | 230.0 | 2802.0 | 3 | 1000 |
| `minimization/lj38` | `rgmin/lj38-fire2` | 0.4.0 | campaign | `24236559f381` | yes | 845.947 | 730.952 | 639.5 | 236.0 | 2637.0 | 6 | 1000 |
| `minimization/lj38` | `rgmin/lj38-fire2` | 0.4.0 | asv | `24236559f381` | no | 845.947 | 730.952 | 639.5 | 236.0 | 2637.0 | 6 | 1000 |
| `minimization/lj38` | `rgmin/lj38-fire2-tuned` | 0.4.0 | campaign | `24236559f381` | yes | 845.947 | 730.952 | 639.5 | 236.0 | 2637.0 | 6 | 1000 |
| `minimization/lj38` | `rgmin/lj38-fire2-tuned` | 0.4.0 | asv | `24236559f381` | no | 845.947 | 730.952 | 639.5 | 236.0 | 2637.0 | 6 | 1000 |
| `minimization/lj38` | `ase/lj38-bfgs-default` | 3.29.0 | campaign | `f27c0005ae6a` | yes | 1003.995 | 823.995 | 458.0 | 261.0 | 10000.0 | 18 | 1000 |
| `minimization/lj38` | `rgmin/lj38-bfgs` | 0.4.0 | campaign | `24236559f381` | yes | 1010.446 | 1010.446 | 970.0 | 595.0 | 2985.0 | 0 | 1000 |
| `minimization/lj38` | `rgmin/lj38-bfgs` | 0.4.0 | asv | `24236559f381` | no | 1010.446 | 1010.446 | 970.0 | 595.0 | 2985.0 | 0 | 1000 |
| `minimization/lj38` | `rgmin/lj38-bfgs-tuned` | 0.4.0 | campaign | `24236559f381` | yes | 1010.446 | 1010.446 | 970.0 | 595.0 | 2985.0 | 0 | 1000 |
| `minimization/lj38` | `rgmin/lj38-bfgs-tuned` | 0.4.0 | asv | `24236559f381` | no | 1010.446 | 1010.446 | 970.0 | 595.0 | 2985.0 | 0 | 1000 |
| `minimization/lj38` | `ase/lj38-lbfgs-default` | 3.29.0 | campaign | `f27c0005ae6a` | yes | 1189.647 | 769.647 | 319.0 | 175.0 | 10000.0 | 42 | 1000 |
| `minimization/lj38` | `ase/lj38-preconfire-default` | 3.29.0 | campaign | `f27c0005ae6a` | yes | 1816.586 | 1816.586 | 1562.0 | 659.0 | 8441.0 | 0 | 1000 |
| `minimization/lj38` | `rgmin/lj38-quickmin` | 0.4.0 | campaign | `24236559f381` | yes | 1932.997 | 1897.225 | 1598.0 | 467.0 | 8671.0 | 2 | 1000 |
| `minimization/lj38` | `rgmin/lj38-quickmin` | 0.4.0 | asv | `24236559f381` | no | 1932.997 | 1897.225 | 1598.0 | 467.0 | 8671.0 | 2 | 1000 |
| `minimization/lj38` | `rgmin/lj38-quickmin-tuned` | 0.4.0 | campaign | `24236559f381` | yes | 1932.997 | 1897.225 | 1598.0 | 467.0 | 8671.0 | 2 | 1000 |
| `minimization/lj38` | `rgmin/lj38-quickmin-tuned` | 0.4.0 | asv | `24236559f381` | no | 1932.997 | 1897.225 | 1598.0 | 467.0 | 8671.0 | 2 | 1000 |
| `minimization/lj38` | `rgmin/lj38-diis` | 0.4.1 | campaign | `fd26a60ddd48` | yes | 1967.417 | 1930.918 | 1652.0 | 395.0 | 8624.0 | 2 | 1000 |
| `minimization/lj38` | `rgmin/lj38-sr1` | 0.4.1 | campaign | `fd26a60ddd48` | yes | 2185.659 | 2093.014 | 1728.0 | 292.0 | 10000.0 | 6 | 1000 |
| `minimization/lj38` | `rgmin/lj38-sr2` | 0.4.1 | campaign | `fd26a60ddd48` | yes | 2298.441 | 2230.075 | 1837.5 | 319.0 | 10000.0 | 6 | 1000 |
| `minimization/lj38` | `rgmin/lj38-sd` | 0.4.0 | campaign | `24236559f381` | yes | 8392.748 | 6350.193 | 6099.0 | 1141.0 | 10000.0 | 203 | 1000 |
| `minimization/lj38` | `rgmin/lj38-sd` | 0.4.0 | asv | `24236559f381` | no | 8392.748 | 6350.193 | 6099.0 | 1141.0 | 10000.0 | 203 | 1000 |
| `minimization/lj38` | `rgmin/lj38-sd-tuned` | 0.4.0 | campaign | `24236559f381` | yes | 8392.748 | 6350.193 | 6099.0 | 1141.0 | 10000.0 | 203 | 1000 |
| `minimization/lj38` | `rgmin/lj38-sd-tuned` | 0.4.0 | asv | `24236559f381` | no | 8392.748 | 6350.193 | 6099.0 | 1141.0 | 10000.0 | 203 | 1000 |
| `minimization/lj38` | `rgmin/lj38-adam` | 0.4.1 | campaign | `fd26a60ddd48` | yes | 13835.677 | 8365.677 | 10000.0 | 852.0 | 10000.0 | 547 | 1000 |
| `minimization/lj38` | `rgmin/lj38-adam` | 0.4.0 | campaign | `24236559f381` | yes | 16420.159 | 9110.159 | 10000.0 | 2033.0 | 10000.0 | 731 | 1000 |
| `minimization/lj38` | `rgmin/lj38-adam` | 0.4.0 | asv | `24236559f381` | no | 16420.159 | 9110.159 | 10000.0 | 2033.0 | 10000.0 | 731 | 1000 |
| `minimization/lj38` | `rgmin/lj38-adam-tuned` | 0.4.0 | campaign | `24236559f381` | yes | 16420.159 | 9110.159 | 10000.0 | 2033.0 | 10000.0 | 731 | 1000 |
| `minimization/lj38` | `rgmin/lj38-adam-tuned` | 0.4.0 | asv | `24236559f381` | no | 16420.159 | 9110.159 | 10000.0 | 2033.0 | 10000.0 | 731 | 1000 |
| `minimization/lj38` | `rgmin/lj38-sr1` | 0.4.0 | campaign | `24236559f381` | yes | 19908.901 | 8133.101 | 10000.0 | 234.0 | 10000.0 | 995 | 1000 |
| `minimization/lj38` | `rgmin/lj38-sr1` | 0.4.0 | asv | `24236559f381` | no | 19908.901 | 8133.101 | 10000.0 | 234.0 | 10000.0 | 995 | 1000 |
| `minimization/lj38` | `rgmin/lj38-sr1-tuned` | 0.4.0 | campaign | `24236559f381` | yes | 19908.901 | 8133.101 | 10000.0 | 234.0 | 10000.0 | 995 | 1000 |
| `minimization/lj38` | `rgmin/lj38-sr1-tuned` | 0.4.0 | asv | `24236559f381` | no | 19908.901 | 8133.101 | 10000.0 | 234.0 | 10000.0 | 995 | 1000 |
| `minimization/lj38` | `rgmin/lj38-diis` | 0.4.0 | campaign | `24236559f381` | yes | 19945.03 | 9836.207 | 10000.0 | 1690.0 | 10000.0 | 996 | 1000 |
| `minimization/lj38` | `rgmin/lj38-diis` | 0.4.0 | asv | `24236559f381` | no | 19945.03 | 9836.207 | 10000.0 | 1690.0 | 10000.0 | 996 | 1000 |
| `minimization/lj38` | `rgmin/lj38-diis-tuned` | 0.4.0 | campaign | `24236559f381` | yes | 19945.03 | 9836.207 | 10000.0 | 1690.0 | 10000.0 | 996 | 1000 |
| `minimization/lj38` | `rgmin/lj38-diis-tuned` | 0.4.0 | asv | `24236559f381` | no | 19945.03 | 9836.207 | 10000.0 | 1690.0 | 10000.0 | 996 | 1000 |
| `minimization/lj38` | `rgmin/lj38-sr2` | 0.4.0 | campaign | `24236559f381` | yes | 20000.0 | 9990.449 | 10000.0 | 449.0 | 10000.0 | 1000 | 1000 |
| `minimization/lj38` | `rgmin/lj38-sr2` | 0.4.0 | asv | `24236559f381` | no | 20000.0 | 9990.449 | 10000.0 | 449.0 | 10000.0 | 1000 | 1000 |
| `minimization/lj38` | `rgmin/lj38-sr2-tuned` | 0.4.0 | campaign | `24236559f381` | yes | 20000.0 | 9990.449 | 10000.0 | 449.0 | 10000.0 | 1000 | 1000 |
| `minimization/lj38` | `rgmin/lj38-sr2-tuned` | 0.4.0 | asv | `24236559f381` | no | 20000.0 | 9990.449 | 10000.0 | 449.0 | 10000.0 | 1000 | 1000 |
| `minimization/morse-bulk` | `rgmin/morse-newton-tuned` | 0.4.0 | campaign | `24236559f381` | yes | 8.35 | 8.35 | 9.0 | 4.0 | 13.0 | 0 | 100 |
| `minimization/morse-bulk` | `rgmin/morse-newton-tuned` | 0.4.0 | asv | `24236559f381` | no | 8.35 | 8.35 | 9.0 | 4.0 | 13.0 | 0 | 100 |
| `minimization/morse-bulk` | `rgmin/morse-dogleg` | 0.4.0 | campaign | `24236559f381` | yes | 10.11 | 10.11 | 10.0 | 6.0 | 17.0 | 0 | 100 |
| `minimization/morse-bulk` | `rgmin/morse-dogleg` | 0.4.0 | asv | `24236559f381` | no | 10.11 | 10.11 | 10.0 | 6.0 | 17.0 | 0 | 100 |
| `minimization/morse-bulk` | `rgmin/morse-lbfgs-m20-tuned` | 0.4.0 | campaign | `24236559f381` | yes | 26.87 | 26.87 | 27.0 | 22.0 | 31.0 | 0 | 100 |
| `minimization/morse-bulk` | `rgmin/morse-lbfgs-m20-tuned` | 0.4.0 | asv | `24236559f381` | no | 26.87 | 26.87 | 27.0 | 22.0 | 31.0 | 0 | 100 |
| `minimization/morse-bulk` | `rgmin/morse-lbfgs-m100-tuned` | 0.4.0 | campaign | `24236559f381` | yes | 26.91 | 26.91 | 27.0 | 22.0 | 31.0 | 0 | 100 |
| `minimization/morse-bulk` | `rgmin/morse-lbfgs-m100-tuned` | 0.4.0 | asv | `24236559f381` | no | 26.91 | 26.91 | 27.0 | 22.0 | 31.0 | 0 | 100 |
| `minimization/morse-bulk` | `rgmin/morse-lbfgs-tuned` | 0.4.0 | campaign | `24236559f381` | yes | 26.93 | 26.93 | 27.0 | 22.0 | 31.0 | 0 | 100 |
| `minimization/morse-bulk` | `rgmin/morse-lbfgs-tuned` | 0.4.0 | asv | `24236559f381` | no | 26.93 | 26.93 | 27.0 | 22.0 | 31.0 | 0 | 100 |
| `minimization/morse-bulk` | `rgmin/morse-lbfgs-m20` | 0.4.0 | campaign | `24236559f381` | yes | 27.05 | 27.05 | 27.0 | 23.0 | 34.0 | 0 | 100 |
| `minimization/morse-bulk` | `rgmin/morse-lbfgs-m20` | 0.4.0 | asv | `24236559f381` | no | 27.05 | 27.05 | 27.0 | 23.0 | 34.0 | 0 | 100 |
| `minimization/morse-bulk` | `rgmin/morse-lbfgs-m100` | 0.4.0 | campaign | `24236559f381` | yes | 27.12 | 27.12 | 27.0 | 23.0 | 35.0 | 0 | 100 |
| `minimization/morse-bulk` | `rgmin/morse-lbfgs-m100` | 0.4.0 | asv | `24236559f381` | no | 27.12 | 27.12 | 27.0 | 23.0 | 35.0 | 0 | 100 |
| `minimization/morse-bulk` | `rgmin/morse-lbfgs` | 0.4.0 | campaign | `24236559f381` | yes | 27.18 | 27.18 | 27.5 | 23.0 | 34.0 | 0 | 100 |
| `minimization/morse-bulk` | `rgmin/morse-lbfgs` | 0.4.0 | asv | `24236559f381` | no | 27.18 | 27.18 | 27.5 | 23.0 | 34.0 | 0 | 100 |
| `minimization/morse-bulk` | `rgmin/morse-diis-tuned` | 0.4.1 | campaign | `fd26a60ddd48` | yes | 27.44 | 27.44 | 28.0 | 21.0 | 32.0 | 0 | 100 |
| `minimization/morse-bulk` | `sella/morse-min-default` | 2.6.0 | campaign | `6cfb60cdb100` | yes | 28.89 | 28.89 | 28.0 | 20.0 | 41.0 | 0 | 100 |
| `minimization/morse-bulk` | `ase/morse-bfgslinesearch-default` | 3.29.0 | campaign | `f27c0005ae6a` | yes | 28.91 | 28.91 | 28.0 | 21.0 | 37.0 | 0 | 100 |
| `minimization/morse-bulk` | `rgmin/morse-diis` | 0.4.1 | campaign | `fd26a60ddd48` | yes | 28.94 | 28.94 | 29.0 | 21.0 | 42.0 | 0 | 100 |
| `minimization/morse-bulk` | `ase/morse-preconlbfgs-tuned` | 3.29.0 | campaign | `f27c0005ae6a` | yes | 28.98 | 28.98 | 29.0 | 16.0 | 42.0 | 0 | 100 |
| `minimization/morse-bulk` | `rgmin/morse-dogleg-tuned` | 0.4.0 | campaign | `24236559f381` | yes | 31.26 | 31.26 | 32.5 | 7.0 | 47.0 | 0 | 100 |
| `minimization/morse-bulk` | `rgmin/morse-dogleg-tuned` | 0.4.0 | asv | `24236559f381` | no | 31.26 | 31.26 | 32.5 | 7.0 | 47.0 | 0 | 100 |
| `minimization/morse-bulk` | `ase/morse-lbfgs-tuned` | 3.29.0 | campaign | `f27c0005ae6a` | yes | 32.03 | 32.03 | 32.0 | 26.0 | 36.0 | 0 | 100 |
| `minimization/morse-bulk` | `rgmin/morse-bb-tuned` | 0.4.0 | campaign | `24236559f381` | yes | 33.22 | 33.22 | 34.0 | 24.0 | 39.0 | 0 | 100 |
| `minimization/morse-bulk` | `rgmin/morse-bb-tuned` | 0.4.0 | asv | `24236559f381` | no | 33.22 | 33.22 | 34.0 | 24.0 | 39.0 | 0 | 100 |
| `minimization/morse-bulk` | `ase/morse-preconode12r-default` | 3.29.0 | campaign | `f27c0005ae6a` | yes | 41.78 | 41.78 | 44.0 | 21.0 | 56.0 | 0 | 100 |
| `minimization/morse-bulk` | `ase/morse-preconode12r-tuned` | 3.29.0 | campaign | `f27c0005ae6a` | yes | 41.78 | 41.78 | 44.0 | 21.0 | 56.0 | 0 | 100 |
| `minimization/morse-bulk` | `rgmin/morse-cg-tuned` | 0.4.0 | campaign | `24236559f381` | yes | 50.49 | 50.49 | 50.0 | 36.0 | 70.0 | 0 | 100 |
| `minimization/morse-bulk` | `rgmin/morse-cg-tuned` | 0.4.0 | asv | `24236559f381` | no | 50.49 | 50.49 | 50.0 | 36.0 | 70.0 | 0 | 100 |
| `minimization/morse-bulk` | `ase/morse-bfgs-default` | 3.29.0 | campaign | `f27c0005ae6a` | yes | 53.36 | 53.36 | 54.0 | 44.0 | 56.0 | 0 | 100 |
| `minimization/morse-bulk` | `ase/morse-lbfgs-default` | 3.29.0 | campaign | `f27c0005ae6a` | yes | 53.36 | 53.36 | 54.0 | 44.0 | 56.0 | 0 | 100 |
| `minimization/morse-bulk` | `rgmin/morse-bfgs` | 0.4.0 | campaign | `24236559f381` | yes | 54.04 | 54.04 | 50.0 | 28.0 | 175.0 | 0 | 100 |
| `minimization/morse-bulk` | `rgmin/morse-bfgs` | 0.4.0 | asv | `24236559f381` | no | 54.04 | 54.04 | 50.0 | 28.0 | 175.0 | 0 | 100 |
| `minimization/morse-bulk` | `eon/morse-lbfgs-m100` | 3.6.0 | campaign | `1d71c9445018` | yes | 54.18 | 54.18 | 53.0 | 36.0 | 101.0 | 0 | 100 |
| `minimization/morse-bulk` | `eon/morse-lbfgs` | 3.6.0 | campaign | `1d71c9445018` | yes | 54.31 | 54.31 | 52.0 | 35.0 | 85.0 | 0 | 100 |
| `minimization/morse-bulk` | `rgmin/morse-cg` | 0.4.0 | campaign | `24236559f381` | yes | 54.72 | 54.72 | 54.0 | 38.0 | 85.0 | 0 | 100 |
| `minimization/morse-bulk` | `rgmin/morse-cg` | 0.4.0 | asv | `24236559f381` | no | 54.72 | 54.72 | 54.0 | 38.0 | 85.0 | 0 | 100 |
| `minimization/morse-bulk` | `ase/morse-bfgs-tuned` | 3.29.0 | campaign | `f27c0005ae6a` | yes | 54.92 | 54.92 | 55.0 | 44.0 | 60.0 | 0 | 100 |
| `minimization/morse-bulk` | `rgmin/morse-sr2-tuned` | 0.4.1 | campaign | `fd26a60ddd48` | yes | 59.53 | 59.53 | 61.0 | 40.0 | 71.0 | 0 | 100 |
| `minimization/morse-bulk` | `ase/morse-preconlbfgs-default` | 3.29.0 | campaign | `f27c0005ae6a` | yes | 59.83 | 59.83 | 44.0 | 16.0 | 242.0 | 0 | 100 |
| `minimization/morse-bulk` | `rgmin/morse-sr2` | 0.4.1 | campaign | `fd26a60ddd48` | yes | 60.2 | 60.2 | 60.0 | 45.0 | 74.0 | 0 | 100 |
| `minimization/morse-bulk` | `rgmin/morse-sr1-tuned` | 0.4.1 | campaign | `fd26a60ddd48` | yes | 61.53 | 61.53 | 65.5 | 30.0 | 87.0 | 0 | 100 |
| `minimization/morse-bulk` | `rgmin/morse-quickmin-tuned` | 0.4.0 | campaign | `24236559f381` | yes | 61.75 | 61.75 | 61.0 | 45.0 | 76.0 | 0 | 100 |
| `minimization/morse-bulk` | `rgmin/morse-quickmin-tuned` | 0.4.0 | asv | `24236559f381` | no | 61.75 | 61.75 | 61.0 | 45.0 | 76.0 | 0 | 100 |
| `minimization/morse-bulk` | `rgmin/morse-quickmin` | 0.4.0 | campaign | `24236559f381` | yes | 62.03 | 62.03 | 62.0 | 45.0 | 74.0 | 0 | 100 |
| `minimization/morse-bulk` | `rgmin/morse-quickmin` | 0.4.0 | asv | `24236559f381` | no | 62.03 | 62.03 | 62.0 | 45.0 | 74.0 | 0 | 100 |
| `minimization/morse-bulk` | `rgmin/morse-sr1` | 0.4.1 | campaign | `fd26a60ddd48` | yes | 66.77 | 66.77 | 71.5 | 30.0 | 85.0 | 0 | 100 |
| `minimization/morse-bulk` | `rgmin/morse-bfgs-tuned` | 0.4.0 | campaign | `24236559f381` | yes | 70.53 | 70.53 | 71.0 | 28.0 | 115.0 | 0 | 100 |
| `minimization/morse-bulk` | `rgmin/morse-bfgs-tuned` | 0.4.0 | asv | `24236559f381` | no | 70.53 | 70.53 | 71.0 | 28.0 | 115.0 | 0 | 100 |
| `minimization/morse-bulk` | `sella/morse-min-tuned` | 2.6.0 | campaign | `6cfb60cdb100` | yes | 71.43 | 71.43 | 72.5 | 49.0 | 84.0 | 0 | 100 |
| `minimization/morse-bulk` | `rgmin/morse-sd-tuned` | 0.4.0 | campaign | `24236559f381` | yes | 90.48 | 90.48 | 89.0 | 64.0 | 111.0 | 0 | 100 |
| `minimization/morse-bulk` | `rgmin/morse-sd-tuned` | 0.4.0 | asv | `24236559f381` | no | 90.48 | 90.48 | 89.0 | 64.0 | 111.0 | 0 | 100 |
| `minimization/morse-bulk` | `ase/morse-bfgslinesearch-tuned` | 3.29.0 | campaign | `f27c0005ae6a` | yes | 93.44 | 93.44 | 95.0 | 70.0 | 104.0 | 0 | 100 |
| `minimization/morse-bulk` | `rgmin/morse-sd` | 0.4.0 | campaign | `24236559f381` | yes | 96.59 | 96.59 | 97.0 | 64.0 | 119.0 | 0 | 100 |
| `minimization/morse-bulk` | `rgmin/morse-sd` | 0.4.0 | asv | `24236559f381` | no | 96.59 | 96.59 | 97.0 | 64.0 | 119.0 | 0 | 100 |
| `minimization/morse-bulk` | `rgmin/morse-fire-tuned` | 0.4.0 | campaign | `24236559f381` | yes | 113.19 | 113.19 | 113.0 | 92.0 | 127.0 | 0 | 100 |
| `minimization/morse-bulk` | `rgmin/morse-fire-tuned` | 0.4.0 | asv | `24236559f381` | no | 113.19 | 113.19 | 113.0 | 92.0 | 127.0 | 0 | 100 |
| `minimization/morse-bulk` | `ase/morse-fire-default` | 3.29.0 | campaign | `f27c0005ae6a` | yes | 113.36 | 113.36 | 112.0 | 98.0 | 135.0 | 0 | 100 |
| `minimization/morse-bulk` | `ase/morse-fire-tuned` | 3.29.0 | campaign | `f27c0005ae6a` | yes | 113.36 | 113.36 | 112.0 | 98.0 | 135.0 | 0 | 100 |
| `minimization/morse-bulk` | `rgmin/morse-fire` | 0.4.0 | campaign | `24236559f381` | yes | 116.19 | 116.19 | 119.0 | 92.0 | 128.0 | 0 | 100 |
| `minimization/morse-bulk` | `rgmin/morse-fire` | 0.4.0 | asv | `24236559f381` | no | 116.19 | 116.19 | 119.0 | 92.0 | 128.0 | 0 | 100 |
| `minimization/morse-bulk` | `rgmin/morse-fire2` | 0.4.0 | campaign | `24236559f381` | yes | 121.85 | 121.85 | 124.0 | 101.0 | 136.0 | 0 | 100 |
| `minimization/morse-bulk` | `rgmin/morse-fire2` | 0.4.0 | asv | `24236559f381` | no | 121.85 | 121.85 | 124.0 | 101.0 | 136.0 | 0 | 100 |
| `minimization/morse-bulk` | `rgmin/morse-fire2-tuned` | 0.4.0 | campaign | `24236559f381` | yes | 121.87 | 121.87 | 124.0 | 101.0 | 136.0 | 0 | 100 |
| `minimization/morse-bulk` | `rgmin/morse-fire2-tuned` | 0.4.0 | asv | `24236559f381` | no | 121.87 | 121.87 | 124.0 | 101.0 | 136.0 | 0 | 100 |
| `minimization/morse-bulk` | `rgmin/morse-adam` | 0.4.1 | campaign | `fd26a60ddd48` | yes | 219.34 | 219.34 | 222.0 | 170.0 | 248.0 | 0 | 100 |
| `minimization/morse-bulk` | `rgmin/morse-adam-tuned` | 0.4.1 | campaign | `fd26a60ddd48` | yes | 219.34 | 219.34 | 222.0 | 170.0 | 248.0 | 0 | 100 |
| `minimization/morse-bulk` | `rgmin/morse-adam` | 0.4.0 | campaign | `24236559f381` | yes | 285.73 | 285.73 | 285.0 | 219.0 | 343.0 | 0 | 100 |
| `minimization/morse-bulk` | `rgmin/morse-adam` | 0.4.0 | asv | `24236559f381` | no | 285.73 | 285.73 | 285.0 | 219.0 | 343.0 | 0 | 100 |
| `minimization/morse-bulk` | `rgmin/morse-adam-tuned` | 0.4.0 | campaign | `24236559f381` | yes | 285.73 | 285.73 | 285.0 | 219.0 | 343.0 | 0 | 100 |
| `minimization/morse-bulk` | `rgmin/morse-adam-tuned` | 0.4.0 | asv | `24236559f381` | no | 285.73 | 285.73 | 285.0 | 219.0 | 343.0 | 0 | 100 |
| `minimization/morse-bulk` | `rgmin/morse-sr2` | 0.4.0 | campaign | `24236559f381` | yes | 341.13 | 341.13 | 334.0 | 235.0 | 821.0 | 0 | 100 |
| `minimization/morse-bulk` | `rgmin/morse-sr2` | 0.4.0 | asv | `24236559f381` | no | 341.13 | 341.13 | 334.0 | 235.0 | 821.0 | 0 | 100 |
| `minimization/morse-bulk` | `ase/morse-preconfire-default` | 3.29.0 | campaign | `f27c0005ae6a` | yes | 363.32 | 363.32 | 371.0 | 272.0 | 407.0 | 0 | 100 |
| `minimization/morse-bulk` | `ase/morse-preconfire-tuned` | 3.29.0 | campaign | `f27c0005ae6a` | yes | 365.52 | 365.52 | 372.0 | 291.0 | 408.0 | 0 | 100 |
| `minimization/morse-bulk` | `rgmin/morse-newton` | 0.4.0 | campaign | `24236559f381` | yes | 396.69 | 296.69 | 8.0 | 4.0 | 10000.0 | 1 | 100 |
| `minimization/morse-bulk` | `rgmin/morse-newton` | 0.4.0 | asv | `24236559f381` | no | 396.69 | 296.69 | 8.0 | 4.0 | 10000.0 | 1 | 100 |
| `minimization/morse-bulk` | `rgmin/morse-sr1` | 0.4.0 | campaign | `24236559f381` | yes | 566.31 | 366.31 | 137.0 | 54.0 | 10000.0 | 2 | 100 |
| `minimization/morse-bulk` | `rgmin/morse-sr1` | 0.4.0 | asv | `24236559f381` | no | 566.31 | 366.31 | 137.0 | 54.0 | 10000.0 | 2 | 100 |
| `minimization/morse-bulk` | `rgmin/morse-sr1-tuned` | 0.4.0 | campaign | `24236559f381` | yes | 1187.31 | 787.31 | 160.5 | 54.0 | 10000.0 | 4 | 100 |
| `minimization/morse-bulk` | `rgmin/morse-sr1-tuned` | 0.4.0 | asv | `24236559f381` | no | 1187.31 | 787.31 | 160.5 | 54.0 | 10000.0 | 4 | 100 |
| `minimization/morse-bulk` | `rgmin/morse-sr2-tuned` | 0.4.0 | campaign | `24236559f381` | yes | 1198.65 | 798.65 | 362.5 | 235.0 | 10000.0 | 4 | 100 |
| `minimization/morse-bulk` | `rgmin/morse-sr2-tuned` | 0.4.0 | asv | `24236559f381` | no | 1198.65 | 798.65 | 362.5 | 235.0 | 10000.0 | 4 | 100 |
| `minimization/morse-bulk` | `rgmin/morse-bb` | 0.4.0 | campaign | `24236559f381` | yes | 15606.64 | 7806.64 | 10000.0 | 24.0 | 10000.0 | 78 | 100 |
| `minimization/morse-bulk` | `rgmin/morse-bb` | 0.4.0 | asv | `24236559f381` | no | 15606.64 | 7806.64 | 10000.0 | 24.0 | 10000.0 | 78 | 100 |
| `minimization/morse-bulk` | `rgmin/morse-diis-tuned` | 0.4.0 | campaign | `24236559f381` | yes | 15819.42 | 8019.42 | 10000.0 | 156.0 | 10000.0 | 78 | 100 |
| `minimization/morse-bulk` | `rgmin/morse-diis-tuned` | 0.4.0 | asv | `24236559f381` | no | 15819.42 | 8019.42 | 10000.0 | 156.0 | 10000.0 | 78 | 100 |
| `minimization/morse-bulk` | `rgmin/morse-diis` | 0.4.0 | campaign | `24236559f381` | yes | 16153.43 | 8153.43 | 10000.0 | 153.0 | 10000.0 | 80 | 100 |
| `minimization/morse-bulk` | `rgmin/morse-diis` | 0.4.0 | asv | `24236559f381` | no | 16153.43 | 8153.43 | 10000.0 | 153.0 | 10000.0 | 80 | 100 |
| `neb/Pt-heptamer` | `ase/neb-pt-heptamer-tuned` | 3.29.0 | campaign | `f27c0005ae6a` | yes | 55.32692307692308 | 55.32692307692308 | 56.25 | 31.25 | 74.25 | 0 | 13 |
| `neb/Pt-heptamer` | `ase/neb-pt-heptamer-default` | 3.29.0 | campaign | `f27c0005ae6a` | yes | 75.86538461538461 | 75.86538461538461 | 77.25 | 47.25 | 99.25 | 0 | 13 |
| `neb/Pt-heptamer` | `rgsaddle/idpp-ci-neb-lbfgs` | 0.2.2 | asv | `c0db2b8d333c` | yes | 318.3076923076923 | 318.3076923076923 | 314.0 | 210.0 | 458.0 | 0 | 13 |
| `neb/Pt-heptamer` | `rgsaddle/sidpp-ci-neb-lbfgs` | 0.2.2 | asv | `c0db2b8d333c` | yes | 318.9230769230769 | 318.9230769230769 | 314.0 | 210.0 | 466.0 | 0 | 13 |
| `neb/Pt-heptamer` | `rgsaddle/ci-neb-lbfgs` | 0.2.2 | asv | `c0db2b8d333c` | yes | 319.53846153846155 | 319.53846153846155 | 330.0 | 218.0 | 410.0 | 0 | 13 |
| `neb/Pt-heptamer` | `rgsaddle/oci-neb-lbfgs` | 0.2.2 | asv | `c0db2b8d333c` | yes | 345.53846153846155 | 345.53846153846155 | 356.0 | 244.0 | 436.0 | 0 | 13 |
| `neb/Pt-heptamer` | `rgsaddle/ew-ci-neb-lbfgs` | 0.2.2 | asv | `c0db2b8d333c` | yes | 357.2307692307692 | 357.2307692307692 | 332.0 | 220.0 | 556.0 | 0 | 13 |
| `neb/Pt-heptamer` | `rgsaddle/sidpp-ci-neb` | 0.2.2 | asv | `c0db2b8d333c` | yes | 732.4615384615385 | 732.4615384615385 | 682.0 | 650.0 | 978.0 | 0 | 13 |
| `neb/Pt-heptamer` | `rgsaddle/idpp-ci-neb` | 0.2.2 | asv | `c0db2b8d333c` | yes | 735.5384615384615 | 735.5384615384615 | 706.0 | 650.0 | 978.0 | 0 | 13 |
| `neb/Pt-heptamer` | `rgsaddle/ci-neb` | 0.2.2 | asv | `c0db2b8d333c` | yes | 736.7692307692307 | 736.7692307692307 | 698.0 | 594.0 | 1058.0 | 0 | 13 |
| `neb/Pt-heptamer` | `rgsaddle/ew-ci-neb` | 0.2.2 | asv | `c0db2b8d333c` | yes | 759.6923076923077 | 759.6923076923077 | 740.0 | 548.0 | 1028.0 | 0 | 13 |
| `neb/Pt-heptamer` | `rgsaddle/oci-neb` | 0.2.2 | asv | `c0db2b8d333c` | yes | 760.7692307692307 | 760.7692307692307 | 722.0 | 618.0 | 1082.0 | 0 | 13 |
| `neb/Pt-heptamer` | `rgsaddle/neb` | 0.2.2 | asv | `c0db2b8d333c` | yes | 20000.0 | 970.6153846153846 | 962.0 | 786.0 | 1226.0 | 13 | 13 |
| `neb/Pt-heptamer` | `rgsaddle/neb-lbfgs` | 0.2.2 | asv | `c0db2b8d333c` | yes | 20000.0 | 422.9230769230769 | 410.0 | 266.0 | 522.0 | 13 | 13 |
| `tsopt/LJ38` | `sella/lj38-ts-default` | 2.6.0 | campaign | `6cfb60cdb100` | yes | 68.74 | 68.74 | 59.5 | 25.0 | 257.0 | 0 | 200 |
| `tsopt/LJ38` | `rgsaddle/prfo-tuned` | 0.2.2 | campaign | `c0db2b8d333c` | yes | 192.6 | 192.6 | 173.0 | 135.0 | 1691.0 | 0 | 200 |
| `tsopt/LJ38` | `rgsaddle/prfo-tuned` | 0.2.2 | asv | `c0db2b8d333c` | no | 192.6 | 192.6 | 173.0 | 135.0 | 1691.0 | 0 | 200 |
| `tsopt/LJ38` | `sella/lj38-ts-tuned` | 2.6.0 | campaign | `6cfb60cdb100` | yes | 274.46 | 92.775 | 68.0 | 34.0 | 1840.0 | 2 | 200 |
| `tsopt/LJ38` | `rgsaddle/dimer-rotskip` | 0.2.2 | campaign | `c0db2b8d333c` | yes | 7803.19 | 2926.815 | 856.5 | 83.0 | 10006.0 | 74 | 200 |
| `tsopt/LJ38` | `rgsaddle/dimer-rotskip` | 0.2.2 | asv | `c0db2b8d333c` | no | 7803.19 | 2926.815 | 856.5 | 83.0 | 10006.0 | 74 | 200 |
| `tsopt/LJ38` | `rgsaddle/prfo` | 0.2.2 | campaign | `c0db2b8d333c` | yes | 15146.995 | 723.265 | 460.5 | 230.0 | 10000.0 | 150 | 200 |
| `tsopt/LJ38` | `rgsaddle/prfo` | 0.2.2 | asv | `c0db2b8d333c` | no | 15146.995 | 723.265 | 460.5 | 230.0 | 10000.0 | 150 | 200 |
| `tsopt/LJ38` | `rgsaddle/dimer-fire-tuned` | 0.2.2 | campaign | `c0db2b8d333c` | yes | 16338.965 | 3321.315 | 2795.5 | 568.0 | 10009.0 | 153 | 200 |
| `tsopt/LJ38` | `rgsaddle/dimer-fire-tuned` | 0.2.2 | asv | `c0db2b8d333c` | no | 16338.965 | 3321.315 | 2795.5 | 568.0 | 10009.0 | 153 | 200 |
| `tsopt/LJ38` | `rgsaddle/dimer-rotskip-tuned` | 0.2.2 | campaign | `c0db2b8d333c` | yes | 16738.105 | 6506.53 | 8407.0 | 80.0 | 10002.0 | 166 | 200 |
| `tsopt/LJ38` | `rgsaddle/dimer-rotskip-tuned` | 0.2.2 | asv | `c0db2b8d333c` | no | 16738.105 | 6506.53 | 8407.0 | 80.0 | 10002.0 | 166 | 200 |
| `tsopt/LJ38` | `rgsaddle/lanczos-tuned` | 0.2.2 | campaign | `c0db2b8d333c` | yes | 19146.12 | 9646.12 | 10000.0 | 447.0 | 10000.0 | 190 | 200 |
| `tsopt/LJ38` | `rgsaddle/lanczos-tuned` | 0.2.2 | asv | `c0db2b8d333c` | no | 19146.12 | 9646.12 | 10000.0 | 447.0 | 10000.0 | 190 | 200 |
| `tsopt/LJ38` | `rgsaddle/lanczos` | 0.2.2 | campaign | `c0db2b8d333c` | yes | 19418.82 | 9779.385 | 10011.0 | 860.0 | 10011.0 | 193 | 200 |
| `tsopt/LJ38` | `rgsaddle/lanczos` | 0.2.2 | asv | `c0db2b8d333c` | no | 19418.82 | 9779.385 | 10011.0 | 860.0 | 10011.0 | 193 | 200 |
| `tsopt/LJ38` | `rgsaddle/dimer-fire` | 0.2.2 | campaign | `c0db2b8d333c` | yes | 19454.98 | 4792.34 | 3666.0 | 997.0 | 10021.0 | 192 | 200 |
| `tsopt/LJ38` | `rgsaddle/dimer-fire` | 0.2.2 | asv | `c0db2b8d333c` | no | 19454.98 | 4792.34 | 3666.0 | 997.0 | 10021.0 | 192 | 200 |
| `tsopt/LJ38` | `rgsaddle/dimer-lbfgs-tuned` | 0.2.2 | campaign | `c0db2b8d333c` | yes | 19924.765 | 9983.535 | 10009.0 | 4953.0 | 10010.0 | 199 | 200 |
| `tsopt/LJ38` | `rgsaddle/dimer-lbfgs-tuned` | 0.2.2 | asv | `c0db2b8d333c` | no | 19924.765 | 9983.535 | 10009.0 | 4953.0 | 10010.0 | 199 | 200 |
| `tsopt/LJ38` | `rgsaddle/dimer-lbfgs` | 0.2.2 | campaign | `c0db2b8d333c` | yes | 20000.0 | 9991.44 | 10011.0 | 6110.0 | 10011.0 | 200 | 200 |
| `tsopt/LJ38` | `rgsaddle/dimer-lbfgs` | 0.2.2 | asv | `c0db2b8d333c` | no | 20000.0 | 9991.44 | 10011.0 | 6110.0 | 10011.0 | 200 | 200 |
| `tsopt/pt-island` | `sella/pt-island-ts-default` | 2.6.0 | campaign | `6cfb60cdb100` | yes | 74.28571428571429 | 74.28571428571429 | 71.0 | 41.0 | 128.0 | 0 | 49 |
| `tsopt/pt-island` | `sella/pt-island-ts-tuned` | 2.6.0 | campaign | `6cfb60cdb100` | yes | 87.55102040816327 | 87.55102040816327 | 88.0 | 67.0 | 141.0 | 0 | 49 |
| `tsopt/pt-island` | `rgsaddle/pt-dimer-rotskip` | 0.2.2 | campaign | `c0db2b8d333c` | yes | 223.89795918367346 | 223.89795918367346 | 145.0 | 105.0 | 3962.0 | 0 | 49 |
| `tsopt/pt-island` | `rgsaddle/pt-dimer-rotskip` | 0.2.2 | asv | `c0db2b8d333c` | no | 223.89795918367346 | 223.89795918367346 | 145.0 | 105.0 | 3962.0 | 0 | 49 |
| `tsopt/pt-island` | `rgsaddle/pt-dimer-rotskip-tuned` | 0.2.2 | campaign | `c0db2b8d333c` | yes | 299.6326530612245 | 299.6326530612245 | 167.0 | 103.0 | 1589.0 | 0 | 49 |
| `tsopt/pt-island` | `rgsaddle/pt-dimer-rotskip-tuned` | 0.2.2 | asv | `c0db2b8d333c` | no | 299.6326530612245 | 299.6326530612245 | 167.0 | 103.0 | 1589.0 | 0 | 49 |
| `tsopt/pt-island` | `rgsaddle/pt-dimer-fire-tuned` | 0.2.2 | campaign | `c0db2b8d333c` | yes | 1096.8979591836735 | 1096.8979591836735 | 918.0 | 536.0 | 3039.0 | 0 | 49 |
| `tsopt/pt-island` | `rgsaddle/pt-dimer-fire-tuned` | 0.2.2 | asv | `c0db2b8d333c` | no | 1096.8979591836735 | 1096.8979591836735 | 918.0 | 536.0 | 3039.0 | 0 | 49 |
| `tsopt/pt-island` | `rgsaddle/pt-dimer-fire` | 0.2.2 | campaign | `c0db2b8d333c` | yes | 3274.1428571428573 | 3070.1428571428573 | 2337.0 | 1206.0 | 10004.0 | 1 | 49 |
| `tsopt/pt-island` | `rgsaddle/pt-dimer-fire` | 0.2.2 | asv | `c0db2b8d333c` | no | 3274.1428571428573 | 3070.1428571428573 | 2337.0 | 1206.0 | 10004.0 | 1 | 49 |
| `tsopt/pt-island` | `rgsaddle/pt-dimer-lbfgs-tuned` | 0.2.2 | campaign | `c0db2b8d333c` | yes | 7701.714285714285 | 4031.5510204081634 | 542.0 | 225.0 | 10009.0 | 18 | 49 |
| `tsopt/pt-island` | `rgsaddle/pt-dimer-lbfgs-tuned` | 0.2.2 | asv | `c0db2b8d333c` | no | 7701.714285714285 | 4031.5510204081634 | 542.0 | 225.0 | 10009.0 | 18 | 49 |
| `tsopt/pt-island` | `rgsaddle/pt-lanczos` | 0.2.2 | campaign | `c0db2b8d333c` | yes | 9736.795918367347 | 5048.081632653061 | 974.0 | 298.0 | 10011.0 | 23 | 49 |
| `tsopt/pt-island` | `rgsaddle/pt-lanczos` | 0.2.2 | asv | `c0db2b8d333c` | no | 9736.795918367347 | 5048.081632653061 | 974.0 | 298.0 | 10011.0 | 23 | 49 |
| `tsopt/pt-island` | `rgsaddle/pt-lanczos-tuned` | 0.2.2 | campaign | `c0db2b8d333c` | yes | 9990.326530612245 | 5092.367346938776 | 635.0 | 198.0 | 10000.0 | 24 | 49 |
| `tsopt/pt-island` | `rgsaddle/pt-lanczos-tuned` | 0.2.2 | asv | `c0db2b8d333c` | no | 9990.326530612245 | 5092.367346938776 | 635.0 | 198.0 | 10000.0 | 24 | 49 |
| `tsopt/pt-island` | `rgsaddle/pt-dimer-lbfgs` | 0.2.2 | campaign | `c0db2b8d333c` | yes | 11854.510204081633 | 6146.448979591837 | 10011.0 | 502.0 | 10018.0 | 28 | 49 |
| `tsopt/pt-island` | `rgsaddle/pt-dimer-lbfgs` | 0.2.2 | asv | `c0db2b8d333c` | no | 11854.510204081633 | 6146.448979591837 | 10011.0 | 502.0 | 10018.0 | 28 | 49 |

## Partial (90)

These rows were scored and kept out of the ranked table.

| Problem | Template | Source | n | Reason | Path |
| --- | --- | --- | --- | --- | --- |
| `double-ended-saddle-search/lj38` | `eon/develop-de-lj38-cap` | asv | 1 | scored 1 of 50 starts | `results/asv/e172a4b674067643a20f44d30d9ca4a93025acac` |
| `double-ended-saddle-search/lj38` | `eon/develop-de-lj38-ew` | asv | 1 | scored 1 of 50 starts | `results/asv/e172a4b674067643a20f44d30d9ca4a93025acac` |
| `double-ended-saddle-search/lj38` | `eon/develop-de-lj38-oci` | asv | 1 | scored 1 of 50 starts | `results/asv/e172a4b674067643a20f44d30d9ca4a93025acac` |
| `double-ended-saddle-search/lj38` | `ase/de-lj38-default` | asv | 1 | scored 1 of 50 starts | `results/asv/806ecaca75a456ba67e7f95621c51ab84e94adaf` |
| `double-ended-saddle-search/lj38` | `sella/de-lj38-default` | asv | 1 | scored 1 of 50 starts | `results/asv/806ecaca75a456ba67e7f95621c51ab84e94adaf` |
| `double-ended-saddle-search/lj38` | `sella/de-lj38-tuned` | asv | 1 | scored 1 of 50 starts | `results/asv/806ecaca75a456ba67e7f95621c51ab84e94adaf` |
| `double-ended-saddle-search/lj38` | `ase/de-lj38-tuned` | asv | 1 | scored 1 of 50 starts | `results/asv/806ecaca75a456ba67e7f95621c51ab84e94adaf` |
| `double-ended-saddle-search/lj38` | `eon/develop-de-lj38-cap` | asv | 1 | scored 1 of 50 starts | `results/asv/773f8e16a8c96df68d7a05af605491f9b7272596` |
| `double-ended-saddle-search/lj38` | `eon/develop-de-lj38-ew` | asv | 1 | scored 1 of 50 starts | `results/asv/773f8e16a8c96df68d7a05af605491f9b7272596` |
| `double-ended-saddle-search/lj38` | `eon/develop-de-lj38-oci` | asv | 1 | scored 1 of 50 starts | `results/asv/773f8e16a8c96df68d7a05af605491f9b7272596` |
| `double-ended-saddle-search/lj38_2ts_separation` | `eon/develop-de-2ts` | asv | 1 | scored 1 of 51 starts | `results/asv/e172a4b674067643a20f44d30d9ca4a93025acac` |
| `double-ended-saddle-search/lj38_2ts_separation` | `eon/develop-de-2ts-ew` | asv | 1 | scored 1 of 51 starts | `results/asv/e172a4b674067643a20f44d30d9ca4a93025acac` |
| `double-ended-saddle-search/lj38_2ts_separation` | `eon/develop-de-2ts-oci` | asv | 1 | scored 1 of 51 starts | `results/asv/e172a4b674067643a20f44d30d9ca4a93025acac` |
| `double-ended-saddle-search/lj38_2ts_separation` | `sella/de-lj38-2ts-default` | asv | 1 | scored 1 of 51 starts | `results/asv/806ecaca75a456ba67e7f95621c51ab84e94adaf` |
| `double-ended-saddle-search/lj38_2ts_separation` | `sella/de-lj38-2ts-tuned` | asv | 1 | scored 1 of 51 starts | `results/asv/806ecaca75a456ba67e7f95621c51ab84e94adaf` |
| `double-ended-saddle-search/lj38_2ts_separation` | `ase/de-lj38-2ts-default` | asv | 1 | scored 1 of 51 starts | `results/asv/806ecaca75a456ba67e7f95621c51ab84e94adaf` |
| `double-ended-saddle-search/lj38_2ts_separation` | `ase/de-lj38-2ts-tuned` | asv | 1 | scored 1 of 51 starts | `results/asv/806ecaca75a456ba67e7f95621c51ab84e94adaf` |
| `double-ended-saddle-search/lj38_2ts_separation` | `eon/develop-de-2ts` | asv | 1 | scored 1 of 51 starts | `results/asv/773f8e16a8c96df68d7a05af605491f9b7272596` |
| `double-ended-saddle-search/lj38_2ts_separation` | `eon/develop-de-2ts-ew` | asv | 1 | scored 1 of 51 starts | `results/asv/773f8e16a8c96df68d7a05af605491f9b7272596` |
| `double-ended-saddle-search/lj38_2ts_separation` | `eon/develop-de-2ts-oci` | asv | 1 | scored 1 of 51 starts | `results/asv/773f8e16a8c96df68d7a05af605491f9b7272596` |
| `double-ended-saddle-search/pt-island` | `eon/develop-de-pt-ew` | asv | 1 | scored 1 of 59 starts | `results/asv/e172a4b674067643a20f44d30d9ca4a93025acac` |
| `double-ended-saddle-search/pt-island` | `eon/develop-de-pt-oci` | asv | 1 | scored 1 of 59 starts | `results/asv/e172a4b674067643a20f44d30d9ca4a93025acac` |
| `double-ended-saddle-search/pt-island` | `ase/de-pt-island-tuned` | asv | 1 | scored 1 of 59 starts | `results/asv/806ecaca75a456ba67e7f95621c51ab84e94adaf` |
| `double-ended-saddle-search/pt-island` | `sella/de-pt-island-default` | asv | 1 | scored 1 of 59 starts | `results/asv/806ecaca75a456ba67e7f95621c51ab84e94adaf` |
| `double-ended-saddle-search/pt-island` | `sella/de-pt-island-tuned` | asv | 1 | scored 1 of 59 starts | `results/asv/806ecaca75a456ba67e7f95621c51ab84e94adaf` |
| `double-ended-saddle-search/pt-island` | `ase/de-pt-island-default` | asv | 1 | scored 1 of 59 starts | `results/asv/806ecaca75a456ba67e7f95621c51ab84e94adaf` |
| `double-ended-saddle-search/pt-island` | `eon/develop-de-pt-ew` | asv | 1 | scored 1 of 59 starts | `results/asv/773f8e16a8c96df68d7a05af605491f9b7272596` |
| `double-ended-saddle-search/pt-island` | `eon/develop-de-pt-oci` | asv | 1 | scored 1 of 59 starts | `results/asv/773f8e16a8c96df68d7a05af605491f9b7272596` |
| `lowest-eigenvector/lj38` | `eon/develop-low-lj38-dimer` | asv | 1 | scored 1 of 200 starts | `results/asv/e172a4b674067643a20f44d30d9ca4a93025acac` |
| `lowest-eigenvector/lj38` | `eon/develop-low-lj38-lanczos` | asv | 1 | scored 1 of 200 starts | `results/asv/e172a4b674067643a20f44d30d9ca4a93025acac` |
| `lowest-eigenvector/lj38` | `eon/develop-low-lj38-dimer` | asv | 1 | scored 1 of 200 starts | `results/asv/773f8e16a8c96df68d7a05af605491f9b7272596` |
| `lowest-eigenvector/lj38` | `eon/develop-low-lj38-lanczos` | asv | 1 | scored 1 of 200 starts | `results/asv/773f8e16a8c96df68d7a05af605491f9b7272596` |
| `minimization/lj38` | `ase/lj38-lbfgs-default` | asv | 1 | scored 1 of 1000 starts | `results/asv/806ecaca75a456ba67e7f95621c51ab84e94adaf` |
| `minimization/lj38` | `ase/lj38-gpmin-default` | asv | 1 | scored 1 of 1000 starts | `results/asv/806ecaca75a456ba67e7f95621c51ab84e94adaf` |
| `minimization/lj38` | `ase/lj38-bfgs-default` | asv | 1 | scored 1 of 1000 starts | `results/asv/806ecaca75a456ba67e7f95621c51ab84e94adaf` |
| `minimization/lj38` | `ase/lj38-bfgslinesearch-default` | asv | 1 | scored 1 of 1000 starts | `results/asv/806ecaca75a456ba67e7f95621c51ab84e94adaf` |
| `minimization/lj38` | `ase/lj38-fire-default` | asv | 1 | scored 1 of 1000 starts | `results/asv/806ecaca75a456ba67e7f95621c51ab84e94adaf` |
| `minimization/lj38` | `ase/lj38-lbfgs-tuned` | asv | 1 | scored 1 of 1000 starts | `results/asv/806ecaca75a456ba67e7f95621c51ab84e94adaf` |
| `minimization/lj38` | `ase/lj38-preconfire-tuned` | asv | 1 | scored 1 of 1000 starts | `results/asv/806ecaca75a456ba67e7f95621c51ab84e94adaf` |
| `minimization/lj38` | `sella/lj38-min-default` | asv | 1 | scored 1 of 1000 starts | `results/asv/806ecaca75a456ba67e7f95621c51ab84e94adaf` |
| `minimization/lj38` | `ase/lj38-gpmin-tuned` | asv | 1 | scored 1 of 1000 starts | `results/asv/806ecaca75a456ba67e7f95621c51ab84e94adaf` |
| `minimization/lj38` | `ase/lj38-preconode12r-tuned` | asv | 1 | scored 1 of 1000 starts | `results/asv/806ecaca75a456ba67e7f95621c51ab84e94adaf` |
| `minimization/lj38` | `ase/lj38-preconlbfgs-default` | asv | 1 | scored 1 of 1000 starts | `results/asv/806ecaca75a456ba67e7f95621c51ab84e94adaf` |
| `minimization/lj38` | `sella/lj38-min-tuned` | asv | 1 | scored 1 of 1000 starts | `results/asv/806ecaca75a456ba67e7f95621c51ab84e94adaf` |
| `minimization/lj38` | `ase/lj38-preconode12r-default` | asv | 1 | scored 1 of 1000 starts | `results/asv/806ecaca75a456ba67e7f95621c51ab84e94adaf` |
| `minimization/lj38` | `ase/lj38-bfgs-tuned` | asv | 1 | scored 1 of 1000 starts | `results/asv/806ecaca75a456ba67e7f95621c51ab84e94adaf` |
| `minimization/lj38` | `ase/lj38-bfgslinesearch-tuned` | asv | 1 | scored 1 of 1000 starts | `results/asv/806ecaca75a456ba67e7f95621c51ab84e94adaf` |
| `minimization/lj38` | `ase/lj38-fire-tuned` | asv | 1 | scored 1 of 1000 starts | `results/asv/806ecaca75a456ba67e7f95621c51ab84e94adaf` |
| `minimization/lj38` | `ase/lj38-preconfire-default` | asv | 1 | scored 1 of 1000 starts | `results/asv/806ecaca75a456ba67e7f95621c51ab84e94adaf` |
| `minimization/lj38` | `ase/lj38-preconlbfgs-tuned` | asv | 1 | scored 1 of 1000 starts | `results/asv/806ecaca75a456ba67e7f95621c51ab84e94adaf` |
| `minimization/morse-bulk` | `ase/morse-preconode12r-tuned` | asv | 1 | scored 1 of 100 starts | `results/asv/806ecaca75a456ba67e7f95621c51ab84e94adaf` |
| `minimization/morse-bulk` | `ase/morse-bfgs-default` | asv | 1 | scored 1 of 100 starts | `results/asv/806ecaca75a456ba67e7f95621c51ab84e94adaf` |
| `minimization/morse-bulk` | `ase/morse-bfgslinesearch-default` | asv | 1 | scored 1 of 100 starts | `results/asv/806ecaca75a456ba67e7f95621c51ab84e94adaf` |
| `minimization/morse-bulk` | `ase/morse-bfgslinesearch-tuned` | asv | 1 | scored 1 of 100 starts | `results/asv/806ecaca75a456ba67e7f95621c51ab84e94adaf` |
| `minimization/morse-bulk` | `ase/morse-fire-tuned` | asv | 1 | scored 1 of 100 starts | `results/asv/806ecaca75a456ba67e7f95621c51ab84e94adaf` |
| `minimization/morse-bulk` | `ase/morse-preconfire-default` | asv | 1 | scored 1 of 100 starts | `results/asv/806ecaca75a456ba67e7f95621c51ab84e94adaf` |
| `minimization/morse-bulk` | `ase/morse-bfgs-tuned` | asv | 1 | scored 1 of 100 starts | `results/asv/806ecaca75a456ba67e7f95621c51ab84e94adaf` |
| `minimization/morse-bulk` | `ase/morse-fire-default` | asv | 1 | scored 1 of 100 starts | `results/asv/806ecaca75a456ba67e7f95621c51ab84e94adaf` |
| `minimization/morse-bulk` | `ase/morse-lbfgs-default` | asv | 1 | scored 1 of 100 starts | `results/asv/806ecaca75a456ba67e7f95621c51ab84e94adaf` |
| `minimization/morse-bulk` | `ase/morse-preconlbfgs-default` | asv | 1 | scored 1 of 100 starts | `results/asv/806ecaca75a456ba67e7f95621c51ab84e94adaf` |
| `minimization/morse-bulk` | `sella/morse-min-tuned` | asv | 1 | scored 1 of 100 starts | `results/asv/806ecaca75a456ba67e7f95621c51ab84e94adaf` |
| `minimization/morse-bulk` | `ase/morse-preconode12r-default` | asv | 1 | scored 1 of 100 starts | `results/asv/806ecaca75a456ba67e7f95621c51ab84e94adaf` |
| `minimization/morse-bulk` | `ase/morse-preconfire-tuned` | asv | 1 | scored 1 of 100 starts | `results/asv/806ecaca75a456ba67e7f95621c51ab84e94adaf` |
| `minimization/morse-bulk` | `sella/morse-min-default` | asv | 1 | scored 1 of 100 starts | `results/asv/806ecaca75a456ba67e7f95621c51ab84e94adaf` |
| `minimization/morse-bulk` | `ase/morse-lbfgs-tuned` | asv | 1 | scored 1 of 100 starts | `results/asv/806ecaca75a456ba67e7f95621c51ab84e94adaf` |
| `minimization/morse-bulk` | `ase/morse-preconlbfgs-tuned` | asv | 1 | scored 1 of 100 starts | `results/asv/806ecaca75a456ba67e7f95621c51ab84e94adaf` |
| `neb/Pt-heptamer` | `ase/neb-pt-heptamer-default` | asv | 1 | scored 1 of 13 starts | `results/asv/806ecaca75a456ba67e7f95621c51ab84e94adaf` |
| `neb/Pt-heptamer` | `ase/neb-pt-heptamer-tuned` | asv | 1 | scored 1 of 13 starts | `results/asv/806ecaca75a456ba67e7f95621c51ab84e94adaf` |
| `saddle-search/pt-island` | `ase/dimer-pt-island-default` | campaign | 16 | result file sets ranked false | `results/campaign/ase-3.29.0-dimer-pt-island-default.dat` |
| `saddle-search/pt-island` | `ase/dimer-pt-island-tuned` | campaign | 16 | result file sets ranked false | `results/campaign/ase-3.29.0-dimer-pt-island-tuned.dat` |
| `saddle-search/pt-island` | `eon/develop-saddle-pt-dimer` | asv | 1 | one-start grid; the problem states no full set | `results/asv/e172a4b674067643a20f44d30d9ca4a93025acac` |
| `saddle-search/pt-island` | `eon/develop-saddle-pt-lanczos` | asv | 1 | one-start grid; the problem states no full set | `results/asv/e172a4b674067643a20f44d30d9ca4a93025acac` |
| `saddle-search/pt-island` | `ase/dimer-pt-island-default` | asv | 1 | one-start grid; the problem states no full set | `results/asv/806ecaca75a456ba67e7f95621c51ab84e94adaf` |
| `saddle-search/pt-island` | `sella/pt-island-search-tuned` | asv | 1 | one-start grid; the problem states no full set | `results/asv/806ecaca75a456ba67e7f95621c51ab84e94adaf` |
| `saddle-search/pt-island` | `ase/dimer-pt-island-tuned` | asv | 1 | one-start grid; the problem states no full set | `results/asv/806ecaca75a456ba67e7f95621c51ab84e94adaf` |
| `saddle-search/pt-island` | `sella/pt-island-search-default` | asv | 1 | one-start grid; the problem states no full set | `results/asv/806ecaca75a456ba67e7f95621c51ab84e94adaf` |
| `saddle-search/pt-island` | `eon/develop-saddle-pt-dimer` | asv | 1 | one-start grid; the problem states no full set | `results/asv/ba1ac41f2075b1a1ef890b096a0d55c09c6390d4` |
| `saddle-search/pt-island` | `eon/develop-saddle-pt-lanczos` | asv | 1 | one-start grid; the problem states no full set | `results/asv/ba1ac41f2075b1a1ef890b096a0d55c09c6390d4` |
| `tsopt/LJ38` | `eon/develop-ts-lj38-dimer` | asv | 1 | scored 1 of 200 starts | `results/asv/e172a4b674067643a20f44d30d9ca4a93025acac` |
| `tsopt/LJ38` | `eon/develop-ts-lj38-lanczos` | asv | 1 | scored 1 of 200 starts | `results/asv/e172a4b674067643a20f44d30d9ca4a93025acac` |
| `tsopt/LJ38` | `sella/lj38-ts-default` | asv | 1 | scored 1 of 200 starts | `results/asv/806ecaca75a456ba67e7f95621c51ab84e94adaf` |
| `tsopt/LJ38` | `sella/lj38-ts-tuned` | asv | 1 | scored 1 of 200 starts | `results/asv/806ecaca75a456ba67e7f95621c51ab84e94adaf` |
| `tsopt/LJ38` | `eon/develop-ts-lj38-dimer` | asv | 1 | scored 1 of 200 starts | `results/asv/773f8e16a8c96df68d7a05af605491f9b7272596` |
| `tsopt/LJ38` | `eon/develop-ts-lj38-lanczos` | asv | 1 | scored 1 of 200 starts | `results/asv/773f8e16a8c96df68d7a05af605491f9b7272596` |
| `tsopt/pt-island` | `eon/develop-ts-pt-dimer` | asv | 1 | scored 1 of 49 starts | `results/asv/e172a4b674067643a20f44d30d9ca4a93025acac` |
| `tsopt/pt-island` | `eon/develop-ts-pt-lanczos` | asv | 1 | scored 1 of 49 starts | `results/asv/e172a4b674067643a20f44d30d9ca4a93025acac` |
| `tsopt/pt-island` | `sella/pt-island-ts-default` | asv | 1 | scored 1 of 49 starts | `results/asv/806ecaca75a456ba67e7f95621c51ab84e94adaf` |
| `tsopt/pt-island` | `sella/pt-island-ts-tuned` | asv | 1 | scored 1 of 49 starts | `results/asv/806ecaca75a456ba67e7f95621c51ab84e94adaf` |
| `tsopt/pt-island` | `eon/develop-ts-pt-dimer` | asv | 1 | scored 1 of 49 starts | `results/asv/773f8e16a8c96df68d7a05af605491f9b7272596` |
| `tsopt/pt-island` | `eon/develop-ts-pt-lanczos` | asv | 1 | scored 1 of 49 starts | `results/asv/773f8e16a8c96df68d7a05af605491f9b7272596` |

## Not scored (44)

| Path | Note |
| --- | --- |
| `results/campaign/eon-3.5.0-de-lj38-ci-neb.dat` | no per-start record |
| `results/campaign/eon-3.5.0-de-lj38-ci-swdneb.dat` | no per-start record |
| `results/campaign/eon-3.5.0-neb-pt-cg.dat` | no per-start record |
| `results/campaign/eon-3.5.0-neb-pt-fire.dat` | no per-start record |
| `results/campaign/eon-3.5.0-neb-pt-lbfgs.dat` | no per-start record |
| `results/campaign/eon-3.5.0-neb-pt-qm.dat` | no per-start record |
| `results/campaign/eon-client-cg.dat` | no per-start record |
| `results/campaign/eon-client-fire.dat` | no per-start record |
| `results/campaign/eon-client-lbfgs-m100.dat` | no per-start record |
| `results/campaign/eon-client-lbfgs-m20.dat` | no per-start record |
| `results/campaign/eon-client-morse-cg.dat` | no per-start record |
| `results/campaign/eon-client-morse-fire.dat` | no per-start record |
| `results/campaign/eon-client-morse-lbfgs.dat` | no per-start record |
| `results/campaign/eon-client-morse-sd.dat` | no per-start record |
| `results/campaign/eon-client-qm.dat` | no per-start record |
| `results/campaign/eon-client-sd-twopoint.dat` | no per-start record |
| `results/campaign/eon-client-sd.dat` | no per-start record |
| `results/campaign/eon-svn-cg.dat` | no per-start record |
| `results/campaign/eon-svn-fire.dat` | no per-start record |
| `results/campaign/eon-svn-lbfgs-m100.dat` | no per-start record |
| `results/campaign/eon-svn-lbfgs-m20.dat` | no per-start record |
| `results/campaign/eon-svn-morse-cg.dat` | no per-start record |
| `results/campaign/eon-svn-morse-fire.dat` | no per-start record |
| `results/campaign/eon-svn-morse-lbfgs.dat` | no per-start record |
| `results/campaign/eon-svn-morse-sd.dat` | no per-start record |
| `results/campaign/eon-svn-qm.dat` | no per-start record |
| `results/campaign/eon-svn-sd-twopoint.dat` | no per-start record |
| `results/campaign/eon-svn-sd.dat` | no per-start record |
| `results/campaign/pele-0.1.2-lbfgs-M1-maxstep0.1.dat` | no per-start record |
| `results/campaign/pele-0.1.2-lbfgs-M10-maxstep0.1.dat` | no per-start record |
| `results/campaign/pele-0.1.2-lbfgs-M100-maxstep0.2.dat` | no per-start record |
| `results/campaign/pele-0.1.2-lbfgs-M20-maxstep0.1.dat` | no per-start record |
| `results/campaign/pele-0.1.2-lbfgs-M4-maxstep0.05.dat` | no per-start record |
| `results/campaign/pele-0.1.2-lbfgs-M4-maxstep0.1.dat` | no per-start record |
| `results/campaign/pele-0.1.2-lbfgs-M4-maxstep0.2.dat` | no per-start record |
| `results/campaign/sella-2.6.0-ts-lj38.dat` | no per-start record |
| `results/asv/d4513f42b39edd0b69ade46ce63fd106e0f406c3` | Lj38EonCg ("'0'", "'lj38-clusters/cluster_0000.con'") track_force_calls was skipped or failed |
| `results/asv/d4513f42b39edd0b69ade46ce63fd106e0f406c3` | Lj38EonFire ("'0'", "'lj38-clusters/cluster_0000.con'") track_force_calls was skipped or failed |
| `results/asv/d4513f42b39edd0b69ade46ce63fd106e0f406c3` | Lj38EonLbfgsM100 ("'0'", "'lj38-clusters/cluster_0000.con'") track_force_calls was skipped or failed |
| `results/asv/d4513f42b39edd0b69ade46ce63fd106e0f406c3` | Lj38EonLbfgsM20 ("'0'", "'lj38-clusters/cluster_0000.con'") track_force_calls was skipped or failed |
| `results/asv/d4513f42b39edd0b69ade46ce63fd106e0f406c3` | Lj38EonQm ("'0'", "'lj38-clusters/cluster_0000.con'") track_force_calls was skipped or failed |
| `results/asv/d4513f42b39edd0b69ade46ce63fd106e0f406c3` | Lj38EonSd ("'0'", "'lj38-clusters/cluster_0000.con'") track_force_calls was skipped or failed |
| `results/asv/d4513f42b39edd0b69ade46ce63fd106e0f406c3` | Lj38EonSdTwopoint ("'0'", "'lj38-clusters/cluster_0000.con'") track_force_calls was skipped or failed |
| `results/asv/d4513f42b39edd0b69ade46ce63fd106e0f406c3` | Pending.track_success matched nothing in commit d4513f42b39edd0b69ade46ce63fd106e0f406c3 |

## Not a problem template (14)

These asv tracks are the quadratic smoke sample. They are not a campaign problem.

| Entry | n |
| --- | --- |
| `quadratic.QuadraticDimer` | 8 |
| `quadratic.QuadraticDisplacement` | 8 |
| `quadratic.QuadraticFailureCap` | 1 |
| `quadratic.QuadraticFailureCrash` | 1 |
| `quadratic.QuadraticFailureDivergence` | 1 |
| `quadratic.QuadraticFailureWrongStationary` | 1 |
| `quadratic.QuadraticKept` | 1 |
| `quadratic.QuadraticNoise0` | 1 |
| `quadratic.QuadraticNoise1em02` | 1 |
| `quadratic.QuadraticNoise1em03` | 1 |
| `quadratic.QuadraticNoise1em04` | 1 |
| `quadratic.QuadraticProfileCap30` | 5 |
| `quadratic.QuadraticProfileStep001` | 5 |
| `quadratic.QuadraticProfileStep005` | 5 |

## Not a harness score (12)

These files have a full start count and no `counter harness`. They stay off the ranked table.

| Problem | Template | n | Path |
| --- | --- | --- | --- |
| `double-ended-saddle-search/lj38` | `eon/de-lj38-lbfgs-m100` | 50 | `results/campaign/eon-main-de-lj38-lbfgs-m100.dat` |
| `double-ended-saddle-search/lj38` | `eon/de-lj38-lbfgs` | 50 | `results/campaign/eon-main-de-lj38-lbfgs.dat` |
| `double-ended-saddle-search/pt-island` | `eon/de-pt-lbfgs-m100` | 59 | `results/campaign/eon-main-de-pt-lbfgs-m100.dat` |
| `double-ended-saddle-search/pt-island` | `eon/de-pt-lbfgs` | 59 | `results/campaign/eon-main-de-pt-lbfgs.dat` |
| `neb/Pt-heptamer` | `eon/neb-pt-lbfgs-m100` | 13 | `results/campaign/eon-main-neb-pt-lbfgs-m100.dat` |
| `neb/Pt-heptamer` | `eon/neb-pt-lbfgs` | 13 | `results/campaign/eon-main-neb-pt-lbfgs.dat` |
| `saddle-search/pt-island` | `eon-3.5.0-saddle-pt-classic` | 10 | `results/campaign/eon-3.5.0-saddle-pt-classic.dat` |
| `saddle-search/pt-island` | `eon-3.5.0-saddle-pt-dimer` | 10 | `results/campaign/eon-3.5.0-saddle-pt-dimer.dat` |
| `tsopt/LJ38` | `eon/ts-lj38-dimer-m100` | 200 | `results/campaign/eon-main-ts-lj38-dimer-m100.dat` |
| `tsopt/LJ38` | `eon/ts-lj38-dimer` | 200 | `results/campaign/eon-main-ts-lj38-dimer.dat` |
| `tsopt/pt-island` | `eon/ts-pt-dimer-m100` | 49 | `results/campaign/eon-main-ts-pt-dimer-m100.dat` |
| `tsopt/pt-island` | `eon/ts-pt-dimer` | 49 | `results/campaign/eon-main-ts-pt-dimer.dat` |

