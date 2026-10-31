Older files in this directory name the machine that measured them, and newer runs record the `BENCH_HOST_LABEL` label instead (`workstation-a` for campaign runs, `unspecified` when unset).
New asv results go under `asv/<label>/`, with every other machine field `unspecified`.
The one older directory under `asv/` keeps the machine record asv wrote then.
`docs/harness.md` lists each place the label is written.
