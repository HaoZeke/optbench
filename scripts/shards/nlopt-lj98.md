# Lennard-Jones 98 NLopt basin-hopping shards

`problems/globalopt/lj98.toml` sets the sample and the cap: 100 starts, 1000000000 force calls. Success is an energy at or below the reference −543.665361 plus 0.001. The template `nloptglobal/lj98-lbfgs-bh` uses that same cap. These commands do not take a cap override.

Run one command from the repository root. Shard k of 12 runs only the indexes on its line and writes only `results/campaign/shards/nlopt-2.11.0-lj98-lbfgs-bh/NNN.progress.json` and `NNN.done.json` for those indexes. Do not pass `--shard-dir`. Do not pass `--assemble`; that writes the ranked campaign row only when all 100 done files exist. Do not delete another shard's files.

Starts 0–7 already have `optbench.bh-progress/v3` checkpoints at 1000000 calls in that directory. Shard 0 resumes them. Omitting `--checkpoint-calls` keeps the default, a snapshot after every finished hop.

The first four shards have 9 starts and the other eight have 8. `--workers 8` matches this VM. A 9-start shard runs eight starts, then the last one.

```text
uv run python -m optbench.runners.nlopt_global --problem globalopt/lj98 --template nloptglobal/lj98-lbfgs-bh --indexes 0-8 --workers 8
uv run python -m optbench.runners.nlopt_global --problem globalopt/lj98 --template nloptglobal/lj98-lbfgs-bh --indexes 9-17 --workers 8
uv run python -m optbench.runners.nlopt_global --problem globalopt/lj98 --template nloptglobal/lj98-lbfgs-bh --indexes 18-26 --workers 8
uv run python -m optbench.runners.nlopt_global --problem globalopt/lj98 --template nloptglobal/lj98-lbfgs-bh --indexes 27-35 --workers 8
uv run python -m optbench.runners.nlopt_global --problem globalopt/lj98 --template nloptglobal/lj98-lbfgs-bh --indexes 36-43 --workers 8
uv run python -m optbench.runners.nlopt_global --problem globalopt/lj98 --template nloptglobal/lj98-lbfgs-bh --indexes 44-51 --workers 8
uv run python -m optbench.runners.nlopt_global --problem globalopt/lj98 --template nloptglobal/lj98-lbfgs-bh --indexes 52-59 --workers 8
uv run python -m optbench.runners.nlopt_global --problem globalopt/lj98 --template nloptglobal/lj98-lbfgs-bh --indexes 60-67 --workers 8
uv run python -m optbench.runners.nlopt_global --problem globalopt/lj98 --template nloptglobal/lj98-lbfgs-bh --indexes 68-75 --workers 8
uv run python -m optbench.runners.nlopt_global --problem globalopt/lj98 --template nloptglobal/lj98-lbfgs-bh --indexes 76-83 --workers 8
uv run python -m optbench.runners.nlopt_global --problem globalopt/lj98 --template nloptglobal/lj98-lbfgs-bh --indexes 84-91 --workers 8
uv run python -m optbench.runners.nlopt_global --problem globalopt/lj98 --template nloptglobal/lj98-lbfgs-bh --indexes 92-99 --workers 8
```
