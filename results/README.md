# Saved results

Output files from the evaluation commands, kept so the numbers in the main README and the project report can be checked.

| File | Command | What it shows |
|---|---|---|
| `backtest_baseline_permits-supply.json` | `retrend backtest --models baseline` with `RETREND_PERMITS_RULE=supply` | Rules-only backtest, 30 cities, jobs, permits and schools shown as extra variants |
| `backtest_baseline_permits-demand.json` | same, with `RETREND_PERMITS_RULE=demand` | The same run with the permits rule read the other way |
| `compare_models_qwen_llama_gemma.json` | `retrend compare-models --models qwen,llama,gemma` | Same-case comparison of qwen3:4b-instruct, llama3.2:1b and gemma3:4b |
| `backtest_qwen_all_signals.json` | `retrend backtest --models qwen --with-schools --with-jobs --with-permits` | qwen judging every signal, with ablations and weight tuning |
| `schools_test.json` | `retrend schools-test` | Enrollment growth against later price growth, 16 of 30 cities |

Zillow and Redfin revise their history, so rerunning these later will give somewhat different numbers.
