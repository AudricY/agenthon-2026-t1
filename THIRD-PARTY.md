# Third-party material

The MIT licence in [LICENSE](LICENSE) covers the code written for this entry. Everything below keeps its own terms.
The build fetches most of it into `vendor/` instead of committing it.

| Material | Terms | In this repository |
|---|---|---|
| The organisers' Track 1 kit (`github.com/Agenthon-2026/track1-coding-public`, commit `84049db`): the units, their checkers and the sandbox base's Dockerfile | CC BY-NC 4.0, "Pawel Polak." (the kit's `LICENSE` and `DATA-LICENSE.md`), except where a unit's `manifest.json` names another licence | Not committed. `scripts/fetch_vendor.sh` clones it |
| `agent/n11/examples.jsonl`: 56 worked examples, each a task's instruction and input-file headers beside a script the model wrote for it during my development runs | The task text keeps its source's licence, listed per entry below; the MIT licence does not cover it | Committed, because it is part of the Final image's `/opt/agent`. Under the image's settings the agent never reads it |
| The organisers' toolkit `qfbench2-common` (`github.com/Agenthon-2026/Agenthon2026-public`, tag `v2.6.0`) | CC BY-NC 4.0, Pawel Polak (the package's metadata) | Not committed. `scripts/run_units.py` imports it to build each unit's tree |
| The tokenizer of [`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16) (`tokenizer.json`, revision `2dc98e2`) | The NVIDIA Nemotron Open Model License, as the model's page names it | Not committed. Fetched and checked by sha256 |
| `tokenizers` 0.23.3 (Hugging Face) | Apache-2.0 | Not committed. The wheel is fetched from PyPI and checked by sha256 |
| The sandbox base's packages (`docker/base.lock`: numpy, pandas, scipy, TA-Lib and 57 more) and `python:3.13-slim` | Each package's own licence | Only the list of versions is committed |
| The model, `nvidia/nemotron-3-super-120b-a12b` | The hosting provider's terms | Reached over an API with your own key. No weights are used or distributed |

`agent/ARTIFACT_PROVENANCE.md` is the provenance note that went into the Final image. It describes all 16 agents I
built, of which this repository holds n11 and the two it imports, n0 and w1.

## Sources of `agent/n11/examples.jsonl`

The entries are extracts: each keeps a task's instruction and the first lines of its input files, and drops the
data and the checks. Fifteen come from the kit's public units. The other 41 come from open task pull requests to
[`QF-Bench/QuantitativeFinance-Bench`](https://github.com/QF-Bench/QuantitativeFinance-Bench) (CC BY-NC 4.0, by the
Finance-Bench Team), which I converted to the kit's format. CC BY-NC 4.0: https://creativecommons.org/licenses/by-nc/4.0/

| Entry (`unit`) | Source | Licence |
|---|---|---|
| `t1-EXAMPLE-bs-greeks-pde` | Agenthon 2026 Track 1 kit, `units/t1-EXAMPLE-bs-greeks-pde` at `84049db` | CC0 1.0 |
| `t1-asian-option-levy-curran` | Agenthon 2026 Track 1 kit, `units/t1-asian-option-levy-curran` at `84049db` | CC BY-NC 4.0 |
| `t1-brinson-sector-attribution` | Agenthon 2026 Track 1 kit, `units/t1-brinson-sector-attribution` at `84049db` | CC BY-NC 4.0 |
| `t1-copula-equity-fitting` | Agenthon 2026 Track 1 kit, `units/t1-copula-equity-fitting` at `84049db` | CC BY-NC 4.0 |
| `t1-copula-sampling-rank-correlation` | Agenthon 2026 Track 1 kit, `units/t1-copula-sampling-rank-correlation` at `84049db` | CC BY-NC 4.0 |
| `t1-digital-barrier-options` | Agenthon 2026 Track 1 kit, `units/t1-digital-barrier-options` at `84049db` | CC BY-NC 4.0 |
| `t1-earnings-surprise-calculator` | Agenthon 2026 Track 1 kit, `units/t1-earnings-surprise-calculator` at `84049db` | CC BY-NC 4.0 |
| `t1-historical-var-data-prep` | Agenthon 2026 Track 1 kit, `units/t1-historical-var-data-prep` at `84049db` | CC BY-NC 4.0 |
| `t1-interest-rate-cap-floor` | Agenthon 2026 Track 1 kit, `units/t1-interest-rate-cap-floor` at `84049db` | CC BY-NC 4.0 |
| `t1-momentum-backtest` | Agenthon 2026 Track 1 kit, `units/t1-momentum-backtest` at `84049db` | CC BY-NC 4.0 |
| `t1-ohlc-realized-vol-estimators` | Agenthon 2026 Track 1 kit, `units/t1-ohlc-realized-vol-estimators` at `84049db` | CC BY-NC 4.0 |
| `t1-ou-jump-commodity` | Agenthon 2026 Track 1 kit, `units/t1-ou-jump-commodity` at `84049db` | CC BY-NC 4.0 |
| `t1-pca-factor-portfolio` | Agenthon 2026 Track 1 kit, `units/t1-pca-factor-portfolio` at `84049db` | CC BY-NC 4.0 |
| `t1-realized-vol-estimators` | Agenthon 2026 Track 1 kit, `units/t1-realized-vol-estimators` at `84049db` | CC BY-NC 4.0 |
| `t1-sma-crossover-spy` | Agenthon 2026 Track 1 kit, `units/t1-sma-crossover-spy` at `84049db` | CC BY-NC 4.0 |
| `t1-asian-option-pricing` | [QF-Bench PR #4](https://github.com/QF-Bench/QuantitativeFinance-Bench/pull/4) at `6919421a0986` | CC BY-NC 4.0 |
| `t1-beta-hedging` | [QF-Bench PR #4](https://github.com/QF-Bench/QuantitativeFinance-Bench/pull/4) at `6919421a0986` | CC BY-NC 4.0 |
| `t1-binary-option-pricing` | [QF-Bench PR #4](https://github.com/QF-Bench/QuantitativeFinance-Bench/pull/4) at `6919421a0986` | CC BY-NC 4.0 |
| `t1-black-scholes-pricing` | [QF-Bench PR #4](https://github.com/QF-Bench/QuantitativeFinance-Bench/pull/4) at `6919421a0986` | CC BY-NC 4.0 |
| `t1-bond-convexity` | [QF-Bench PR #4](https://github.com/QF-Bench/QuantitativeFinance-Bench/pull/4) at `6919421a0986` | CC BY-NC 4.0 |
| `t1-carry-trade-fx` | [QF-Bench PR #4](https://github.com/QF-Bench/QuantitativeFinance-Bench/pull/4) at `6919421a0986` | CC BY-NC 4.0 |
| `t1-cds-pricing` | [QF-Bench PR #4](https://github.com/QF-Bench/QuantitativeFinance-Bench/pull/4) at `6919421a0986` | CC BY-NC 4.0 |
| `t1-chart-pattern-backtest` | [QF-Bench PR #4](https://github.com/QF-Bench/QuantitativeFinance-Bench/pull/4) at `6919421a0986` | CC BY-NC 4.0 |
| `t1-credit-portfolio-var` | [QF-Bench PR #4](https://github.com/QF-Bench/QuantitativeFinance-Bench/pull/4) at `6919421a0986` | CC BY-NC 4.0 |
| `t1-credit-risk` | [QF-Bench PR #4](https://github.com/QF-Bench/QuantitativeFinance-Bench/pull/4) at `6919421a0986` | CC BY-NC 4.0 |
| `t1-dividend-adjusted-returns` | [QF-Bench PR #4](https://github.com/QF-Bench/QuantitativeFinance-Bench/pull/4) at `6919421a0986` | CC BY-NC 4.0 |
| `t1-drawdown-analysis` | [QF-Bench PR #4](https://github.com/QF-Bench/QuantitativeFinance-Bench/pull/4) at `6919421a0986` | CC BY-NC 4.0 |
| `t1-expected-shortfall` | [QF-Bench PR #4](https://github.com/QF-Bench/QuantitativeFinance-Bench/pull/4) at `6919421a0986` | CC BY-NC 4.0 |
| `t1-factor-risk-decomposition` | [QF-Bench PR #4](https://github.com/QF-Bench/QuantitativeFinance-Bench/pull/4) at `6919421a0986` | CC BY-NC 4.0 |
| `t1-financial-statements` | [QF-Bench PR #4](https://github.com/QF-Bench/QuantitativeFinance-Bench/pull/4) at `6919421a0986` | CC BY-NC 4.0 |
| `t1-fx-forward-pricing` | [QF-Bench PR #4](https://github.com/QF-Bench/QuantitativeFinance-Bench/pull/4) at `6919421a0986` | CC BY-NC 4.0 |
| `t1-fx-option-garman-kohlhagen` | [QF-Bench PR #4](https://github.com/QF-Bench/QuantitativeFinance-Bench/pull/4) at `6919421a0986` | CC BY-NC 4.0 |
| `t1-fx-pnl-attribution` | [QF-Bench PR #4](https://github.com/QF-Bench/QuantitativeFinance-Bench/pull/4) at `6919421a0986` | CC BY-NC 4.0 |
| `t1-implied-vol-surface-calibration` | [QF-Bench PR #4](https://github.com/QF-Bench/QuantitativeFinance-Bench/pull/4) at `6919421a0986` | CC BY-NC 4.0 |
| `t1-interest-rate-swap-pricing` | [QF-Bench PR #4](https://github.com/QF-Bench/QuantitativeFinance-Bench/pull/4) at `6919421a0986` | CC BY-NC 4.0 |
| `t1-liquidity-adjusted-var` | [QF-Bench PR #4](https://github.com/QF-Bench/QuantitativeFinance-Bench/pull/4) at `6919421a0986` | CC BY-NC 4.0 |
| `t1-lookback-option-pricing` | [QF-Bench PR #4](https://github.com/QF-Bench/QuantitativeFinance-Bench/pull/4) at `6919421a0986` | CC BY-NC 4.0 |
| `t1-mean-reversion` | [QF-Bench PR #4](https://github.com/QF-Bench/QuantitativeFinance-Bench/pull/4) at `6919421a0986` | CC BY-NC 4.0 |
| `t1-momentum-factor` | [QF-Bench PR #4](https://github.com/QF-Bench/QuantitativeFinance-Bench/pull/4) at `6919421a0986` | CC BY-NC 4.0 |
| `t1-monte-carlo-option-pricing` | [QF-Bench PR #4](https://github.com/QF-Bench/QuantitativeFinance-Bench/pull/4) at `6919421a0986` | CC BY-NC 4.0 |
| `t1-multi-asset-correlation` | [QF-Bench PR #4](https://github.com/QF-Bench/QuantitativeFinance-Bench/pull/4) at `6919421a0986` | CC BY-NC 4.0 |
| `t1-nav-calculation` | [QF-Bench PR #4](https://github.com/QF-Bench/QuantitativeFinance-Bench/pull/4) at `6919421a0986` | CC BY-NC 4.0 |
| `t1-options-covered-call-backtest` | [QF-Bench PR #4](https://github.com/QF-Bench/QuantitativeFinance-Bench/pull/4) at `6919421a0986` | CC BY-NC 4.0 |
| `t1-parametric-var` | [QF-Bench PR #4](https://github.com/QF-Bench/QuantitativeFinance-Bench/pull/4) at `6919421a0986` | CC BY-NC 4.0 |
| `t1-performance-attribution` | [QF-Bench PR #4](https://github.com/QF-Bench/QuantitativeFinance-Bench/pull/4) at `6919421a0986` | CC BY-NC 4.0 |
| `t1-portfolio-analytics-report` | [QF-Bench PR #4](https://github.com/QF-Bench/QuantitativeFinance-Bench/pull/4) at `6919421a0986` | CC BY-NC 4.0 |
| `t1-portfolio-beta` | [QF-Bench PR #4](https://github.com/QF-Bench/QuantitativeFinance-Bench/pull/4) at `6919421a0986` | CC BY-NC 4.0 |
| `t1-portfolio-optimization` | [QF-Bench PR #4](https://github.com/QF-Bench/QuantitativeFinance-Bench/pull/4) at `6919421a0986` | CC BY-NC 4.0 |
| `t1-quanto-option-pricing` | [QF-Bench PR #4](https://github.com/QF-Bench/QuantitativeFinance-Bench/pull/4) at `6919421a0986` | CC BY-NC 4.0 |
| `t1-sharpe-ratio` | [QF-Bench PR #4](https://github.com/QF-Bench/QuantitativeFinance-Bench/pull/4) at `6919421a0986` | CC BY-NC 4.0 |
| `t1-sharpe-sortino-ratio` | [QF-Bench PR #4](https://github.com/QF-Bench/QuantitativeFinance-Bench/pull/4) at `6919421a0986` | CC BY-NC 4.0 |
| `t1-swaption-pricing` | [QF-Bench PR #4](https://github.com/QF-Bench/QuantitativeFinance-Bench/pull/4) at `6919421a0986` | CC BY-NC 4.0 |
| `t1-tracking-error-analysis` | [QF-Bench PR #4](https://github.com/QF-Bench/QuantitativeFinance-Bench/pull/4) at `6919421a0986` | CC BY-NC 4.0 |
| `t1-trade-blotter-reconciliation` | [QF-Bench PR #4](https://github.com/QF-Bench/QuantitativeFinance-Bench/pull/4) at `6919421a0986` | CC BY-NC 4.0 |
| `t1-var-stress-test` | [QF-Bench PR #4](https://github.com/QF-Bench/QuantitativeFinance-Bench/pull/4) at `6919421a0986` | CC BY-NC 4.0 |
| `t1-yield-to-maturity` | [QF-Bench PR #4](https://github.com/QF-Bench/QuantitativeFinance-Bench/pull/4) at `6919421a0986` | CC BY-NC 4.0 |
