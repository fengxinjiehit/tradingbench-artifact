# TradingBench — Anonymous Artifact

**TradingBench** is a benchmark that scores large language models with
*realized trading outcomes* rather than multiple-choice accuracy. At each of
**2,000 decision points**, evenly sampled from 11 years of real XAUUSD (gold)
minute-level data (2015–2026), the model receives hourly (H1) market data up
to time *T* and must decide **LONG / SHORT / FLAT** over the next **24
hours**. Decisions are scored with ATR-normalized realized PnL (stop loss
1.5×ATR, take profit 2.0×ATR, conservative stop-first within a bar,
otherwise time exit at the horizon close), plus win rate, profit factor,
max drawdown, Sharpe, and directional accuracy. The construction uses only
completed H1 bars at or before *T* — strictly no lookahead.

This repository contains the released dataset (decision points and future
price paths), all 12,000 prompts exactly as shown to the models, the
standalone scoring harness, and the full experiment pipeline used in the
paper.

## Contents

```
artifact/
├── README.md                      ← this file
├── code/
│   ├── score_predictions.py       ← standalone scorer (start here)
│   ├── export_dataset.py          ← regenerates data/ and prompts/ from raw M1 CSV
│   ├── tradingbench_pipeline.py   ← end-to-end pipeline (features, prompts, LLM calls, backtest)
│   ├── analyze_results.py         ← analysis of cached LLM responses
│   ├── learned_baseline.py        ← ridge-regression learned baseline
│   ├── significance.py            ← significance tests
│   └── spread_sensitivity.py      ← transaction-cost sensitivity check
├── data/
│   ├── decision_points.csv        ← 2,000 rows: features at each decision point
│   └── future_paths.npz           ← realized 24h paths (scoring ground truth)
└── prompts/
    ├── prompts_features_zh.jsonl  ┐
    ├── prompts_features_en.jsonl  │  6 files × 2,000 prompts:
    ├── prompts_numeric_zh.jsonl   │  {features, numeric, orderflow} × {zh, en}
    ├── prompts_numeric_en.jsonl   │  (zh = Chinese prompts used for most runs;
    ├── prompts_orderflow_zh.jsonl │   en = English control)
    └── prompts_orderflow_en.jsonl ┘
```

## Quickstart: score your own model

Requirements: Python 3 with `numpy` and `pandas` (nothing else).

1. Run your model on the prompts. Each line of a prompt file is:

   ```json
   {"id": 0, "timestamp": "2015-01-23 07:00:00", "prompt": "..."}
   ```

2. Write predictions as JSONL, one object per line. `signal` accepts
   `LONG` / `SHORT` / `FLAT` (any case) or `+1` / `0` / `-1`:

   ```json
   {"id": 0, "signal": "LONG"}
   {"id": 1, "signal": -1}
   ```

   Missing ids are treated as FLAT (no trade).

3. Score, from the `artifact/` root:

   ```bash
   python3 code/score_predictions.py --predictions my_predictions.jsonl
   ```

   Output: trade count and signal distribution, win rate, mean/total PnL in
   ATR multiples, profit factor, max drawdown, per-trade Sharpe, and
   directional accuracy (fraction of traded points where the signal sign
   matches the realized 24h return).

4. Sanity-check your setup against the built-in RSI mean-reversion baseline
   (rsi < 30 → LONG, rsi > 70 → SHORT, else FLAT), which does not need any
   model:

   ```bash
   python3 code/score_predictions.py --selftest
   ```

   Expected result: 220 trades, mean PnL ≈ **+0.52 ATR per trade**,
   win rate ≈ 57.7%, profit factor ≈ 1.85, Sharpe ≈ +0.31.

## Prompt formats

All prompts ask for exactly one word: `LONG / SHORT / FLAT`.

- **features**: last 6 H1 closes plus engineered features (3-bar momentum,
  momentum/volume z-scores, ATR(14), RSI(14)).
- **numeric**: the raw last 24 H1 candlesticks as an OHLCV table
  (`Time,Open,High,Low,Close,TickVol`).
- **orderflow**: the `features` prompt plus order-flow proxies derived from
  minute bars (signed-volume imbalance, imbalance z-score, close position
  within the bar, spread z-score).

`zh` prompts are the Chinese templates used in the paper's main experiments;
`en` prompts are the English control. Decision keywords are always the
English LONG/SHORT/FLAT.

## Data schemas

`data/decision_points.csv` — one row per decision point:

| column | meaning |
|---|---|
| `id` | 0–1999 |
| `timestamp` | decision time (last completed H1 bar) |
| `close` | close at decision time (trade entry price) |
| `atr` | ATR(14), Wilder smoothing |
| `r3` | 3-bar momentum (return) |
| `z_mom`, `z_vol`, `z_imb`, `z_spr` | rolling z-scores (window 336 H1 bars) of momentum, volume, order-flow imbalance, spread |
| `rsi` | RSI(14), Wilder smoothing |
| `of_imb`, `of_pos` | order-flow proxies: signed-volume imbalance ∈ [−1,1], close position in bar ∈ [0,1] |
| `fwd_ret` | realized 24h forward return (scoring only — never shown to models) |

`data/future_paths.npz` — arrays `high`, `low`, `close`, each of shape
(2000, 24): row *i* holds the 24 H1 bars following decision point *i*. This
is the ground truth the scorer replays for the stop/limit simulation.

## Regenerating from raw data

The raw minute-level CSV is **not redistributed** (see License below). To
regenerate the dataset with your own XAUUSD M1 feed, place a CSV with this
schema at `data/XAUUSD_M1_2015_2026.csv` relative to the project root that
contains this `artifact/` directory:

```
Date,Time,Open,High,Low,Close,TickVol,Vol,Spread
2015.01.01,00:00,1183.00,1183.20,1182.80,1183.05,42,0,20
```

(`Date` as `YYYY.MM.DD`, `Time` as `HH:MM`, one row per minute.) Then run,
from the project root:

```bash
python3 artifact/code/export_dataset.py
```

This reproduces `data/` and `prompts/` exactly: M1 → H1 resampling,
feature computation, evenly spaced sampling of 2,000 decision points, and
prompt generation with the pipeline's own `build_prompt()` (the released
prompts are byte-identical to those used in the paper's experiments).
Sampling is deterministic; no seed is needed.

`code/tradingbench_pipeline.py` additionally runs LLM decisions over any
OpenAI-compatible endpoint (API base URL and key via CLI flags /
environment variables) and prints the full baseline comparison table.

## License

The derived benchmark — decision points, future paths, prompts, and all
code in this repository — is released under **CC-BY-4.0**. The raw
minute-level price data originates from a commercial broker feed and is
**not** redistributed; substitute your own XAUUSD M1 data using the schema
above.

## Citation

Anonymous, under review. Citation information will be added upon
publication.
