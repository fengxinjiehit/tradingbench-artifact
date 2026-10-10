#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TradingBench dataset/prompt exporter.

Run from the project root (the directory that contains data/ and artifact/):

    python3 artifact/code/export_dataset.py

Reproduces the benchmark's 2,000 decision points from the raw M1 CSV and
writes, into artifact/:

  data/decision_points.csv           one row per decision point
  data/future_paths.npz              (2000, 24) arrays: high / low / close
  prompts/prompts_{style}_{lang}.jsonl   6 files x 2,000 prompts

Prompts are generated with the pipeline's own build_prompt(), so they are
byte-identical to the prompts used in the paper's experiments.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

# import the pipeline from the same directory as this script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tradingbench_pipeline as tb

CSV_PATH = "data/XAUUSD_M1_2015_2026.csv"   # relative to project root
OUT_DATA = "artifact/data"
OUT_PROMPTS = "artifact/prompts"
MAX_SAMPLES = 2000
HORIZON = tb.HORIZON  # 24
STYLES = ("features", "numeric", "orderflow")
LANGS = ("zh", "en")


def sample_positions(feats, max_samples=MAX_SAMPLES):
    """Exact replica of the sampling in tradingbench_pipeline.run():
    rows with all features + forward return known, evenly spaced via
    np.unique(np.linspace(...))."""
    valid = feats[['z_mom', 'atr', 'z_vol', 'rsi', 'fwd_ret']].notna().all(axis=1)
    positions = np.where(valid.values)[0]
    if max_samples and len(positions) > max_samples:
        idx = np.unique(np.linspace(0, len(positions) - 1, max_samples)
                        .round().astype(int))
        positions = positions[idx]
    return positions


def main():
    if not os.path.exists(CSV_PATH):
        sys.exit(f"[export] raw M1 CSV not found at {CSV_PATH} "
                 f"(run from the project root)")
    os.makedirs(OUT_DATA, exist_ok=True)
    os.makedirs(OUT_PROMPTS, exist_ok=True)

    h1 = tb.load_h1(CSV_PATH)
    feats = tb.add_features(h1, horizon=HORIZON)
    positions = sample_positions(feats)
    n = len(positions)
    print(f"[export] decision points: {n}")
    if n != MAX_SAMPLES:
        sys.exit(f"[export] expected {MAX_SAMPLES} decision points, got {n}")

    ts = feats.index[positions]

    # ---- decision_points.csv ----
    cols = ['close', 'atr', 'r3', 'z_mom', 'z_vol', 'rsi',
            'of_imb', 'of_pos', 'z_imb', 'z_spr', 'fwd_ret']
    dp = pd.DataFrame({c: feats[c].values[positions] for c in cols})
    dp.insert(0, 'timestamp', ts.strftime('%Y-%m-%d %H:%M:%S'))
    dp.insert(0, 'id', np.arange(n))
    out_csv = os.path.join(OUT_DATA, 'decision_points.csv')
    dp.to_csv(out_csv, index=False)
    print(f"[export] wrote {out_csv} ({len(dp)} rows)")

    # ---- future_paths.npz: the 24 H1 bars after each decision point ----
    fut_high = np.full((n, HORIZON), np.nan)
    fut_low = np.full((n, HORIZON), np.nan)
    fut_close = np.full((n, HORIZON), np.nan)
    for i, p in enumerate(positions):
        fut = h1.iloc[p + 1:p + 1 + HORIZON]
        assert len(fut) == HORIZON, f"decision point {i}: only {len(fut)} future bars"
        fut_high[i] = fut['High'].values
        fut_low[i] = fut['Low'].values
        fut_close[i] = fut['Close'].values
    out_npz = os.path.join(OUT_DATA, 'future_paths.npz')
    np.savez_compressed(out_npz, high=fut_high, low=fut_low, close=fut_close)
    print(f"[export] wrote {out_npz} (high/low/close, shape {fut_high.shape})")

    # ---- prompts: 6 style x lang combos, via the pipeline's build_prompt ----
    for style in STYLES:
        for lang in LANGS:
            path = os.path.join(OUT_PROMPTS, f"prompts_{style}_{lang}.jsonl")
            with open(path, 'w', encoding='utf-8') as f:
                for i, p in enumerate(positions):
                    prompt = tb.build_prompt(style, h1, feats, p,
                                             horizon=HORIZON, lang=lang)
                    f.write(json.dumps(
                        {"id": int(i),
                         "timestamp": ts[i].strftime('%Y-%m-%d %H:%M:%S'),
                         "prompt": prompt},
                        ensure_ascii=False) + "\n")
            print(f"[export] wrote {path} ({n} prompts)")

    print("[export] done.")


if __name__ == '__main__':
    main()
