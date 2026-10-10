#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TradingBench standalone scorer.

Scores model predictions against realized market outcomes using the
benchmark's ATR-multiple backtest (SL = 1.5 x ATR, TP = 2.0 x ATR,
conservative stop-first within a bar, otherwise time exit at the 24h
horizon close).

Usage (run from the artifact/ root, or pass --data-dir explicitly):

    python3 code/score_predictions.py --predictions my_predictions.jsonl
    python3 code/score_predictions.py --selftest

Prediction file format (one JSON object per line):

    {"id": 0, "signal": "LONG"}     # LONG / SHORT / FLAT (any case)
    {"id": 1, "signal": -1}         # +1 / 0 / -1 also accepted

ids must refer to data/decision_points.csv. Missing ids are treated as
FLAT (no trade). This file is self-contained: it does NOT import anything
from the rest of the repository.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

SL_ATR = 1.5   # stop loss = 1.5 x ATR
TP_ATR = 2.0   # take profit = 2.0 x ATR
HORIZON = 24   # scored future window (H1 bars)


def trade_pnl(entry, direction, atr, fut_high, fut_low, fut_close):
    """direction: +1 long / -1 short. PnL is expressed in ATR multiples.
    Identical logic to the benchmark pipeline (stop-first within a bar)."""
    sl = entry - direction * SL_ATR * atr
    tp = entry + direction * TP_ATR * atr
    for i in range(len(fut_high)):
        h, l = fut_high[i], fut_low[i]
        if direction == 1:
            if l <= sl:      # check stop first (conservative)
                return -SL_ATR
            if h >= tp:
                return +TP_ATR
        else:
            if h >= sl:
                return -SL_ATR
            if l <= tp:
                return +TP_ATR
    # neither triggered: exit at the horizon close
    return direction * (fut_close[-1] - entry) / atr


def parse_signal_value(v):
    """Accept LONG/SHORT/FLAT (any case) or +1/0/-1 (int or string)."""
    if isinstance(v, (int, float)):
        s = int(v)
        if s in (-1, 0, 1):
            return s
        raise ValueError(f"numeric signal must be -1/0/1, got {v!r}")
    t = str(v).strip().upper()
    if t in ('LONG', '+1', '1'):
        return 1
    if t in ('SHORT', '-1'):
        return -1
    if t in ('FLAT', '0'):
        return 0
    raise ValueError(f"unrecognized signal {v!r} "
                     f"(expected LONG/SHORT/FLAT or +1/0/-1)")


def load_predictions(path):
    signals = {}
    with open(path, encoding='utf-8') as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                signals[int(rec['id'])] = parse_signal_value(rec['signal'])
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                sys.exit(f"[score] bad prediction at {path}:{ln}: {e}")
    return signals


def evaluate(signals, dp, paths):
    """signals: dict id -> +1/-1/0. Returns a metrics dict."""
    n = len(dp)
    entry = dp['close'].values
    atr = dp['atr'].values
    fwd = dp['fwd_ret'].values

    pnls = []
    hits = 0
    traded = 0
    n_long = n_short = n_flat = 0
    for i in range(n):
        s = signals.get(i, 0)   # missing id -> FLAT
        if s == 1:
            n_long += 1
        elif s == -1:
            n_short += 1
        else:
            n_flat += 1
            continue
        pnls.append(trade_pnl(entry[i], s, atr[i],
                              paths['high'][i], paths['low'][i],
                              paths['close'][i]))
        traded += 1
        if np.sign(s) == np.sign(fwd[i]):
            hits += 1

    pnls = np.array(pnls)
    wins = int((pnls > 0).sum())
    gw = pnls[pnls > 0].sum()
    gl = -pnls[pnls < 0].sum()
    pf = gw / gl if gl > 0 else float('inf')
    cum = np.cumsum(pnls)
    mdd = max(cum.max() - cum.min(), 0.0) if len(cum) else 0.0
    sharpe = pnls.mean() / pnls.std() if pnls.std() > 0 else 0.0
    return {
        'decision_points': n,
        'trades': traded,
        'win_rate': wins / traded if traded else 0.0,
        'mean_pnl_atr': float(pnls.mean()) if traded else 0.0,
        'total_pnl_atr': float(pnls.sum()) if traded else 0.0,
        'profit_factor': pf,
        'max_drawdown_atr': float(mdd),
        'sharpe': float(sharpe),
        'directional_accuracy': hits / traded if traded else 0.0,
        'signal_distribution': {'long': n_long, 'short': n_short,
                                'flat': n_flat},
    }


def print_metrics(name, m):
    pf = f"{m['profit_factor']:.2f}" if np.isfinite(m['profit_factor']) else 'inf'
    d = m['signal_distribution']
    print(f"{name}")
    print(f"  decision points   : {m['decision_points']}")
    print(f"  trades            : {m['trades']} "
          f"(long {d['long']}, short {d['short']}, flat {d['flat']})")
    print(f"  win rate          : {m['win_rate'] * 100:.1f}%")
    print(f"  mean PnL / trade  : {m['mean_pnl_atr']:+.3f} ATR")
    print(f"  total PnL         : {m['total_pnl_atr']:+.1f} ATR")
    print(f"  profit factor     : {pf}")
    print(f"  max drawdown      : {m['max_drawdown_atr']:.1f} ATR")
    print(f"  Sharpe (per trade): {m['sharpe']:+.2f}")
    print(f"  directional acc.  : {m['directional_accuracy'] * 100:.1f}% "
          f"(vs realized 24h return, traded points only)")


def main():
    default_data = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'data')
    ap = argparse.ArgumentParser(description="TradingBench standalone scorer")
    ap.add_argument('--predictions',
                    help="JSONL file: {\"id\": int, \"signal\": LONG/SHORT/FLAT}")
    ap.add_argument('--data-dir', default=default_data,
                    help="directory containing decision_points.csv and "
                         "future_paths.npz (default: <artifact>/data)")
    ap.add_argument('--selftest', action='store_true',
                    help="score the built-in RSI mean-reversion baseline "
                         "(rsi<30 -> LONG, rsi>70 -> SHORT, else FLAT)")
    ap.add_argument('--json', action='store_true',
                    help="print metrics as a single JSON line")
    a = ap.parse_args()
    if not a.selftest and not a.predictions:
        ap.error("give --predictions FILE or --selftest")

    dp_path = os.path.join(a.data_dir, 'decision_points.csv')
    npz_path = os.path.join(a.data_dir, 'future_paths.npz')
    for p in (dp_path, npz_path):
        if not os.path.exists(p):
            sys.exit(f"[score] missing {p} "
                     f"(run export_dataset.py first, or fix --data-dir)")
    dp = pd.read_csv(dp_path)
    paths = np.load(npz_path)
    if len(dp) != paths['high'].shape[0]:
        sys.exit("[score] decision_points.csv and future_paths.npz disagree")

    if a.selftest:
        rsi = dp['rsi'].values
        signals = {i: (1 if rsi[i] < 30 else -1 if rsi[i] > 70 else 0)
                   for i in range(len(dp))}
        m = evaluate(signals, dp, paths)
        if a.json:
            print(json.dumps({'name': 'RSI mean-reversion (selftest)', **m}))
        else:
            print_metrics("RSI mean-reversion baseline (selftest)", m)
        return

    signals = load_predictions(a.predictions)
    unknown = sorted(set(signals) - set(range(len(dp))))
    if unknown:
        print(f"[score] warning: {len(unknown)} prediction ids out of range "
              f"(ignored), e.g. {unknown[:5]}", file=sys.stderr)
        signals = {k: v for k, v in signals.items() if 0 <= k < len(dp)}
    m = evaluate(signals, dp, paths)
    if a.json:
        print(json.dumps({'name': os.path.basename(a.predictions), **m}))
    else:
        print_metrics(f"predictions: {a.predictions}", m)


if __name__ == '__main__':
    main()
