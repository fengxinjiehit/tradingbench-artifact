#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TradingBench 显著性检验: 配对 bootstrap

把 FLAT 视为 PnL=0 (观望的真实收益就是 0), 在每个决策点上形成
(模型A, 模型B) 的配对 PnL, 对均值差做 10,000 次 bootstrap 重抽样,
报告 95% 置信区间。检验两组对比:
  1) features - numeric   (特征接口是否显著)
  2) 各模型(features) - RSI反转基线  (LLM 是否显著跑输基线)

用法 (从项目根目录): python3 tradingbench/significance.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tradingbench.tradingbench_pipeline as tb

N_BOOT = 10_000
MODELS = ['deepseek-chat', 'deepseek-reasoner',
          'Qwen/Qwen3-32B', 'Qwen/Qwen3-8B', 'THUDM/GLM-4-32B-0414']
SHORT = {'deepseek-chat': 'DS-V3', 'deepseek-reasoner': 'DS-R1',
         'Qwen/Qwen3-32B': 'Qwen3-32B', 'Qwen/Qwen3-8B': 'Qwen3-8B',
         'THUDM/GLM-4-32B-0414': 'GLM-4-32B'}


def pnl_per_point(sig, positions, feats, h1):
    """每个决策点的 PnL (ATR); FLAT = 0。"""
    out = np.zeros(len(positions))
    for i, p in enumerate(positions):
        s = int(sig[i])
        if s == 0:
            continue
        fut = h1.iloc[p + 1:p + 1 + tb.HORIZON]
        if len(fut) < tb.HORIZON:
            continue
        out[i] = tb.trade_pnl(feats['close'].values[p], s,
                              feats['atr'].values[p],
                              fut['High'].values, fut['Low'].values,
                              fut['Close'].values)
    return out


def boot_ci(a, b, rng):
    """配对 bootstrap: mean(a-b) 的 95% CI。"""
    d = a - b
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    means = d[idx].mean(axis=1)
    return d.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


BLOCK = 20  # block bootstrap 块长: 20 个连续决策点 ≈ 20 个交易日


def boot_ci_block(a, b, rng, block=BLOCK):
    """配对 block bootstrap: 重采样连续块, 保留时序相关性, CI 更保守。"""
    d = a - b
    n = len(d)
    k = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(N_BOOT, k))
    idx = (starts[..., None] + np.arange(block)).reshape(N_BOOT, k * block)[:, :n]
    means = d[idx].mean(axis=1)
    return d.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def main():
    rng = np.random.default_rng(42)
    h1 = tb.load_h1(tb.CSV_PATH)
    feats = tb.add_features(h1)
    valid = feats[['z_mom', 'atr', 'z_vol', 'rsi', 'fwd_ret']].notna().all(axis=1)
    positions = np.where(valid.values)[0]
    idx = np.unique(np.linspace(0, len(positions) - 1, 2000).round().astype(int))
    positions = positions[idx]

    cache = tb._load_cache()
    print(f"[sig] 缓存 {len(cache)} 条")

    def signals(model, style):
        tb.LLM_BASE_URL, tb.LLM_API_KEY_ENV = (
            ('https://api.siliconflow.cn/v1', 'SILICONFLOW_API_KEY')
            if model.startswith(('Qwen/', 'THUDM/')) else
            ('https://api.deepseek.com', 'DEEPSEEK_API_KEY'))
        tb._LLM_CLIENT = None
        return np.array([tb.call_llm(tb.build_prompt(style, h1, feats, p),
                                     feats.iloc[p], model=model)
                         for p in positions], dtype=int)

    rsi = feats['rsi'].values[positions]
    pnl = {'RSI反转': pnl_per_point(
        np.where(rsi < 30, 1, np.where(rsi > 70, -1, 0)), positions, feats, h1)}

    print("\n== 检验 1: features - numeric (配对 bootstrap, 95% CI, ATR) ==")
    print("  (iid = 逐点重抽样; block = 20点连续块重抽样, 更保守)")
    rows1 = []
    for m in MODELS:
        pf = pnl_per_point(signals(m, 'features'), positions, feats, h1)
        pn = pnl_per_point(signals(m, 'numeric'), positions, feats, h1)
        pnl[f'{m}(f)'] = pf
        d, lo, hi = boot_ci(pf, pn, rng)
        _, blo, bhi = boot_ci_block(pf, pn, rng)
        rows1.append((SHORT[m], d, lo, hi))
        print(f"  {SHORT[m]:<12} Δ={d:+.3f}  iid=[{lo:+.3f},{hi:+.3f}]  "
              f"block=[{blo:+.3f},{bhi:+.3f}]  "
              f"{'显著' if blo > 0 or bhi < 0 else '不显著(block口径)'}")

    print("\n== 检验 2: LLM(features) - RSI反转 (95% CI, ATR) ==")
    for m in MODELS:
        d, lo, hi = boot_ci(pnl[f'{m}(f)'], pnl['RSI反转'], rng)
        _, blo, bhi = boot_ci_block(pnl[f'{m}(f)'], pnl['RSI反转'], rng)
        print(f"  {SHORT[m]:<12} Δ={d:+.3f}  iid=[{lo:+.3f},{hi:+.3f}]  "
              f"block=[{blo:+.3f},{bhi:+.3f}]  "
              f"{'显著跑输' if bhi < 0 else ('显著跑赢' if blo > 0 else '不显著(block口径)')}")


if __name__ == '__main__':
    main()
