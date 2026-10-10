#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TradingBench 结果固化 + 失败模式分析

复用 tradingbench_pipeline 的数据/特征/打分逻辑, 对缓存中已有的
(模型 × style) 组合重新汇总 (全部走 llm_cache.jsonl, 不发起 API 调用),
输出:
  - 总表 (各模型 vs 基线)
  - 按年份拆分 (做空偏置是否全程一致)
  - 失败模式: LLM 信号 vs 动量 z-score 的列联表 (趋势外推 or 均值回归)
  - 写入 tradingbench/results.md

用法 (从项目根目录):
  python3 tradingbench/analyze_results.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tradingbench.tradingbench_pipeline as tb

MODELS = [
    ('deepseek-chat', 'features'),
    ('deepseek-chat', 'numeric'),
    ('deepseek-chat', 'orderflow'),
    ('deepseek-reasoner', 'features'),
    ('deepseek-reasoner', 'numeric'),
    ('deepseek-reasoner', 'orderflow'),
]
# SiliconFlow 模型矩阵 (需对应批量跑完、缓存齐全后加 --with_sf 才纳入)
SF_MODELS = [
    ('Qwen/Qwen3-32B', 'features'),
    ('Qwen/Qwen3-32B', 'numeric'),
    ('Qwen/Qwen3-32B', 'orderflow'),
    ('Qwen/Qwen3-8B', 'features'),
    ('Qwen/Qwen3-8B', 'numeric'),
    ('Qwen/Qwen3-8B', 'orderflow'),
    ('THUDM/GLM-4-32B-0414', 'features'),
    ('THUDM/GLM-4-32B-0414', 'numeric'),
    ('THUDM/GLM-4-32B-0414', 'orderflow'),
]
OUT = 'tradingbench/results.md'


def score_signals(sig, positions, feats, h1):
    """与 pipeline 主流程一致的打分, 返回 (pnls数组, 方向命中率, 交易掩码)。"""
    pnls, hits, traded_mask = [], 0, np.zeros(len(positions), dtype=bool)
    for i, p in enumerate(positions):
        s = int(sig[i])
        if s == 0:
            continue
        fut = h1.iloc[p + 1:p + 1 + tb.HORIZON]
        if len(fut) < tb.HORIZON:
            continue
        pnls.append(tb.trade_pnl(feats['close'].values[p], s,
                                 feats['atr'].values[p],
                                 fut['High'].values, fut['Low'].values,
                                 fut['Close'].values))
        traded_mask[i] = True
        hits += int(np.sign(s) == np.sign(feats['fwd_ret'].values[p]))
    pnls = np.array(pnls)
    acc = hits / len(pnls) if len(pnls) else 0.0
    return pnls, acc, traded_mask


def row(name, pnls, acc, n):
    if len(pnls) == 0:
        return f"| {name} | 0 | - | - | - | - |"
    sharpe = pnls.mean() / pnls.std() if pnls.std() > 0 else 0.0
    return (f"| {name} | {len(pnls)} | {pnls.mean():+.3f} | "
            f"{(pnls > 0).mean() * 100:.1f}% | {sharpe:+.2f} | {acc * 100:.1f}% |")


def main():
    import sys
    models = MODELS + (SF_MODELS if '--with_sf' in sys.argv else [])
    h1 = tb.load_h1(tb.CSV_PATH)
    feats = tb.add_features(h1)
    valid = feats[['z_mom', 'atr', 'z_vol', 'rsi', 'fwd_ret']].notna().all(axis=1)
    positions = np.where(valid.values)[0]
    idx = np.unique(np.linspace(0, len(positions) - 1, 2000).round().astype(int))
    positions = positions[idx]
    n = len(positions)
    years = feats.index[positions].year
    z = feats['z_mom'].values[positions]
    rsi = feats['rsi'].values[positions]
    print(f"[analyze] 决策点={n}, 缓存命中={len(tb._load_cache())} 条")

    rng = np.random.default_rng(42)
    z_imb = feats['z_imb'].values[positions]
    all_sig = {
        '随机': rng.integers(0, 3, size=n) - 1,
        '买入持有': np.ones(n, dtype=int),
        'RSI反转': np.where(rsi < 30, 1, np.where(rsi > 70, -1, 0)),
        'Z均值回归': np.where(z < -tb.Z_THR, 1, np.where(z > tb.Z_THR, -1, 0)),
        'OF均值回归': np.where(np.nan_to_num(z_imb) < -tb.Z_THR, 1,
                     np.where(np.nan_to_num(z_imb) > tb.Z_THR, -1, 0)),
    }
    for model, style in models:
        # 缓存未命中的条目会走真实 API, 按模型切换供应商配置
        if model.startswith(('Qwen/', 'THUDM/', 'zai-org/')):
            tb.LLM_BASE_URL = 'https://api.siliconflow.cn/v1'
            tb.LLM_API_KEY_ENV = 'SILICONFLOW_API_KEY'
        else:
            tb.LLM_BASE_URL = 'https://api.deepseek.com'
            tb.LLM_API_KEY_ENV = 'DEEPSEEK_API_KEY'
        tb._LLM_CLIENT = None
        sig = np.array([
            tb.call_llm(tb.build_prompt(style, h1, feats, p),
                        feats.iloc[p], model=model)
            for p in positions
        ], dtype=int)
        all_sig[f'{model}({style})'] = sig

    scored = {name: score_signals(sig, positions, feats, h1)
              for name, sig in all_sig.items()}

    L = []
    L.append(f"# TradingBench 结果 (n={n}, 等间距 2015-2026, "
             f"horizon={tb.HORIZON}h, SL={tb.SL_ATR}ATR, TP={tb.TP_ATR}ATR)\n")
    L.append("PnL 以 ATR 倍数计 (量纲无关)。LLM 响应存于 llm_cache.jsonl。\n")

    # ---- 总表 ----
    L.append("\n## 总表\n")
    L.append("| 模型 | 交易数 | 均Pnl | 胜率 | Sharpe | 方向命中 |")
    L.append("|---|---|---|---|---|---|")
    for name, (pnls, acc, _) in scored.items():
        L.append(row(name, pnls, acc, n))

    # ---- 信号分布 ----
    L.append("\n## 信号分布 (多/空/观望)\n")
    L.append("| 模型 | 多 | 空 | 观望 |")
    L.append("|---|---|---|---|")
    for name, sig in all_sig.items():
        L.append(f"| {name} | {(sig == 1).mean() * 100:.1f}% | "
                 f"{(sig == -1).mean() * 100:.1f}% | {(sig == 0).mean() * 100:.1f}% |")

    # ---- 按年份拆分 (均Pnl, ATR) ----
    L.append("\n## 按年份拆分 (均Pnl, ATR)\n")
    names = list(scored)
    L.append("| 年份 | " + " | ".join(names) + " |")
    L.append("|---" * (len(names) + 1) + "|")
    for yr in sorted(set(years)):
        cells = []
        for name in names:
            pnls, _, traded_mask = scored[name]
            trade_years = years[traded_mask]          # pnls 只含交易, 先对齐年份
            sub = pnls[trade_years == yr]
            cells.append(f"{sub.mean():+.2f}" if len(sub) else "-")
        L.append(f"| {yr} | " + " | ".join(cells) + " |")

    # ---- 失败模式: LLM 信号 vs 动量 z-score 列联表 ----
    L.append("\n## 失败模式: 信号 × 动量 z-score\n")
    L.append("z_mom 分桶后各信号占比, 用于判断 LLM 是趋势外推还是均值回归:\n")
    zbin = pd.cut(z, [-np.inf, -2, -1, 1, 2, np.inf],
                  labels=['z<-2', '-2~-1', '-1~1', '1~2', 'z>2'])
    for name in [f"{m}({s})" for m, s in models]:
        sig = all_sig[name]
        ct = pd.crosstab(pd.Series(zbin, name='z_mom'),
                         pd.Series(sig, name='sig').map({1: '多', 0: '观望', -1: '空'}),
                         normalize='index')
        L.append(f"\n**{name}** (每行合计 100%)\n")
        L.append("| z_mom 桶 | 样本 | 多 | 观望 | 空 |")
        L.append("|---|---|---|---|---|")
        for b in ct.index:
            cnt = int((zbin == b).sum())
            L.append(f"| {b} | {cnt} | {ct.loc[b].get('多', 0) * 100:.0f}% | "
                     f"{ct.loc[b].get('观望', 0) * 100:.0f}% | "
                     f"{ct.loc[b].get('空', 0) * 100:.0f}% |")

    text = "\n".join(L) + "\n"
    with open(OUT, 'w') as f:
        f.write(text)
    print(f"[analyze] 已写入 {OUT}")
    # 控制台同时打印总表
    print("\n".join(L[:len(models) + 12]))


if __name__ == '__main__':
    main()
