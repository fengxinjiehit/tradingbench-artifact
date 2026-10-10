#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TradingBench 点差成本敏感性分析

主结果表中的 PnL 是毛收益 (未扣交易成本)。数据含 Spread 列
(XAUUSD, 2 位小数报价 → 1 point = $0.01), 本脚本对每笔交易扣除
入场时的点差成本 (spread_points × $0.01, 换算为 ATR 倍数),
比较扣成本前后各方法的每笔均 PnL, 检验结论是否稳健。

用法 (从项目根目录): python3 tradingbench/spread_sensitivity.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tradingbench.tradingbench_pipeline as tb
from tradingbench.significance import MODELS, SHORT, pnl_per_point

POINT_USD = 0.01  # XAUUSD 2 位小数报价: 1 point = $0.01


def spread_cost_per_point(positions, feats, h1):
    """每个决策点的单笔点差成本 (ATR 倍数); 不交易则为 0。"""
    spr = h1['Spread'].values[positions] * POINT_USD   # 美元
    atr = feats['atr'].values[positions]
    return spr / atr


def main():
    h1 = tb.load_h1(tb.CSV_PATH)
    feats = tb.add_features(h1)
    valid = feats[['z_mom', 'atr', 'z_vol', 'rsi', 'fwd_ret']].notna().all(axis=1)
    positions = np.where(valid.values)[0]
    idx = np.unique(np.linspace(0, len(positions) - 1, 2000).round().astype(int))
    positions = positions[idx]

    tb._load_cache()
    cost = spread_cost_per_point(positions, feats, h1)
    print(f"[spread] 单笔点差成本 (ATR): 中位数 {np.median(cost):.3f}, "
          f"均值 {cost.mean():.3f}, p90 {np.percentile(cost, 90):.3f}")

    z = feats['z_mom'].values[positions]
    rsi = feats['rsi'].values[positions]
    sigs = {
        'RSI反转': np.where(rsi < 30, 1, np.where(rsi > 70, -1, 0)),
        'Z均值回归': np.where(z < -tb.Z_THR, 1, np.where(z > tb.Z_THR, -1, 0)),
    }
    for m in MODELS:
        tb.LLM_BASE_URL, tb.LLM_API_KEY_ENV = (
            ('https://api.siliconflow.cn/v1', 'SILICONFLOW_API_KEY')
            if m.startswith(('Qwen/', 'THUDM/')) else
            ('https://api.deepseek.com', 'DEEPSEEK_API_KEY'))
        tb._LLM_CLIENT = None
        sigs[f'{SHORT[m]}(f)'] = np.array(
            [tb.call_llm(tb.build_prompt('features', h1, feats, p),
                         feats.iloc[p], model=m)
             for p in positions], dtype=int)

    print(f"\n{'方法':<14}{'交易数':>6}{'毛均Pnl':>9}{'净均Pnl':>9}{'单笔成本':>9}")
    print('-' * 50)
    for name, sig in sigs.items():
        gross = pnl_per_point(sig, positions, feats, h1)
        traded = sig != 0
        if traded.sum() == 0:
            continue
        net = gross - np.where(traded, cost, 0.0)
        print(f"{name:<14}{int(traded.sum()):>6}"
              f"{gross[traded].mean():>+9.3f}{net[traded].mean():>+9.3f}"
              f"{cost[traded].mean():>9.3f}")


if __name__ == '__main__':
    main()
