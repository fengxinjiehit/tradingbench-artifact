#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TradingBench 审稿修补包: 学习型基线 + 信号一致性 + equity curve

1) 岭回归基线 (numpy 闭式解, 无 sklearn 依赖):
   与 features prompt 相同的特征 (r3/z_mom/z_vol/rsi/z_imb/z_spr),
   扩展窗口滚动重训 (每 250 点), 无未来函数; 死区阈值取训练集
   |pred| 的 75 分位 (约 25% 时间交易, 与规则基线的选择性对齐)。
2) 信号一致性: 各 LLM(features) 与 RSI/Z-MR 规则的一致率。
3) equity curve: 累计 ATR PnL 曲线 → paper/figs/equity_curves.pdf

用法 (从项目根目录):
  export SILICONFLOW_API_KEY=...   # 缓存齐全时不会真正调用
  python3 tradingbench/learned_baseline.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tradingbench.tradingbench_pipeline as tb

MODELS = ['deepseek-chat', 'deepseek-reasoner',
          'Qwen/Qwen3-32B', 'Qwen/Qwen3-8B', 'THUDM/GLM-4-32B-0414']
SHORT = {'deepseek-chat': 'DS-V3', 'deepseek-reasoner': 'DS-R1',
         'Qwen/Qwen3-32B': 'Qwen3-32B', 'Qwen/Qwen3-8B': 'Qwen3-8B',
         'THUDM/GLM-4-32B-0414': 'GLM-4-32B'}
FEATS = ['r3', 'z_mom', 'z_vol', 'rsi', 'z_imb', 'z_spr']
REFIT = 250          # 每 250 个决策点重训一次
WARMUP = 250         # 前 250 点只用于训练
RIDGE = 1.0          # L2 强度


def pnl_of(sig, positions, feats, h1):
    pnls, hits = [], 0
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
        hits += int(np.sign(s) == np.sign(feats['fwd_ret'].values[p]))
    return np.array(pnls), (hits / len(pnls) if pnls else 0.0)


def ridge_signals(X, y, n):
    """扩展窗口岭回归 → 每点信号 (+1/-1/0), 前 WARMUP 点不预测。"""
    sig = np.zeros(n, dtype=int)
    coef = None
    mu = sd = None
    thr = 0.0
    for i in range(WARMUP, n):
        if (i - WARMUP) % REFIT == 0:
            Xtr, ytr = X[:i], y[:i]
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
            Xs = (Xtr - mu) / sd
            A = Xs.T @ Xs + RIDGE * len(Xtr) * np.eye(Xs.shape[1])
            coef = np.linalg.solve(A, Xs.T @ ytr)
            ptr = Xs @ coef
            thr = np.quantile(np.abs(ptr), 0.75)   # 死区: 训练集 |pred| 75 分位
        p = float(((X[i] - mu) / sd) @ coef)
        sig[i] = 1 if p > thr else (-1 if p < -thr else 0)
    return sig


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    h1 = tb.load_h1(tb.CSV_PATH)
    feats = tb.add_features(h1)
    valid = feats[['z_mom', 'atr', 'z_vol', 'rsi', 'fwd_ret',
                   'z_imb', 'z_spr']].notna().all(axis=1)
    allpos = np.where(valid.values)[0]
    idx = np.unique(np.linspace(0, len(allpos) - 1, 2000).round().astype(int))
    positions = allpos[idx]
    n = len(positions)
    print(f"[sig] 决策点={n} (注: z_imb/z_spr 纳入 valid, 点数可能略变)")

    X = feats[FEATS].values[positions]
    X = np.nan_to_num(X)
    y = feats['fwd_ret'].values[positions]

    # ---- 规则基线 + 岭回归 ----
    z = feats['z_mom'].values[positions]
    rsi = feats['rsi'].values[positions]
    sigs = {
        'RSI reversal': np.where(rsi < 30, 1, np.where(rsi > 70, -1, 0)),
        'Z-score MR': np.where(z < -tb.Z_THR, 1, np.where(z > tb.Z_THR, -1, 0)),
        'Ridge (learned)': ridge_signals(X, y, n),
    }

    # ---- LLM 信号 (走缓存) ----
    for m in MODELS:
        tb.LLM_BASE_URL, tb.LLM_API_KEY_ENV = (
            ('https://api.siliconflow.cn/v1', 'SILICONFLOW_API_KEY')
            if m.startswith(('Qwen/', 'THUDM/')) else
            ('https://api.deepseek.com', 'DEEPSEEK_API_KEY'))
        tb._LLM_CLIENT = None
        sigs[SHORT[m]] = np.array([
            tb.call_llm(tb.build_prompt('features', h1, feats, int(p)),
                        feats.iloc[int(p)], model=m)
            for p in positions], dtype=int)

    # ---- 打分: 全部 2000 点 (LLM/规则) + 评估子集 (岭回归从 250 起) ----
    print("\n== 主口径 (全部 2000 点; Ridge 仅在后 1750 点可评) ==")
    curves = {}
    for name, sig in sigs.items():
        pnls, acc = pnl_of(sig, positions, feats, h1)
        curves[name] = pnls
        mean = pnls.mean() if len(pnls) else np.nan
        print(f"  {name:<18} 交易{len(pnls):>5} 均Pnl{mean:+.3f} 方向命中{acc*100:.1f}%")

    print("\n== 公平口径 (仅点后 1750 个评估子集) ==")
    sub = np.arange(WARMUP, n)
    for name, sig in sigs.items():
        pnls, acc = pnl_of(sig[sub], positions[sub], feats, h1)
        mean = pnls.mean() if len(pnls) else np.nan
        print(f"  {name:<18} 交易{len(pnls):>5} 均Pnl{mean:+.3f} 方向命中{acc*100:.1f}%")

    # ---- 信号一致性 ----
    print("\n== 信号一致性 (features) ==")
    print(f"  {'模型':<18} vs RSI(全体) vs RSI(出手点) vs Z-MR(全体)")
    for name in [SHORT[m] for m in MODELS]:
        sig = sigs[name]
        a_all = (sig == sigs['RSI reversal']).mean()
        mask = sigs['RSI reversal'] != 0
        a_tr = (sig[mask] == sigs['RSI reversal'][mask]).mean()
        a_z = (sig == sigs['Z-score MR']).mean()
        print(f"  {name:<18} {a_all*100:>8.0f}% {a_tr*100:>13.0f}% {a_z*100:>12.0f}%")

    # ---- equity curve ----
    os.makedirs('paper/figs', exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, pnls in curves.items():
        if len(pnls) == 0:
            continue
        lw = 2.2 if name in ('RSI reversal', 'Ridge (learned)') else 1.0
        ax.plot(np.cumsum(pnls), label=name, lw=lw,
                alpha=0.95 if lw > 1 else 0.7)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('Trades (chronological)')
    ax.set_ylabel('Cumulative PnL (ATR)')
    ax.set_title('TradingBench equity curves (features format, n=2000)')
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig('paper/figs/equity_curves.pdf')
    fig.savefig('paper/figs/equity_curves.png', dpi=150)
    print("\n[fig] 已写 paper/figs/equity_curves.{pdf,png}")


if __name__ == '__main__':
    main()
