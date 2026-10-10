#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TradingBench — LLM 金融时序推理评测基准 (端到端 pipeline)

核心思想:
  给模型"截至时刻 T"的市场数据, 让它判断未来 HORIZON 根 H1 的方向(多/空/观望),
  用真实事后行情 + ATR 止损/止盈回测来打分 (PnL/胜率/Sharpe/回撤), 而非选 A/B/C。

链路:
  M1 CSV → 聚合成 H1 → 特征(无未来函数) → 采样决策点 → 构造 prompt
        → 模型决策(当前为 mock) → 走 ATR 止损止盈的简易回测 → 汇总指标 → 与基线对比

用法 (从项目根目录运行):
  python3 tradingbench/tradingbench_pipeline.py --max_samples 2000 --style features --seed 42
  python3 tradingbench/tradingbench_pipeline.py --model deepseek-chat --max_samples 100   # 真实 LLM

注意:
  - 默认 --model mock 为占位规则, 跑通链路用; 真实模型走 OpenAI 兼容接口
    (默认 DeepSeek, 需环境变量 DEEPSEEK_API_KEY), 响应缓存在 tradingbench/llm_cache.jsonl。
  - 特征严格只用 ≤T 的已完成 H1 bar, 无 lookahead; fwd_ret 只用于打分, 绝不喂给模型。
"""
import argparse
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

CSV_PATH = "data/XAUUSD_M1_2015_2026.csv"          # 从项目根目录运行
CACHE_PATH = "tradingbench/llm_cache.jsonl"   # prompt 哈希 → 决策, 避免重复付费

# ---- 特征参数 (对齐 V3 EA 的均值回归参数) ----
Z_WIN     = 336     # z-score 滚动窗口 (根 H1)
MOM_BARS  = 3       # 动量回看
ATR_PERIOD= 14      # ATR 周期 (Wilder 平滑)
VOL_WIN   = 48      # 量能求和窗口
RSI_PERIOD= 14
HORIZON   = 24      # 预测未来 24 根 H1 (≈24小时)
SL_ATR    = 1.5     # 止损 = 1.5×ATR
TP_ATR    = 2.0     # 止盈 = 2.0×ATR
Z_THR     = 2.0     # 均值回归基线的 z 阈值


# =====================================================================
# 1. 数据: M1 CSV → H1
# =====================================================================
def load_h1(csv_path, max_rows=None):
    df = pd.read_csv(
        csv_path,
        nrows=max_rows,
        dtype={'Open': 'float32', 'High': 'float32', 'Low': 'float32',
               'Close': 'float32', 'TickVol': 'float32', 'Vol': 'float32',
               'Spread': 'float32'},
    )
    dt = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df.set_index(dt).sort_index()

    # ---- 订单流代理特征 (M1 级, 聚到 H1) ----
    # 无真实盘口/主动买卖标记, 用两个经典代理:
    #   1) 签名量: sign(收-开)×TickVol —— 主动买/卖压力的代理
    #   2) 收盘位置: (收-低)/(高-低), TickVol 加权 —— 多空谁收下影线
    df['_stv'] = np.sign(df['Close'] - df['Open']) * df['TickVol']
    rng1 = (df['High'] - df['Low']).replace(0, np.nan)
    df['_posw'] = (df['Close'] - df['Low']) / rng1 * df['TickVol']

    h1 = df.resample('1h').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last',
        'TickVol': 'sum', 'Vol': 'sum', 'Spread': 'mean',
        '_stv': 'sum', '_posw': 'sum',
    }).dropna(subset=['Close'])
    tv = h1['TickVol'].replace(0, np.nan)
    h1['OF_imb'] = h1['_stv'] / tv          # 订单流不平衡 ∈ [-1,1]
    h1['OF_pos'] = h1['_posw'] / tv         # 收盘位置 ∈ [0,1]
    h1 = h1.drop(columns=['_stv', '_posw'])
    print(f"[data] M1 行数={len(df):,}  →  H1 根数={len(h1):,}  "
          f"({h1.index[0]} ~ {h1.index[-1]})")
    return h1


# =====================================================================
# 2. 特征 (无未来函数)
# =====================================================================
def add_features(h1, horizon=HORIZON):
    c = h1['Close'].astype('float64')
    h = h1['High'].astype('float64')
    l = h1['Low'].astype('float64')

    # 3-bar 动量 + 滚动 z-score (总体标准差 ddof=0, 对齐 V3)
    r3 = c / c.shift(MOM_BARS) - 1.0
    z_mom = (r3 - r3.rolling(Z_WIN).mean()) / r3.rolling(Z_WIN).std(ddof=0)

    # ATR (Wilder: alpha=1/period, 等价于 MT5 的平滑)
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()],
                   axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / ATR_PERIOD, adjust=False).mean()

    # 量能 z-score: 近 VOL_WIN 根 TickVol 之和 的 z-score
    vol_sum = h1['TickVol'].astype('float64').rolling(VOL_WIN).sum()
    z_vol = (vol_sum - vol_sum.rolling(Z_WIN).mean()) / vol_sum.rolling(Z_WIN).std(ddof=0)

    # RSI (Wilder)
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1.0 / RSI_PERIOD, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1.0 / RSI_PERIOD, adjust=False).mean()
    rsi = 100 - 100 / (1 + gain / loss)

    # 订单流代理的 z-score (与 z_mom 同窗口)
    z_imb = (h1['OF_imb'] - h1['OF_imb'].rolling(Z_WIN).mean()) / \
            h1['OF_imb'].rolling(Z_WIN).std(ddof=0)
    z_spr = (h1['Spread'] - h1['Spread'].rolling(Z_WIN).mean()) / \
            h1['Spread'].rolling(Z_WIN).std(ddof=0)

    # 目标: 未来 horizon 根收益 (仅打分用)
    fwd_ret = c.shift(-horizon) / c - 1.0

    return pd.DataFrame({
        'close': c, 'high': h, 'low': l,
        'r3': r3, 'z_mom': z_mom, 'atr': atr, 'z_vol': z_vol,
        'rsi': rsi, 'fwd_ret': fwd_ret,
        'of_imb': h1['OF_imb'], 'of_pos': h1['OF_pos'],
        'z_imb': z_imb, 'z_spr': z_spr,
    }, index=h1.index)


# =====================================================================
# 3. prompt 构造 (features/numeric/orderflow × zh/en)
# =====================================================================
def build_prompt(style, h1, feats, pos, horizon=HORIZON, lang='zh'):
    """pos = 当前已完成 H1 bar 的整数位置。只用 ≤pos 的数据。
    prompt 文本含 horizon 与 lang, 故缓存 key 天然按配置区分。"""
    if style in ('features', 'orderflow'):
        # 最近几根 H1 的收盘 + 特征键值对
        recent = feats['close'].iloc[max(0, pos - 5):pos + 1]
        if lang == 'en':
            lines = [f"You are a gold (XAUUSD) trader. Below is market data up to "
                     f"the current time. Predict the price direction over the next "
                     f"{horizon} hours. Answer with exactly one word: LONG / SHORT / FLAT.\n"]
            lines.append("Last 6 H1 closes: " +
                         ", ".join(f"{x:.1f}" for x in recent))
            lines.append(f"3-bar momentum (return): {feats['r3'].iloc[pos]:+.4f}")
            lines.append(f"Momentum z-score:        {feats['z_mom'].iloc[pos]:+.2f}")
            lines.append(f"Volume z-score:          {feats['z_vol'].iloc[pos]:+.2f}")
            lines.append(f"ATR(14):                 {feats['atr'].iloc[pos]:.2f} USD")
            lines.append(f"RSI(14):                 {feats['rsi'].iloc[pos]:.1f}")
            if style == 'orderflow':
                lines.append(f"Order-flow imbalance (-1~1): {feats['of_imb'].iloc[pos]:+.3f}"
                             "  (positive = aggressive buying)")
                lines.append(f"Imbalance z-score:           {feats['z_imb'].iloc[pos]:+.2f}")
                lines.append(f"Close position in bar (0~1): {feats['of_pos'].iloc[pos]:.2f}"
                             "  (1 = closed at high)")
                lines.append(f"Spread z-score:              {feats['z_spr'].iloc[pos]:+.2f}")
            return "\n".join(lines)
        lines = ["你是一名黄金(XAUUSD)交易员。以下是截至当前时刻的市场数据，"
                 f"请判断未来 {horizon} 小时价格方向，只回答: LONG / SHORT / FLAT。\n"]
        lines.append("最近 6 根 H1 收盘价: " +
                     ", ".join(f"{x:.1f}" for x in recent))
        lines.append(f"3bar动量(收益率): {feats['r3'].iloc[pos]:+.4f}")
        lines.append(f"动量 z-score:     {feats['z_mom'].iloc[pos]:+.2f}")
        lines.append(f"量能 z-score:     {feats['z_vol'].iloc[pos]:+.2f}")
        lines.append(f"ATR(14):          {feats['atr'].iloc[pos]:.2f} 美元")
        lines.append(f"RSI(14):          {feats['rsi'].iloc[pos]:.1f}")
        if style == 'orderflow':
            # 订单流代理: 签名量不平衡 / 收盘位置 / 点差 (M1 聚合到 H1)
            lines.append(f"订单流不平衡(-1~1): {feats['of_imb'].iloc[pos]:+.3f}"
                         "  (正=主动买占优)")
            lines.append(f"不平衡 z-score:     {feats['z_imb'].iloc[pos]:+.2f}")
            lines.append(f"K线收盘位置(0~1):   {feats['of_pos'].iloc[pos]:.2f}"
                         "  (1=收在最高)")
            lines.append(f"点差 z-score:       {feats['z_spr'].iloc[pos]:+.2f}")
        return "\n".join(lines)

    # style == 'numeric': 原始 OHLCV 表格
    recent = h1.iloc[max(0, pos - 23):pos + 1]
    if lang == 'en':
        lines = [f"You are a gold (XAUUSD) trader. Below are the last 24 H1 "
                 f"candlesticks (OHLCV). Predict the direction over the next "
                 f"{horizon} hours. Answer with exactly one word: LONG / SHORT / FLAT.\n"]
    else:
        lines = ["你是黄金(XAUUSD)交易员。以下是最近 24 根 H1 K 线(OHLCV)，"
                 f"请判断未来 {horizon} 小时方向，只回答: LONG / SHORT / FLAT。\n"]
    lines.append("Time,Open,High,Low,Close,TickVol")
    for t, r in recent.iterrows():
        lines.append(f"{t.strftime('%m-%d %H:%M')},{r['Open']:.1f},"
                     f"{r['High']:.1f},{r['Low']:.1f},{r['Close']:.1f},"
                     f"{r['TickVol']:.0f}")
    return "\n".join(lines)


# =====================================================================
# 4. 模型决策接口
# =====================================================================
_LLM_CLIENT = None
_LLM_CACHE = None
_LLM_FAILS = 0
_CACHE_LOCK = threading.Lock()
# 供应商配置 (run() 里由 CLI 参数覆盖, 默认 DeepSeek)
LLM_BASE_URL = 'https://api.deepseek.com'
LLM_API_KEY_ENV = 'DEEPSEEK_API_KEY'


def _load_cache():
    global _LLM_CACHE
    if _LLM_CACHE is None:
        _LLM_CACHE = {}
        if os.path.exists(CACHE_PATH):
            with open(CACHE_PATH) as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        _LLM_CACHE[rec['h']] = rec['sig']
                    except (json.JSONDecodeError, KeyError):
                        pass
    return _LLM_CACHE


def _cache_put(h, sig):
    with _CACHE_LOCK:
        _load_cache()[h] = sig
        with open(CACHE_PATH, 'a') as f:
            f.write(json.dumps({'h': h, 'sig': sig}) + '\n')


def parse_signal(text):
    """LONG/SHORT/FLAT → +1/-1/0。先出现的关键词优先, 解析失败按 FLAT 计。"""
    t = text.upper()
    pos = {k: t.find(k) for k in ('LONG', 'SHORT', 'FLAT') if t.find(k) >= 0}
    if not pos:
        return 0
    return {'LONG': 1, 'SHORT': -1, 'FLAT': 0}[min(pos, key=pos.get)]


def call_llm(prompt, feats_row, model='mock'):
    """返回: +1(多) / -1(空) / 0(观望)。model='mock' 时为占位规则。"""
    global _LLM_CLIENT
    if model == 'mock':
        # MOCK: 跟随动量方向(趋势), 带死区 —— 仅用于跑通链路, 非真实 LLM。
        z = feats_row['z_mom']
        if z > 0.5:
            return 1
        if z < -0.5:
            return -1
        return 0

    # ---- 真实 LLM (OpenAI 兼容接口, 默认 DeepSeek) ----
    h = hashlib.md5((model + '\n' + prompt).encode()).hexdigest()
    if h in _load_cache():
        return _LLM_CACHE[h]

    if _LLM_CLIENT is None:
        from openai import OpenAI
        _LLM_CLIENT = OpenAI(
            api_key=os.environ[LLM_API_KEY_ENV],
            base_url=LLM_BASE_URL,
            timeout=180,  # 防止供应商端点挂死时无限阻塞
        )

    last_err = None
    # reasoner 类模型: 不支持 temperature, 且 CoT 会占用 max_tokens
    kwargs = {'max_tokens': 1024} if 'reasoner' in model else \
             {'temperature': 0.0, 'max_tokens': 16}
    # Qwen3 混合思考模型 (SiliconFlow): 关掉思考, 否则 max_tokens 被 CoT 吃光
    if 'Qwen3' in model:
        kwargs['max_tokens'] = 64
        kwargs['extra_body'] = {'enable_thinking': False}
    for attempt in range(6):
        try:
            r = _LLM_CLIENT.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
            sig = parse_signal(r.choices[0].message.content or '')
            _cache_put(h, sig)
            return sig
        except Exception as e:
            msg = str(e)
            # 401/402/403 等不可重试错误 (欠费/鉴权失败): 立即终止,
            # 否则 2000 个样本会全部静默记为 FLAT, 产出看似正常的假结果
            if any(c in msg for c in ('401', '402', '403')) and '429' not in msg:
                raise SystemExit(f"[llm] 不可重试错误, 终止运行: {msg}")
            last_err = e
            # 429 (TPM 限流) 按分钟计窗口, 用长退避; 其他错误短退避
            time.sleep(min(60, 5 * 2 ** attempt) if '429' in msg
                       else 2 ** attempt)
    global _LLM_FAILS
    _LLM_FAILS += 1
    print(f"  [llm] 调用失败(第{_LLM_FAILS}次), 按 FLAT 计: {last_err}")
    return 0


# =====================================================================
# 5. 单笔 ATR 止损/止盈 回测 (walk-forward, 无 lookahead)
# =====================================================================
def trade_pnl(entry, direction, atr, fut_high, fut_low, fut_close):
    """direction: +1 多 / -1 空。收益以 ATR 倍数计(量纲无关)。"""
    sl = entry - direction * SL_ATR * atr
    tp = entry + direction * TP_ATR * atr
    for i in range(len(fut_high)):
        h, l = fut_high[i], fut_low[i]
        if direction == 1:
            if l <= sl:      # 先看止损(保守: 同根先触发止损)
                return -SL_ATR
            if h >= tp:
                return +TP_ATR
        else:
            if h >= sl:
                return -SL_ATR
            if l <= tp:
                return +TP_ATR
    # 未触发, 时间到按最后收盘离场
    return direction * (fut_close[-1] - entry) / atr


# =====================================================================
# 6. 汇总指标
# =====================================================================
def summarize(name, pnls, n_total, acc):
    n = len(pnls)
    if n == 0:
        return f"{name:<14} 无交易"
    wins = (np.array(pnls) > 0).sum()
    tot = sum(pnls)
    gw = sum(p for p in pnls if p > 0)
    gl = -sum(p for p in pnls if p < 0)
    pf = gw / gl if gl > 0 else float('inf')
    cum = np.cumsum(pnls)
    mdd = max(cum.max() - cum.min(), 0.0) if len(cum) else 0.0
    sharpe = np.mean(pnls) / np.std(pnls) if np.std(pnls) > 0 else 0.0
    return (f"{name:<14} 交易{n:>5} 胜率{wins/n*100:>4.1f}%  "
            f"均Pnl{np.mean(pnls):+.3f}ATR 总Pnl{tot:+.1f}ATR  "
            f"PF{pf:>4.2f} 回撤{mdd:.1f}ATR Sharpe{sharpe:+.2f}  "
            f"方向命中{acc*100:>4.1f}% ({n_total}决策)")


# =====================================================================
# 7. 主流程
# =====================================================================
def run(csv_path, max_samples, style, seed, model,
        base_url='https://api.deepseek.com', api_key_env='DEEPSEEK_API_KEY',
        workers=8, horizon=HORIZON, lang='zh'):
    global LLM_BASE_URL, LLM_API_KEY_ENV, _LLM_FAILS
    LLM_BASE_URL, LLM_API_KEY_ENV = base_url, api_key_env
    _LLM_FAILS = 0
    rng = np.random.default_rng(seed)
    h1 = load_h1(csv_path)
    feats = add_features(h1, horizon=horizon)

    # 采样: 特征齐全 + 未来收益已知 的已完成 bar, 等间距抽取以覆盖各年份/行情
    valid = feats[['z_mom', 'atr', 'z_vol', 'rsi', 'fwd_ret']].notna().all(axis=1)
    positions = np.where(valid.values)[0]
    if max_samples and len(positions) > max_samples:
        idx = np.unique(np.linspace(0, len(positions) - 1, max_samples).round().astype(int))
        positions = positions[idx]
    n = len(positions)
    ts = feats.index[positions]
    print(f"[sample] 决策点={n:,} (等间距, {ts[0]:%Y-%m-%d} ~ {ts[-1]:%Y-%m-%d}, "
          f"horizon={horizon}h, lang={lang}, SL={SL_ATR}ATR, TP={TP_ATR}ATR)")

    # ---- 各模型信号 (与 positions 对齐的数组) ----
    z = feats['z_mom'].values[positions]
    rsi = feats['rsi'].values[positions]
    z_imb = feats['z_imb'].values[positions]

    models = {
        "随机":     rng.integers(0, 3, size=n) - 1,                  # -1/0/1
        "买入持有": np.ones(n, dtype=int),
        "RSI反转":  np.where(rsi < 30, 1, np.where(rsi > 70, -1, 0)),
        "Z均值回归": np.where(z < -Z_THR, 1, np.where(z > Z_THR, -1, 0)),
        # 订单流均值回归: 不平衡极端 → 反向 (检验 OF 代理本身是否含信息)
        "OF均值回归": np.where(np.nan_to_num(z_imb) < -Z_THR, 1,
                     np.where(np.nan_to_num(z_imb) > Z_THR, -1, 0)),
    }

    # ---- LLM: 逐个决策点构造 prompt + 调用 (mock 串行, 真实模型并发) ----
    def decide(p):
        prompt = build_prompt(style, h1, feats, p, horizon=horizon, lang=lang)
        return call_llm(prompt, feats.iloc[p], model=model)

    if model == 'mock':
        llm_sig = np.array([decide(p) for p in positions], dtype=int)
    else:
        llm_sig = np.zeros(n, dtype=int)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for i, sig in enumerate(ex.map(decide, positions)):
                llm_sig[i] = sig
                if (i + 1) % 200 == 0:
                    print(f"  [llm] {i + 1}/{n} 已完成")
        if _LLM_FAILS:
            print(f"  [llm] ⚠️ 共 {_LLM_FAILS} 次调用失败被记为 FLAT "
                  f"(重跑可免费补全这些样本)")
    models[f"LLM:{model}"] = llm_sig

    # ---- 逐决策点打分 (真实行情 + ATR 回测) ----
    print(f"[eval] 计算 {n} 个决策点的真实收益...")
    results = {}
    for name, sig in models.items():
        pnls = []
        hits = 0
        traded = 0
        for i, p in enumerate(positions):
            s = int(sig[i])
            if s == 0:
                continue
            entry = feats['close'].values[p]
            atr = feats['atr'].values[p]
            fwd = feats['fwd_ret'].values[p]
            fut = h1.iloc[p + 1:p + 1 + horizon]
            if len(fut) < horizon:
                continue
            pnls.append(trade_pnl(entry, s, atr,
                                  fut['High'].values, fut['Low'].values,
                                  fut['Close'].values))
            traded += 1
            if np.sign(s) == np.sign(fwd):
                hits += 1
        acc = hits / traded if traded else 0.0
        results[name] = (pnls, n, acc)

    print("\n" + "=" * 90)
    print(f"TradingBench 结果 (n={n}, style={style}) — PnL 单位为 ATR 倍数")
    print("=" * 90)
    for name, (pnls, nn, acc) in results.items():
        print(summarize(name, pnls, nn, acc))
    print("-" * 90)
    print("信号分布 (多/空/观望):")
    for name, sig in models.items():
        n_long = int((sig == 1).sum())
        n_short = int((sig == -1).sum())
        n_flat = int((sig == 0).sum())
        print(f"  {name:<14} 多{n_long:>5} ({n_long/n*100:>4.1f}%)  "
              f"空{n_short:>5} ({n_short/n*100:>4.1f}%)  "
              f"观望{n_flat:>5} ({n_flat/n*100:>4.1f}%)")
    print("=" * 90)
    if model == 'mock':
        print("注意: mock 是占位(趋势跟随规则), 非真实 LLM。")
        print("      接真实模型: --model deepseek-chat (需 DEEPSEEK_API_KEY)。")
    return results


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--max_samples', type=int, default=2000)
    ap.add_argument('--style', choices=['features', 'numeric', 'orderflow'],
                    default='features')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--model', default='mock',
                    help="mock | deepseek-chat | deepseek-reasoner 等 OpenAI 兼容模型名")
    ap.add_argument('--base_url', default='https://api.deepseek.com',
                    help="OpenAI 兼容 endpoint, 如 DashScope: "
                         "https://dashscope.aliyuncs.com/compatible-mode/v1")
    ap.add_argument('--api_key_env', default='DEEPSEEK_API_KEY',
                    help="存放 API key 的环境变量名")
    ap.add_argument('--workers', type=int, default=8,
                    help="并发请求数; numeric 等长 prompt 建议降到 2-3 防 TPM 限流")
    ap.add_argument('--horizon', type=int, default=HORIZON,
                    help="预测/打分窗口 (H1 根数), 默认 24")
    ap.add_argument('--lang', choices=['zh', 'en'], default='zh',
                    help="prompt 语言 (决策关键词恒为 LONG/SHORT/FLAT)")
    a = ap.parse_args()
    run(CSV_PATH, a.max_samples, a.style, a.seed, a.model,
        a.base_url, a.api_key_env, a.workers, a.horizon, a.lang)
