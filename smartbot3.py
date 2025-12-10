# smc_bot.py
import numpy as np
import pandas as pd
import csv
from dataclasses import dataclass
from typing import Tuple, Optional

# ------------------ Load Data ------------------
# Adjust path to your GBPJPY 15M CSV
df = pd.read_csv('/Users/admin/Downloads/GBPJPY_15m_2025-09.csv', header=None)
df.columns = ['pair', 'timestamp', 'open', 'close']
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Simple approximation of high/low if only open/close exist
df['high'] = df[['open', 'close']].max(axis=1)
df['low'] = df[['open', 'close']].min(axis=1)

print(df.head())
print(df.info())

# ------------------ Data Classes ------------------


@dataclass
class Signal:
    side: str            # 'buy' or 'sell'
    entry: float
    stop: float
    take_profit: float
    reason: str

# ------------------ Swing Detection ------------------


def detect_swings(df: pd.DataFrame, lookback=3) -> pd.DataFrame:
    """
    Detect swing highs and lows in the DataFrame.
    Adds 'is_swing_high' and 'is_swing_low' boolean columns.
    """
    df = df.copy()
    df['is_swing_high'] = False
    df['is_swing_low'] = False
    for i in range(lookback, len(df) - lookback):
        high = df['high'].iloc[i]
        low = df['low'].iloc[i]
        if all(high > df['high'].iloc[i - j] for j in range(1, lookback + 1)) and \
           all(high > df['high'].iloc[i + j] for j in range(1, lookback + 1)):
            df.at[df.index[i], 'is_swing_high'] = True
        if all(low < df['low'].iloc[i - j] for j in range(1, lookback + 1)) and \
           all(low < df['low'].iloc[i + j] for j in range(1, lookback + 1)):
            df.at[df.index[i], 'is_swing_low'] = True
    return df
# ---------- Detect Fair Value Gaps (FVG) ----------


def detect_fvg(df):
    df['fvg_type'] = None
    df['fvg_high'] = np.nan
    df['fvg_low'] = np.nan

    for i in range(len(df) - 2):
        # Bullish FVG: candle3.low > candle1.high
        if df['low'].iloc[i + 2] > df['high'].iloc[i]:
            df.loc[i + 1, 'fvg_type'] = 'bullish'
            df.loc[i + 1, 'fvg_low'] = df['high'].iloc[i]
            df.loc[i + 1, 'fvg_high'] = df['low'].iloc[i + 2]

        # Bearish FVG: candle3.high < candle1.low
        elif df['high'].iloc[i + 2] < df['low'].iloc[i]:
            df.loc[i + 1, 'fvg_type'] = 'bearish'
            df.loc[i + 1, 'fvg_high'] = df['low'].iloc[i]
            df.loc[i + 1, 'fvg_low'] = df['high'].iloc[i + 2]

    return df

# ------------------ BOS Detection ------------------


def detect_bos(df: pd.DataFrame) -> Optional[Tuple[str, int, float]]:
    """
    Returns (direction, index, price) if a Break of Structure occurred at the last bar.
    direction: 'bull' or 'bear'
    """
    swings = df[df['is_swing_high'] | df['is_swing_low']].copy()
    if len(swings) < 3:
        return None
    last = swings.iloc[-3:]
    # Bullish BOS
    lows = last[last['is_swing_low']]
    if len(lows) >= 2 and lows.iloc[-1]['low'] > lows.iloc[-2]['low']:
        return ('bull', df.index[-1], df['close'].iloc[-1])
    # Bearish BOS
    highs = last[last['is_swing_high']]
    if len(highs) >= 2 and highs.iloc[-1]['high'] < highs.iloc[-2]['high']:
        return ('bear', df.index[-1], df['close'].iloc[-1])
    return None
# ---------- Detect Liquidity Zones ----------


def detect_liquidity(df, window=5, threshold=0.0005):
    df['liquidity'] = None

    for i in range(window, len(df)):
        recent_highs = df['high'].iloc[i - window:i]
        recent_lows = df['low'].iloc[i - window:i]

        # Range liquidity
        if (recent_highs.max() - recent_lows.min()) / df['close'].iloc[i] < threshold:
            df.loc[i, 'liquidity'] = 'range'
        # Trendline liquidity: equal highs
        elif np.std(recent_highs) < (threshold * 2):
            df.loc[i, 'liquidity'] = 'equal_highs'
        # Trendline liquidity: equal lows
        elif np.std(recent_lows) < (threshold * 2):
            df.loc[i, 'liquidity'] = 'equal_lows'

    return df

# ------------------ Order Block Detection ------------------


def find_order_block_before_bos(df: pd.DataFrame, bos_index: int, direction: str, lookback=20) -> Optional[Tuple[float, float, int]]:
    """
    Find the most recent large candle (range > median*1.5) in opposite direction before BOS.
    Returns (ob_high, ob_low, ob_index)
    """
    start = max(0, bos_index - lookback)
    window = df.iloc[start:bos_index].copy()
    window['range'] = window['high'] - window['low']
    threshold = window['range'].median() * 1.5
    if direction == 'bull':
        candidates = window[(window['close'] < window['open'])
                            & (window['range'] > threshold)]
    else:
        candidates = window[(window['close'] > window['open'])
                            & (window['range'] > threshold)]
    if candidates.empty:
        return None
    ob = candidates.iloc[-1]
    return (ob['high'], ob['low'], ob.name)

# ------------------ Signal Generation ------------------


def generate_signal(df: pd.DataFrame) -> Optional[Signal]:
    df2 = detect_swings(df, lookback=3)
    bos = detect_bos(df2)
    if not bos:
        return None
    direction, idx, price = bos
    ob = find_order_block_before_bos(df2, idx, direction, lookback=40)
    if not ob:
        return None
    ob_high, ob_low, ob_index = ob
    if direction == 'bull':
        entry = (ob_low + ob_high)/2
        stop = ob_low - 0.0008
        tp = price + (price - stop) * 1.5
        return Signal('buy', entry, stop, tp, reason=f'BOS bull + OB at idx {ob_index}')
    else:
        entry = (ob_low + ob_high)/2
        stop = ob_high + 0.0008
        tp = price - (stop - price) * 1.5
        return Signal('sell', entry, stop, tp, reason=f'BOS bear + OB at idx {ob_index}')

# ------------------ Position Sizing ------------------


def position_size(account_balance: float, risk_pct: float, entry: float, stop: float, lot_value_per_pip=10) -> float:
    """
    Compute lot size (approx). risk_pct: fraction of account (e.g. 0.01 = 1%)
    """
    risk_amount = account_balance * risk_pct
    pip_dist = abs(entry - stop) * 100  # GBPJPY 3-digit pip approximation
    if pip_dist == 0:
        return 0.0
    lots = risk_amount / (pip_dist * lot_value_per_pip)
    return lots


# ------------------ Backtest & CSV Logging ------------------
if __name__ == "__main__":
    # Resample to 15-minute candles
    df_15m = df.resample('15min', on='timestamp').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()
    df_15m.reset_index(inplace=True)

    # CSV logging
    with open('signals_log.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'side', 'entry',
                        'stop', 'take_profit', 'reason'])

        for i in range(len(df_15m)):
            current_data = df_15m.iloc[:i+1].copy()
            sig = generate_signal(current_data)
            if sig:
                writer.writerow([
                    df_15m.iloc[i]['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
                    sig.side,
                    sig.entry,
                    sig.stop,
                    sig.take_profit,
                    sig.reason
                ])

    print("Backtest complete. Signals saved to signals_log.csv")

signals = pd.read_csv('signals_log.csv')
# Example: simple profit/loss calculation
signals['pnl'] = 0
for i, row in signals.iterrows():
    if row['side'] == 'buy':
        signals.at[i, 'pnl'] = row['take_profit'] - row['entry']
    else:
        signals.at[i, 'pnl'] = row['entry'] - row['take_profit']

print("Total PnL:", signals['pnl'].sum())
print(signals.head())

# Load signals
df = pd.read_csv('signals_log.csv')

# Assume 1 lot = 100,000 units for Forex, pip value ~10 USD for GBP/JPY
# Simplified P/L calculation


def calc_profit(row):
    if row['side'] == 'buy':
        if row['take_profit'] > row['entry']:
            return row['take_profit'] - row['entry']
        else:
            return row['stop'] - row['entry']
    else:  # sell
        if row['take_profit'] < row['entry']:
            return row['entry'] - row['take_profit']
        else:
            return row['entry'] - row['stop']


df['profit'] = df.apply(calc_profit, axis=1)

# Convert to USD assuming pip value 10 USD per 1 lot
df['profit_usd'] = df['profit'] * 10000 * 1  # 1 lot

# Check total profit
total_profit = df['profit_usd'].sum()
print("Total Profit (USD):", total_profit)

# Save updated CSV
df.to_csv('signals_with_profit.csv', index=False)
win_rate = (df['profit_usd'] > 0).mean() * 100
print(f"Win Rate: {win_rate:.2f}%")
# Save updated CSV
