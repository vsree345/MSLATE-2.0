"""
fetch_yfinance.py - NIFTY 50 (^NSEI) and India VIX (^INDIAVIX) from Yahoo Finance.

Source  : Yahoo Finance via the open-source `yfinance` package
          (https://github.com/ranaroussi/yfinance, Apache-2.0).  No key, no login.
Terms   : Yahoo data is for personal use only (Yahoo ToS); do not redistribute.

Fetches -> data/external/yfinance/
  daily_NSEI.csv, daily_INDIAVIX.csv   daily OHLC, last 3 years   (need 4: PDH/PDL/PDC, VIX)
  1m_NSEI.csv,    1m_INDIAVIX.csv      1-min OHLC, last ~29 days   (need 1: spot)
  (merged with any rows already on disk, so running it weekly builds up history)

Limitations
  - Yahoo keeps 1-minute bars only ~30 days back and returns max 8 days per request.
  - ^NSEI volume is 0 / unreliable; NO futures (NIFTY futures are not on Yahoo) and
    NO options OI.
  - Timestamps converted to naive IST.  Occasional missing minutes; last daily bar
    during market hours is a partial (live) bar.
"""
import datetime as dt
import os

import pandas as pd
import yfinance as yf

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "data", "external", "yfinance")
TICKERS = {"^NSEI": "NSEI", "^INDIAVIX": "INDIAVIX"}


def clean(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    idx = df.index
    if idx.tz is not None:
        idx = idx.tz_convert("Asia/Kolkata").tz_localize(None)
    df.index = idx
    df.index.name = "timestamp"
    return df.round(2)


def save_merged(df, path):
    if os.path.exists(path):
        old = pd.read_csv(path, index_col="timestamp", parse_dates=True)
        df = pd.concat([old, df])
        df = df[~df.index.duplicated(keep="last")]
    df.sort_index().to_csv(path)
    print(path, len(df), "rows", df.index.min(), "->", df.index.max())


def main():
    os.makedirs(OUT, exist_ok=True)
    for ticker, name in TICKERS.items():
        d = yf.download(ticker, period="3y", interval="1d",
                        auto_adjust=False, progress=False)
        save_merged(clean(d), os.path.join(OUT, "daily_" + name + ".csv"))

        parts = []
        end = dt.datetime.now(dt.timezone.utc)
        for _ in range(4):                       # 4 x 7 days = 28 days of 1m
            start = end - dt.timedelta(days=7)
            p = yf.download(ticker, start=start, end=end, interval="1m",
                            auto_adjust=False, progress=False)
            if len(p):
                parts.append(clean(p))
            end = start
        if parts:
            save_merged(pd.concat(parts), os.path.join(OUT, "1m_" + name + ".csv"))


if __name__ == "__main__":
    main()
