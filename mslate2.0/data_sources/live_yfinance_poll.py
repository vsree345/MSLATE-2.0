"""
live_yfinance_poll.py - near-live NIFTY 50 / India VIX 1-min bars by polling Yahoo
Finance (yfinance).  No key, no login.  Personal use only (Yahoo ToS).

Each poll fetches today's 1-min bars for ^NSEI and ^INDIAVIX and prints:
  last bar time (IST), last close, wall-clock IST, and
  lag_s = wall-clock - (last bar start + 60 s)   i.e. how late the latest
          *completed* minute arrived.  Also Yahoo's regularMarketTime lag.
Rows are appended to data/external/yfinance_live/poll_log.csv.

Usage : python live_yfinance_poll.py [POLL_SECONDS=30] [N_POLLS=10]
Limits: no futures, no options OI, ^NSEI volume = 0.  Poll politely (>= 15-30 s).
"""
import csv
import datetime as dt
import os
import sys
import time
import zoneinfo

import yfinance as yf

IST = zoneinfo.ZoneInfo("Asia/Kolkata")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "data", "external", "yfinance_live")


def poll_once(ticker):
    t = yf.Ticker(ticker)
    bars = t.history(period="1d", interval="1m")
    now = dt.datetime.now(IST)
    if bars.empty:
        return [now.isoformat(timespec="seconds"), ticker, "", "", "", ""]
    last_ts = bars.index[-1].tz_convert(IST)
    lag = (now - (last_ts + dt.timedelta(seconds=60))).total_seconds()
    try:
        rmt = t.fast_info.get("lastPrice"), t.info.get("regularMarketTime")
        rmt_lag = now.timestamp() - rmt[1] if rmt[1] else ""
    except Exception:
        rmt_lag = ""
    return [now.isoformat(timespec="seconds"), ticker,
            last_ts.isoformat(timespec="minutes"),
            round(float(bars["Close"].iloc[-1]), 2), round(lag, 1),
            round(rmt_lag, 1) if rmt_lag != "" else ""]


def main():
    every = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "poll_log.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["wall_ist", "ticker", "last_bar_ist", "last_close",
                        "bar_lag_s", "regular_market_time_lag_s"])
        for i in range(n):
            for tk in ("^NSEI", "^INDIAVIX"):
                row = poll_once(tk)
                w.writerow(row)
                print(row)
            fh.flush()
            if i < n - 1:
                time.sleep(every)


if __name__ == "__main__":
    main()
