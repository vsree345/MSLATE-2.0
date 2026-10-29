"""
fetch_nse_bhavcopy.py - NSE F&O UDiFF bhavcopy (END-OF-DAY) -> NIFTY futures daily
OHLC + OI (need 4: futures PDH/PDL/PDC) and EOD per-strike option OI.

Source  : https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_<YYYYMMDD>_F_0000.csv.zip
          (NSE's publicly published daily report archive; no login)
Terms   : !! NSE's website terms of use prohibit automated scraping / bulk download of
          nseindia.com.  This script is therefore OFF by default for bulk use: it
          fetches only the dates you pass, sleeps 3 s between requests, and should be
          run by hand for a modest range.  Prefer the HF / Yahoo sources when they
          cover what you need.  Personal research only.

Output  : data/external/nse_bhavcopy_udiff/BhavCopy_NSE_FO_..._F_0000.csv.zip  (raw)
          data/external/nse_bhavcopy_udiff/nifty_fut_daily.csv  (NIFTY IDF rows,
          all expiries: date, contract, expiry, OHLC, settle, OI, volume in contracts)

Limitations: EOD only - no intraday.  Holidays return 404 (skipped).  UDiFF format
          exists from 2024-07-08; older dates use the legacy fo<DD><MON><YYYY>bhav.csv.zip.

Usage   : python fetch_nse_bhavcopy.py 2026-10-01 2026-10-28
"""
import datetime as dt
import os
import sys
import time
import urllib.error
import urllib.request
import zipfile

import pandas as pd

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "data", "external", "nse_bhavcopy_udiff")
URL = ("https://nsearchives.nseindia.com/content/fo/"
       "BhavCopy_NSE_FO_0_0_0_{d}_F_0000.csv.zip")


def fetch(day):
    name = "BhavCopy_NSE_FO_0_0_0_%s_F_0000.csv.zip" % day.strftime("%Y%m%d")
    dest = os.path.join(OUT, name)
    if os.path.exists(dest):
        return dest
    req = urllib.request.Request(URL.format(d=day.strftime("%Y%m%d")),
                                 headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read()
    except urllib.error.HTTPError as e:
        print(day, "HTTP", e.code, "(holiday?)")
        return None
    with open(dest, "wb") as fh:
        fh.write(data)
    print(day, "ok", len(data))
    time.sleep(3)
    return dest


def main():
    start = dt.date.fromisoformat(sys.argv[1])
    end = dt.date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else start
    os.makedirs(OUT, exist_ok=True)
    day = start
    while day <= end:
        if day.weekday() < 5:
            fetch(day)
        day += dt.timedelta(days=1)

    rows = []
    for f in sorted(os.listdir(OUT)):
        if f.endswith("_F_0000.csv.zip"):
            z = zipfile.ZipFile(os.path.join(OUT, f))
            d = pd.read_csv(z.open(z.namelist()[0]))
            rows.append(d[(d.TckrSymb == "NIFTY") & (d.FinInstrmTp == "IDF")])
    if rows:
        fut = pd.concat(rows)[["TradDt", "FinInstrmNm", "XpryDt", "OpnPric", "HghPric",
                               "LwPric", "ClsPric", "SttlmPric", "OpnIntrst",
                               "TtlTradgVol"]]
        fut.columns = ["date", "contract", "expiry", "open", "high", "low", "close",
                       "settle", "oi", "volume_contracts"]
        fut.sort_values(["date", "expiry"]).to_csv(
            os.path.join(OUT, "nifty_fut_daily.csv"), index=False)
        print("nifty_fut_daily.csv", len(fut), "rows")


if __name__ == "__main__":
    main()
