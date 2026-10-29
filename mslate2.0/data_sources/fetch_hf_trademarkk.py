"""
fetch_hf_trademarkk.py - NIFTY 1-min spot + weekly option chains from Hugging Face.

Source  : https://huggingface.co/datasets/thetrademarkk/india-index-options-1m
          (identical mirror: https://huggingface.co/datasets/codepyx23/india-index-options-1m)
Licence : CC-BY-NC-4.0 (non-commercial, attribution required). "Educational use
          only. Provided as-is ... verify against official exchange data."
Login   : none - public, ungated, plain HTTPS download.

Fetches
  index/NIFTY.parquet                  NIFTY 50 spot 1-min OHLC, 2021-05-24 .. 2026-07-02
  options/NIFTY/<EXPIRY>.parquet       1-min OHLC + volume + open_interest for every
                                       strike of one weekly/monthly expiry; each file
                                       holds only the ~5-7 trading days before expiry.

Limitations
  - NO futures data at all.  Spot `volume` is always 0.
  - Option `open_interest` is only populated about every 3 minutes (other rows 0) ->
    forward-fill per contract.  Units = shares / 100 (multiply by 100 to match the
    engine's data/signal_dtls_*.csv current_oi; verified exactly on 2026-02-20).
  - Some expiries are missing (e.g. 2026-06-16, -23, -30), so a few trading days
    have no option chain.  Spot ends 2026-07-02.
  - Timestamps are tz-aware IST (+05:30); pre-open 09:07/09:08 rows exist.

Usage
  python fetch_hf_trademarkk.py [START_EXPIRY] [END_EXPIRY]
  (defaults 2025-07-01 2026-08-31; files already present are skipped)
"""
import json
import os
import sys
import urllib.request

REPO = "thetrademarkk/india-index-options-1m"
BASE = "https://huggingface.co/datasets/" + REPO + "/resolve/main/"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "data", "external", "hf_trademarkk")


def download(rel_path):
    dest = os.path.join(OUT, *rel_path.split("/"))
    if os.path.exists(dest):
        print("skip", rel_path)
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print("get ", rel_path)
    urllib.request.urlretrieve(BASE + rel_path, dest + ".part")
    os.replace(dest + ".part", dest)


def list_nifty_expiries():
    url = "https://huggingface.co/api/datasets/" + REPO
    with urllib.request.urlopen(url) as r:
        files = [s["rfilename"] for s in json.load(r)["siblings"]]
    return sorted(f for f in files if f.startswith("options/NIFTY/"))


def main():
    start = sys.argv[1] if len(sys.argv) > 1 else "2025-07-01"
    end = sys.argv[2] if len(sys.argv) > 2 else "2026-08-31"
    download("index/NIFTY.parquet")
    for f in list_nifty_expiries():
        expiry = f.split("/")[-1].replace(".parquet", "")
        if start <= expiry <= end:
            download(f)


if __name__ == "__main__":
    main()
