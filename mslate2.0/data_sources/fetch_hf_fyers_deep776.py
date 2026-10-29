"""
fetch_hf_fyers_deep776.py - NIFTY current-month FUTURES 1-min (+volume, OI), NIFTY 50
index 1-min and monthly-expiry option strikes 1-min (+OI), from Hugging Face.

Source  : https://huggingface.co/datasets/deep776/FYERS_NSE_NIFTY50_INDEX
          (data originally pulled from the FYERS broker API by the uploader; we only
          download the public HF copy - no FYERS account involved)
Licence : NOT STATED (no dataset card).  Public & ungated, but redistribution rights
          are unclear -> treat as personal-research only, do not republish.
Login   : none.

Layout  : NSE_NIFTY50_INDEX/option_data/parquet/<RUN>/<SYMBOL>_<TF>_<from>_to_<to>.parquet
          RUN = weekly snapshot (e.g. NSE_NIFTY50_INDEX_20261027_121510), each covering a
          rolling ~100-day window.  TF in {5S,1,5,60,D}.  SYMBOL = NIFTY50-INDEX,
          NIFTY<YY><MON>FUT, NIFTY<YY><MON><STRIKE><CE|PE> (MONTHLY expiry only).
Columns : time (unix epoch seconds, UTC), open, high, low, close, volume (per bar), oi

Fetches (TF=1 and D only):
  - FUT and INDEX files from every snapshot (small),
  - option strike files only from the LATEST snapshot per monthly contract.

Limitations
  - Repo history starts 2026-09-26; earliest bar 2026-06-21.  Rolling window: re-run
    weekly to keep accumulating, older snapshots may be deleted upstream.
  - Only the current monthly FUT per snapshot (SEP-26 and OCT-26 so far), so
    "current-month" futures are only true near-month from ~Aug-26 onward; before that
    the bars are a next/far-month contract (price ok-ish, volume/OI not near-month).
  - Options are MONTHLY expiry only (no weekly chain).

Usage   : python fetch_hf_fyers_deep776.py
"""
import json
import os
import re
import urllib.request

REPO = "deep776/FYERS_NSE_NIFTY50_INDEX"
BASE = "https://huggingface.co/datasets/" + REPO + "/resolve/main/"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "data", "external", "hf_deep776_fyers")
PREFIX = "NSE_NIFTY50_INDEX/option_data/parquet/"


def download(rel_path, dest):
    if os.path.exists(dest):
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print("get ", rel_path)
    urllib.request.urlretrieve(BASE + rel_path, dest + ".part")
    os.replace(dest + ".part", dest)


def main():
    with urllib.request.urlopen("https://huggingface.co/api/datasets/" + REPO) as r:
        files = [s["rfilename"] for s in json.load(r)["siblings"]]
    files = [f for f in files if f.startswith(PREFIX)]
    for meta in ("NSE_NIFTY50_INDEX/NSE_NIFTY50_INDEX_expiries.csv",
                 "NSE_NIFTY50_INDEX/NSE_NIFTY50_INDEX_tracker.csv"):
        download(meta, os.path.join(OUT, os.path.basename(meta)))

    latest_opt = {}   # monthly contract code (e.g. 26OCT) -> latest run name
    for f in files:
        run, name = f[len(PREFIX):].split("/")
        m = re.match(r"NIFTY(\d\d[A-Z]{3})\d+(CE|PE)_1_", name)
        if m:
            code = m.group(1)
            latest_opt[code] = max(latest_opt.get(code, ""), run)

    for f in files:
        run, name = f[len(PREFIX):].split("/")
        sym, tf = name.split("_")[0], name.split("_")[1]
        if tf not in ("1", "D"):
            continue
        if sym.startswith("NIFTY50-INDEX") or sym.endswith("FUT"):
            download(f, os.path.join(OUT, run, name))
        elif tf == "1":
            m = re.match(r"NIFTY(\d\d[A-Z]{3})\d+(CE|PE)$", sym)
            if m and latest_opt.get(m.group(1)) == run:
                download(f, os.path.join(OUT, run, "options", name))


if __name__ == "__main__":
    main()
