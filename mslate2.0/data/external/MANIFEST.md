# External market data: MANIFEST

Free data, no account needed, for multi-day TRAP-X backtests and a near-live feed.
Compiled 2026-10-29 (IST). Every file here was downloaded from a public URL without
login or API key. **Treat these downloads as untrusted data.** The fetch scripts live in
`../../data_sources/`. Run any Python that reads these files with `python -I`.

Engine needs, as referred to below:
1. NIFTY 50 spot 1-min OHLC, 09:15–15:30 IST
2. NIFTY current-month futures 1-min OHLC + volume + OI
3. Options flow: per-strike CE/PE 1-min OI around ATM (±7 strikes, 50-pt spacing)
4. Previous-day OHLC for NIFTY 50 and NIFTY futures, plus the India VIX close

## Coverage at a glance

| Source (dir) | Need 1 spot | Need 2 futures | Need 3 options OI | Need 4 PDC | Range obtained | Licence |
|---|---|---|---|---|---|---|
| `hf_trademarkk/` | YES (1m) | **NO** | YES, weekly chain, OI every ~3 min | spot only (aggregate 1m) | spot 2021-05-24→2026-07-02; options for expiries 2025-07-03→2026-08-04 (trading days 2025-06-25→2026-07-02) | CC-BY-NC-4.0 |
| `hf_deep776_fyers/` | YES (1m) | YES (1m, vol + OI), current monthly contract per snapshot | MONTHLY expiry only, OI every minute | NIFTY + FUT daily (D files) | 2026-06-22→2026-10-27 | **not stated** |
| `yfinance/` | YES, last ~28 days only | NO | NO | YES: NIFTY + VIX daily, 3 yrs | daily 2023-10-30→2026-10-29; 1m 2026-10-05→2026-10-29 | Yahoo ToS, personal use |
| `nse_bhavcopy_udiff/` | NO (EOD) | daily only (all NIFTY FUT expiries + OI) | EOD OI only | YES: futures PDH/PDL/PDC | 1 test file: 2026-10-28 | NSE published file; **NSE ToU bars automated scraping** |
| `yfinance_live/` | near-live 1m poll | NO | NO | n/a | test poll 2026-10-29 20:43 IST | Yahoo ToS |

---

## 1. `hf_trademarkk/`: Hugging Face `thetrademarkk/india-index-options-1m`

- **URL:** https://huggingface.co/datasets/thetrademarkk/india-index-options-1m
  (byte-identical mirror: `codepyx23/india-index-options-1m`). Public and ungated.
- **Licence:** CC-BY-NC-4.0. Non-commercial use, attribution required. The card says
  "Educational use only … verify against official exchange data."
- **Fetch script:** `data_sources/fetch_hf_trademarkk.py [START_EXPIRY END_EXPIRY]`
  (default 2025-07-01..2026-08-31)
- **Files:** `index/NIFTY.parquet` (8.4 MB) and `options/NIFTY/<expiry>.parquet` (54 files),
  about 162 MB in total.
- **Index schema:** `timestamp` (tz-aware IST), open, high, low, close, volume (always 0),
  trading_day, symbol. 486,050 rows over 1,262 days.
- **Options schema:** timestamp, open, high, low, close, volume, `open_interest`, trading_day,
  symbol, strike (int), option_type CE/PE, expiry. 15.2 M rows. Each expiry file holds
  every strike for only the ~5–7 trading days before that expiry. Use the nearest
  expiry for each trading day.
- **Coverage (QA, 2025-06-25→2026-07-02):** 246 spot days. The median day has 376 bars
  (09:15–15:30); pre-open rows from 09:01 to 09:08 are also present. On 243 of the 246 days
  a chain is available, and the ATM±7 window (30 contracts) is complete on almost all of
  them. The median contract has 375 bars and about 124 OI updates per day.
- **Validation:** the OI agrees exactly with the engine's own `data/signal_dtls_2026-02-20.csv`
  after ×100. For example, 25400CE 09:16 is 37,640 here and 3,764,085 in the engine;
  09:19 is 49,348 here and 4,934,865 in the engine. The OI update timing also matches.
  The index bars match the FYERS index bars (source 2): over 3,000 overlapping minutes the
  maximum difference is 0.45 pts.
- **Gaps and quirks:**
  - There are no futures.
  - `open_interest` is 0 between exchange OI updates, roughly every 3 minutes. Forward-fill
    it per contract and never read the 0 as real OI.
  - OI units are shares/100. Multiply by 100 to get the engine's units.
  - Some expiries are missing: 2025-10-21 and 2026-06-16/23/30. 2026-06-09 is almost
    empty. As a result:
    - 2025-10-15/16/17 have no chain.
    - 2026-06-04→06-17 only have the far 2026-07-07 expiry. On 06-04 and 06-05 fewer
      than 20 of the 30 ATM±7 contracts are present.
  - Short spot days: 2025-09-26 (321 bars), 2025-10-21 (95 bars, Muhurat), 2026-05-22 (337),
    and 2026-06-03 (6 bars, bad).
- **Sample rows:**
  ```
  timestamp,open,high,low,close,volume,trading_day,symbol
  2026-02-20 09:01:00+05:30,25306.2,25462.3,25296.95,25435.1,0,2026-02-20,NIFTY
  timestamp,open,high,low,close,volume,open_interest,trading_day,symbol,strike,option_type,expiry
  2026-02-20 09:16:00+05:30,171.8,183.85,167.0,175.65,2755155,37640,2026-02-20,NIFTY,25400,CE,2026-02-24
  2026-02-20 09:17:00+05:30,177.95,179.2,157.6,157.6,1430585,0,2026-02-20,NIFTY,25400,CE,2026-02-24
  ```

## 2. `hf_deep776_fyers/`: Hugging Face `deep776/FYERS_NSE_NIFTY50_INDEX`

- **URL:** https://huggingface.co/datasets/deep776/FYERS_NSE_NIFTY50_INDEX. Public and
  ungated. The uploader pulled the data from the FYERS broker API. We only download the
  public HF copy, so no FYERS account is involved.
- **Licence:** **none stated**, as there is no dataset card. Treat it as personal research
  only and do not redistribute.
- **Fetch script:** `data_sources/fetch_hf_fyers_deep776.py`. It fetches only the TF=1 and D
  files: FUT and INDEX from every snapshot, and option strikes from the latest snapshot
  of each monthly contract.
- **Layout:** `<RUN>/<SYMBOL>_<TF>_<from>_to_<to>.parquet`. Each RUN is a weekly snapshot of
  a rolling ~100-day window. Option strikes are under `<RUN>/options/`.
- **Schema:** `time` (int epoch), open, high, low, close, volume (per bar), oi (shares, per minute).
  **Timestamp quirk:** `time` is IST wall-clock encoded as if it were UTC. Use
  `pd.to_datetime(time, unit="s")` with no tz conversion, which gives 09:15 for the first bar.
- **Obtained:**
  - NIFTY50-INDEX 1m: 2026-06-22→2026-10-27, 375 bars/day (09:15–15:29).
  - NIFTY26SEPFUT 1m: 2026-06-22→2026-09-29.
  - NIFTY26OCTFUT 1m: 2026-06-29→2026-10-27.
  - Both futures have about 385 bars/day (09:15–15:39), with volume and OI changing almost
    every minute.
  - Monthly options (100 strikes × CE/PE): SEP-26 expiry from the 20260929 run, and OCT-26
    expiry from the 20261027 run. SEP-26 has 200 files and 2.97 M rows (2026-06-22→09-29).
    OCT-26 has 200 files and 2.66 M rows (2026-07-29→10-27). Strikes run from 21050 to
    26000 in 50-pt steps. Illiquid strikes are sparse. In total there are 422 files (420 parquet + 2 metadata CSVs, 95 MB).
- **Validation:** daily index OHLC equals the yfinance daily data exactly over 83 days. Index 1m
  bars equal the trademarkk 1m bars. Futures OI is on the same scale as the engine
  (about 14–19 M shares).
- **Gaps and quirks:**
  - The repo history begins 2026-09-26. Nothing exists before 2026-06-21, and old
    snapshots may be dropped upstream, so re-run the script weekly to accumulate data.
  - The current-month contract is only correct when it is the snapshot's own monthly. In
    Jul–Aug 2026 the available contract is SEP or OCT, a far month: price and basis are
    usable, but volume and OI are not near-month.
  - The NOV-26 future is not yet in any snapshot.
  - Options are monthly only (`NIFTY26OCT25500CE`), not the weekly chain the engine used.
  - Futures bars continue to 15:39. Drop anything after 15:30.
- **Sample rows:**
  ```
  time,open,high,low,close,volume,oi            -> ts (IST)
  1788442500,24049.5,24059.0,24045.9,24051.1,2990,16779815   2026-09-03 13:35  NIFTY26SEPFUT
  1783086600,387.0,387.0,386.0,387.0,390,260455              2026-07-03 13:50  NIFTY26SEP25000CE
  ```

## 3. `yfinance/`: Yahoo Finance via `yfinance` 1.7.0

- **URL:** Yahoo Finance chart API, accessed through https://github.com/ranaroussi/yfinance
  (Apache-2.0). Tickers `^NSEI` and `^INDIAVIX`. No key is needed.
- **Terms:** Yahoo ToS allow personal, non-commercial use only. Do not redistribute.
- **Fetch script:** `data_sources/fetch_yfinance.py`. It merges with existing rows, so running
  it weekly builds 1m history beyond Yahoo's ~30-day window.
- **Files:**
  - `daily_NSEI.csv`: 740 rows, 2023-10-30→2026-10-29.
  - `daily_INDIAVIX.csv`: 735 rows, same range.
  - `1m_NSEI.csv` and `1m_INDIAVIX.csv`: 6,738 rows each, 2026-10-05→2026-10-29.
- **Columns:** timestamp (naive IST), open, high, low, close, volume.
- **Validation:** daily NIFTY 2026-02-19 OHLC is 25873.35/25885.30/25388.75/25454.35, and
  INDIA VIX closed at 13.46 that day. Both are **identical** to `data/pdc_2026-02-20.csv`.
  The 1m bars are close to FYERS but not identical: mean |close diff| is 0.15 pts, the
  maximum is 3.95, and 44% of minutes match exactly.
- **Gaps and quirks:**
  - 1m data is kept for only about 30 days, and each request returns at most 8 days.
  - ^NSEI 1m volume is 0.
  - There are 375 bars/day (09:15–15:29) and no 15:30 bar.
  - 2026-10-15 has 363 bars.
  - 2026-10-20 is absent (holiday/Muhurat).
  - The daily bar for the current day is partial during market hours.
  - There are no futures and no options.
- **Sample rows:** `2026-10-05 09:15:00,22532.4,22552.35,22507.55,22543.85,0` and
  `2026-10-29,16.25,16.57,15.09,15.12,0` (VIX daily).

## 4. `nse_bhavcopy_udiff/`: NSE F&O UDiFF bhavcopy (EOD)

- **URL:** `https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_<YYYYMMDD>_F_0000.csv.zip`
- **Terms:** this is a file NSE publishes, and it downloaded without login. However,
  **NSE's terms of use prohibit automated scraping/bulk download from nseindia.com**, so
  only one test file was fetched. The script is rate-limited (3 s) and should be run by
  hand for small ranges. Prefer yfinance/FYERS daily data where they suffice.
- **Fetch script:** `data_sources/fetch_nse_bhavcopy.py START [END]`. It writes
  `nifty_fut_daily.csv`.
- **Obtained:** the 2026-10-28 file (1.07 MB zip, 35,033 rows, 34 UDiFF columns). Its
  NIFTY IDF rows give NOV/DEC/JAN futures OHLC, settlement price, OI and volume.
- **Satisfies:** need 4, futures PDH/PDL/PDC, for any date, including the trademarkk
  window where no futures exist. It also gives EOD per-strike option OI. It has **no
  intraday data**.
- **Sample:** `2026-10-28,NIFTY26NOVFUT,2026-11-23,22690.0,22875.5,22567.0,22778.0,22778.0,16049995,75166`

## 5. `yfinance_live/`: near-live feed test

- **Script:** `data_sources/live_yfinance_poll.py [POLL_SECONDS] [N_POLLS]`. It polls
  `history(period="1d", interval="1m")` for ^NSEI and ^INDIAVIX and appends the results to
  `poll_log.csv`, recording `bar_lag_s = wall_IST - (last_bar_start + 60 s)`.
- **Test on 2026-10-29 at 20:43 IST (market closed):** both tickers returned within about 1 s per poll, with no key. The last bar was 15:29 IST (close 22936.75, VIX 15.12), and
  regularMarketTime was about 15:31 IST. **Intraday delay could not be measured because
  the session had already closed.** To get a real figure, run
  `python -I live_yfinance_poll.py 30 20` between 09:20 and 15:25 IST and read the
  `bar_lag_s` column.
- **Limitations:** it gives spot and VIX only. It has no futures, no volume and no option OI.
  Yahoo is unofficial and unsupported, and can rate-limit or change without notice.
- **No other free, no-login source** provides live futures or option OI. The NSE
  option-chain JSON exists but is covered by the NSE scraping prohibition, so it was not used.

## Evaluated but not downloaded

| Candidate | Finding |
|---|---|
| `rissin/nse-options-intraday` (HF, licence "other") | Upstox 1m NIFTY options from Oct 2024 to 2026, but `oi` is NaN for all intraday rows, so it does not meet need 3. Its daily part is the NSE bhavcopy. 2.5 GB. |
| `manishbaliii/nifty-options-data` (HF, no licence) | 19.25 M rows of Upstox 1m options with OI (first rows dated 2024-10-03; full range not checked). Possibly useful as an older backup, 202 MB. |
| `artist-23/nifty-options-data` (HF, no licence) | ATM-relative ±10 1m options with OI and IV (first rows dated 2020-12-29; full range not checked). 1.27 GB, and absolute strikes are not fixed. |
| `kalpeshWalke/nifty-oi-history` (HF, no licence) | ATM-relative 1m snapshot of CE/PE OI, but the first rows show `Timestamp` as HH:MM only, with no date. 93 k rows. |
| `nagarhimanshu37/nifty-session-logs` (HF, MIT) | Bot telemetry; includes files named `spot_1m_candles.json` and `futures_1m_candles.json` for 18 days in Oct 2026 (contents not inspected). FYERS covers the same data. |
| `average-developer/stocks-INDIAVIX-1D-candles` | Encoded delta format. yfinance is simpler. |
| GitHub search | No repo has recent free NIFTY futures 1m CSVs. |
| Kaggle | Requires login, so it was skipped. |
| niftyindices.com historical data | EOD only, and yfinance daily duplicates it exactly. Not used. |
