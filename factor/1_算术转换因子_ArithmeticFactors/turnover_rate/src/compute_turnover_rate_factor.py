"""Compute monthly 20-day average turnover-rate factor from Tushare daily_basic.

Factor definition:
- At each market month-end trade date, for each stock, compute the mean of
  ``turnover_rate`` over the latest 20 trading records up to that date.
- Require 20 valid observations (``min_periods=20``).

Data source:
- Tushare Pro ``daily_basic`` endpoint.

Outputs:
- output/turnover_rate/class_by_stock/{ts_code}.csv
- output/turnover_rate/complete_factor/{ts_code}.csv
Both files contain two columns:
- ``trade_date``
- ``turnover_rate_20d``

Example:
    python src/compute_turnover_rate_factor.py
    python src/compute_turnover_rate_factor.py --stock 000001.SZ --start-date 20100101
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import tushare as ts
from tqdm import tqdm


_THIS_DIR = Path(__file__).resolve().parent
FACTOR_DIR = _THIS_DIR.parent
PROJECT_ROOT = FACTOR_DIR.parent.parent.parent

BASE_CONF = PROJECT_ROOT / "conf" / "base.json"
TRADE_CALENDAR_PATH = PROJECT_ROOT / "data" / "交易日历" / "trade_calendar.csv"

OUT_CLASS_BY_STOCK = FACTOR_DIR / "output" / "class_by_stock"
OUT_COMPLETE = FACTOR_DIR / "output" / "complete_factor"

FACTOR_COL = "turnover_rate_20d"
DEFAULT_START = "20140201"
DEFAULT_END = "20211031"
SLEEP_SEC = 0.20
MAX_RETRIES = 5


def load_token() -> str:
    """Load Tushare token from conf/base.json."""
    with BASE_CONF.open("r", encoding="utf-8-sig") as f:
        conf = json.load(f)
    token = conf.get("tushare_token", "")
    if not token:
        raise ValueError(f"Missing tushare_token in {BASE_CONF}")
    return token


def fetch_with_retry(api_func, **kwargs) -> pd.DataFrame:
    """Call a Tushare API with retry and simple backoff."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            df = api_func(**kwargs)
            time.sleep(SLEEP_SEC)
            return df if df is not None else pd.DataFrame()
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            if "每分钟最多访问" in msg or "频次" in msg:
                wait_sec = 15 * attempt
            else:
                wait_sec = 3 * attempt
            print(f"[retry {attempt}/{MAX_RETRIES}] {msg}; sleep {wait_sec}s")
            time.sleep(wait_sec)
    raise RuntimeError(f"Tushare request failed after retries: {kwargs}")


def load_month_end_dates(start_date: str, end_date: str) -> set[str]:
    """Return market month-end trade dates (SSE open days) within range."""
    cal = pd.read_csv(TRADE_CALENDAR_PATH, dtype={"cal_date": str})
    cal = cal[(cal["exchange"] == "SSE") & (cal["is_open"] == 1)].copy()
    cal = cal[(cal["cal_date"] >= start_date) & (cal["cal_date"] <= end_date)]

    cal["month"] = cal["cal_date"].str[:6]
    month_end = cal.groupby("month", as_index=False)["cal_date"].max()
    return set(month_end["cal_date"].tolist())


def get_stock_list(pro, stock: str | None) -> list[str]:
    """Get target stock list from Tushare, or a single user-specified stock."""
    if stock:
        return [stock]

    dfs = []
    for status in ["L", "D", "P"]:
        df = fetch_with_retry(pro.stock_basic, exchange="", list_status=status, fields="ts_code")
        if not df.empty:
            dfs.append(df)

    if not dfs:
        raise RuntimeError("Failed to load stock list from Tushare")

    stocks = pd.concat(dfs, ignore_index=True)["ts_code"].dropna().drop_duplicates().sort_values().tolist()
    return stocks


def compute_monthly_factor(df_daily: pd.DataFrame, month_end_dates: set[str], window: int) -> pd.DataFrame:
    """Compute month-end rolling mean turnover rate for a stock."""
    if df_daily.empty:
        return pd.DataFrame(columns=["trade_date", FACTOR_COL])

    df = df_daily.copy()
    df["trade_date"] = df["trade_date"].astype(str)
    df["turnover_rate"] = pd.to_numeric(df["turnover_rate"], errors="coerce")
    df = df.sort_values("trade_date")

    df[FACTOR_COL] = df["turnover_rate"].rolling(window=window, min_periods=window).mean()
    df = df[df["trade_date"].isin(month_end_dates)]
    out = df[["trade_date", FACTOR_COL]].dropna(subset=[FACTOR_COL]).copy()
    return out


def process_one_stock(
    pro,
    ts_code: str,
    start_date: str,
    end_date: str,
    month_end_dates: set[str],
    window: int,
    overwrite: bool,
) -> str:
    """Fetch one stock's daily_basic, compute factor, and write outputs."""
    out_path = OUT_CLASS_BY_STOCK / f"{ts_code}.csv"
    if out_path.exists() and not overwrite:
        return "skipped"

    df_daily = fetch_with_retry(
        pro.daily_basic,
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date,
        fields="ts_code,trade_date,turnover_rate",
    )

    out = compute_monthly_factor(df_daily, month_end_dates=month_end_dates, window=window)

    out.to_csv(out_path, index=False, encoding="utf-8")
    out.to_csv(OUT_COMPLETE / f"{ts_code}.csv", index=False, encoding="utf-8")
    return "ok" if not out.empty else "empty"


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Compute turnover-rate monthly factor from Tushare daily_basic.")
    parser.add_argument("--stock", type=str, default=None, help="Single stock code, e.g. 000001.SZ")
    parser.add_argument("--start-date", type=str, default=DEFAULT_START, help="YYYYMMDD")
    parser.add_argument("--end-date", type=str, default=DEFAULT_END, help="YYYYMMDD")
    parser.add_argument("--window", type=int, default=20, help="Rolling window size in trading days")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    args = parser.parse_args()

    OUT_CLASS_BY_STOCK.mkdir(parents=True, exist_ok=True)
    OUT_COMPLETE.mkdir(parents=True, exist_ok=True)

    month_end_dates = load_month_end_dates(start_date=args.start_date, end_date=args.end_date)
    if not month_end_dates:
        raise RuntimeError("No month-end dates found in selected range.")

    ts.set_token(load_token())
    pro = ts.pro_api()

    stocks = get_stock_list(pro, stock=args.stock)
    print(f"Total stocks to process: {len(stocks)}")

    stats = {"ok": 0, "empty": 0, "skipped": 0, "failed": 0}
    for ts_code in tqdm(stocks, desc="turnover_rate"):
        try:
            status = process_one_stock(
                pro=pro,
                ts_code=ts_code,
                start_date=args.start_date,
                end_date=args.end_date,
                month_end_dates=month_end_dates,
                window=args.window,
                overwrite=args.overwrite,
            )
            stats[status] += 1
        except Exception as exc:  # noqa: BLE001
            stats["failed"] += 1
            print(f"[failed] {ts_code}: {exc}")

    print("Done.")
    print(
        f"ok={stats['ok']}, empty={stats['empty']}, skipped={stats['skipped']}, failed={stats['failed']}"
    )
    print(f"Output class_by_stock: {OUT_CLASS_BY_STOCK}")
    print(f"Output complete_factor: {OUT_COMPLETE}")


if __name__ == "__main__":
    main()
