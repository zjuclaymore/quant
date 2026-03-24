"""
将截面数据 log_mv_2000_2025.csv 转换为按股票保存的时序数据。

输入: class_by_year/log_mv_2000_2025.csv
    列: trade_date, ts_code, log_mv

输出: class_by_stock/{ts_code}.csv
    每个股票一个文件，按 trade_date 升序排列
    列: trade_date, log_mv
"""

import os
import time
import pandas as pd
from collections import defaultdict

# ── 路径配置 ──────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(SCRIPT_DIR, "class_by_year", "log_mv_2000_2025.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "class_by_stock")

# ── 参数 ──────────────────────────────────────────────────
CHUNK_SIZE = 500_000  # 每次读取的行数，降低内存占用


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"[INFO] 输入文件: {INPUT_CSV}")
    print(f"[INFO] 输出目录: {OUTPUT_DIR}")
    print(f"[INFO] 分块大小: {CHUNK_SIZE:,} 行")
    print()

    t0 = time.time()

    # ── 第一步：分块读取，按股票代码聚合 ────────────────────
    stock_data: dict[str, list[pd.DataFrame]] = defaultdict(list)
    total_rows = 0

    reader = pd.read_csv(
        INPUT_CSV,
        dtype={"trade_date": str, "ts_code": str, "log_mv": float},
        chunksize=CHUNK_SIZE,
    )

    for i, chunk in enumerate(reader, 1):
        total_rows += len(chunk)
        elapsed = time.time() - t0
        print(f"  [chunk {i}] 已读取 {total_rows:>12,} 行  |  耗时 {elapsed:.1f}s")

        for ts_code, group in chunk.groupby("ts_code"):
            stock_data[ts_code].append(group[["trade_date", "log_mv"]])

    t1 = time.time()
    print(f"\n[INFO] 读取完毕: {total_rows:,} 行, {len(stock_data):,} 只股票, 耗时 {t1 - t0:.1f}s")

    # ── 第二步：合并并保存每只股票的时序文件 ─────────────────
    print(f"[INFO] 正在写入按股票分类的 CSV 文件...")

    for idx, (ts_code, dfs) in enumerate(sorted(stock_data.items()), 1):
        df = pd.concat(dfs, ignore_index=True)
        df.sort_values("trade_date", inplace=True)
        df.reset_index(drop=True, inplace=True)

        out_path = os.path.join(OUTPUT_DIR, f"{ts_code}.csv")
        df.to_csv(out_path, index=False)

        if idx % 500 == 0 or idx == len(stock_data):
            print(f"  [write] {idx:>5,} / {len(stock_data):,} 股票已保存")

    t2 = time.time()
    print(f"\n[DONE] 全部完成!")
    print(f"  总行数:   {total_rows:,}")
    print(f"  股票数:   {len(stock_data):,}")
    print(f"  读取耗时: {t1 - t0:.1f}s")
    print(f"  写入耗时: {t2 - t1:.1f}s")
    print(f"  总耗时:   {t2 - t0:.1f}s")


if __name__ == "__main__":
    main()
