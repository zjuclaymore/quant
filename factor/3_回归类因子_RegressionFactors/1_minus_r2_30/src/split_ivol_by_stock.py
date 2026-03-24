# -*- coding: utf-8 -*-
"""
split_ivol_by_stock.py
----------------------
将 class_by_year/ 目录下按年份存储的因子 CSV 文件
转换为 class_by_stock/ 目录下按股票存储的时序 CSV 文件。

源数据: 每个文件 = 一个年份，行 = 各股票在各交易日的因子值
        列: trade_date, ts_code, 1_minus_r2
目标:   每个文件 = 一只股票，行 = 各交易日的因子值时序
        列: trade_date, 1_minus_r2

用法:
  python split_ivol_by_stock.py
"""

import os
import glob
import time
import pandas as pd

# ──────────────────────────── 路径配置 ────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

SRC_DIR = os.path.join(PROJECT_DIR, "output", "class_by_year")
DST_DIR = os.path.join(PROJECT_DIR, "output", "class_by_stock")


def main():
    os.makedirs(DST_DIR, exist_ok=True)

    # 1. 读取所有年度 CSV 文件
    csv_files = sorted(glob.glob(os.path.join(SRC_DIR, "ivol_*.csv")))
    if not csv_files:
        print(f"[错误] 源目录中没有 ivol_*.csv 文件: {SRC_DIR}")
        return

    print(f"源目录:   {SRC_DIR}")
    print(f"输出目录: {DST_DIR}")
    print(f"待处理年度文件: {len(csv_files)} 个")
    print()

    # 2. 逐文件加载并合并
    t0 = time.time()
    frames = []
    for i, fp in enumerate(csv_files, 1):
        year_label = os.path.basename(fp)
        df = pd.read_csv(fp, dtype={"trade_date": str})
        frames.append(df)
        print(f"  [{i}/{len(csv_files)}] {year_label}: {len(df):,} 行")

    all_data = pd.concat(frames, ignore_index=True)
    del frames  # 释放内存

    load_time = time.time() - t0
    print(f"\n加载完成: 共 {len(all_data):,} 行, "
          f"{all_data['ts_code'].nunique():,} 只股票, "
          f"耗时 {load_time:.1f}s")

    # 3. 按股票分组写入
    print("\n开始按股票写入时序数据 ...")
    t1 = time.time()
    grouped = all_data.groupby("ts_code")
    total_stocks = len(grouped)
    saved = 0

    for code, group in grouped:
        # 按交易日排序，去掉股票代码列
        group = group.sort_values("trade_date").reset_index(drop=True)
        group = group.drop(columns=["ts_code"])

        # 去重（保留最后一条）
        group = group.drop_duplicates(subset=["trade_date"], keep="last")

        out_path = os.path.join(DST_DIR, f"{code}.csv")
        group.to_csv(out_path, index=False)
        saved += 1

        if saved % 500 == 0 or saved == total_stocks:
            elapsed = time.time() - t1
            print(f"  写入进度: {saved}/{total_stocks} 只股票, "
                  f"耗时 {elapsed:.1f}s")

    total_time = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"完成! 共写入 {saved} 个股票时序文件")
    print(f"输出目录: {DST_DIR}")
    print(f"总耗时: {total_time:.1f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
