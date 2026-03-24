"""
convert_pickle_to_csv.py
------------------------
将 tr_adjust_umr_1 因子的 pickle 切片文件批量转换为标准两列 CSV 格式。

输入目录：output/class_by_stock/        （每只股票一个 .pickle）
输出目录：output/class_by_stock_csv/    （每只股票一个 .csv，与其他因子格式一致）

标准输出格式（与 log_mv_1 等因子一致）：
    - 文件名：<股票代码>.csv，如 000001.SZ.csv
    - 列1：trade_date （YYYYMMDD 整数格式）
    - 列2：tr_adjust_umr_1 （因子值）
"""

import os
import glob
import pandas as pd
from pathlib import Path
from tqdm import tqdm

FACTOR_NAME = "tr_adjust_umr_1"
# 原始 pickle 列名（来自计算脚本 compute_factor.py）
SRC_DATE_COL    = "交易日期"
SRC_FACTOR_COL  = "factor_value"

BASE = Path(r"E:\1_basement\quant_research\factor\2_统计描述因子_StatisticalFactors\tr_adjust_umr_1\output")
SRC_DIR = BASE / "class_by_stock"
DST_DIR = BASE / "class_by_stock_csv"
DST_DIR.mkdir(parents=True, exist_ok=True)


def convert_one(pickle_path: Path, dst_dir: Path) -> bool:
    """
    将单个股票的 pickle 文件读取并转换为标准 CSV 格式。

    标准格式：
        trade_date  - YYYYMMDD 整数（去掉日期分隔符）
        tr_adjust_umr_1 - 因子值

    参数：
        pickle_path (Path): 源 pickle 文件路径
        dst_dir (Path): 输出 CSV 文件目录

    返回：
        bool: 转换成功返回 True，失败返回 False
    """
    try:
        df = pd.read_pickle(pickle_path)

        # 自动适配列名（兼容中英文、大小写）
        date_col = None
        factor_col = None
        for c in df.columns:
            if c == SRC_DATE_COL or "date" in str(c).lower() or "日期" in str(c):
                date_col = c
            if c == SRC_FACTOR_COL or "factor" in str(c).lower():
                factor_col = c

        if date_col is None or factor_col is None:
            return False

        res = df[[date_col, factor_col]].rename(
            columns={date_col: "trade_date", factor_col: FACTOR_NAME}
        )
        res = res.dropna(subset=["trade_date", FACTOR_NAME])

        # 规范日期为 YYYYMMDD 整数
        # 公式：去掉分隔符后转 int，如 "2020-01-02" -> 20200102
        res["trade_date"] = (
            res["trade_date"].astype(str).str.replace("-", "", regex=False)
            .astype(float).astype(int)
        )

        res = res.drop_duplicates(subset=["trade_date"], keep="last")
        res = res.sort_values("trade_date").reset_index(drop=True)

        out_name = pickle_path.stem + ".csv"   # 000001.SZ.csv
        res.to_csv(dst_dir / out_name, index=False)
        return True
    except Exception:
        return False


def main():
    """
    批量读取 class_by_stock 下所有 pickle 文件，转换后写入 class_by_stock_csv。
    """
    files = list(SRC_DIR.glob("*.pickle"))
    print(f"发现 {len(files)} 个 pickle 文件，开始转换 → {DST_DIR}")

    ok, fail = 0, 0
    for f in tqdm(files, desc="Converting"):
        if convert_one(f, DST_DIR):
            ok += 1
        else:
            fail += 1

    print(f"\n转换完成：成功 {ok} 个，失败 {fail} 个。")
    print(f"CSV 文件已保存至：{DST_DIR}")


if __name__ == "__main__":
    main()
