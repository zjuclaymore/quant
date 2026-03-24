"""
CSV 转 Parquet 格式转换脚本

将按股票存储的CSV文件转换为按日期存储的Parquet格式，
大幅提升回测加载速度。

用法:
    python convert_to_parquet.py --input-dir <input> --output-dir <output>
"""

import os
import glob
import json
import argparse
import pandas as pd
from tqdm import tqdm


def convert_class_by_stock_to_parquet(input_dir, output_dir, factor_name=None):
    """
    将 class_by_stock 格式的CSV转换为 class_by_date_parquet 格式

    参数:
        input_dir: 输入目录 (包含按股票存储的CSV文件)
        output_dir: 输出目录 (将创建class_by_date_parquet子目录)
        factor_name: 因子名称 (如未指定则自动检测)
    """
    os.makedirs(output_dir, exist_ok=True)

    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    if not csv_files:
        print(f"错误: 未找到CSV文件: {input_dir}")
        return

    print(f"[1/3] 加载 {len(csv_files)} 个CSV文件...")

    all_data = []
    detected_factor_name = None

    for csv_file in tqdm(csv_files, desc="读取CSV"):
        symbol = os.path.basename(csv_file).replace(".csv", "")

        try:
            df = pd.read_csv(csv_file)
            if df.empty:
                continue

            df["symbol"] = symbol

            if detected_factor_name is None:
                cols = [c for c in df.columns if c not in ["trade_date", "symbol"]]
                if cols:
                    detected_factor_name = cols[0]

            all_data.append(df)
        except Exception as e:
            print(f"警告: 读取 {csv_file} 失败: {e}")

    if not all_data:
        print("错误: 未能加载任何数据")
        return

    factor_name = factor_name or detected_factor_name
    print(f"[2/3] 合并数据并按日期拆分 (因子名: {factor_name})...")

    full_df = pd.concat(all_data, ignore_index=True)

    date_col = "trade_date"
    if date_col in full_df.columns:
        full_df[date_col] = pd.to_datetime(
            full_df[date_col].astype(str), format="%Y%m%d"
        )

    print(f"总记录数: {len(full_df)}, 股票数: {full_df['symbol'].nunique()}")

    grouped = full_df.groupby(full_df[date_col].dt.strftime("%Y%m%d"))

    parquet_dir = os.path.join(output_dir, "class_by_date_parquet")
    os.makedirs(parquet_dir, exist_ok=True)

    for date_str, group in tqdm(grouped, desc="写入Parquet"):
        output_file = os.path.join(parquet_dir, f"{date_str}.parquet")
        group_export = group[["symbol", factor_name]].copy()
        group_export.to_parquet(output_file, index=False, engine="pyarrow")

    meta = {
        "factor_name": factor_name,
        "format": "class_by_date_parquet",
        "total_records": len(full_df),
        "total_symbols": full_df["symbol"].nunique(),
        "date_range": {
            "start": full_df[date_col].min().strftime("%Y-%m-%d"),
            "end": full_df[date_col].max().strftime("%Y-%m-%d"),
        },
    }

    meta_file = os.path.join(parquet_dir, "_meta.json")
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[3/3] 转换完成!")
    print(f"  输出目录: {parquet_dir}")
    print(f"  文件数: {len(grouped)} 个Parquet文件")
    print(f"  因子名: {factor_name}")
    print(f"  元数据: {meta_file}")


def main():
    parser = argparse.ArgumentParser(description="CSV转Parquet格式转换工具")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="输入目录 (class_by_stock格式的CSV文件)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录 (默认为输入目录的同级目录)",
    )
    parser.add_argument(
        "--factor-name",
        type=str,
        default=None,
        help="因子名称 (默认自动检测)",
    )

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir or os.path.dirname(input_dir)

    convert_class_by_stock_to_parquet(
        input_dir=input_dir,
        output_dir=output_dir,
        factor_name=args.factor_name,
    )


if __name__ == "__main__":
    main()
