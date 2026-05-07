r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  市值数据加载与合并器 (Market Cap Data Loader & Merger)                       ║
║  模块: factor_engineering / 阶段: p01-数据加载                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

功能:
  加载对数市值数据，与因子数据按 (code, date) 合并
  自动检测多种格式（Parquet/CSV），智能列名标准化

数据来源:
  E:\1_basement\quant_research\factor\1_算术转换因子_ArithmeticFactors\log_mv_1\output\class_by_stock

CLI 用法:
  python p01_data_loading__merge_market_cap.py \\
    --factor-path <path> \\              # 因子文件路径（必填）
    [--mv-path <path>] \\                # 市值数据路径（自动检测）
    [--merge-method {left|inner|outer}] # 合并方式，默认 left
    --output <path> \\                  # 输出文件路径（必填）
    [--preview-rows <int>]              # 预览行数，默认 10

示例:
  # 基础用法：因子 + 市值合并
  python p01_data_loading__merge_market_cap.py \\
    --factor-path "factor_data.parquet" \\
    --output "factor_with_mv.parquet"

  # 自定义市值路径 + inner合并
  python p01_data_loading__merge_market_cap.py \\
    --factor-path "factor.csv" \\
    --mv-path "E:\\custom_mv" \\
    --merge-method inner \\
    --output "merged.parquet"

合并方式:
  left: 以因子数据为主（默认），保留所有因子记录
  inner: 仅保留两边都有的数据
  outer: 保留所有数据，缺失值填 NaN

输出列: [原因子列] + [code, date, lncap]
"""

import os
import pandas as pd
import glob
import argparse
import sys
from datetime import datetime
from pathlib import Path


def _standardize_code(code: str) -> str:
    """
    标准化股票代码为 6 位数字格式。

    例如: 000001.SZ -> 000001, 600000.SH -> 600000, 837592.BJ -> 837592
    """
    code = str(code).strip()
    # 移除后缀 .SZ/.SH/.BJ 等
    if "." in code:
        code = code.split(".")[0]
    # 补零到 6 位
    return code.zfill(6)


def load_market_cap_data(mv_path=None):
    """
    加载市值数据（对数市值）

    参数:
        mv_path (str): 市值数据路径。若为None，使用默认路径。

    返回:
        pd.DataFrame: 市值数据，包含 code/symbol, date, lncap/log_mv 列
    """
    # 默认市值路径
    if mv_path is None:
        mv_path = r"E:\1_basement\quant_research\factor\1_算术转换因子_ArithmeticFactors\log_mv_1\output\class_by_stock"

    print(f"[Market Cap Loader] 市值路径: {mv_path}")

    if not os.path.exists(mv_path):
        raise FileNotFoundError(f"市值数据路径不存在: {mv_path}")

    # 查找市值数据文件
    parquet_files = sorted(glob.glob(os.path.join(mv_path, "*.parquet")))
    csv_files = sorted(glob.glob(os.path.join(mv_path, "*.csv")))

    all_dfs = []

    # 加载 Parquet 文件
    if parquet_files:
        print(f"[Market Cap Loader] 发现 {len(parquet_files)} 个 Parquet 文件")
        for f in parquet_files:
            try:
                df = pd.read_parquet(f)
                if not df.empty:
                    all_dfs.append(df)
            except Exception as e:
                print(f"[Warning] Parquet读取失败 ({os.path.basename(f)}): {e}")

    # 加载 CSV 文件
    if csv_files:
        print(f"[Market Cap Loader] 发现 {len(csv_files)} 个 CSV 文件")
        for f in csv_files:
            try:
                df = pd.read_csv(f)
                if not df.empty:
                    # 从文件名提取股票代码
                    filename_code = Path(f).stem  # e.g., "000001.SZ"
                    df = standardize_market_cap_columns(df, filename_code)
                    # 标准化代码格式
                    if "code" in df.columns:
                        df["code"] = df["code"].astype(str).apply(_standardize_code)
                    all_dfs.append(df)
            except Exception as e:
                print(f"[Warning] CSV读取失败 ({os.path.basename(f)}): {e}")

    if not all_dfs:
        raise ValueError(f"无有效的市值数据文件: {mv_path}")

    # 合并所有文件
    df_mv = pd.concat(all_dfs, ignore_index=True)
    print(f"[Market Cap Loader] 初始加载: {len(df_mv)} 行")

    # 标准化列名
    df_mv = standardize_market_cap_columns(df_mv)

    # 数据类型转换和排序
    if "date" in df_mv.columns:
        # 确保日期格式正确
        if df_mv["date"].dtype in ["int64", "Int64"]:
            # 如果是整数格式（YYYYMMDD），转换为 datetime64[ns]
            df_mv["date"] = pd.to_datetime(df_mv["date"].astype(str), format="%Y%m%d")
        elif df_mv["date"].dtype == "object":
            # 如果是字符串，尝试转换为 datetime
            df_mv["date"] = pd.to_datetime(df_mv["date"])
    if "code" in df_mv.columns:
        df_mv["code"] = df_mv["code"].astype(str)

    # 检查是否存在市值列
    if "lncap" not in df_mv.columns:
        # 尝试找到市值列
        potential_cols = [
            c
            for c in df_mv.columns
            if "mv" in c.lower() or "cap" in c.lower() or "log" in c.lower()
        ]
        if potential_cols:
            df_mv = df_mv.rename(columns={potential_cols[0]: "lncap"})
            print(f"[Market Cap Loader] 自动检测到市值列: {potential_cols[0]} -> lncap")
        else:
            raise ValueError(f"无法找到市值列。可用列: {list(df_mv.columns)}")

    # 排除缺失值
    df_mv = df_mv.dropna(subset=["lncap"])
    df_mv = (
        df_mv[["code", "date", "lncap"]]
        .drop_duplicates()
        .sort_values(["date", "code"])
        .reset_index(drop=True)
    )

    print(f"[Market Cap Loader] [+] 加载完成: {len(df_mv)} 行")
    print(
        f"[Market Cap Loader] 日期范围: {df_mv['date'].min()} ~ {df_mv['date'].max()}"
    )
    print(f"[Market Cap Loader] 股票数: {df_mv['code'].nunique()}")

    return df_mv


def standardize_market_cap_columns(df, filename_code=None):
    """
    标准化市值数据的列名

    code/symbol → 'code'
    date/* → 'date'
    {log_mv/lncap/mv} → 'lncap'
    """
    df = df.copy()

    # 如果没有 code 列，但提供了文件名中的 code，则添加
    if "code" not in df.columns and filename_code:
        df["code"] = filename_code

    # 标准化 code 列
    code_cols = {"code", "symbol", "stock_code", "stk_code", "Code", "Symbol"}
    for col in code_cols:
        if col in df.columns and col != "code":
            df = df.rename(columns={col: "code"})
            break

    # 标准化 date 列
    date_cols = {"date", "trade_date", "trading_date", "Date", "DATE", "datetime"}
    for col in date_cols:
        if col in df.columns and col != "date":
            df = df.rename(columns={col: "date"})
            break

    return df


def merge_market_cap_to_factor(df_factor, df_mv, on=["code", "date"], how="left"):
    """
    将市值数据合并到因子数据

    对于月度因子数据，会匹配到当月最后一个交易日的市值数据。

    参数:
        df_factor (pd.DataFrame): 因子数据
        df_mv (pd.DataFrame): 市值数据
        on (list): 合并键
        how (str): 合并方式 ('left', 'inner', 'outer')

    返回:
        pd.DataFrame: 合并后的数据
    """
    print(f"\n[Merger] 开始合并市值数据")
    print(f"[Merger] 因子数据形状: {df_factor.shape}")
    print(f"[Merger] 市值数据形状: {df_mv.shape}")
    print(f"[Merger] 合并方式: {how}, 合并键: {on}")

    # 检查必要列
    for col in on:
        if col not in df_factor.columns:
            raise ValueError(f"因子数据缺少列: {col}")
        if col not in df_mv.columns:
            raise ValueError(f"市值数据缺少列: {col}")

    # 确保 date 列为 datetime 类型以便调用 .dt
    df_factor_date = df_factor["date"]
    if df_factor_date.dtype in ["int64", "Int64", "object"]:
        df_factor_date = pd.to_datetime(df_factor_date.astype(str), format="%Y%m%d", errors='coerce')
        if df_factor_date.isna().all():
             df_factor_date = pd.to_datetime(df_factor["date"].astype(str), errors='coerce')

    # 对于月度因子数据，匹配当月最后一个交易日的市值
    if len(df_factor_date.dt.day.unique()) <= 31:
        print(f"[Merger] 检测到月度因子数据，匹配当月最后一个交易日的市值")

        # 创建市值数据的月度汇总（每个月最后一个交易日）
        df_mv_copy = df_mv.copy()
        df_mv_copy["year_month"] = df_mv_copy["date"].dt.to_period("M")

        # 找到每个股票每个月的最后一个交易日
        monthly_mv = (
            df_mv_copy.sort_values(["code", "year_month", "date"])
            .groupby(["code", "year_month"])
            .last()
            .reset_index()[["code", "year_month", "lncap"]]
        )

        # 为因子数据添加 year_month 列
        df_factor_copy = df_factor.copy()
        df_factor_copy["year_month"] = df_factor_date.dt.to_period("M")

        # 合并
        df_merged = pd.merge(
            df_factor_copy,
            monthly_mv,
            left_on=["code", "year_month"],
            right_on=["code", "year_month"],
            how=how,
        )

        # 清理临时列
        df_merged = df_merged.drop(columns=["year_month"])
    else:
        # 直接合并（适用于日频因子数据）
        df_merged = pd.merge(df_factor, df_mv[on + ["lncap"]], on=on, how=how)

    print(f"[Merger] [+] 合并完成")
    print(f"[Merger] 合并后数据形状: {df_merged.shape}")
    print(
        f"[Merger] 市值列缺失率: {(df_merged['lncap'].isna().sum() / len(df_merged) * 100):.2f}%"
    )

    return df_merged


def main():
    """CLI入口"""
    parser = argparse.ArgumentParser(
        description="市值数据加载和合并器",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--factor-path", type=str, required=True, help="因子数据文件路径 (必填)"
    )

    parser.add_argument(
        "--mv-path",
        type=str,
        default=None,
        help=(
            "市值数据路径 (默认自动使用)\n"
            "  默认: E:\\...\\factor\\1_算术转换因子_ArithmeticFactors\\log_mv_1\\output\\class_by_stock"
        ),
    )

    parser.add_argument(
        "--merge-method",
        type=str,
        choices=["left", "inner", "outer"],
        default="left",
        help="合并方式 (默认: left = 以因子数据为主)",
    )

    parser.add_argument(
        "--output", type=str, required=True, help="输出文件路径 (parquet 或 csv)"
    )

    parser.add_argument(
        "--preview-rows", type=int, default=10, help="预览行数 (默认: 10)"
    )

    args = parser.parse_args()

    try:
        # 验证因子文件
        if not os.path.exists(args.factor_path):
            raise FileNotFoundError(f"因子文件不存在: {args.factor_path}")

        # 加载因子数据
        print("[Main] 加载因子数据...")
        if args.factor_path.endswith(".parquet"):
            df_factor = pd.read_parquet(args.factor_path)
        elif args.factor_path.endswith(".csv"):
            df_factor = pd.read_csv(args.factor_path)
        else:
            raise ValueError(f"不支持的因子文件格式: {args.factor_path}")

        print(f"[Main] 因子数据加载完成: {len(df_factor)} 行")

        # 确保code列存在并标准化
        if "code" not in df_factor.columns:
            code_cols = {"symbol", "stock_code", "stk_code"}
            for col in code_cols:
                if col in df_factor.columns:
                    df_factor = df_factor.rename(columns={col: "code"})
                    break

        if "code" not in df_factor.columns:
            raise ValueError(f"因子数据缺少code列。可用列: {list(df_factor.columns)}")

        df_factor["code"] = df_factor["code"].astype(str).apply(_standardize_code)

        # 加载市值数据
        print("\n[Main] 加载市值数据...")
        df_mv = load_market_cap_data(mv_path=args.mv_path)

        # 合并
        df_merged = merge_market_cap_to_factor(
            df_factor, df_mv, on=["code", "date"], how=args.merge_method
        )

        # 预览
        print(f"\n{'=' * 60}")
        print(f"数据预览 (前 {args.preview_rows} 行):")
        print(f"{'=' * 60}")
        print(df_merged.head(args.preview_rows))

        # 保存
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        if args.output.endswith(".parquet"):
            df_merged.to_parquet(args.output, index=False)
        else:
            df_merged.to_csv(args.output, index=False)

        print(f"\n[Main] [+] 已保存到: {args.output}")

        return 0

    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
