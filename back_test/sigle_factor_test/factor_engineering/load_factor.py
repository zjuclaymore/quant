r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  多格式因子数据加载器 (Multi-Format Factor Data Loader)                        ║
║  模块: factor_engineering / 阶段: p01-数据加载                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

功能:
  自动检测并加载多种格式的因子数据（Parquet/CSV/Pickle）
  标准化列名 (code, date, {factor})

CLI 用法:
  python p01_data_loading__load_factor_parquet.py \\
    --factor-path <path> \\              # 因子文件或目录路径（必填）
    [--factor-col <name>] \\             # 因子列名，默认自动检测
    [--format {parquet|csv|pickle|auto}] # 文件格式，默认 auto
    [--output <path>] \\                # 输出路径（可选）
    [--preview-rows <int>]              # 预览行数，默认 10

示例:
  # 自动检测格式
  python p01_data_loading__load_factor_parquet.py \\
    --factor-path "E:\\factor\\momentum.parquet"

  # 指定格式和因子列
  python p01_data_loading__load_factor_parquet.py \\
    --factor-path "E:\\factor\\data" \\
    --factor-col "zscore_momentum" \\
    --format csv \\
    --output "E:\\output\\momentum_clean.parquet"

数据格式:
  输入: Parquet/CSV/Pickle 文件，需包含 code, date, {factor} 列
  输出: 标准化数据框 [code, date, factor_value]
"""
import os
import pandas as pd
import pickle
import json
import glob
import argparse
import sys
from datetime import datetime
from pathlib import Path


def detect_factor_column(df, exclude_cols=None):
    """
    自动检测因子列（排除 code/symbol 和 date/trade_date）
    
    参数:
        df (pd.DataFrame): 数据框
        exclude_cols (list): 排除的列名
    
    返回:
        str: 因子列名，若无则返回None
    """
    if exclude_cols is None:
        exclude_cols = []
    
    exclude_set = set(exclude_cols) | {
        'code', 'symbol', 'stock_code', 'stk_code',
        'date', 'trade_date', 'trading_date', 'datetime'
    }
    
    potential_cols = [c for c in df.columns if c not in exclude_set]
    
    if len(potential_cols) == 0:
        return None
    elif len(potential_cols) == 1:
        return potential_cols[0]
    else:
        # 多个列时，返回第一个非数值聚合列
        return potential_cols[0]


def standardize_column_names(df):
    """
    标准化列名：code/symbol → 'code', date/* → 'date'
    
    参数:
        df (pd.DataFrame): 输入数据框
    
    返回:
        pd.DataFrame: 列名标准化后的数据框
    """
    df = df.copy()
    
    # 标准化 code 列
    code_cols = {'code', 'symbol', 'stock_code', 'stk_code'}
    for col in code_cols:
        if col in df.columns and col != 'code':
            df = df.rename(columns={col: 'code'})
            break
    
    # 标准化 date 列
    date_cols = {'date', 'trade_date', 'trading_date', 'datetime', 'Date', 'DATE'}
    for col in date_cols:
        if col in df.columns and col != 'date':
            df = df.rename(columns={col: 'date'})
            break
    
    return df


def load_factor_from_parquet(factor_path, factor_col=None):
    """
    加载 Parquet 格式因子数据
    
    参数:
        factor_path (str): 文件或目录路径
        factor_col (str): 因子列名，若为None则自动检测
    
    返回:
        pd.DataFrame: 因子数据
    """
    print(f"[Parquet Loader] 加载路径: {factor_path}")
    
    if os.path.isfile(factor_path):
        # 单个 Parquet 文件
        df = pd.read_parquet(factor_path)
        print(f"[Parquet Loader] 单文件加载: {len(df)} 行")
    
    elif os.path.isdir(factor_path):
        # 目录：逐日期读取多个 Parquet 文件
        files = sorted(glob.glob(os.path.join(factor_path, "*.parquet")))
        if not files:
            raise FileNotFoundError(f"目录中无 Parquet 文件: {factor_path}")
        
        all_dfs = []
        for f in files:
            try:
                df_day = pd.read_parquet(f)
                if not df_day.empty:
                    # 从文件名提取日期 (YYYYMMDD)
                    fname = os.path.basename(f).replace(".parquet", "")
                    if fname.isdigit() and len(fname) == 8:
                        df_day["date"] = pd.to_datetime(fname, format="%Y%m%d")
                    all_dfs.append(df_day)
            except Exception as e:
                print(f"[Warning] Parquet读取失败 ({f}): {e}")
        
        if not all_dfs:
            raise ValueError(f"无有效的Parquet数据: {factor_path}")
        
        df = pd.concat(all_dfs, ignore_index=True)
        print(f"[Parquet Loader] 多文件加载: {len(df)} 行，{len(files)} 个文件")
    
    else:
        raise FileNotFoundError(f"路径不存在: {factor_path}")
    
    # 标准化列名
    df = standardize_column_names(df)
    
    # 检测因子列
    if factor_col is None:
        factor_col = detect_factor_column(df)
    
    if factor_col is None:
        raise ValueError(f"无法检测因子列。可用列: {list(df.columns)}")
    
    print(f"[Parquet Loader] 因子列: {factor_col}")
    
    return df, factor_col


def load_factor_from_csv(factor_path, factor_col=None):
    """
    加载 CSV 格式因子数据
    
    参数:
        factor_path (str): 文件或目录路径
        factor_col (str): 因子列名
    
    返回:
        pd.DataFrame: 因子数据
    """
    print(f"[CSV Loader] 加载路径: {factor_path}")
    
    if os.path.isfile(factor_path):
        df = pd.read_csv(factor_path)
        print(f"[CSV Loader] 单文件加载: {len(df)} 行")
    
    elif os.path.isdir(factor_path):
        files = sorted(glob.glob(os.path.join(factor_path, "*.csv")))
        if not files:
            raise FileNotFoundError(f"目录中无 CSV 文件: {factor_path}")
        
        all_dfs = []
        for f in files:
            try:
                df_day = pd.read_csv(f)
                if not df_day.empty:
                    all_dfs.append(df_day)
            except Exception as e:
                print(f"[Warning] CSV读取失败 ({f}): {e}")
        
        if not all_dfs:
            raise ValueError(f"无有效的CSV数据: {factor_path}")
        
        df = pd.concat(all_dfs, ignore_index=True)
        print(f"[CSV Loader] 多文件加载: {len(df)} 行，{len(files)} 个文件")
    
    else:
        raise FileNotFoundError(f"路径不存在: {factor_path}")
    
    # 标准化列名
    df = standardize_column_names(df)
    
    # 检测因子列
    if factor_col is None:
        factor_col = detect_factor_column(df)
    
    if factor_col is None:
        raise ValueError(f"无法检测因子列。可用列: {list(df.columns)}")
    
    print(f"[CSV Loader] 因子列: {factor_col}")
    
    return df, factor_col


def load_factor_from_pickle(factor_path, factor_col=None):
    """
    加载 Pickle 格式因子数据
    
    参数:
        factor_path (str): 文件路径
        factor_col (str): 因子列名
    
    返回:
        pd.DataFrame: 因子数据
    """
    print(f"[Pickle Loader] 加载路径: {factor_path}")
    
    if not os.path.isfile(factor_path):
        raise FileNotFoundError(f"文件不存在: {factor_path}")
    
    try:
        with open(factor_path, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, pd.DataFrame):
            df = data
        elif isinstance(data, dict):
            df = pd.DataFrame(data)
        else:
            raise ValueError(f"Pickle 数据类型不支持: {type(data)}")
        
        print(f"[Pickle Loader] 加载完成: {len(df)} 行")
    
    except Exception as e:
        raise ValueError(f"Pickle加载失败: {e}")
    
    # 标准化列名
    df = standardize_column_names(df)
    
    # 检测因子列
    if factor_col is None:
        factor_col = detect_factor_column(df)
    
    if factor_col is None:
        raise ValueError(f"无法检测因子列。可用列: {list(df.columns)}")
    
    print(f"[Pickle Loader] 因子列: {factor_col}")
    
    return df, factor_col


def load_factor_data(factor_path, factor_col=None, format_type=None):
    """
    通用因子加载函数（自动检测格式）
    
    参数:
        factor_path (str): 文件或目录路径
        factor_col (str): 因子列名，若为None则自动检测
        format_type (str): 格式类型 ('parquet', 'csv', 'pickle')，若为None则自动检测
    
    返回:
        tuple: (df, factor_col) - 标准化的因子数据框和因子列名
    """
    print(f"\n[Factor Loader] 开始加载因子数据")
    print(f"[Factor Loader] 路径: {factor_path}")
    
    # 自动检测格式
    if format_type is None:
        if factor_path.endswith('.parquet'):
            format_type = 'parquet'
        elif factor_path.endswith('.csv'):
            format_type = 'csv'
        elif factor_path.endswith(('.pickle', '.pkl')):
            format_type = 'pickle'
        elif os.path.isdir(factor_path):
            # 检查目录中的文件类型
            parquet_files = glob.glob(os.path.join(factor_path, "*.parquet"))
            csv_files = glob.glob(os.path.join(factor_path, "*.csv"))
            if parquet_files:
                format_type = 'parquet'
            elif csv_files:
                format_type = 'csv'
            else:
                raise ValueError(f"无法检测目录格式。目录: {factor_path}")
        else:
            raise ValueError(f"无法检测文件格式。路径: {factor_path}")
    
    print(f"[Factor Loader] 检测格式: {format_type}")
    
    # 按格式加载
    if format_type == 'parquet':
        df, col = load_factor_from_parquet(factor_path, factor_col)
    elif format_type == 'csv':
        df, col = load_factor_from_csv(factor_path, factor_col)
    elif format_type == 'pickle':
        df, col = load_factor_from_pickle(factor_path, factor_col)
    else:
        raise ValueError(f"不支持的格式类型: {format_type}")
    
    # 验证必要列
    required_cols = {'code', 'date'}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"缺少必要列。期望: {required_cols}, 实际: {set(df.columns)}")
    
    # 数据类型转换
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['date', 'code']).reset_index(drop=True)
    
    print(f"[Factor Loader] [+] 加载成功")
    print(f"[Factor Loader] 数据形状: {df.shape}")
    print(f"[Factor Loader] 日期范围: {df['date'].min()} ~ {df['date'].max()}")
    print(f"[Factor Loader] 股票数: {df['code'].nunique()}")
    print(f"[Factor Loader] 因子列: {col}")
    
    return df, col


def main():
    """CLI入口"""
    parser = argparse.ArgumentParser(
        description="多格式因子数据加载器 (支持 Parquet/CSV/Pickle)",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        '--factor-path',
        type=str,
        required=True,
        help='因子数据路径 (文件或目录)\n  支持: .parquet, .csv, .pickle, .pkl'
    )
    
    parser.add_argument(
        '--factor-col',
        type=str,
        default=None,
        help='因子列名 (默认: 自动检测，一般为非code/date的列)'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        choices=['parquet', 'csv', 'pickle', 'auto'],
        default='auto',
        help='数据格式 (默认: auto = 自动检测)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='输出文件路径 (可选，格式: parquet/csv)'
    )
    
    parser.add_argument(
        '--preview-rows',
        type=int,
        default=10,
        help='预览行数 (默认: 10)'
    )
    
    args = parser.parse_args()
    
    try:
        # 加载因子数据
        format_arg = None if args.format == 'auto' else args.format
        df_factor, factor_col = load_factor_data(
            factor_path=args.factor_path,
            factor_col=args.factor_col,
            format_type=format_arg
        )
        
        print(f"\n{'='*60}")
        print(f"数据预览 (前 {args.preview_rows} 行):")
        print(f"{'='*60}")
        print(df_factor.head(args.preview_rows))
        
        print(f"\n{'='*60}")
        print(f"数据统计:")
        print(f"{'='*60}")
        print(f"总行数: {len(df_factor)}")
        print(f"股票数: {df_factor['code'].nunique()}")
        print(f"交易日数: {df_factor['date'].nunique()}")
        print(f"因子列: {factor_col}")
        print(f"因子统计:")
        print(df_factor[factor_col].describe())
        
        # 保存输出
        if args.output:
            os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
            if args.output.endswith('.parquet'):
                df_factor.to_parquet(args.output, index=False)
            else:
                df_factor.to_csv(args.output, index=False)
            print(f"\n[Saved] 已保存到: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
"""
自动拆分视图文件（仅用于阅读理解，不参与运行）。
来源文件: back_test/sigle_factor_test/src/data_loader.py
函数: load_factor_parquet
类型: module_function
行号: 275-331
签名: def load_factor_parquet(factor_dir)
作用概述: 高性能 Parquet 因子加载引擎 (Optimized Parquet Loader)
"""
def load_factor_parquet(factor_dir):
    """
    高性能 Parquet 因子加载引擎 (Optimized Parquet Loader)

    针对按日期存储的 Parquet 文件进行快速扫描。每个文件通常代表一个交易日的全市场截面。

    逻辑细节:
        - 信号自发现: 优先从 `_meta.json` 中读取 `factor_name`。若不存在，则猜测首个非索引列。
        - 自动日期注入: 从文件名 YYYYMMDD 提取日期信息，确保数据的时序准确性。
        - 并发安全: 采用顺序读取配合 `pd.concat`，在大规模数据集下保持性能优势。

    参数:
        factor_dir (str): 包含多个日期 Parquet 文件的目录。

    返回:
        tuple: (合并后的因子 DataFrame, 因子列名, 股票名单集合)。
    """
    meta_file = os.path.join(factor_dir, "_meta.json")
    if os.path.exists(meta_file):
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        factor_col = meta.get("factor_name")
    else:
        factor_col = None

    files = sorted(glob.glob(os.path.join(factor_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"未找到Parquet文件: {factor_dir}")

    all_dfs = []
    symbols_whitelist = set()

    for f in tqdm(files, desc="[Phase 1] 加载Parquet因子数据"):
        try:
            df = pd.read_parquet(f)
            if df.empty:
                continue

            date_str = os.path.basename(f).replace(".parquet", "")
            df["date"] = pd.to_datetime(date_str, format="%Y%m%d")

            if factor_col is None:
                potential_cols = [c for c in df.columns if c not in ["date", "symbol"]]
                factor_col = potential_cols[0] if potential_cols else None

            symbols_whitelist.update(df["symbol"].tolist())
            all_dfs.append(df)
        except Exception as e:
            print(f"[Error] 因子Parquet加载失败({f}): {type(e).__name__}: {e}")

    full_factor_df = pd.concat(all_dfs, ignore_index=True)
    print(
        f"[Phase 1] Parquet因子加载完成: {len(full_factor_df)} 条记录, "
        f"股票数: {len(symbols_whitelist)}, 因子: {factor_col}"
    )

    return full_factor_df, factor_col, symbols_whitelist
