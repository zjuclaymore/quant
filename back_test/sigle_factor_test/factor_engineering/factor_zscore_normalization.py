r"""
╔═══════════════════════════════════════════════════════════════════╗
║  因子Zscore标准化处理器 (Zscore Normalization Processor)          ║
║  模块: factor_engineering / 阶段: p02-因子处理                   ║
╚═══════════════════════════════════════════════════════════════════╝

功能:
  对因子进行Zscore标准化（z = (x - mean) / std）
  支持截面、时间序列、全局三种标准化方法

CLI 用法:
  python factor_zscore_normalization.py \
    --input <path>                  # 输入因子文件（必填）
    --factor-col <name>             # 因子列名（必填）
    [--output <path>]               # 输出路径，默认 原文件 + _zscore后缀
    [--method {cross_sectional|time_series|global}] # 标准化方法
    [--groupby <col>]               # 分组列，默认 date
    [--enable {yes|no}]             # 是否启用处理，默认 yes
    [--preview-rows <int>]          # 预览行数，默认 10

Zscore标准化方法:
  cross_sectional: 截面标准化（每日独立）- 每一个交易日有不同的平均、标准差、水平（默认）
  time_series: 时间序列标准化（每股票独立）- 每一股票上下整个历史水平
  global: 全局标准化 - 整个输入数据的标准化

示例:
  # 基础：截面标准化（推荐）
  python factor_zscore_normalization.py \
    --input "factor_mad3.parquet" \
    --factor-col "momentum" \
    --method cross_sectional

  # 时间序列标准化
  python factor_zscore_normalization.py \
    --input "factor_mad3.parquet" \
    --factor-col "momentum" \
    --method time_series

  # 自定义输出路径
  python factor_zscore_normalization.py \
    --input "factor_mad3.parquet" \
    --factor-col "momentum" \
    --output "factor_zscore.parquet"

输出文件命名: {input}_zscore.parquet
"""
import os
import pandas as pd
import numpy as np
import argparse
import sys
from datetime import datetime


def zscore_normalization(df, factor_col, groupby_col='date', method='cross_sectional'):
    """
    Zscore标准化处理
    
    参数:
        df (pd.DataFrame): 输入数据
        factor_col (str): 因子列名
        groupby_col (str): 分组列（默认date）
        method (str): 标准化方法
                     'cross_sectional': 截面标准化（按date分组）
                     'time_series': 时间序列标准化（按code分组）
                     'global': 全局标准化
    
    返回:
        pd.DataFrame: 标准化后的数据
    """
    print(f"\n[Zscore] 开始Zscore标准化")
    print(f"[Zscore] 因子列: {factor_col}")
    print(f"[Zscore] 方法: {method}")
    
    df = df.copy()
    factor_col_new = f"{factor_col}_zscore"
    
    # 初始统计
    init_mean = df[factor_col].mean()
    init_std = df[factor_col].std()
    
    print(f"[Zscore] 标准化前统计:")
    print(f"  mean: {init_mean:.6f}, std: {init_std:.6f}")
    
    if method == 'cross_sectional':
        # 截面标准化（每日独立）
        if groupby_col and groupby_col in df.columns:
            df[factor_col_new] = df.groupby(groupby_col)[factor_col].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-8)
            )
            print(f"[Zscore] 按 {groupby_col} 进行截面标准化")
        else:
            df[factor_col_new] = (df[factor_col] - df[factor_col].mean()) / (df[factor_col].std() + 1e-8)
    
    elif method == 'time_series':
        # 时间序列标准化（每只股票独立）
        if 'code' in df.columns:
            df[factor_col_new] = df.groupby('code')[factor_col].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-8)
            )
            print("[Zscore] 按 code 进行时间序列标准化")
        else:
            raise ValueError("时间序列标准化需要 'code' 列")
    
    else:  # global
        # 全局标准化
        mean_val = df[factor_col].mean()
        std_val = df[factor_col].std()
        df[factor_col_new] = (df[factor_col] - mean_val) / (std_val + 1e-8)
        print("[Zscore] 进行全局标准化")
    
    # 处理后统计
    after_mean = df[factor_col_new].mean()
    after_std = df[factor_col_new].std()
    after_min = df[factor_col_new].min()
    after_max = df[factor_col_new].max()
    
    print(f"\n[Zscore] 标准化后统计:")
    print(f"  mean: {after_mean:.6f}, std: {after_std:.6f}")
    print(f"  min: {after_min:.6f}, max: {after_max:.6f}")
    print(f"[Zscore] 缺失值: {df[factor_col_new].isna().sum()}")
    
    # 删除原列，保留标准化列
    df = df.drop(columns=[factor_col])
    df = df.rename(columns={factor_col_new: factor_col})
    
    return df


def main():
    """CLI入口"""
    parser = argparse.ArgumentParser(
        description="因子Zscore标准化处理器",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='输入因子文件路径 (parquet或csv)'
    )
    
    parser.add_argument(
        '--factor-col',
        type=str,
        required=True,
        help='因子列名'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='输出文件路径 (默认: 原路径+_zscore后缀)'
    )
    
    parser.add_argument(
        '--method',
        type=str,
        choices=['cross_sectional', 'time_series', 'global'],
        default='cross_sectional',
        help=(
            '标准化方法 (默认: cross_sectional)\n'
            '  cross_sectional: 截面标准化（每日独立）\n'
            '  time_series: 时间序列标准化（每只股票独立）\n'
            '  global: 全局标准化'
        )
    )
    
    parser.add_argument(
        '--groupby',
        type=str,
        default='date',
        help='分组列 (仅在cross_sectional时使用, 默认: date)'
    )
    
    parser.add_argument(
        '--enable',
        type=str,
        choices=['yes', 'no'],
        default='yes',
        help='是否启用Zscore标准化 (默认: yes)'
    )
    
    parser.add_argument(
        '--preview-rows',
        type=int,
        default=10,
        help='预览行数 (默认: 10)'
    )
    
    args = parser.parse_args()
    
    try:
        # 检查启用状态
        if args.enable == 'no':
            print("[Zscore] ✗ 已禁用Zscore标准化，跳过")
            return 0
        
        # 加载因子数据
        print("[Zscore] 加载输入文件...")
        if args.input.endswith('.parquet'):
            df = pd.read_parquet(args.input)
        elif args.input.endswith('.csv'):
            df = pd.read_csv(args.input)
        else:
            raise ValueError(f"不支持的文件格式: {args.input}")
        
        print(f"[Zscore] 输入数据: {len(df)} 行")
        
        # 检查因子列
        if args.factor_col not in df.columns:
            raise ValueError(f"因子列不存在: {args.factor_col}")
        
        # Zscore标准化
        df_processed = zscore_normalization(
            df, args.factor_col,
            groupby_col=args.groupby,
            method=args.method
        )
        
        # 确定输出路径
        if args.output is None:
            base, ext = os.path.splitext(args.input)
            args.output = f"{base}_zscore{ext}"
        
        # 保存
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        if args.output.endswith('.parquet'):
            df_processed.to_parquet(args.output, index=False)
        else:
            df_processed.to_csv(args.output, index=False)
        
        print(f"\n[Zscore] [+] 已保存到: {args.output}")
        print(f"\n预览 (前 {args.preview_rows} 行):")
        print(df_processed.head(args.preview_rows))
        
        return 0
        
    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
