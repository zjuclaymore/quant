r"""
╔═══════════════════════════════════════════════════════════════════╗
║  因子3MAD极值处理器 (3MAD Winsorization Processor)                ║
║  模块: factor_engineering / 阶段: p02-因子处理                   ║
╚═══════════════════════════════════════════════════════════════════╝

功能:
  对因子进行3倍中位绝对偏差（MAD）的极值缺尾处理
  隐藏极端值对回测的影响

CLI 用法:
  python factor_mad3_winsorization.py \
    --input <path>                  # 输入因子文件（必填）
    --factor-col <name>             # 因子列名（必填）
    [--output <path>]               # 输出路径，默认 原文件 + _mad3后缀
    [--groupby <col>]               # 分组列，默认 date（按交易日分组）
    [--enable {yes|no}]             # 是否启用处理，默认 yes
    [--preview-rows <int>]          # 预览行数，默认 10

标准化方法:
  按交易日 (date) 分组进行极值缺尾
  需大于0的MAD值方可执行缺尾处理

示例:
  # 基础用法：一键3MAD处理
  python factor_mad3_winsorization.py \
    --input "factor.parquet" \
    --factor-col "momentum"

  # 禁用处理（跳过）
  python factor_mad3_winsorization.py \
    --input "factor.parquet" \
    --factor-col "momentum" \
    --enable no

  # 自定义输出路径
  python factor_mad3_winsorization.py \
    --input "factor.parquet" \
    --factor-col "momentum" \
    --output "factor_winsorized.parquet"

输出文件命名: {input}_mad3.parquet
"""
import os
import pandas as pd
import numpy as np
import argparse
import sys
from datetime import datetime


def mad_3_winsorization(df, factor_col, groupby_col='date'):
    """
    3倍中位绝对偏差(MAD)极值处理
    
    参数:
        df (pd.DataFrame): 输入数据，包含 code, date, {factor} 列
        factor_col (str): 因子列名
        groupby_col (str): 分组列（默认按date分组）
    
    返回:
        pd.DataFrame: 极值处理后的数据
    """
    print(f"\n[3MAD] 开始3MAD极值处理")
    print(f"[3MAD] 因子列: {factor_col}")
    print(f"[3MAD] 分组列: {groupby_col}")
    
    df = df.copy()
    
    # 初始统计
    init_mean = df[factor_col].mean()
    init_std = df[factor_col].std()
    init_min = df[factor_col].min()
    init_max = df[factor_col].max()
    outlier_count = 0
    
    print(f"[3MAD] 处理前统计:")
    print(f"  mean: {init_mean:.6f}, std: {init_std:.6f}")
    print(f"  min: {init_min:.6f}, max: {init_max:.6f}")
    
    # 按分组列进行3MAD处理
    if groupby_col and groupby_col in df.columns:
        for group_key, group_df in df.groupby(groupby_col):
            group_values = group_df[factor_col].values
            
            # 计算中位数和MAD
            median = np.nanmedian(group_values)
            mad = np.nanmedian(np.abs(group_values - median))
            
            if mad == 0:
                continue
            
            # 计算3MAD的上下限
            upper = median + 3 * mad
            lower = median - 3 * mad
            
            # 标记异常值
            mask = (df[groupby_col] == group_key)
            outlier_mask = mask & ((df[factor_col] > upper) | (df[factor_col] < lower))
            outlier_count += outlier_mask.sum()
            
            # 缩尾处理
            df.loc[outlier_mask & mask, factor_col] = df.loc[outlier_mask & mask, factor_col].clip(lower, upper)
    else:
        # 全局处理
        group_values = df[factor_col].values
        median = np.nanmedian(group_values)
        mad = np.nanmedian(np.abs(group_values - median))
        
        if mad > 0:
            upper = median + 3 * mad
            lower = median - 3 * mad
            
            outlier_mask = (df[factor_col] > upper) | (df[factor_col] < lower)
            outlier_count = outlier_mask.sum()
            
            df.loc[outlier_mask, factor_col] = df.loc[outlier_mask, factor_col].clip(lower, upper)
    
    # 处理后统计
    after_mean = df[factor_col].mean()
    after_std = df[factor_col].std()
    after_min = df[factor_col].min()
    after_max = df[factor_col].max()
    
    print(f"\n[3MAD] 处理后统计:")
    print(f"  mean: {after_mean:.6f}, std: {after_std:.6f}")
    print(f"  min: {after_min:.6f}, max: {after_max:.6f}")
    print(f"[3MAD] 缩尾数据点: {outlier_count} ({outlier_count/len(df)*100:.2f}%)")
    
    return df


def main():
    """CLI入口"""
    parser = argparse.ArgumentParser(
        description="因子3MAD极值处理器",
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
        help='输出文件路径 (默认: 原路径+_mad3后缀)'
    )
    
    parser.add_argument(
        '--groupby',
        type=str,
        default='date',
        help='分组列 (默认: date 按交易日分组)'
    )
    
    parser.add_argument(
        '--enable',
        type=str,
        choices=['yes', 'no'],
        default='yes',
        help='是否启用3MAD处理 (默认: yes)'
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
            print("[3MAD] ✗ 已禁用3MAD处理，跳过")
            return 0
        
        # 加载因子数据
        print("[3MAD] 加载输入文件...")
        if args.input.endswith('.parquet'):
            df = pd.read_parquet(args.input)
        elif args.input.endswith('.csv'):
            df = pd.read_csv(args.input)
        else:
            raise ValueError(f"不支持的文件格式: {args.input}")
        
        print(f"[3MAD] 输入数据: {len(df)} 行")
        
        # 检查因子列
        if args.factor_col not in df.columns:
            raise ValueError(f"因子列不存在: {args.factor_col}")
        
        # 3MAD处理
        df_processed = mad_3_winsorization(df, args.factor_col, groupby_col=args.groupby)
        
        # 确定输出路径
        if args.output is None:
            base, ext = os.path.splitext(args.input)
            args.output = f"{base}_mad3{ext}"
        
        # 保存
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        if args.output.endswith('.parquet'):
            df_processed.to_parquet(args.output, index=False)
        else:
            df_processed.to_csv(args.output, index=False)
        
        print(f"\n[3MAD] ✓ 已保存到: {args.output}")
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
