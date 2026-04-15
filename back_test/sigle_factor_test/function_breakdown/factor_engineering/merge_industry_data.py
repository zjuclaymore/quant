r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  产业分类数据加载与合并器 (Industry Classification Data Loader & Merger)     ║
║  模块: factor_engineering / 阶段: p01-数据加载                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

功能:
  加载申万行业分类数据（2021版），与因子数据按 code 合并
  自动检测多种格式（Pickle/Parquet/CSV），智能列名标准化

数据来源:
  E:\1_basement\quant_research\data\申万行业分类2021版

CLI 用法:
  python p01_data_loading__merge_industry_data.py \\
    --factor-path <path> \\              # 因子文件路径（必填）
    [--ind-path <path>] \\               # 产业数据路径（自动检测）
    [--merge-method {left|inner|outer}] # 合并方式，默认 left
    --output <path> \\                  # 输出文件路径（必填）
    [--preview-rows <int>]              # 预览行数，默认 10

示例:
  # 基础用法：因子 + 产业合并
  python p01_data_loading__merge_industry_data.py \\
    --factor-path "factor_data.parquet" \\
    --output "factor_with_ind.parquet"

  # 自定义产业路径
  python p01_data_loading__merge_industry_data.py \\
    --factor-path "factor.csv" \\
    --ind-path "E:\\custom_industry" \\
    --merge-method left \\
    --output "merged.parquet"

合并方式:
  left: 以因子数据为主（默认）
  inner: 仅保留两边都有的数据
  outer: 保留所有数据

输出列: [原因子列] + [code, ind_code, ind_name]
"""
import os
import pandas as pd
import pickle
import glob
import argparse
import sys
from datetime import datetime
from pathlib import Path


def load_industry_data(ind_path=None):
    """
    加载产业分类数据（申万行业分类2021版）
    
    参数:
        ind_path (str): 产业数据路径。若为None，使用默认路径。
    
    返回:
        pd.DataFrame: 产业数据，包含 code/symbol, industry/ind_code, ind_name/industry_name 列
    """
    # 默认产业路径
    if ind_path is None:
        ind_path = r'E:\1_basement\quant_research\data\申万行业分类2021版'
    
    print(f"[Industry Loader] 产业数据路径: {ind_path}")
    
    if not os.path.exists(ind_path):
        raise FileNotFoundError(f"产业数据路径不存在: {ind_path}")
    
    df_ind = None
    
    # 尝试加载 Pickle 文件（优先）
    pickle_files = glob.glob(os.path.join(ind_path, "*.pickle")) + glob.glob(os.path.join(ind_path, "*.pkl"))
    if pickle_files:
        for f in pickle_files:
            try:
                df_ind = pd.read_pickle(f)
                print(f"[Industry Loader] 从Pickle加载: {os.path.basename(f)}")
                break
            except Exception as e:
                print(f"[Warning] Pickle读取失败 ({os.path.basename(f)}): {e}")
    
    # 如果没有Pickle，尝试加载 Parquet
    if df_ind is None:
        parquet_files = sorted(glob.glob(os.path.join(ind_path, "*.parquet")))
        if parquet_files:
            try:
                df_ind = pd.read_parquet(parquet_files[0])
                print(f"[Industry Loader] 从Parquet加载: {os.path.basename(parquet_files[0])}")
            except Exception as e:
                print(f"[Warning] Parquet读取失败: {e}")
    
    # 尝试加载 CSV
    if df_ind is None:
        csv_files = sorted(glob.glob(os.path.join(ind_path, "*.csv")))
        if csv_files:
            try:
                df_ind = pd.read_csv(csv_files[0])
                print(f"[Industry Loader] 从CSV加载: {os.path.basename(csv_files[0])}")
            except Exception as e:
                print(f"[Warning] CSV读取失败: {e}")
    
    if df_ind is None:
        raise ValueError(f"无有效的产业数据文件: {ind_path}")
    
    print(f"[Industry Loader] 初始加载: {len(df_ind)} 行")
    
    # 标准化列名
    df_ind = standardize_industry_columns(df_ind)
    
    # 数据类型转换
    if 'code' in df_ind.columns:
        df_ind['code'] = df_ind['code'].astype(str)
    
    # 保留必要列
    keep_cols = []
    if 'code' in df_ind.columns:
        keep_cols.append('code')
    
    # 产业代码列
    ind_code_cols = {'sw_code', 'ind_code', 'industry_code', 'sw1_code', 'industry'}
    for col in ind_code_cols:
        if col in df_ind.columns:
            keep_cols.append(col)
            break
    
    # 产业名称列
    ind_name_cols = {'sw_name', 'ind_name', 'industry_name', 'sw1_name', 'industry_name'}
    for col in ind_name_cols:
        if col in df_ind.columns:
            keep_cols.append(col)
            break
    
    if len(keep_cols) < 2:
        raise ValueError(f"无法找到产业代码和名称列。可用列: {list(df_ind.columns)}")
    
    # 去重和排序
    df_ind = df_ind[keep_cols].drop_duplicates().sort_values('code').reset_index(drop=True)
    
    # 标准化产业列名
    if 'sw_code' in df_ind.columns:
        df_ind = df_ind.rename(columns={'sw_code': 'ind_code'})
    elif 'industry_code' in df_ind.columns:
        df_ind = df_ind.rename(columns={'industry_code': 'ind_code'})
    elif 'sw1_code' in df_ind.columns:
        df_ind = df_ind.rename(columns={'sw1_code': 'ind_code'})
    elif 'industry' in df_ind.columns:
        df_ind = df_ind.rename(columns={'industry': 'ind_code'})
    
    if 'sw_name' in df_ind.columns:
        df_ind = df_ind.rename(columns={'sw_name': 'ind_name'})
    elif 'sw1_name' in df_ind.columns:
        df_ind = df_ind.rename(columns={'sw1_name': 'ind_name'})
    elif 'industry_name' in df_ind.columns:
        df_ind = df_ind.rename(columns={'industry_name': 'ind_name'})
    
    print(f"[Industry Loader] ✓ 加载完成: {len(df_ind)} 行")
    print(f"[Industry Loader] 股票数: {df_ind['code'].nunique()}")
    print(f"[Industry Loader] 产业数: {df_ind['ind_code'].nunique() if 'ind_code' in df_ind.columns else '?'}")
    
    return df_ind


def standardize_industry_columns(df):
    """
    标准化产业数据的列名
    
    code/symbol → 'code'
    {sw_code/ind_code/industry_code} → 'ind_code'
    {sw_name/ind_name/industry_name} → 'ind_name'
    """
    df = df.copy()
    
    # 标准化 code 列
    code_cols = {'code', 'symbol', 'stock_code', 'stk_code', 'Code', 'Symbol'}
    for col in code_cols:
        if col in df.columns and col != 'code':
            df = df.rename(columns={col: 'code'})
            break
    
    return df


def merge_industry_to_factor(df_factor, df_ind, on='code', how='left'):
    """
    将产业数据合并到因子数据
    
    参数:
        df_factor (pd.DataFrame): 因子数据
        df_ind (pd.DataFrame): 产业数据
        on (str): 合并键（一般为code）
        how (str): 合并方式 ('left', 'inner', 'outer')
    
    返回:
        pd.DataFrame: 合并后的数据
    """
    print(f"\n[Merger] 开始合并产业数据")
    print(f"[Merger] 因子数据形状: {df_factor.shape}")
    print(f"[Merger] 产业数据形状: {df_ind.shape}")
    print(f"[Merger] 合并方式: {how}, 合并键: {on}")
    
    # 检查必要列
    if on not in df_factor.columns:
        raise ValueError(f"因子数据缺少列: {on}")
    if on not in df_ind.columns:
        raise ValueError(f"产业数据缺少列: {on}")
    
    # 确定要合并的产业列
    merge_cols = [on]
    if 'ind_code' in df_ind.columns:
        merge_cols.append('ind_code')
    if 'ind_name' in df_ind.columns:
        merge_cols.append('ind_name')
    
    # 合并
    df_merged = pd.merge(df_factor, df_ind[merge_cols], on=on, how=how)
    
    print(f"[Merger] ✓ 合并完成")
    print(f"[Merger] 合并后数据形状: {df_merged.shape}")
    
    # 统计产业缺失情况
    if 'ind_code' in df_merged.columns:
        miss_rate = (df_merged['ind_code'].isna().sum() / len(df_merged) * 100)
        print(f"[Merger] 产业代码缺失率: {miss_rate:.2f}%")
    
    return df_merged


def main():
    """CLI入口"""
    parser = argparse.ArgumentParser(
        description="产业数据加载和合并器（申万行业分类2021版）",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        '--factor-path',
        type=str,
        required=True,
        help='因子数据文件路径 (必填, parquet或csv)'
    )
    
    parser.add_argument(
        '--ind-path',
        type=str,
        default=None,
        help=(
            '产业数据路径 (默认自动使用)\n'
            '  默认: E:\\1_basement\\quant_research\\data\\申万行业分类2021版'
        )
    )
    
    parser.add_argument(
        '--merge-method',
        type=str,
        choices=['left', 'inner', 'outer'],
        default='left',
        help='合并方式 (默认: left = 以因子数据为主)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='输出文件路径 (parquet 或 csv)'
    )
    
    parser.add_argument(
        '--preview-rows',
        type=int,
        default=10,
        help='预览行数 (默认: 10)'
    )
    
    args = parser.parse_args()
    
    try:
        # 验证因子文件
        if not os.path.exists(args.factor_path):
            raise FileNotFoundError(f"因子文件不存在: {args.factor_path}")
        
        # 加载因子数据
        print("[Main] 加载因子数据...")
        if args.factor_path.endswith('.parquet'):
            df_factor = pd.read_parquet(args.factor_path)
        elif args.factor_path.endswith('.csv'):
            df_factor = pd.read_csv(args.factor_path)
        else:
            raise ValueError(f"不支持的因子文件格式: {args.factor_path}")
        
        print(f"[Main] 因子数据加载完成: {len(df_factor)} 行")
        
        # 确保code列存在并标准化
        if 'code' not in df_factor.columns:
            code_cols = {'symbol', 'stock_code', 'stk_code'}
            for col in code_cols:
                if col in df_factor.columns:
                    df_factor = df_factor.rename(columns={col: 'code'})
                    break
        
        if 'code' not in df_factor.columns:
            raise ValueError(f"因子数据缺少code列。可用列: {list(df_factor.columns)}")
        
        df_factor['code'] = df_factor['code'].astype(str)
        
        # 加载产业数据
        print("\n[Main] 加载产业数据...")
        df_ind = load_industry_data(ind_path=args.ind_path)
        
        # 合并
        df_merged = merge_industry_to_factor(
            df_factor, df_ind,
            on='code',
            how=args.merge_method
        )
        
        # 预览
        print(f"\n{'='*60}")
        print(f"数据预览 (前 {args.preview_rows} 行):")
        print(f"{'='*60}")
        print(df_merged.head(args.preview_rows))
        
        # 保存
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        if args.output.endswith('.parquet'):
            df_merged.to_parquet(args.output, index=False)
        else:
            df_merged.to_csv(args.output, index=False)
        
        print(f"\n[Main] ✓ 已保存到: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
