r"""
因子市值行业中性化处理器 (Market Cap & Industry Neutralization with CLI)
功能: 对因子进行市值和行业中性化处理
方法: 截面回归残差法
"""
import os
import pandas as pd
import numpy as np
import argparse
import sys
from datetime import datetime


def _to_year_month(s: pd.Series) -> pd.Series:
    """将日期列统一映射到 YYYYMM（Int64）。"""
    if np.issubdtype(s.dtype, np.number):
        x = pd.to_numeric(s, errors='coerce').astype('Int64')
        return (x // 100).astype('Int64')
    dt = pd.to_datetime(s, errors='coerce')
    return (dt.dt.year * 100 + dt.dt.month).astype('Int64')


def _impute_mv_ind_by_stock_month(df: pd.DataFrame, mv_col: str, ind_col: str, date_col: str) -> pd.DataFrame:
    """按股票+月份补齐市值/行业缺失值。"""
    if 'code' not in df.columns or date_col not in df.columns:
        print("[Neutral] [Impute] 缺少 code 或 date 列，跳过按股票-月份补齐")
        return df

    out = df.copy()
    out['_year_month_tmp_'] = _to_year_month(out[date_col])

    # lncap 优先用股票-月份均值补齐
    if mv_col in out.columns:
        out[mv_col] = out.groupby(['code', '_year_month_tmp_'])[mv_col].transform(
            lambda x: x.fillna(x.mean())
        )
        # 兜底: 当月全市场截面均值
        out[mv_col] = out.groupby('_year_month_tmp_')[mv_col].transform(lambda x: x.fillna(x.mean()))

    # 行业优先用股票-月份众数补齐（若无众数则继续兜底）
    if ind_col in out.columns:
        def _fill_mode(x: pd.Series) -> pd.Series:
            m = x.mode(dropna=True)
            if m.empty:
                return x
            return x.fillna(m.iloc[0])

        out[ind_col] = out.groupby(['code', '_year_month_tmp_'])[ind_col].transform(_fill_mode)
        # 兜底: 当月全市场截面众数
        out[ind_col] = out.groupby('_year_month_tmp_')[ind_col].transform(_fill_mode)

    out = out.drop(columns=['_year_month_tmp_'])
    return out


def market_cap_industry_neutralization(df, factor_col, mv_col='lncap', ind_col='ind_code', groupby_col='date'):
    """
    市值行业中性化处理
    
    参数:
        df (pd.DataFrame): 输入数据，包含 code, date, {factor}, {mv}, {ind} 等列
        factor_col (str): 因子列名
        mv_col (str): 市值列名（默认 lncap）
        ind_col (str): 行业列名（默认 ind_code）
        groupby_col (str): 分组列（默认 date，按交易日分组）
    
    返回:
        pd.DataFrame: 中性化后的数据
    """
    print(f"\n[Neutral] 开始市值行业中性化")
    print(f"[Neutral] 因子列: {factor_col}")
    print(f"[Neutral] 市值列: {mv_col}, 行业列: {ind_col}")
    print(f"[Neutral] 分组列: {groupby_col}")
    
    df = df.copy()
    factor_col_new = f"{factor_col}_neutral"
    
    # 检查必要列
    if mv_col not in df.columns:
        raise ValueError(f"市值列不存在: {mv_col}")
    if ind_col not in df.columns:
        raise ValueError(f"行业列不存在: {ind_col}")
    
    # 初始统计
    init_mean = df[factor_col].mean()
    init_std = df[factor_col].std()
    
    print(f"[Neutral] 中性化前统计:")
    print(f"  mean: {init_mean:.6f}, std: {init_std:.6f}")

    mv_missing_before = int(df[mv_col].isna().sum())
    ind_missing_before = int(df[ind_col].isna().sum())
    df = _impute_mv_ind_by_stock_month(df, mv_col=mv_col, ind_col=ind_col, date_col=groupby_col)
    mv_missing_after = int(df[mv_col].isna().sum())
    ind_missing_after = int(df[ind_col].isna().sum())
    print(f"[Neutral] [Impute] {mv_col} 缺失: {mv_missing_before} -> {mv_missing_after}")
    print(f"[Neutral] [Impute] {ind_col} 缺失: {ind_missing_before} -> {ind_missing_after}")
    
    try:
        from sklearn.linear_model import LinearRegression
    except ImportError:
        raise ImportError("需要安装 scikit-learn: pip install scikit-learn")
    
    neutral_series = pd.Series(np.nan, index=df.index, dtype="float64")
    processed_count = 0
    error_count = 0
    
    # 按分组列进行截面回归
    if groupby_col and groupby_col in df.columns:
        print(f"[Neutral] 按 {groupby_col} 进行截面中性化处理...")
        
        for group_key, group_df in df.groupby(groupby_col):
            try:
                # 提取该截面的数据
                y = group_df[factor_col].values.reshape(-1, 1)
                
                # 准备自变量：市值 + 行业哑变量
                X_data = [group_df[mv_col].values.reshape(-1, 1)]
                
                # 行业哑变量
                industries = pd.get_dummies(group_df[ind_col], drop_first=True)
                X_data.append(industries.values)
                
                X = np.hstack(X_data)
                
                # 检查缺失值
                valid_mask = ~(np.isnan(y.flatten()) | np.isnan(X).any(axis=1))
                if valid_mask.sum() < 3:
                    neutral_series.loc[group_df.index] = np.nan
                    error_count += 1
                    continue
                
                X_valid = X[valid_mask]
                y_valid = y[valid_mask].flatten()
                
                # 回归计算残差
                model = LinearRegression()
                model.fit(X_valid, y_valid)
                residuals = y_valid - model.predict(X_valid)
                
                # 映射回完整数据
                residuals_full = np.full(len(group_df), np.nan)
                residuals_full[valid_mask] = residuals
                neutral_series.loc[group_df.index] = residuals_full
                processed_count += 1
                
            except Exception as e:
                print(f"[Warning] {groupby_col}={group_key} 处理失败: {e}")
                neutral_series.loc[group_df.index] = np.nan
                error_count += 1

        df[factor_col_new] = neutral_series
    
    else:
        # 全局回归
        print("[Neutral] 进行全局中性化处理...")
        y = df[factor_col].values.reshape(-1, 1)
        X_data = [df[mv_col].values.reshape(-1, 1)]
        industries = pd.get_dummies(df[ind_col], drop_first=True)
        X_data.append(industries.values)
        X = np.hstack(X_data)
        
        valid_mask = ~(np.isnan(y.flatten()) | np.isnan(X).any(axis=1))
        X_valid = X[valid_mask]
        y_valid = y[valid_mask].flatten()
        
        model = LinearRegression()
        model.fit(X_valid, y_valid)
        residuals = y_valid - model.predict(X_valid)
        
        residuals_full = np.full(len(df), np.nan)
        residuals_full[valid_mask] = residuals
        df[factor_col_new] = residuals_full
        processed_count = 1
    
    # 处理后统计
    after_mean = df[factor_col_new].mean()
    after_std = df[factor_col_new].std()
    after_min = df[factor_col_new].min()
    after_max = df[factor_col_new].max()
    miss_count = df[factor_col_new].isna().sum()
    
    print(f"\n[Neutral] 中性化后统计:")
    print(f"  mean: {after_mean:.6f}, std: {after_std:.6f}")
    print(f"  min: {after_min:.6f}, max: {after_max:.6f}")
    print(f"  缺失值: {miss_count} ({miss_count/len(df)*100:.2f}%)")
    print(f"[Neutral] 处理成功: {processed_count}个截面, 处理失败: {error_count}个")
    
    # 删除原列，保留中性化列
    df = df.drop(columns=[factor_col])
    df = df.rename(columns={factor_col_new: factor_col})
    
    return df


def main():
    """CLI入口"""
    parser = argparse.ArgumentParser(
        description="因子市值行业中性化处理器",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='输入因子文件路径 (parquet或csv，需包含 lncap 和 ind_code 列)'
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
        help='输出文件路径 (默认: 原路径+_neutral后缀)'
    )
    
    parser.add_argument(
        '--mv-col',
        type=str,
        default='lncap',
        help='市值列名 (默认: lncap)'
    )
    
    parser.add_argument(
        '--ind-col',
        type=str,
        default='ind_code',
        help='行业列名 (默认: ind_code)'
    )
    
    parser.add_argument(
        '--groupby',
        type=str,
        default='date',
        help='分组列 (默认: date 按交易日截面中性化)'
    )
    
    parser.add_argument(
        '--enable',
        type=str,
        choices=['yes', 'no'],
        default='yes',
        help='是否启用中性化处理 (默认: yes)'
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
            print("[Neutral] ✗ 已禁用中性化处理，跳过")
            return 0
        
        # 加载因子数据
        print("[Neutral] 加载输入文件...")
        if args.input.endswith('.parquet'):
            df = pd.read_parquet(args.input)
        elif args.input.endswith('.csv'):
            df = pd.read_csv(args.input)
        else:
            raise ValueError(f"不支持的文件格式: {args.input}")
        
        print(f"[Neutral] 输入数据: {len(df)} 行")
        
        # 检查必要列
        for col in [args.factor_col, args.mv_col, args.ind_col]:
            if col not in df.columns:
                raise ValueError(f"列不存在: {col}")
        
        # 市值行业中性化
        df_processed = market_cap_industry_neutralization(
            df, args.factor_col,
            mv_col=args.mv_col,
            ind_col=args.ind_col,
            groupby_col=args.groupby
        )
        
        # 确定输出路径
        if args.output is None:
            base, ext = os.path.splitext(args.input)
            args.output = f"{base}_neutral{ext}"
        
        # 保存
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        if args.output.endswith('.parquet'):
            df_processed.to_parquet(args.output, index=False)
        else:
            df_processed.to_csv(args.output, index=False)
        
        print(f"\n[Neutral] [+] 已保存到: {args.output}")
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
