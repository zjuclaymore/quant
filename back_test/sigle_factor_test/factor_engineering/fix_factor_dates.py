r"""
信号日期对齐脚本 (Signal Date Alignment Script)

功能:
    修正因子数据中出现的非交易日日期（如普通的自然月末）。
    逻辑是将日期平移到当前日期之前的最近一个有效交易日（即股票池中存在的日期）。
    这通常用于将自然月度因子对齐到月末最后一个交易日。

输入:
    - 因子文件: E:\1_basement\quant_research\back_test\sigle_factor_test\function_breakdown\output\factor_A021_200804_202410.parquet
    - 股票池文件: E:\1_basement\quant_research\back_test\sigle_factor_test\function_breakdown\stock_pool\stock_pool_base.parquet (提供有效交易日序列)

输出:
    - 修正后的因子文件: E:\1_basement\quant_research\back_test\sigle_factor_test\function_breakdown\output\factor_A021_200804_202410_fixed.parquet

用法:
    uv run python fix_factor_dates.py
"""
import pandas as pd
import os
from bisect import bisect_right

def fix_factor_dates():
    """
    主函数: 加载数据，执行日期对齐逻辑，并保存结果。
    """
    # 路径配置
    factor_path = r'E:\1_basement\quant_research\back_test\sigle_factor_test\function_breakdown\output\factor_A021_200804_202410.parquet'
    stock_pool_path = r'E:\1_basement\quant_research\back_test\sigle_factor_test\function_breakdown\stock_pool\stock_pool_base.parquet'
    output_path = r'E:\1_basement\quant_research\back_test\sigle_factor_test\function_breakdown\output\factor_A021_200804_202410_fixed.parquet'

    print(f"Loading factor data from: {factor_path}")
    df_factor = pd.read_parquet(factor_path)
    
    print(f"Loading trading dates from: {stock_pool_path}")
    df_stock_pool = pd.read_parquet(stock_pool_path)

    # 提取所有有效交易日并排序 (确保是 int64 格式以匹配因子日期)
    valid_dates = sorted(df_stock_pool['date'].unique())
    valid_dates_set = set(valid_dates)

    # 提取因子中的唯一日期
    factor_dates = sorted(df_factor['date'].unique())

    # 构建日期映射表
    date_mapping = {}
    modified_count = 0
    
    for d in factor_dates:
        if d in valid_dates_set:
            date_mapping[d] = d
        else:
            # 找到小于或等于 d 的最大交易日
            # 使用 bisect_right 查找索引，idx-1 即为所需元素
            idx = bisect_right(valid_dates, d)
            if idx > 0:
                fixed_date = valid_dates[idx - 1]
                date_mapping[d] = fixed_date
                modified_count += 1
                print(f"  Mapping non-trading day {d} -> {fixed_date}")
            else:
                print(f"  Warning: No valid trading date found for {d}, keeping original.")
                date_mapping[d] = d

    # 应用映射
    df_factor['date_original'] = df_factor['date'] # 保留原日期以供参考(可选)
    df_factor['date'] = df_factor['date'].map(date_mapping)
    
    # 移除临时列并保存
    df_factor = df_factor.drop(columns=['date_original'])
    
    print(f"\nCorrection Summary:")
    print(f"  Total unique dates in factor data: {len(factor_dates)}")
    print(f"  Dates modified: {modified_count}")
    
    # 保存结果
    df_factor.to_parquet(output_path, index=False)
    print(f"\nSaved fixed factor data to: {output_path}")

if __name__ == "__main__":
    fix_factor_dates()
