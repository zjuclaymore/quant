"""
Announcement Gap 因子月度加工脚本
加工链:
    1. 提取每月最后一个交易日 (调仓日).
    2. 加载各股因子数据集，并仅保留调仓日数据.
    3. 合并为截面数据进行处理.
    4. 3MAD 去极值, 行业+市值中性化, Z-Score 截面标准化.
输出: factor_package/announcement_gap_processed_parquet/<YYYYMMDD>.parquet
"""

import json
import os
import warnings
import pandas as pd
from factor_processor import FactorProcessor

def get_rebalance_dates(cal_path: str) -> set:
    """
    自交易日历提取每月最后一个有效交易日
    """
    cal = pd.read_csv(cal_path, dtype=str)
    # 筛选开盘日
    trade_days = cal[cal["is_open"] == "1"]["cal_date"]
    td_dt = pd.to_datetime(trade_days)
    
    # 按月分组，取索引最大的（即最后一个交易日）
    monthly_dates = trade_days[td_dt.groupby(td_dt.dt.to_period('M')).idxmax()].tolist()
    return set(monthly_dates)

def batch_process():
    config_path = 'data_config.json'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    raw_dir    = config['paths']['announcement_gap_raw']
    output_dir = config['paths']['announcement_gap_processed']
    cal_path   = r"E:\1_basement\quant_research\data\交易日历\trade_calendar.csv"
    
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(cal_path):
        print(f"Error: Calendar file not found at {cal_path}")
        return

    target_dates = get_rebalance_dates(cal_path)
    print(f"提取了 {len(target_dates)} 个月度调仓决策日.")

    # 写 _meta.json
    meta = {
        'factor_name': 'announcement_gap',
        'description': 'Monthly Processed Announcement Gap (Winsorized, Neutralized, Standardized on rebalance dates)',
        'processing_steps': [
            'Filter by monthly rebalance dates',
            '3MAD Winsorization (Factor & Log Market Value)',
            'Neutralization (Industry + Log Market Value)',
            'Z-Score Standardization'
        ],
        'columns': ['symbol', 'announcement_gap'],
        'format': 'parquet',
    }
    with open(os.path.join(output_dir, '_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    processor = FactorProcessor(config_path)
    warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)

    summary = []

    print("\n[开始加载并横截面预处理月度数据...]")
    for date_str in sorted(target_dates):
        # 查找目标日期对应的截面 Parquet
        file_path = os.path.join(raw_dir, f"{date_str}.parquet")
        if not os.path.exists(file_path):
            continue
            
        try:
            df = pd.read_parquet(file_path)
            if df.empty or 'symbol' not in df.columns or 'announcement_gap' not in df.columns:
                continue
                
            # 转换为 process_day 期望的 Series: index=symbol, values=factor
            factor_series = df.set_index('symbol')['announcement_gap']
            
            # 截面加工 (fill_na=False 针对事件选股因子更科学)
            processed_series = processor.process_day(factor_series, date_str, fill_na=False)

            if not processed_series.empty:
                save_df = processed_series.reset_index()
                save_df.columns = ['symbol', 'announcement_gap']
                
                output_path = os.path.join(output_dir, f"{date_str}.parquet")
                save_df.to_parquet(output_path, index=False)

                mean = processed_series.mean()
                std  = processed_series.std()
                print(f'[{date_str}] {len(processed_series)} stocks  mean={mean:.4f}  std={std:.4f}')
                summary.append({
                    'date': date_str,
                    'count': len(processed_series),
                    'mean': float(mean),
                    'std': float(std)
                })
        except Exception as e:
             # print(f"[{date_str}] Error processing: {e}")
             pass

    with open(os.path.join(output_dir, 'processing_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f'\nDone. {len(summary)} dates processed -> {output_dir}')

if __name__ == '__main__':
    batch_process()
