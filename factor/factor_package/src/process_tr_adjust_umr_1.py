"""
tr_adjust_umr_1 因子批处理脚本

处理链:
    1. 丢弃缺失值 (fill_na=False)
    2. 3MAD 去极值 (因子 & 对数市值)
    3. 行业 + 市值中性化
    4. Z-Score 截面标准化
输出: factor_package/tr_adjust_umr_1_processed_parquet/<YYYYMMDD>.parquet
"""

import json
import os
import warnings
import pandas as pd
from factor_processor import FactorProcessor


def batch_process():
    config_path = 'data_config.json'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    raw_dir    = config['paths']['tr_adjust_umr_1_raw']
    output_dir = config['paths']['tr_adjust_umr_1_processed']
    os.makedirs(output_dir, exist_ok=True)

    # 写 _meta.json
    meta = {
        'factor_name': 'tr_adjust_umr_1',
        'description': 'Processed tr_adjust_umr_1 factor (Winsorized, Neutralized, Standardized)',
        'processing_steps': [
            'Drop missing values (fill_na=False)',
            '3MAD Winsorization (Factor & Log Market Value)',
            'Neutralization (Industry + Log Market Value)',
            'Z-Score Standardization',
        ],
        'columns': ['symbol', 'tr_adjust_umr_1'],
        'format': 'parquet',
    }
    with open(os.path.join(output_dir, '_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    processor = FactorProcessor(config_path)
    warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)

    files = sorted(f for f in os.listdir(raw_dir) if f.endswith('.parquet'))

    log_mv_processed = config['paths'].get('log_mv_processed')
    if log_mv_processed and os.path.isdir(log_mv_processed):
        mv_dates = {os.path.splitext(f)[0] for f in os.listdir(log_mv_processed) if f.endswith('.parquet')}
        files = [f for f in files if os.path.splitext(f)[0] in mv_dates]
        print(f'Found {len(files)} files to process (intersected with log_mv_processed dates).')
    else:
        print(f'Found {len(files)} files to process.')

    summary = []

    for filename in files:
        trade_date = filename.split('.')[0]
        file_path  = os.path.join(raw_dir, filename)

        try:
            df = pd.read_parquet(file_path)

            if 'symbol' not in df.columns:
                print(f'[{trade_date}] symbol column not found, skip.')
                continue

            # symbol 格式归一化 (兼容纯数字代码)
            def format_symbol(s):
                s = str(s).zfill(6)
                if '.' in s:
                    return s
                return s + '.SH' if s.startswith('6') else s + '.SZ'

            df['symbol'] = df['symbol'].apply(format_symbol)

            factor_cols = [c for c in df.columns if c != 'symbol']
            if not factor_cols:
                continue
            factor_name   = factor_cols[0]
            factor_series = df.set_index('symbol')[factor_name]

            processed_series = processor.process_day(factor_series, trade_date, fill_na=False)

            if not processed_series.empty:
                output_path  = os.path.join(output_dir, f'{trade_date}.parquet')
                processed_df = processed_series.reset_index()
                processed_df.columns = ['symbol', 'tr_adjust_umr_1']
                processed_df.to_parquet(output_path, index=False)

                mean = processed_series.mean()
                std  = processed_series.std()
                print(f'[{trade_date}] {len(processed_series)} stocks  mean={mean:.4f}  std={std:.4f}')
                summary.append({
                    'date': str(trade_date),
                    'count': int(len(processed_series)),
                    'mean': float(mean) if pd.notna(mean) else None,
                    'std': float(std) if pd.notna(std) else None,
                })

        except Exception as e:
            print(f'[{trade_date}] Error: {e}')

    with open(os.path.join(output_dir, 'processing_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f'\nDone. {len(summary)} dates processed -> {output_dir}')


if __name__ == '__main__':
    batch_process()
