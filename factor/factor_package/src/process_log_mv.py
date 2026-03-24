"""
市值因子处理脚本 (Market Cap Processing Script)

该脚本负责将分散的市值因子 (log_mv) 原始 CSV 数据聚合，
利用 FactorProcessor 进行截面去极值与标准化处理，
并最终以按日期命名的 Parquet 格式存储于 factor_package 目录中。

处理逻辑:
    1. 加载所有个股 CSV 文件。
    2. 按交易日进行截面归集。
    3. 调用 FactorProcessor.process_cross_section 进行MAD去极值与 Z-Score。
    4. 跳过中性化 (市值因子本身是中性化的基准)。
    5. 跳过缺失值填充 (按用户要求直接丢弃)。
"""

import os
import json
import pandas as pd
import glob
from tqdm import tqdm
from factor_processor import FactorProcessor

def process_log_mv():
    """
    执行市值因子的全量批处理
    """
    # 路径配置
    config_path = r'E:\1_basement\quant_research\factor\src\data_config.json'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    raw_dir = config['paths']['log_mv']
    out_dir = config['paths']['log_mv_processed']
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f"Created output directory: {out_dir}")

    # --- Phase 1: 加载并聚合所有个股 CSV ---
    print(f"Loading raw log_mv CSVs from: {raw_dir}")
    csv_files = glob.glob(os.path.join(raw_dir, "*.csv"))
    
    all_data_list = []
    error_count = 0
    for f in tqdm(csv_files, desc="Reading individual stock CSVs"):
        symbol = os.path.basename(f).replace(".csv", "")
        try:
            # 增加检查，防止某些特殊编码或损坏文件导致崩溃
            df_stock = pd.read_csv(f, engine='c', on_bad_lines='warn')
            if df_stock.empty:
                continue
            
            # 确保日期列名为 trade_date
            if 'trade_date' not in df_stock.columns:
                date_col = next((c for c in df_stock.columns if 'date' in c.lower()), None)
                if date_col:
                    df_stock.rename(columns={date_col: 'trade_date'}, inplace=True)
                else:
                    continue
            
            df_stock['symbol'] = symbol
            all_data_list.append(df_stock[['trade_date', 'symbol', 'log_mv']])
        except Exception as e:
            print(f"\n[Error] Failed to load {symbol} from {f}: {e}")
            error_count += 1

    print(f"\nLoaded {len(all_data_list)} stocks, encountered {error_count} errors.")

    if not all_data_list:
        print("No data found to process.")
        return

    full_df = pd.concat(all_data_list, ignore_index=True)
    full_df['trade_date'] = full_df['trade_date'].astype(str)
    
    # --- Phase 2: 按日期进行截面处理 ---
    processor = FactorProcessor(config_path=config_path)
    
    print(f"Processing cross-sections for {full_df['trade_date'].nunique()} dates...")
    
    # 使用 groupby 代替显式过滤，大幅提高速度
    for trade_date, df_cs in tqdm(full_df.groupby('trade_date'), desc="Cross-sectional processing"):
        try:
            # 去极值与标准化
            processed_cs = processor.process_cross_section(
                df_cs.copy(), 
                factor_col='log_mv',
                mv_col=None, 
                ind_col=None,
                do_neutralization=False,
                do_standardization=False,
                do_imputes=False # 缺失值直接舍去
            )
            
            if processed_cs is not None and not processed_cs.empty:
                # 保存为 Parquet
                output_file = os.path.join(out_dir, f"{trade_date}.parquet")
                processed_cs[['symbol', 'log_mv']].to_parquet(output_file, index=False)
        except Exception as e:
            print(f"\n[Error] Failed processing date {trade_date}: {e}")

    # --- Phase 3: 生成元数据说明 ---
    meta = {
        "factor_name": "log_mv",
        "description": "Cross-sectionally winsorized and standardized Market Cap (Log)",
        "processing_steps": [
            "MAD Winsorization (3.0x)",
            "Skipped Standardization (per instruction)",
            "Missing values dropped (no imputation)"
        ],
        "schema": ["symbol", "log_mv"],
        "source_dir": raw_dir,
        "processed_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(os.path.join(out_dir, "_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
        
    print(f"Successfully processed log_mv. Results saved to: {out_dir}")

if __name__ == "__main__":
    process_log_mv()
