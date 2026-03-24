import pandas as pd
import tushare as ts
import os
import time
import sys
from tqdm import tqdm
import numpy as np

# Config
TOKEN = '2d4f555869182905bfd48ce1fd0f649015f2bf10b3ef4a7e558573bb'
START_DATE_DATA = '20190101'
END_DATE_DATA = '20251231'
START_DATE_OUTPUT = '20200101'
END_DATE_OUTPUT = '20251231'

BASE_DIR = r'E:\1_basement\ml\factors\valuation_percentile_ttm_1'
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
SRC_DIR = os.path.join(BASE_DIR, 'src')
CALENDAR_PATH = r'E:\1_basement\ml\utility\trade_calendar\trade_calendar_1990_2025_processed.csv'

# Setup
ts.set_token(TOKEN)
pro = ts.pro_api()

def get_trade_cal():
    print(f"Loading calendar from {CALENDAR_PATH}")
    df = pd.read_csv(CALENDAR_PATH)
    # Ensure standard format (remove hyphens for filtering but keep datetime for ops)
    # The file has 'cal_date' like '1990-12-19'.
    df['cal_date'] = pd.to_datetime(df['cal_date'])
    return df

def get_daily_basic(start_date, end_date):
    # Check full cache
    cache_file = os.path.join(DATA_DIR, f'daily_basic_{start_date}_{end_date}.pkl')
    if os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}")
        return pd.read_pickle(cache_file)
    
    print(f"Downloading daily_basic from {start_date} to {end_date}")
    
    # Create temp dir for chunks
    temp_dir = os.path.join(DATA_DIR, 'temp_daily')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Get trade days
    cal_df = get_trade_cal()
    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)
    
    trade_days = cal_df[(cal_df['cal_date'] >= start_dt) & 
                        (cal_df['cal_date'] <= end_dt) & 
                        (cal_df['is_open'] == 1)]['cal_date'].sort_values().dt.strftime('%Y%m%d').tolist()
    
    print(f"Total trade days to fetch: {len(trade_days)}")
    
    # Fetch Loop
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    def fetch_one(date):
        # Check if daily file exists
        daily_file = os.path.join(temp_dir, f"{date}.pkl")
        if os.path.exists(daily_file):
            return "Skipped"

        # Retry logic
        max_retries = 5
        success = False
        for i in range(max_retries):
            try:
                # Limit fields to save bandwidth/memory
                time.sleep(0.1) # slight delay to stagger
                # Re-init pro in thread? No, pro is thread-safe usually or just REST
                # But safer to create local instance if needed. Global pro works for Tushare.
                df = pro.daily_basic(trade_date=date, fields='ts_code,trade_date,pe_ttm')
                # Save immediately
                df.to_pickle(daily_file)
                return "Success"
            except Exception as e:
                time.sleep(1 + i) # Backoff
        
        return f"Failed: {date}"

    print("Starting parallel download...")
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_one, date): date for date in trade_days}
        
        for future in tqdm(as_completed(futures), total=len(trade_days), desc="Fetching parallel"):
            res = future.result()
            if res.startswith("Failed"):
                print(res)
            
    # Combine
    print("Combining daily files...")
    all_chunks = []
    chunk_files = sorted(os.listdir(temp_dir))
    for f in tqdm(chunk_files, desc="Loading chunks"):
        if f.endswith('.pkl'):
             all_chunks.append(pd.read_pickle(os.path.join(temp_dir, f)))
             
    if not all_chunks:
        print("No temp data found!")
        return pd.DataFrame()

    full_df = pd.concat(all_chunks, ignore_index=True)
    
    # Save full cache
    print(f"Saving combined cache to {cache_file}")
    full_df.to_pickle(cache_file)
    
    # Clean up temp? Maybe keep until confirmed success
    return full_df

def process_vp():
    # 1. Get Data
    df = get_daily_basic(START_DATE_DATA, END_DATE_DATA)
    if df.empty:
        print("Data is empty. Exiting.")
        return

    print("Processing data...")
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    
    # Ensure numeric
    df['pe_ttm'] = pd.to_numeric(df['pe_ttm'], errors='coerce')
    
    # Sort for rolling
    df = df.sort_values(['ts_code', 'trade_date'])
    
    # 2. Calculate Inverse PE
    # 1/PE. If PE is 0, results in inf. handled as NaN or Inf.
    with np.errstate(divide='ignore'):
        df['inv_pe'] = 1 / df['pe_ttm']
    
    # Replace inf/-inf with NaN for ranking stability? 
    # Or just let rank handle it (Inf is highest, -Inf is lowest).
    # Ideally PE=0 is undefined (or infinity PE?). Tushare PE TTM shouldn't be exactly 0 often.
    # If 0, 1/0 is Inf.
    df['inv_pe'] = df['inv_pe'].replace([np.inf, -np.inf], np.nan)

    # 3. Rolling Percentile
    print("Calculating rolling percentile (this may take a moment)...")
    
    # Using groupby + rolling
    # min_periods=1 ensures we get a rank even if we don't have full 244 days yet (start of 2019/2020)
    # But usually "past 244 days" implies a full window. 
    # However, for 2020-01 start, we have 2019 history (approx 244 trade days in a year).
    # So by Jan 2020 we should have near full window.
    # We will use min_periods=200 to ensure robust stats, or keep it strict. 
    # Let's use min_periods=120 (approx 6 months) to be safe for new listings?
    # User said: "calculate ... in past 244 trading days".
    # I'll stick to a standard valid window or allow partial. 
    # Let's set min_periods=200 to ensure we rely on roughly a year of data.
    
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=244) # No, we want backward. standard rolling is backward.
    
    # GroupBy Rolling is efficient
    # We transform just the inv_pe column
    # pct=True computes percentile (0 to 1).
    
    # Note: rolling().rank() might be slow on large DF.
    # An alternative is `lambda x: x.rank(pct=True).iloc[-1]` inside rolling? No, rolling.rank is optimized.
    
    df['vp'] = df.groupby('ts_code')['inv_pe'].transform(lambda x: x.rolling(244, min_periods=100).rank(pct=True))
    
    # 4. Handle Loss Making Companies
    # "Current loss -> VP=0"
    # Loss is defined as PE_TTM < 0
    mask_loss = df['pe_ttm'] < 0
    df.loc[mask_loss, 'vp'] = 0
    
    # Fill NaN VP with ...? 
    # If NaN (e.g. not enough history), it remains NaN.
    # User didn't specify. We leave as NaN or filtered out?
    # We'll valid rows output.
    
    # 5. Filter Output Dates (Month Ends)
    print("Filtering for month-end dates...")
    cal_df = get_trade_cal()
    
    target_cal = cal_df[(cal_df['cal_date'] >= pd.Timestamp(START_DATE_OUTPUT)) & 
                        (cal_df['cal_date'] <= pd.Timestamp(END_DATE_OUTPUT)) &
                        (cal_df['is_open'] == 1)].copy()
    
    # Create Year-Month key
    target_cal['ym'] = target_cal['cal_date'].dt.to_period('M')
    
    # Get last trade date per month
    month_ends = target_cal.groupby('ym')['cal_date'].max()
    month_end_dates = month_ends.tolist()
    
    print(f"Found {len(month_end_dates)} month-end trading dates.")
    
    final_output = df[df['trade_date'].isin(month_end_dates)].copy()
    
    # Select columns
    final_output = final_output[['ts_code', 'trade_date', 'vp']]
    
    # Sort
    final_output = final_output.sort_values(['trade_date', 'ts_code'])
    
    # Check output
    if final_output.empty:
        print("Warning: Final output is empty!")
    else:
        print(f"Final output shape: {final_output.shape}")
        sample = final_output.iloc[0]
        print(f"Sample: {sample.to_dict()}")

    # Save
    out_file = os.path.join(OUTPUT_DIR, 'vp_ttm_2020_2025.csv')
    final_output.to_csv(out_file, index=False)
    print(f"Saved result to {out_file}")

if __name__ == "__main__":
    process_vp()
