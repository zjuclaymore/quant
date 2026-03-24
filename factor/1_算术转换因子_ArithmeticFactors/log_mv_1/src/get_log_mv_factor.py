
import tushare as ts
import pandas as pd
import numpy as np
import time
import os
import datetime
from tqdm import tqdm

# Configuration
TOKEN = '2d4f555869182905bfd48ce1fd0f649015f2bf10b3ef4a7e558573bb'
TRADE_CALENDAR_PATH = r'E:\1_basement\ml\utility\trade_calendar\trade_calendar_1990_2025_processed.csv'
OUTPUT_DIR = r'E:\1_basement\ml\factors\log_mv\output'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'log_mv_2000_2025.csv')
LOG_FILE = os.path.join(OUTPUT_DIR, 'download_log.txt')
START_DATE = '20000101'
END_DATE = '20251231'

# Rate limiting settings
CALLS_PER_MINUTE = 80  # Conservative limit
SLEEP_TIME = 60 / CALLS_PER_MINUTE

def init_tushare():
    ts.set_token(TOKEN)
    return ts.pro_api()

def load_trade_calendar():
    df = pd.read_csv(TRADE_CALENDAR_PATH)
    # Ensure correct date format
    df['cal_date'] = df['cal_date'].astype(str).str.replace('-', '')
    
    # Filter for A-share trading days and range
    # Assuming 'SSE' covers the trading days logic for A-share broadly or just use the dates where is_open=1
    # We will just pick unique trading days sorted
    trading_days = df[(df['is_open'] == 1) & 
                      (df['cal_date'] >= START_DATE) & 
                      (df['cal_date'] <= END_DATE)]['cal_date'].unique()
    return sorted(trading_days)

def get_processed_dates():
    if not os.path.exists(OUTPUT_FILE):
        return set()
    try:
        # Check if file has header, if not, careful
        # We assume we write header on creation
        df = pd.read_csv(OUTPUT_FILE, usecols=['trade_date'])
        return set(df['trade_date'].astype(str).unique())
    except Exception as e:
        print(f"Error reading existing file: {e}")
        return set()

def log_message(msg):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{timestamp}] {msg}"
    tqdm.write(full_msg)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(full_msg + '\n')

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    pro = init_tushare()
    trade_dates = load_trade_calendar()
    processed_dates = get_processed_dates()
    
    # Needs to write header if new file
    header = not os.path.exists(OUTPUT_FILE)
    
    log_message(f"Total trading days to process: {len(trade_dates)}")
    log_message(f"Already processed: {len(processed_dates)}")
    
    dates_to_process = [d for d in trade_dates if d not in processed_dates]
    log_message(f"Remaining days: {len(dates_to_process)}")
    
    if not dates_to_process:
        log_message("All dates processed. Exiting.")
        return

    # To be safe against crashes, we append to file directly
    for i, trade_date in enumerate(tqdm(dates_to_process, desc="Downloading")):
        try:
            start_time = time.time()
            
            # Fetch daily_basic
            # fields: ts_code, trade_date, close, turn... we need total_mv
            # total_mv: 总市值 （万元）
            df = pro.daily_basic(ts_code='', trade_date=trade_date, fields='ts_code,trade_date,total_mv')
            
            if df is not None and not df.empty:
                # Calculate log_mv
                # total_mv is in 10k CNY.
                # Usually log market value uses the actual value or consistent unit.
                # We will stick to log(total_mv * 10000) or just log(total_mv).
                # User asked for "total market value's log". Usually implies actual value.
                # But let's check if log(total_mv) is standard. 
                # If total_mv = 1 (10k), log(10000) = 9.21. log(1) = 0.
                # I will calculate log of the raw value provided by tushare (which is 10k) * 10000 to be precise magnitude, 
                # or just keep it simple. Let's precise: log(total_mv * 10000).
                # Handle zeros or negatives just in case (though MV shouldn't be <= 0)
                
                # Filter valid MV
                df = df[df['total_mv'] > 0].copy()
                df['log_mv'] = np.log(df['total_mv'] * 10000)
                
                # Select columns
                out_df = df[['trade_date', 'ts_code', 'log_mv']]
                
                # Append to CSV
                out_df.to_csv(OUTPUT_FILE, mode='a', header=header, index=False)
                
                # Only write header once
                header = False
                
                log_message(f"Processed {trade_date}: {len(out_df)} records.")
            else:
                log_message(f"No data for {trade_date}")
            
            # Rate limiting
            elapsed = time.time() - start_time
            if elapsed < SLEEP_TIME:
                time.sleep(SLEEP_TIME - elapsed)
                
        except Exception as e:
            log_message(f"Failed to process {trade_date}: {e}")
            # Optional: break or continue? Continue is better but huge errors log might fill disk.
            # If it's a rate limit error, we might want to pause longer.
            time.sleep(5) # Wait a bit on error

if __name__ == "__main__":
    main()
