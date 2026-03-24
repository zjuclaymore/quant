import pandas as pd
import numpy as np
import os
import logging
from tqdm import tqdm
import concurrent.futures

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ivol_calculation.log"),
        logging.StreamHandler()
    ]
)

FACTOR_FILE = r"E:\1_basement\ml\factors\1_minus_r\output\fivefactor_daily.csv"
STOCK_DATA_DIR = r"E:\1_basement\ml\data\中国A股日行情_AShareEODPrices"
OUTPUT_DIR = r"E:\1_basement\ml\factors\1_minus_r\output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "ivol_monthly.csv")

def load_factors():
    logging.info("Loading Fama-French factors...")
    df = pd.read_csv(FACTOR_FILE)
    df['trade_date'] = df['trddy'].astype(str).str.replace('-', '')
    cols_to_use = ['mkt_rf', 'smb', 'hml']
    for col in cols_to_use:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.set_index('trade_date')[cols_to_use]

def read_single_pickle(args):
    date_str, file_path = args
    try:
        df = pd.read_pickle(file_path)
        if 'Wind代码' in df.columns and '涨跌幅(%)' in df.columns:
            # Minimal data
            subset = df[['Wind代码', '涨跌幅(%)']].copy()
            subset.columns = ['ts_code', 'pct_chg']
            subset['trade_date'] = date_str
            return subset
    except Exception as e:
        return None
    return None

def load_all_stock_data(file_map):
    logging.info("Loading stock data files into memory...")
    
    tasks = [(date, path) for date, path in file_map.items()]
    # Use only recent years to save time? Or full history.
    # User didn't specify start date, but previous tasks implied 2000+.
    # Let's load everything, parallel reading should be fast.
    
    results = []
    # Using ThreadPoolExecutor because IO bound
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(read_single_pickle, task): task for task in tasks}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(tasks), desc="Reading pickles"):
            res = future.result()
            if res is not None:
                results.append(res)
                
    logging.info("Concatenating daily data...")
    if not results:
        return pd.DataFrame()
        
    full_df = pd.concat(results, ignore_index=True)
    return full_df

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. Load Factors
    factors_df = load_factors()
    
    # 2. Scan Files
    logging.info("Scanning stock files...")
    files = [f for f in os.listdir(STOCK_DATA_DIR) if f.endswith('.pickle')]
    file_map = {}
    for f in files:
        if f[:-7].isdigit() and len(f[:-7]) == 8:
            file_map[f[:-7]] = os.path.join(STOCK_DATA_DIR, f)
            
    sorted_dates = sorted(file_map.keys())
    
    # 3. Load Sample Data (Parallel)
    stock_df = load_all_stock_data(file_map)
    if stock_df.empty:
        logging.error("No stock data loaded.")
        return

    # 4. Pivot to Wide Format (Index=Date, Columns=Stock, Values=Return)
    logging.info("Pivoting data to wide format...")
    # Ensure correct types
    stock_df['pct_chg'] = pd.to_numeric(stock_df['pct_chg'], errors='coerce')
    
    # Check for duplicates
    # Sometimes duplicates exist? Drop them.
    stock_df = stock_df.drop_duplicates(subset=['trade_date', 'ts_code'])
    
    returns_matrix = stock_df.pivot(index='trade_date', columns='ts_code', values='pct_chg')
    returns_matrix = returns_matrix.sort_index()
    
    # Align factors and returns
    common_dates = returns_matrix.index.intersection(factors_df.index)
    returns_matrix = returns_matrix.loc[common_dates]
    factors_df = factors_df.loc[common_dates]
    
    logging.info(f"Aligned Data: {len(common_dates)} days, {returns_matrix.shape[1]} stocks.")
    
    # 5. Identify Month Ends
    df_dates = pd.DataFrame({'date': common_dates})
    df_dates['dt'] = pd.to_datetime(df_dates['date'])
    df_dates['month'] = df_dates['dt'].dt.to_period('M')
    month_ends = df_dates.groupby('month')['date'].max().values
    
    logging.info(f"Target Month-Ends: {len(month_ends)}")
    
    # 6. Calculate 1-R^2 using Rolling Window (Matrix Algebra)
    final_results = []
    
    # Limit to numeric arrays for speed
    all_dates = common_dates.to_numpy()
    
    for i, date in enumerate(tqdm(all_dates, desc="Calculating")):
        if date not in month_ends:
            continue
            
        # Get window indices: [i-19, i] inclusive (20 days)
        if i < 19:
            continue
            
        window_dates = all_dates[i-19 : i+1]
        
        # Get data slice
        # factors: (20, 3)
        F = factors_df.loc[window_dates].values
        # Add constant
        F = np.column_stack([np.ones(len(F)), F]) # (20, 4)
        
        # returns: (20, N)
        R = returns_matrix.loc[window_dates].values
        
        # We need to regress each column of R on F.
        # Handle NaNs: Ideally we want OLS only on valid data points.
        # Vectorized OLS with NaNs is tricky.
        # BUT: most stocks trade most days.
        # Fast approach: 
        # Calculate R^2 for each column independently locally using numpy loop (still fast in C)
        # OR handle purely valid columns.
        
        # Let's iterate over stocks (columns) where count is sufficient.
        # This inner loop is still 3000 items but much faster than pandas overhead.
        
        valid_counts = np.sum(~np.isnan(R), axis=0) # (N,)
        valid_stocks_idx = np.where(valid_counts >= 15)[0]
        
        if len(valid_stocks_idx) == 0:
            continue
            
        # Subset to valid stocks
        R_subset = R[:, valid_stocks_idx] # (20, M)
        stock_cols = returns_matrix.columns[valid_stocks_idx]
        
        # We can solve Y = X * B
        # B = (X'X)^-1 X'Y
        # But X is constant for all stocks!
        # So pseudo-inverse of X is constant.
        # P = (X'X)^-1 X'
        # B = P * Y
        
        try:
            # P_inv = np.linalg.pinv(F) # (4, 20)
            # Betas = P_inv @ R_subset # (4, M)
            # Predicted = F @ Betas # (20, M)
            # Residuals = R_subset - Predicted
            
            # This 'pinv' assumes full rows. But R_subset has NaNs! 
            # Vectorized OLS Fails if NaNs are present in diff positions for diff stocks.
            # Backtrack: Loop over stocks is safer if NaNs are prevalent.
            # With 3000 stocks, a simple python loop with numpy lstsq per stock
            # on 20 data points is actually extremely fast (microseconds per stock).
            # 3000 stocks * 400 months = 1.2M regressions. 1M ops is < 10 secs.
            
            for j in range(len(valid_stocks_idx)):
                y = R_subset[:, j]
                mask = ~np.isnan(y)
                if np.sum(mask) < 15:
                    continue
                    
                y_valid = y[mask]
                X_valid = F[mask]
                
                # Manual OLS
                # beta = (X'X)^-1 X'y
                # Using lstsq for stable calc
                beta, rss, rank, s = np.linalg.lstsq(X_valid, y_valid, rcond=None)
                
                # RSS is returned in 'rss' if rank=4 and len>4
                if len(rss) > 0:
                    ss_res = rss[0]
                else:
                    # manually calc
                    y_hat = X_valid @ beta
                    ss_res = np.sum((y_valid - y_hat)**2)
                
                ss_tot = np.sum((y_valid - np.mean(y_valid))**2)
                
                if ss_tot > 1e-8:
                    r2 = 1 - (ss_res / ss_tot)
                    ivol = 1 - r2
                    # Clip
                    ivol = max(0.0, min(1.0, ivol))
                    
                    final_results.append({
                        'trade_date': date,
                        'ts_code': stock_cols[j],
                        '1_minus_r2': ivol
                    })
                    
        except Exception as e:
            logging.error(f"Error in batch calculation {date}: {e}")
            continue

    logging.info("Saving results...")
    result_df = pd.DataFrame(final_results)
    result_df.to_csv(OUTPUT_FILE, index=False)
    logging.info("Done.")

if __name__ == "__main__":
    main()
