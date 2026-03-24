import pandas as pd
import numpy as np
import os
import logging
from tqdm import tqdm
import concurrent.futures
import gc

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ivol_yearly.log"),
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

def read_single_pickle(file_path):
    try:
        df = pd.read_pickle(file_path)
        if 'Wind代码' in df.columns and '涨跌幅(%)' in df.columns:
            # Minimal data
            subset = df[['Wind代码', '涨跌幅(%)']].copy()
            subset.columns = ['ts_code', 'pct_chg']
            # Trade date is implicitly the filename/key, we add it later
            return subset
    except Exception:
        return None
    return None

def load_data_for_dates(date_list, file_map):
    """
    Load stock data for a specific list of dates.
    """
    tasks = []
    for d in date_list:
        if d in file_map:
            tasks.append((d, file_map[d]))
    
    results = []
    # Using ThreadPoolExecutor because IO bound
    # Batch reading to avoid too many small tasks if list is huge?
    # Actually just submit all might be fine for < 300 files (1 year)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_to_date = {executor.submit(read_single_pickle, path): date for date, path in tasks}
        
        for future in concurrent.futures.as_completed(future_to_date):
            date = future_to_date[future]
            try:
                res = future.result()
                if res is not None:
                    res['trade_date'] = date
                    results.append(res)
            except Exception as e:
                pass
                
    if not results:
        return pd.DataFrame()
        
    return pd.concat(results, ignore_index=True)

def matrix_regression(Y, X):
    """
    Perform vectorized regression Y = X*B + E for multiple columns in Y.
    Y: (T, N) - Returns for N stocks
    X: (T, K) - Factors (including constant)
    Returns: 1 - R^2 array of shape (N,)
    """
    # Hat matrix H = X (X'X)^-1 X'
    # Y_hat = H Y
    # But X is small (20x4), so explicit inverse is fast.
    
    T, K = X.shape
    
    try:
        # Precompute projection matrix P = (X'X)^-1 X'
        # X is (20, 4)
        xtx = X.T @ X
        xtx_inv = np.linalg.inv(xtx) # (4, 4)
        P = xtx_inv @ X.T # (4, 20)
        
        # Coefficients B = P Y  -> (4, N)
        B = P @ Y
        
        # Fitted values Y_hat = X B -> (20, N)
        Y_hat = X @ B
        
        # Residuals
        E = Y - Y_hat
        
        # SSR = sum(E^2) matrix diag logic not needed, simply sum squares axis 0
        SSR = np.sum(E**2, axis=0) # (N,)
        
        # SST
        Y_bar = np.mean(Y, axis=0) # (N,)
        SST = np.sum((Y - Y_bar)**2, axis=0)
        
        # R^2 = 1 - SSR/SST
        # Avoid divide by zero
        mask = SST > 1e-8
        r2 = np.zeros(Y.shape[1])
        r2[mask] = 1 - (SSR[mask] / SST[mask])
        r2[~mask] = np.nan # Or 0? If no variation, R2 is undefined/0.
        
        return 1 - r2
        
    except np.linalg.LinAlgError:
        return np.full(Y.shape[1], np.nan)

def calculate_for_window(returns_df, factors_df, trade_date):
    """
    Calculate IVol for a specific date using the data window.
    """
    common_dates = returns_df.index.intersection(factors_df.index)
    if len(common_dates) < 15:
        return []
        
    R = returns_df.loc[common_dates] # (T, N)
    F = factors_df.loc[common_dates].values # (T, 3)
    
    # Add constant
    F_design = np.column_stack([np.ones(len(F)), F]) # (T, 4)
    
    # Convert R to numpy
    R_vals = R.values
    
    # Check for NaNs
    # stocks with ANY NaN in this window cannot be part of the fast matrix batch easily?
    # Actually, we can check columns.
    
    nan_mask = np.isnan(R_vals).any(axis=0)
    
    results = []
    
    # 1. Full Data Stocks (Fast Path)
    full_data_idx = np.where(~nan_mask)[0]
    if len(full_data_idx) > 0:
        Y_full = R_vals[:, full_data_idx]
        ivol_vals = matrix_regression(Y_full, F_design)
        
        full_codes = R.columns[full_data_idx]
        for code, val in zip(full_codes, ivol_vals):
            if not np.isnan(val):
                results.append({'trade_date': trade_date, 'ts_code': code, '1_minus_r2': max(0.0, min(1.0, val))})
    
    # 2. Partial Data Stocks (Slower Path)
    # Filter columns that have enough valid data (>15) but some NaNs
    partial_data_idx = np.where(nan_mask)[0]
    
    # Optimization: If very few partials, loop is fine. 
    # If many, could batch by NaN pattern, but loop is simpler for now.
    
    for idx in partial_data_idx:
        y = R_vals[:, idx]
        valid_mask = ~np.isnan(y)
        
        if np.sum(valid_mask) < 15:
            continue
            
        y_valid = y[valid_mask]
        X_valid = F_design[valid_mask]
        
        try:
            beta, rss, rank, s = np.linalg.lstsq(X_valid, y_valid, rcond=None)
            
            if len(rss) > 0:
                ss_res = rss[0]
            else:
                y_hat = X_valid @ beta
                ss_res = np.sum((y_valid - y_hat)**2)
            
            ss_tot = np.sum((y_valid - np.mean(y_valid))**2)
            
            if ss_tot > 1e-8:
                r2 = 1 - (ss_res / ss_tot)
                ivol = 1 - r2
                results.append({'trade_date': trade_date, 'ts_code': R.columns[idx], '1_minus_r2': max(0.0, min(1.0, ivol))})
        except:
            continue
            
    return results

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # Clear output file first
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
        
    # Write header
    with open(OUTPUT_FILE, 'w') as f:
        f.write("trade_date,ts_code,1_minus_r2\n")

    factors_df = load_factors()
    
    logging.info("Scanning stock files...")
    files = [f for f in os.listdir(STOCK_DATA_DIR) if f.endswith('.pickle')]
    file_map = {}
    all_dates = []
    for f in files:
        if f[:-7].isdigit() and len(f[:-7]) == 8:
            d_str = f[:-7]
            file_map[d_str] = os.path.join(STOCK_DATA_DIR, f)
            all_dates.append(d_str)
            
    all_dates.sort()
    
    # Identify variables for grouping
    df_dates = pd.DataFrame({'date': all_dates})
    df_dates['dt'] = pd.to_datetime(df_dates['date'])
    df_dates['year'] = df_dates['dt'].dt.year
    df_dates['month'] = df_dates['dt'].dt.to_period('M')
    
    # Month ends needed for CALCULATION
    month_ends = set(df_dates.groupby('month')['date'].max().values)
    
    # Process by Year
    years = sorted(df_dates['year'].unique())
    
    logging.info(f"Processing years: {years}")
    
    for year in years:
        logging.info(f"Processing Year: {year}")
        
        # Define date range for loading:
        # We need dates in this year, PLUS previous 20 trading days roughly (1 month)
        # to calculate for Jan-31.
        
        current_year_dates = df_dates[df_dates['year'] == year]['date'].tolist()
        
        # Get prior dates
        # Find index of first date of year
        first_date_idx = all_dates.index(current_year_dates[0])
        start_idx = max(0, first_date_idx - 25) # Extra buffer
        load_dates = all_dates[start_idx : first_date_idx] + current_year_dates
        
        # Load Data
        stock_df = load_data_for_dates(load_dates, file_map)
        
        if stock_df.empty:
            continue
            
        stock_df['pct_chg'] = pd.to_numeric(stock_df['pct_chg'], errors='coerce')
        stock_df = stock_df.drop_duplicates(subset=['trade_date', 'ts_code'])
        
        # Pivot
        try:
            returns_matrix = stock_df.pivot(index='trade_date', columns='ts_code', values='pct_chg')
            returns_matrix = returns_matrix.sort_index()
        except Exception as e:
            logging.error(f"Error pivoting data for year {year}: {e}")
            continue
            
        # Iterate through Month Ends in THIS YEAR
        full_dates_np = returns_matrix.index.to_numpy() # Strings
        
        year_month_ends = [d for d in month_ends if d in current_year_dates]
        
        results_buffer = []
        
        for m_date in tqdm(year_month_ends, desc=f"Year {year}"):
            if m_date not in returns_matrix.index:
                continue
                
            idx_loc = returns_matrix.index.get_loc(m_date)
            
            if idx_loc < 19:
                continue
                
            # Get 20 day window
            window_dates = full_dates_np[idx_loc-19 : idx_loc+1]
            
            # Extract slice
            window_returns = returns_matrix.loc[window_dates]
            
            # Calculate
            calc_res = calculate_for_window(window_returns, factors_df, m_date)
            results_buffer.extend(calc_res)
            
        # Save batch
        if results_buffer:
            batch_df = pd.DataFrame(results_buffer)
            batch_df.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
            
        # Cleanup
        del stock_df
        del returns_matrix
        del results_buffer
        gc.collect()
        
    logging.info("All Done.")

if __name__ == "__main__":
    main()
