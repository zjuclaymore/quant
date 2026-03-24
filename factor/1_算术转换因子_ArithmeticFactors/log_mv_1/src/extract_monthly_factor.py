import pandas as pd
import os
from tqdm import tqdm

def extract_monthly_factor(start_year, end_year, input_dir, output_file):
    """
    Reads yearly daily factor files, selects the last trading day of each month for each stock,
    and aggregates them into a single file sorted by ts_code and trade_date.
    """
    all_monthly_data = []

    print(f"Extracting monthly data from {start_year} to {end_year}...")
    
    for year in tqdm(range(start_year, end_year + 1)):
        file_path = os.path.join(input_dir, f"log_mv_{year}.csv")
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found. Skipping.")
            continue
        
        # Read daily data
        df = pd.read_csv(file_path, dtype={'trade_date': str, 'ts_code': str})
        
        # Create a month identifier
        df['month'] = df['trade_date'].str[:6]
        
        # Sort by date ensures we get the last date when taking the last item
        df = df.sort_values('trade_date')
        
        # Group by stock and month, keep the last record (month-end)
        monthly_df = df.groupby(['ts_code', 'month'], as_index=False).last()
        
        # Drop the helper column
        monthly_df = monthly_df.drop(columns=['month'])
        
        all_monthly_data.append(monthly_df)
    
    # Concatenate all years
    if all_monthly_data:
        print("Concatenating data...")
        final_df = pd.concat(all_monthly_data, ignore_index=True)
        
        # Sort by ts_code and trade_date
        print("Sorting data...")
        final_df = final_df.sort_values(by=['ts_code', 'trade_date'])
        
        # Save to CSV
        print(f"Saving to {output_file}...")
        final_df.to_csv(output_file, index=False)
        print("Done.")
    else:
        print("No data extracted.")

if __name__ == "__main__":
    input_directory = r"E:\1_basement\ml\factors\log_mv\output"
    output_csv = r"E:\1_basement\ml\factors\log_mv\output\log_mv_monthly_2000_2025.csv"
    
    extract_monthly_factor(2000, 2025, input_directory, output_csv)
