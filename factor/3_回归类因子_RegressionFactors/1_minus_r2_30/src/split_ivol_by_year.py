import pandas as pd
import os
from tqdm import tqdm

def split_csv_by_year(input_file, output_dir):
    """
    Splits a large CSV file into yearly files based on the 'trade_date' column.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Processing {input_file}...")
    
    # Read in chunks to handle large files
    chunksize = 1000000 
    
    # helper to track which files have been initialized with headers
    initialized_files = set()

    reader = pd.read_csv(input_file, chunksize=chunksize, dtype={'trade_date': str})

    for i, chunk in enumerate(reader):
        print(f"Processing chunk {i+1}...")
        
        # Extract year
        chunk['year'] = chunk['trade_date'].str[:4]
        
        # Group by year
        groups = chunk.groupby('year')
        
        for year, group in groups:
            output_file = os.path.join(output_dir, f"ivol_{year}.csv")
            
            # Remove the temporary 'year' column before saving
            save_group = group.drop(columns=['year'])
            
            mode = 'a'
            header = False
            
            if output_file not in initialized_files:
                # Check if file exists on disk to decide header
                if not os.path.exists(output_file):
                    header = True
                # Start fresh if it's the first time this script sees it? 
                # Ideally we should delete existing files first to avoid appending to old runs.
                # But logical 'initialized_files' assumes we are doing it in one run.
                # Let's delete if it exists and we haven't touched it yet.
                if os.path.exists(output_file) and output_file not in initialized_files:
                     try:
                         os.remove(output_file)
                         header = True
                     except:
                         pass
                
                initialized_files.add(output_file)
            
            save_group.to_csv(output_file, mode=mode, header=header, index=False)
            
    print("Splitting complete.")

if __name__ == "__main__":
    input_csv = r"E:\1_basement\ml\factors\1_minus_r\output\ivol_monthly.csv"
    output_directory = r"E:\1_basement\ml\factors\1_minus_r\output"
    
    split_csv_by_year(input_csv, output_directory)
