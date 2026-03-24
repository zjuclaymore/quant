import pandas as pd
import os

pickle_path = r"E:\1_basement\quant_research\data\中国A股三表_快报_预报\000001.SZ.pickle"
if os.path.exists(pickle_path):
    print("--- 验证 000001.SZ.pickle 的报表类型分布 ---")
    df = pd.read_pickle(pickle_path)
    print(f"Total Rows: {len(df)}")
    
    # 统计 _source 和 report_type 联合分布
    if 'report_type' in df.columns:
        print("\nReport Type distribution from tables:")
        print(df.groupby(['_source', 'report_type']).size().to_string())
    else:
        print("\nNo report_type column found (it could be only Notice/Express without it).")
        
    print("\n--- _source Value Counts ---")
    print(df['_source'].value_counts())
    
else:
    print(f"Error: {pickle_path} does not exist!")
