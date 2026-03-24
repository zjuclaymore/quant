import pandas as pd

parquet_path = r"E:\1_basement\quant_research\data\AIndexEODPrices_all.parquet"

try:
    # 只读取前几行看表头
    df = pd.read_parquet(parquet_path)
    print("\n--- Columns in Parquet ---")
    print(df.columns.tolist())
    
    # 查找是否有 000001.SH
    code_col = 'S_INFO_WINDCODE' if 'S_INFO_WINDCODE' in df.columns else 'WIND_CODE' if 'WIND_CODE' in df.columns else 'ts_code'
    if code_col in df.columns:
        codes = df[code_col].unique()
        print(f"\nTotal full codes: {len(codes)}")
        # 看看有没有 000001
        sh_code = [c for c in codes if '000001' in str(c)]
        print(f"Sample matches containing '000001': {sh_code}")
    else:
        print(f"Could not find code column among: {df.columns.tolist()}")
except Exception as e:
    print(f"Error reading {parquet_path}: {e}")
