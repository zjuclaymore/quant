import pandas as pd

parquet_path = r"E:\1_basement\quant_research\data\AIndexEODPrices_all.parquet"

try:
    df = pd.read_parquet(parquet_path)
    print("\n--- Unique Symbols Containing '000001' ---")
    symbols = df['symbol'].dropna().unique().tolist()
    matches = [s for s in symbols if '000001' in str(s)]
    print(f"Matches: {matches}")
    
    # 打印其中一个的数据范围
    if matches:
        sub = df[df['symbol'] == matches[0]]
        print(f"\nDate Range for {matches[0]}: {sub['date'].min()} to {sub['date'].max()}")
        print(sub[['date', 'close']].head(3))
except Exception as e:
    print(f"Error: {e}")
