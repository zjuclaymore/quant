import pandas as pd
import os

path = r"E:\1_basement\quant_research\factor\factor_package\tr_adjust_umr_1_processed_parquet\20100104.parquet"
if os.path.exists(path):
    print("File exists")
    df = pd.read_parquet(path)
    print("Columns:", df.columns.tolist())
    print("Head:\n", df.head(2))
else:
    print("File does not exist")
