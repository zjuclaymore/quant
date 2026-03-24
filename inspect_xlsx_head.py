import pandas as pd
import os

xlsx_path = r"E:\1_basement\quant_research\data\中国A股利润表_AShareIncome\报表类型.xlsx"
df_xlsx = pd.read_excel(xlsx_path)
print("--- 报表类型.xlsx (前15行) ---")
print(df_xlsx.head(15).to_string())
print("\n--- 报表类型.xlsx (尾15行) ---")
print(df_xlsx.tail(15).to_string())
