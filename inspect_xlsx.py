import pandas as pd
import os

xlsx_path = r"E:\1_basement\quant_research\data\中国A股利润表_AShareIncome\报表类型.xlsx"
if os.path.exists(xlsx_path):
    print("--- 报表类型.xlsx ---")
    df_xlsx = pd.read_excel(xlsx_path)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_colwidth', None)
    print(df_xlsx.to_string())
else:
    print("报表类型.xlsx 不存在")
