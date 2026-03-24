import pandas as pd
import os

# 1. 探查 报表类型.xlsx
xlsx_path = r"E:\1_basement\quant_research\data\中国A股利润表_AShareIncome\报表类型.xlsx"
if os.path.exists(xlsx_path):
    print("--- 报表类型.xlsx ---")
    try:
        df_xlsx = pd.read_excel(xlsx_path)
        print(df_xlsx.to_string())
    except Exception as e:
        print(f"Error reading xlsx: {e}")
else:
    print("报表类型.xlsx 不存在")

# 2. 探查 利润表.pickle
pickle_path = r"E:\1_basement\quant_research\data\中国A股利润表_AShareIncome\利润表.pickle"
if os.path.exists(pickle_path):
    print("\n--- 利润表.pickle 探查 ---")
    try:
        # 只取前 10000 行加快速度，如果文件很大
        df = pd.read_pickle(pickle_path)
        print(f"Total Rows: {len(df)}")
        print("\nColumns:", df.columns.tolist())
        
        # 查找包含“类型”或“代码”的列
        type_cols = [c for c in df.columns if "类型" in str(c) or "代码" in str(c)]
        print(f"\nType/Code Columns: {type_cols}")
        
        # 打印相关列的去重计数
        for c in ["报表类型代码", "报表类型", "Wind代码"] + type_cols[:10]:
            if c in df.columns:
                print(f"\n--- Value counts for {c} ---")
                print(df[c].value_counts().head(10))
    except Exception as e:
        print(f"Error reading pickle: {e}")
else:
    print("利润表.pickle 不存在")
