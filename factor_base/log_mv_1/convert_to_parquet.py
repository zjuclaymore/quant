import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

"""
本脚本用于将指定目录下的股票因子 CSV 文件转换为 Parquet 格式，以提高读取效率。
"""

def convert_csv_to_parquet(csv_path: str, parquet_path: str):
    """
    将单个 CSV 文件读取并另存为 Parquet。
    
    Args:
        csv_path: 源 CSV 路径.
        parquet_path: 目标 Parquet 路径.
    """
    try:
        # 读取 CSV，假设第一列是日期，第二列是因子
        df = pd.read_csv(csv_path)
        if not df.empty:
            df.to_parquet(parquet_path, index=False, engine='pyarrow')
    except Exception as e:
        print(f"Error converting {csv_path}: {e}")

def main():
    """主转换逻辑"""
    source_dir = r"E:\1_basement\quant_research\factor_base\log_mv_1\output\class_by_stock"
    target_dir = r"E:\1_basement\quant_research\factor_base\log_mv_1"
    
    # 确保目标目录存在
    os.makedirs(target_dir, exist_ok=True)
    
    files = [f for f in os.listdir(source_dir) if f.endswith(".csv")]
    print(f"Found {len(files)} CSV files in {source_dir}")
    
    tasks = []
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        for f in files:
            csv_path = os.path.join(source_dir, f)
            parquet_name = f.replace(".csv", ".parquet")
            parquet_path = os.path.join(target_dir, parquet_name)
            tasks.append(executor.submit(convert_csv_to_parquet, csv_path, parquet_path))
            
        for _ in tqdm(as_completed(tasks), total=len(tasks), desc="Converting CSV to Parquet"):
            pass

    print(f"Conversion completed. Files are located in {target_dir}")

if __name__ == "__main__":
    main()
