from pathlib import Path

import pandas as pd

def reorganize_data(source_dir: str, target_dir: str):
    """
    将混在一起的、包含多天数据的 index parquet 文件，
    按照 "截面日 (Date) / 指数 (Index)" 的层级结构进行拆分和重组归档。
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    files = [path for path in source_dir.iterdir() if path.suffix == ".parquet"]
    print(f"开始重组数据，发现 {len(files)} 个原始宽基指数聚合文件。")

    by_date_dir = target_dir / "by_date"
    by_index_dir = target_dir / "by_index"

    by_date_dir.mkdir(parents=True, exist_ok=True)
    by_index_dir.mkdir(parents=True, exist_ok=True)

    for file_path in files:
        df = pd.read_parquet(file_path)
        if df.empty:
            continue

        index_name = df['index_name'].iloc[0]
        ts_code = df['ts_code'].iloc[0]

        df['trade_date'] = pd.to_datetime(df['trade_time']).dt.strftime('%Y-%m-%d')
        unique_dates = df['trade_date'].unique()

        index_folder = by_index_dir / f"{ts_code}_{index_name}"
        index_folder.mkdir(parents=True, exist_ok=True)

        for date in unique_dates:
            df_day = df[df['trade_date'] == date].copy()
            df_day = df_day.drop(columns=['trade_date'])

            index_date_path = index_folder / f"{date}.parquet"
            df_day.to_parquet(index_date_path, index=False, engine='pyarrow')

            date_folder = by_date_dir / date
            date_folder.mkdir(parents=True, exist_ok=True)
            date_index_path = date_folder / f"{ts_code}_{index_name}.parquet"
            df_day.to_parquet(date_index_path, index=False, engine='pyarrow')

        print(f"已拆分 {index_name} ({ts_code}) -> 涵盖 {len(unique_dates)} 个交易日")

    print(f"\n[Done] 数据重组与拆分归档完成！")
    print(f"按截面日管理目录: {by_date_dir}")
    print(f"按指数类别管理目录: {by_index_dir}")
    return by_date_dir, by_index_dir

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parents[2]
    SOURCE_DIR = BASE_DIR / "index_distortion_analysis" / "data_1min"
    TARGET_DIR = BASE_DIR / "index_distortion_analysis" / "data_1min_organized"
    reorganize_data(SOURCE_DIR, TARGET_DIR)
