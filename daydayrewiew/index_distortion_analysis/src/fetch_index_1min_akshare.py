import akshare as ak
import pandas as pd
from pathlib import Path

"""
新浪财经版宽基指数分钟线拉取器 (Sina Crawler Index Minute Fetcher)
功能: 通过新浪财经接口获取最近几天（约 8 个交易日）的 1min 数据，
      非常稳定，无东方财富的分块反爬问题。
"""

def fetch_index_1min_sina(output_dir: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    indices = {
        'sh000001': ('000001.SH', '上证指数'),
        'sz399001': ('399001.SZ', '深证成指'),
        'sh000680': ('000680.SH', '科创综指'),
        'sz399006': ('399006.SZ', '创业板指'),
        'bj899050': ('899050.BJ', '北证50'),
    }

    print("启动新浪爬虫: 获取最近 8 个交易日的 1min 数据\n")

    output_files = []
    for symbol, (ts_code, name) in indices.items():
        print(f"正在拉取: {name} ({ts_code}) ...", end=" ")
        try:
            df = ak.stock_zh_a_minute(symbol=symbol, period='1')

            if df is not None and not df.empty:
                rename_map = {
                    'day': 'trade_time',
                    'volume': 'vol'
                }
                df = df.rename(columns=rename_map)

                df['ts_code'] = ts_code
                df['index_name'] = name

                df = df.sort_values('trade_time').reset_index(drop=True)

                output_file = output_dir / f"{ts_code}_{name}_1min.parquet"
                df.to_parquet(output_file, index=False, engine='pyarrow')
                output_files.append(output_file)
                print(f"成功 ({len(df)} 行) -> 已落盘")
            else:
                print("失败: 无数据")

        except Exception as e:
            print(f"异常: {e}")
    return output_files

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parents[2]
    OUTPUT_DIR = BASE_DIR / "index_distortion_analysis" / "data_1min"
    fetch_index_1min_sina(OUTPUT_DIR)
