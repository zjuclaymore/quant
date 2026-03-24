"""
download_trade_cal.py
---------------------
使用 tushare 下载 A 股交易日历并保存至项目本地 utility 目录。

输出文件 : E:\\1_basement\\quant_research\\utility\\trade_calendar\\trade_calendar.csv
对应配置 : data_sources.json → "trade_calendar_file"

用法:
  python download_trade_cal.py                   # 默认 1990-01-01 ~ 2035-12-31
  python download_trade_cal.py --start 20000101 --end 20301231
  python download_trade_cal.py --overwrite        # 覆盖已有文件
"""

import os
import json
import argparse
from pathlib import Path

import tushare as ts
import pandas as pd

# ──────────────────────────────────────────────────────────────
# 从 base.json 获取 token
# ──────────────────────────────────────────────────────────────
_THIS_DIR    = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parent.parent.parent   # universe→quant_research→src→quant_research
BASE_CONF    = PROJECT_ROOT / "conf" / "base.json"

with open(BASE_CONF, "r", encoding="utf-8-sig") as _f:
    _base = json.load(_f)

TOKEN    = _base.get("tushare_token", "")
EXCHANGE = "SSE"   # 上交所日历覆盖全部 A 股交易日

# 输出路径（与 data_sources.json 保持一致）
OUTPUT_DIR  = PROJECT_ROOT / "utility" / "trade_calendar"
OUTPUT_FILE = OUTPUT_DIR / "trade_calendar.csv"


def download(start: str, end: str, overwrite: bool) -> None:
    if OUTPUT_FILE.exists() and not overwrite:
        print(f"文件已存在，跳过下载：{OUTPUT_FILE}")
        print("如需重新下载请加 --overwrite 参数。")
        return

    print(f"初始化 tushare ...")
    ts.set_token(TOKEN)
    pro = ts.pro_api()

    print(f"拉取交易日历 {EXCHANGE}  {start} → {end} ...")
    df = pro.trade_cal(exchange=EXCHANGE, start_date=start, end_date=end,
                       fields="exchange,cal_date,is_open,pretrade_date")

    if df is None or df.empty:
        raise RuntimeError("tushare 返回空数据，请检查 token 或网络。")

    print(f"获取 {len(df)} 行。")

    # 标准化：cal_date 格式为 YYYYMMDD（字符串），is_open 为 0/1
    df = df.sort_values("cal_date").reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"✓ 已保存：{OUTPUT_FILE}")
    print(f"  交易日总计：{df['is_open'].sum()} 天 / {len(df)} 自然日")


def main():
    parser = argparse.ArgumentParser(description="下载 A 股交易日历（tushare）")
    parser.add_argument("--start",     default="19900101", help="起始日期 YYYYMMDD")
    parser.add_argument("--end",       default="20351231", help="结束日期 YYYYMMDD")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有文件")
    args = parser.parse_args()
    download(args.start, args.end, args.overwrite)


if __name__ == "__main__":
    main()
