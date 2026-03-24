"""
generate_report_calendar.py
----------------------------
根据交易日历生成 A 股财报披露日历。

A 股定期报告披露规则（证监会）：
  - 一季报 (Q1)：报告期 YYYY0331，披露窗口 4/1  ~ 4/30
  - 中  报 (Q2)：报告期 YYYY0630，披露窗口 7/1  ~ 8/31
  - 三季报 (Q3)：报告期 YYYY0930，披露窗口 10/1 ~ 10/31
  - 年  报 (Q4)：报告期 YYYY1231，披露窗口 次年1/1 ~ 4/30
                  （实务中大部分在 3-4 月集中披露）

生成逻辑：
  对于每个交易日 trade_date，标注：
    1. report_period       — 当日"可获得的最新报告期"（严格遵循披露截止日）
    2. disclosure_window   — 当日所处的披露窗口名称（如 "Q1_YYYY"），无窗口时为空
    3. next_disclosure_end — 下一个披露截止日（交易日）
    4. is_disclosure_day   — 是否处于某个披露窗口内

输出：
  E:\\1_basement\\quant_research\\data\\交易日历\\report_calendar.csv
"""

import os
import pandas as pd
from datetime import datetime

# ─── 路径 ─────────────────────────────────────────────────
TRADE_CAL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data", "交易日历", "trade_calendar.csv"
)
OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data", "交易日历", "report_calendar.csv"
)

# 如果直接在 data/交易日历/ 运行，也能找到文件
if not os.path.exists(TRADE_CAL_PATH):
    TRADE_CAL_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "trade_calendar.csv"
    )
    OUTPUT_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "report_calendar.csv"
    )


def get_disclosure_deadlines(year: int) -> list[tuple[str, str, str]]:
    """
    返回指定年份的四个报告期及其披露截止日。
    格式: [(report_period, disclosure_start, disclosure_end), ...]

    注意：年报的 report_period 是上一年的 1231，但披露窗口在当年。
    """
    return [
        # 上一年年报 + 当年一季报的披露窗口都截止于 4/30
        (f"{year - 1}1231", f"{year}0101", f"{year}0430"),  # 年报
        (f"{year}0331",     f"{year}0401", f"{year}0430"),  # 一季报
        (f"{year}0630",     f"{year}0701", f"{year}0831"),  # 中报
        (f"{year}0930",     f"{year}1001", f"{year}1031"),  # 三季报
    ]


def determine_latest_report_period(cal_date_str: str) -> str:
    """
    给定一个日历日期（YYYYMMDD），确定到该日为止能确定"全市场强制披露完毕"的最新报告期。

    逻辑（基于披露截止日后第一天才能确认全部公司已披露）：
      - 在 5/1 及之后 → 可获得 Q1 (YYYY0331) 及上年年报 (YYYY-1 1231)
      - 在 9/1 及之后 → 可获得 Q2 (YYYY0630)
      - 在 11/1 及之后 → 可获得 Q3 (YYYY0930)
      - 1/1 ~ 4/30     → 仍在等待年报+Q1，最新可用为上年 Q3
    """
    date = datetime.strptime(cal_date_str, "%Y%m%d")
    year = date.year
    mmdd = int(date.strftime("%m%d"))

    if mmdd >= 1101:
        # 11/1 之后：三季报已全部披露完毕
        return f"{year}0930"
    elif mmdd >= 901:
        # 9/1 ~ 10/31：中报已全部披露完毕
        return f"{year}0630"
    elif mmdd >= 501:
        # 5/1 ~ 8/31：一季报 + 上年年报已全部披露完毕
        return f"{year}0331"
    else:
        # 1/1 ~ 4/30：上年三季报是最新的"确定全部披露"报告期
        return f"{year - 1}0930"


def determine_disclosure_window(cal_date_str: str) -> str:
    """
    判断当日是否处于某个披露窗口内，返回窗口名称。
    例如 "年报+Q1_2025" 表示 2025年的年报和一季报披露窗口。
    """
    date = datetime.strptime(cal_date_str, "%Y%m%d")
    year = date.year
    mmdd = int(date.strftime("%m%d"))

    if 101 <= mmdd <= 430:
        return f"年报({year-1})+Q1({year})"
    elif 701 <= mmdd <= 831:
        return f"中报Q2({year})"
    elif 1001 <= mmdd <= 1031:
        return f"三季报Q3({year})"
    else:
        return ""


def find_next_disclosure_end(cal_date_str: str) -> str:
    """找到下一个披露截止日（自然日）"""
    date = datetime.strptime(cal_date_str, "%Y%m%d")
    year = date.year
    mmdd = int(date.strftime("%m%d"))

    # 所有截止日节点
    deadlines = [
        (430,  f"{year}0430"),
        (831,  f"{year}0831"),
        (1031, f"{year}1031"),
    ]

    for cutoff, deadline_str in deadlines:
        if mmdd <= cutoff:
            return deadline_str

    # 过了10/31，下一个截止日是明年的4/30
    return f"{year + 1}0430"


def main():
    print(f"[INFO] 读取交易日历: {TRADE_CAL_PATH}")
    cal = pd.read_csv(TRADE_CAL_PATH, dtype=str)

    # 仅保留交易日
    trade_days = cal[cal["is_open"] == "1"].copy()
    trade_days = trade_days[["cal_date"]].rename(columns={"cal_date": "trade_date"})
    trade_days = trade_days.sort_values("trade_date").reset_index(drop=True)

    print(f"[INFO] 交易日总数: {len(trade_days):,}")
    print(f"[INFO] 日期范围: {trade_days['trade_date'].iloc[0]} ~ {trade_days['trade_date'].iloc[-1]}")

    # 生成各字段
    print("[INFO] 正在计算 report_period...")
    trade_days["report_period"] = trade_days["trade_date"].apply(
        determine_latest_report_period
    )

    print("[INFO] 正在计算 disclosure_window...")
    trade_days["disclosure_window"] = trade_days["trade_date"].apply(
        determine_disclosure_window
    )

    trade_days["is_disclosure_day"] = (trade_days["disclosure_window"] != "").astype(int)

    print("[INFO] 正在计算 next_disclosure_end...")
    trade_days["next_disclosure_end"] = trade_days["trade_date"].apply(
        find_next_disclosure_end
    )

    # 额外字段：report_period 对应的报告类型
    def report_type(rp: str) -> str:
        mm = rp[4:6]
        if mm == "03":
            return "Q1"
        elif mm == "06":
            return "Q2"
        elif mm == "09":
            return "Q3"
        elif mm == "12":
            return "Q4"
        return ""

    trade_days["report_type"] = trade_days["report_period"].apply(report_type)

    # 重排列
    trade_days = trade_days[[
        "trade_date",
        "report_period",
        "report_type",
        "disclosure_window",
        "is_disclosure_day",
        "next_disclosure_end",
    ]]

    # 保存
    trade_days.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n[DONE] 已保存: {OUTPUT_PATH}")
    print(f"  总行数:        {len(trade_days):,}")
    print(f"  披露窗口交易日: {trade_days['is_disclosure_day'].sum():,}")

    # 预览
    print("\n[预览] 前 15 行:")
    print(trade_days.head(15).to_string(index=False))

    # 2024-2025 年披露窗口切换点抽样
    sample = trade_days[trade_days["trade_date"].between("20240425", "20240510")]
    if not sample.empty:
        print("\n[预览] 2024年4月底~5月初（年报/Q1 → Q2 窗口切换）:")
        print(sample.to_string(index=False))


if __name__ == "__main__":
    main()
