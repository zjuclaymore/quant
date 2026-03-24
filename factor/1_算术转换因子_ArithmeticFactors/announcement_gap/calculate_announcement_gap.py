"""
calculate_announcement_gap.py
------------------------------
此脚本用于计算“公告后个股超额开盘收益因子”(Announcement Gap Factor)。

因子逻辑：
  对于每次财报或重要公告，计算其发布后的首个交易日开盘相对于公告前一交易日收盘的涨跌幅，
  并减去同期市场基准（中证全指）的开盘涨跌幅。

公式定义：
  Announcement Gap = (Stock_Open_T / Stock_Close_T-1 - 1) - (Mkt_Open_T / Mkt_Close_T-1 - 1)
  其中 T 为公告后的第一个交易日 (next_trade_date)。

优化策略：
  1. Phase 1: 扫描公告数据，收集 (ts_code, ann_date, next_td) 元数据。
  2. Phase 2: 按日期分组批量加载 EOD 数据，避免重复加载大型 pickle 文件。
  3. Phase 3: 计算超额收益并按股票保存。

注意项：
  - 此脚本假设公告在公告日收盘后或非交易日发布。
  - 对于交易日早盘（9:00前）发布的公告，由于缺乏具体分钟时间，目前统一在 T+1 交易日捕捉反应。
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from collections import defaultdict

# ─── 路径配置 ─────────────────────────────────────────────
PROJECT_ROOT = r"E:\1_basement\quant_research"

ANN_DIR = os.path.join(PROJECT_ROOT, "data", "中国A股三表_快报_预报", "class_by_stock_dedup")
EOD_DIR = os.path.join(PROJECT_ROOT, "data", "中国A股日行情_AShareEODPrices")
MKT_FILE = os.path.join(PROJECT_ROOT, "data", "市场收益_MarketRevenue", "000985.CSI.xlsx")
CAL_FILE = os.path.join(PROJECT_ROOT, "data", "交易日历", "trade_calendar.csv")

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT, "factor", "1_算术转换因子_ArithmeticFactors",
    "announcement_gap", "output", "class_by_stock"
)

# ─── 公告日期列名候选 ────────────────────────────────────
ANN_DATE_CANDIDATES = ["ANN_DT", "ann_dt", "公告日期", "ACTUAL_ANN_DT", "actual_ann_dt"]
REPORT_PERIOD_CANDIDATES = ["REPORT_PERIOD", "report_period", "报告期"]
REPORT_TYPE_CANDIDATES = ["report_type", "报告类型", "STATEMENT_TYPE", "_source"]


def find_col(columns, candidates):
    """在列名列表中查找第一个匹配的候选列名"""
    col_set = set(columns)
    for c in candidates:
        if c in col_set:
            return c
    for c in columns:
        c_lower = str(c).lower()
        for cand in candidates:
            if cand.lower() in c_lower:
                return c
    return None


# ─── 1. 加载交易日历 → next_td 映射 ──────────────────────
def load_next_trade_day_map():
    """返回 { 日历日YYYYMMDD: 严格之后的下一个交易日 }"""
    cal = pd.read_csv(CAL_FILE, dtype=str)
    trade_days = sorted(cal[cal["is_open"] == "1"]["cal_date"].tolist())
    all_dates = sorted(cal["cal_date"].tolist())

    next_td_map = {}
    td_ptr = 0
    for d in all_dates:
        while td_ptr < len(trade_days) and trade_days[td_ptr] <= d:
            td_ptr += 1
        if td_ptr < len(trade_days):
            next_td_map[d] = trade_days[td_ptr]

    return next_td_map, trade_days


# ─── 2. 加载中证全指开盘收益率 ───────────────────────────
def load_market_open_return():
    """返回 { 'YYYYMMDD': mkt_open_ret }"""
    df = pd.read_excel(MKT_FILE)
    df = df.sort_values("日期").reset_index(drop=True)
    df["date_str"] = pd.to_datetime(df["日期"]).dt.strftime("%Y%m%d")
    df["prev_close"] = df["收盘价(元)"].shift(1)
    df["mkt_open_ret"] = df["开盘价(元)"] / df["prev_close"] - 1.0

    return dict(zip(df["date_str"], df["mkt_open_ret"]))


# ─── Phase 1: 收集所有公告记录 ───────────────────────────
def collect_all_announcements(next_td_map):
    """
    扫描所有公告pickle，收集每条公告的元数据。
    返回 DataFrame: [ts_code, ann_date, next_trade_date, report_period, report_type]
    """
    stock_files = sorted([f for f in os.listdir(ANN_DIR) if f.endswith(".pickle")])
    total = len(stock_files)

    all_records = []
    ann_date_col = None
    rp_col = None
    rt_col = None

    t0 = time.time()
    for i, stock_file in enumerate(stock_files, 1):
        ts_code = os.path.splitext(stock_file)[0]
        fpath = os.path.join(ANN_DIR, stock_file)

        try:
            df = pd.read_pickle(fpath)
        except Exception:
            continue

        if not isinstance(df, pd.DataFrame) or df.empty:
            continue

        # 首次探测列名
        if ann_date_col is None or ann_date_col not in df.columns:
            ann_date_col = find_col(df.columns, ANN_DATE_CANDIDATES)
            rp_col = find_col(df.columns, REPORT_PERIOD_CANDIDATES)
            rt_col = find_col(df.columns, REPORT_TYPE_CANDIDATES)
            if i == 1:
                print(f"  公告数据列: {list(df.columns)[:10]}...")
                print(f"  公告日期列: {ann_date_col}")
                print(f"  报告期列:   {rp_col}")
                print(f"  报告类型列: {rt_col}")

        if ann_date_col is None:
            continue

        # 提取公告日期
        ann_dates = df[ann_date_col].dropna().astype(str)
        ann_dates = ann_dates.str.replace("-", "", regex=False).str.replace("/", "", regex=False).str[:8]

        for idx, ann_date in ann_dates.items():
            if len(ann_date) != 8 or not ann_date.isdigit():
                continue
            next_td = next_td_map.get(ann_date)
            if next_td is None:
                continue

            record = {
                "ts_code": ts_code,
                "ann_date": ann_date,
                "next_trade_date": next_td,
            }
            if rp_col and rp_col in df.columns:
                record["report_period"] = str(df.at[idx, rp_col]) if pd.notna(df.at[idx, rp_col]) else ""
            if rt_col and rt_col in df.columns:
                record["report_type"] = str(df.at[idx, rt_col]) if pd.notna(df.at[idx, rt_col]) else ""

            all_records.append(record)

        if i % 1000 == 0 or i == total:
            elapsed = time.time() - t0
            print(f"  [{i:>5,}/{total:,}] 已收集 {len(all_records):,} 条公告记录, 耗时 {elapsed:.1f}s")

    return pd.DataFrame(all_records)


# ─── Phase 2: 批量计算开盘涨跌幅 ─────────────────────────
def batch_compute_open_returns(ann_df, mkt_open_ret):
    """
    按 next_trade_date 分组，加载 EOD pickle，批量提取个股开盘涨跌幅。
    """
    # 按 next_trade_date 分组
    grouped = ann_df.groupby("next_trade_date")
    unique_dates = sorted(grouped.groups.keys())
    total_dates = len(unique_dates)

    print(f"  需加载 {total_dates:,} 个交易日的 EOD 数据")

    stock_open_rets = {}  # (ts_code, next_td) -> ret
    loaded = 0
    skipped = 0
    t0 = time.time()

    for di, trade_date in enumerate(unique_dates, 1):
        eod_path = os.path.join(EOD_DIR, f"{trade_date}.pickle")
        if not os.path.exists(eod_path):
            skipped += 1
            continue

        try:
            eod_df = pd.read_pickle(eod_path)
        except Exception:
            skipped += 1
            continue

        loaded += 1

        # 计算开盘涨跌幅 (矢量化)
        open_col = "开盘价(元)"
        prev_close_col = "昨收盘价(元)"
        code_col = "Wind代码"

        if open_col in eod_df.columns and prev_close_col in eod_df.columns and code_col in eod_df.columns:
            mask = (eod_df[prev_close_col] != 0) & eod_df[open_col].notna() & eod_df[prev_close_col].notna()
            valid = eod_df[mask]
            rets = (valid[open_col] / valid[prev_close_col] - 1.0)

            # 获取这个日期下需要的股票
            date_group = grouped.get_group(trade_date)
            needed_stocks = set(date_group["ts_code"])

            for code, ret in zip(valid[code_col], rets):
                if code in needed_stocks:
                    stock_open_rets[(code, trade_date)] = ret

        if di % 500 == 0 or di == total_dates:
            elapsed = time.time() - t0
            print(f"  [{di:>5,}/{total_dates:,}] "
                  f"已加载: {loaded:,}, 跳过: {skipped:,}, "
                  f"匹配: {len(stock_open_rets):,}, 耗时: {elapsed:.1f}s")

    return stock_open_rets


# ─── Phase 3: 合并并保存 ─────────────────────────────────
def merge_and_save(ann_df, stock_open_rets, mkt_open_ret, trade_days):
    """
    将提取的个股涨跌幅与市场基准合并，计算因子值，并执行向下填充(ffill)以保证因子持续性。
    最终按股票代码导出 CSV。

    Args:
        ann_df (pd.DataFrame): 包含公告日期映射的 DataFrame。
        stock_open_rets (dict): {(ts_code, next_td): open_ret} 映射。
        mkt_open_ret (dict): {date_str: mkt_ret} 映射。
        trade_days (list): 完整的交易日列表 (str)。

    Returns:
        tuple: (保存的股票数量, 总记录条数, 结果 DataFrame)
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 计算公式应用
    ann_df["stock_open_ret"] = ann_df.apply(
        lambda r: stock_open_rets.get((r["ts_code"], r["next_trade_date"])), axis=1
    )
    ann_df["mkt_open_ret"] = ann_df["next_trade_date"].map(mkt_open_ret)

    # 2. 过滤有效行并计算
    valid = ann_df.dropna(subset=["stock_open_ret", "mkt_open_ret"]).copy()
    valid["announcement_gap"] = (valid["stock_open_ret"] - valid["mkt_open_ret"]).round(6)
    valid = valid.rename(columns={"next_trade_date": "trade_date"})

    # 去重：同一天可能有多个公告，保留第一个
    valid = valid.drop_duplicates(subset=["ts_code", "trade_date"])

    # 3. 按股票执行填充并保存
    saved = 0
    total_records = 0
    td_series = pd.Series(trade_days)
    all_valid_results = []

    for ts_code, group in valid.groupby("ts_code"):
        group = group.sort_values("trade_date")
        
        # 将该股票的信号扩展到整个交易日历并向下填充
        # 我们只从该股票第一次公告日期开始填充，避免空值覆盖过远
        first_date = group["trade_date"].min()
        relevant_tds = td_series[td_series >= first_date]
        
        # 重新索引并填充，设置限制防止过时信号干扰
        dense_df = group.set_index("trade_date").reindex(relevant_tds)
        dense_df["announcement_gap"] = dense_df["announcement_gap"].ffill(limit=60)
        dense_df = dense_df.reset_index().rename(columns={"index": "trade_date"})
        
        # 只保留必要的列
        out_df = dense_df[["trade_date", "announcement_gap"]].dropna(subset=["announcement_gap"])
        
        if not out_df.empty:
            out_path = os.path.join(OUTPUT_DIR, f"{ts_code}.csv")
            out_df.to_csv(out_path, index=False)
            saved += 1
            total_records += len(out_df)
            all_valid_results.append(out_df.assign(ts_code=ts_code))

    final_df = pd.concat(all_valid_results, ignore_index=True) if all_valid_results else pd.DataFrame()
    return saved, total_records, final_df


# ─── 主函数 ───────────────────────────────────────────────
def main():
    """
    因子计算执行入口。
    
    流程：
      1. 加载交易日历与下一交易日映射。
      2. 加载市场基准（中证全指）开盘收益率。
      3. 批量收集所有股票的财报公告日期。
      4. 批量从日行情数据中提取公告后首日个股开盘收益率。
      5. 计算超额收益并执行 60 天向下填充，保存为 CSV 因子文件。
    """
    t_start = time.time()

    print("=" * 60)
    print("公告后超额开盘收益因子 (Announcement Gap)")
    print("=" * 60)

    # Step 1: 交易日历
    print("\n[1/5] 加载交易日历...")
    next_td_map, trade_days = load_next_trade_day_map()
    print(f"  交易日: {len(trade_days):,}")
    print(f"  日期映射: {len(next_td_map):,}")

    # Step 2: 中证全指
    print("\n[2/5] 加载中证全指...")
    mkt_open_ret = load_market_open_return()
    valid_mkt = {k: v for k, v in mkt_open_ret.items() if pd.notna(v)}
    mkt_dates = sorted(valid_mkt.keys())
    print(f"  有效交易日: {len(valid_mkt):,}")
    print(f"  日期范围: {mkt_dates[0]} ~ {mkt_dates[-1]}")

    # Step 3: 收集公告
    print("\n[3/5] 收集所有公告记录...")
    ann_df = collect_all_announcements(next_td_map)
    print(f"  公告总数: {len(ann_df):,}")
    print(f"  涉及股票: {ann_df['ts_code'].nunique():,}")
    print(f"  涉及交易日: {ann_df['next_trade_date'].nunique():,}")

    if ann_df.empty:
        print("\n[ERROR] 未找到任何公告记录！请检查公告数据。")
        return

    # Step 4: 批量计算开盘涨跌幅
    print("\n[4/5] 批量加载 EOD 数据并计算开盘涨跌幅...")
    stock_open_rets = batch_compute_open_returns(ann_df, valid_mkt)
    print(f"  获得开盘涨跌幅: {len(stock_open_rets):,} 条")

    # Step 5: 合并保存
    print("\n[5/5] 合并并保存结果 (包含向下填充)...")
    saved, total_records, result_df = merge_and_save(ann_df, stock_open_rets, valid_mkt, trade_days)

    t_end = time.time()

    print(f"\n{'=' * 60}")
    print(f"[DONE] 完成!")
    print(f"  公告总数:     {len(ann_df):,}")
    print(f"  有效记录:     {total_records:,}")
    print(f"  输出股票数:   {saved:,}")
    print(f"  总耗时:       {t_end - t_start:.1f}s")
    print(f"  输出目录:     {OUTPUT_DIR}")

    # 因子统计
    if len(result_df) > 0:
        gap = result_df["announcement_gap"]
        print(f"\n[因子统计]")
        print(f"  均值:   {gap.mean():.6f}")
        print(f"  标准差: {gap.std():.6f}")
        print(f"  中位数: {gap.median():.6f}")
        print(f"  最小值: {gap.min():.6f}")
        print(f"  最大值: {gap.max():.6f}")

    # 抽样展示
    sample_file = os.path.join(OUTPUT_DIR, "000001.SZ.csv")
    if os.path.exists(sample_file):
        print(f"\n[预览] 000001.SZ 前 10 条:")
        sample = pd.read_csv(sample_file)
        print(sample.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
