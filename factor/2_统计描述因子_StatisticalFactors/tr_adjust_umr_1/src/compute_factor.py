# -*- coding: utf-8 -*-
"""
compute_factor.py
-----------------
TR-Adjusted Unsystematic Momentum Reversal (UMR) 因子计算

算法步骤:
1. 计算 True Range (TR):
   TR = max(high-low, |high-preclose|, |low-preclose|) / preclose

2. 计算过去10个交易日的 TR 均值 (TR_MA10)

3. 系数 = TR_MA10 - TR_today

4. 超额收益 = 个股日收益率 - Wind全A日收益率

5. 调整后收益 = 系数 × 超额收益

6. 因子值 = 过去122个交易日的半衰加权(半衰期=60日)对调整后收益求和

输出: 每只股票一个 pickle 文件，保存在 output/ 目录

用法:
  python compute_factor.py                        # 全量计算
  python compute_factor.py --codes 000001.SZ 600519.SH  # 指定股票
  python compute_factor.py --start 20200101       # 仅保留指定日期后的因子值
"""

import os
import sys
import glob
import argparse
import time
import pandas as pd
import numpy as np


# ──────────────────── 路径配置 ────────────────────

FACTOR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(FACTOR_ROOT)))

STOCK_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "中国A股时序量价数据")
MARKET_XLSX = os.path.join(
    PROJECT_ROOT, "data", "市场收益_MarketRevenue", "881001.WI.xlsx"
)
OUTPUT_DIR = os.path.join(FACTOR_ROOT, "output")

# ──────────────────── 参数配置 ────────────────────

TR_WINDOW = 10       # TR 均值回望窗口
FACTOR_WINDOW = 122  # 因子半衰加权回望窗口
HALF_LIFE = 60       # 半衰期 (天)


def build_halflife_weights(window: int, half_life: int) -> np.ndarray:
    """构建半衰期指数衰减权重向量，最近的权重最大。"""
    # t=0 是最远的一天, t=window-1 是最近的一天
    t = np.arange(window)
    decay = np.log(2) / half_life
    weights = np.exp(decay * (t - (window - 1)))
    return weights / weights.sum()


def load_market_return(xlsx_path: str) -> pd.Series:
    """
    加载 Wind全A (881001.WI) 日收益率。
    返回: pd.Series, index=日期(str YYYYMMDD), values=日涨跌幅(小数)
    """
    print(f"加载 Wind全A 数据: {xlsx_path}")
    df = pd.read_excel(xlsx_path, engine="openpyxl")

    # 自适应列名: 尝试多种可能的列名
    date_col = None
    ret_col = None

    for c in df.columns:
        c_str = str(c)
        if "日期" in c_str or "date" in c_str.lower() or "trade" in c_str.lower():
            date_col = c
        if "涨跌幅" in c_str or "pct" in c_str.lower() or "return" in c_str.lower():
            ret_col = c

    if date_col is None:
        # 如果没有明确的日期列，尝试第一列
        date_col = df.columns[0]
        print(f"  [提示] 使用第一列作为日期列: {date_col}")
    if ret_col is None:
        # 尝试找包含百分比/涨跌的列
        for c in df.columns:
            c_str = str(c)
            if "%" in c_str or "change" in c_str.lower():
                ret_col = c
                break
        if ret_col is None:
            print(f"  [警告] 未找到涨跌幅列，可用列: {df.columns.tolist()}")
            print(f"  [提示] 尝试使用收盘价计算收益率...")
            # 尝试用收盘价算
            close_col = None
            for c in df.columns:
                c_str = str(c)
                if "收盘" in c_str or "close" in c_str.lower():
                    close_col = c
                    break
            if close_col is None:
                print(f"  [错误] 无法确定收益率列。列名: {df.columns.tolist()}")
                sys.exit(1)
            df[close_col] = pd.to_numeric(df[close_col], errors="coerce")
            df["市场日收益率"] = df[close_col].pct_change()
            ret_col = "市场日收益率"

    # 标准化日期并按时间排序，确保收益率时序正确
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    # 如果是收盘价回退路径，这里再计算收益率，避免因原始乱序导致 pct_change 错位
    if ret_col == "市场日收益率":
        close_col = None
        for c in df.columns:
            c_str = str(c)
            if "收盘" in c_str or "close" in c_str.lower():
                close_col = c
                break
        if close_col is None:
            print(f"  [错误] 市场收益率回退路径未找到收盘列。列名: {df.columns.tolist()}")
            sys.exit(1)
        df[close_col] = pd.to_numeric(df[close_col], errors="coerce")
        df[ret_col] = df[close_col].pct_change()

    df[date_col] = df[date_col].dt.strftime("%Y%m%d")
    df[ret_col] = pd.to_numeric(df[ret_col], errors="coerce")

    # 如果涨跌幅是百分比形式 (>1 说明是百分比)，转为小数
    if df[ret_col].abs().median() > 1:
        df[ret_col] = df[ret_col] / 100.0

    market_ret = df.set_index(date_col)[ret_col].dropna()
    market_ret = market_ret[~market_ret.index.duplicated(keep="last")]
    market_ret = market_ret.sort_index()
    print(f"  Wind全A 数据: {len(market_ret)} 个交易日, {market_ret.index[0]} ~ {market_ret.index[-1]}")
    return market_ret


def compute_stock_factor(
    stock_path: str,
    market_ret: pd.Series,
    weights: np.ndarray,
    tr_window: int = TR_WINDOW,
    factor_window: int = FACTOR_WINDOW,
) -> pd.DataFrame | None:
    """
    计算单只股票的 TR-Adjusted UMR 因子。

    返回 DataFrame: columns = [交易日期, factor_value]
    """
    try:
        df = pd.read_pickle(stock_path)
    except Exception as e:
        print(f"  [警告] 读取失败: {stock_path} ({e})")
        return None

    if df.empty or len(df) < factor_window + tr_window:
        return None

    # 确保列存在
    required = ["交易日期", "最高价(元)", "最低价(元)", "昨收盘价(元)", "收盘价(元)"]
    for col in required:
        if col not in df.columns:
            return None

    # 转换数据类型
    df = df.copy()
    df["交易日期"] = df["交易日期"].astype(str).str[:8]
    for col in ["最高价(元)", "最低价(元)", "昨收盘价(元)", "收盘价(元)"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("交易日期").reset_index(drop=True)
    df = df.dropna(subset=["最高价(元)", "最低价(元)", "昨收盘价(元)", "收盘价(元)"])

    # 过滤掉 preclose=0 的行 (涨停/停牌等异常数据)
    df = df[df["昨收盘价(元)"] > 0].copy()

    if len(df) < factor_window + tr_window:
        return None

    high = df["最高价(元)"].values
    low = df["最低价(元)"].values
    preclose = df["昨收盘价(元)"].values
    close = df["收盘价(元)"].values

    # ---- Step 1: TR ----
    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - preclose), np.abs(low - preclose))
    ) / preclose

    # ---- Step 2: TR_MA10 ----
    # 用过去 tr_window 日(不含当日)的 TR 均值，避免与当日 TR 重叠
    tr_series = pd.Series(tr)
    tr_ma = tr_series.rolling(window=tr_window, min_periods=tr_window).mean().shift(1).values

    # ---- Step 3: 系数 = TR_MA10 - TR ----
    coeff = tr_ma - tr

    # ---- Step 4: 个股日收益率 & 超额收益 ----
    stock_ret = (close - preclose) / preclose  # 日收益率

    # 匹配 market return
    dates = df["交易日期"].values
    mkt_ret_aligned = market_ret.reindex(dates).to_numpy(dtype=float)

    # 对齐后剔除市场收益缺失的日期，避免 rolling 窗口被 NaN 大面积污染
    valid_mask = np.isfinite(mkt_ret_aligned)
    if valid_mask.sum() < factor_window + tr_window:
        return None

    dates = dates[valid_mask]
    tr = tr[valid_mask]
    coeff = coeff[valid_mask]
    preclose = preclose[valid_mask]
    close = close[valid_mask]
    mkt_ret_aligned = mkt_ret_aligned[valid_mask]

    excess_ret = stock_ret - mkt_ret_aligned

    # ---- Step 5: 调整后收益 ----
    adjusted_ret = coeff * excess_ret

    # ---- Step 6: 半衰加权 ----
    adj_series = pd.Series(adjusted_ret)
    factor_values = adj_series.rolling(
        window=factor_window, min_periods=factor_window
    ).apply(lambda x: np.dot(x, weights), raw=True).shift(1).values

    # 构建结果
    result = pd.DataFrame({
        "交易日期": dates,
        "factor_value": factor_values,
    })
    result = result.dropna(subset=["factor_value"])

    if result.empty:
        return None

    return result


def main():
    parser = argparse.ArgumentParser(
        description="计算 TR-Adjusted UMR 因子",
    )
    parser.add_argument("--codes", nargs="+", default=None, help="仅计算指定股票")
    parser.add_argument("--start", type=str, default=None, help="仅保留该日期之后的因子值")
    parser.add_argument("--skip-existing", action="store_true", help="跳过已有输出")
    parser.add_argument("--stock-dir", type=str, default=STOCK_DATA_DIR)
    parser.add_argument("--market-xlsx", type=str, default=MARKET_XLSX)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 加载 Wind全A 收益率
    market_ret = load_market_return(args.market_xlsx)

    # 构建半衰权重
    weights = build_halflife_weights(FACTOR_WINDOW, HALF_LIFE)

    # 获取股票文件列表
    if args.codes:
        stock_files = [
            os.path.join(args.stock_dir, f"{code}.pickle") for code in args.codes
        ]
        stock_files = [f for f in stock_files if os.path.exists(f)]
    else:
        stock_files = sorted(glob.glob(os.path.join(args.stock_dir, "*.pickle")))

    print(f"\n待计算股票: {len(stock_files)} 只")
    print(f"输出目录: {args.output_dir}")
    print(f"TR窗口={TR_WINDOW}, 因子窗口={FACTOR_WINDOW}, 半衰期={HALF_LIFE}")
    print()

    t0 = time.time()
    saved = 0
    skipped = 0
    failed = 0

    for i, fp in enumerate(stock_files, 1):
        code = os.path.splitext(os.path.basename(fp))[0]
        out_path = os.path.join(args.output_dir, f"{code}.pickle")

        if args.skip_existing and os.path.exists(out_path):
            skipped += 1
            continue

        result = compute_stock_factor(fp, market_ret, weights)

        if result is None:
            failed += 1
            continue

        # 可选: 过滤起始日期
        if args.start:
            result = result[result["交易日期"] >= args.start]

        if result.empty:
            failed += 1
            continue

        result.to_pickle(out_path)
        saved += 1

        if saved % 500 == 0 or i == len(stock_files):
            elapsed = time.time() - t0
            print(f"  进度: {i}/{len(stock_files)}, 已保存 {saved}, 跳过 {skipped}, 失败 {failed}, 耗时 {elapsed:.1f}s")

    total_time = time.time() - t0
    print(f"\n{'='*60}")
    print(f"完成! 保存 {saved} 个, 跳过 {skipped} 个, 失败 {failed} 个")
    print(f"输出目录: {args.output_dir}")
    print(f"总耗时: {total_time:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
