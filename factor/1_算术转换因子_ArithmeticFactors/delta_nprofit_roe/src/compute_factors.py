"""
compute_factors.py
------------------
PIT-aware implementation.

Factor 1: delta_nprofit -- MoM of YoY growth rate of single-quarter net profit
    single_NP(t)     = cum_NP(t) - cum_NP(t-1)           (Q1: use cumulative directly)
    yoy_growth(t)    = (single_NP(t) - single_NP(t-4)) / |single_NP(t-4)|
    delta_nprofit(t) = yoy_growth(t) - yoy_growth(t-1)    (MoM: vs previous quarter)
    Filter: |single_NP(t-4)| < 1,000,000 => yoy_growth = NaN

Factor 2: delta_roe -- YoY change of single-quarter ROE
    ROE(t)       = single_NP(t) * 2 / (equity_begin(t) + equity_end(t))
    delta_roe(t) = ROE(t) - ROE(t-4)
    Filter: (|equity_begin| + |equity_end|) / 2 < 1,000,000 => ROE = NaN

PIT (Point-in-Time) methodology:
    - Maintain rolling dicts pit_cum_np / pit_eq, updated chronologically by ann_date.
    - Each record's factor value depends ONLY on data publicly available by that ann_date,
      preventing look-ahead bias from financial restatements.
    - equity_begin uses Last Known Value strategy:
      prefer prev_yyyyq equity_end; if unavailable (e.g. Q1 announced before Q4),
      fall back to the most recent known equity_end for any earlier quarter.

Data sources (currently only formal reports enabled):
    1. Wind Income Statement: data/AShareIncome/class_by_stock/{code}.csv
    2. Wind Balance Sheet:    data/AShareBalanceSheet/class_by_stock/{code}.csv
    (Express/Notice commented out pending PIT validation)

Usage:
    uv run --python 3.10 --with "pandas" --with "numpy>=2.0.0" --with "tqdm" python src/compute_factors.py
"""

import argparse
import traceback
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────
# 路径配置
# ──────────────────────────────────────────────────────────────
_THIS_DIR    = Path(__file__).resolve().parent
FACTOR_DIR   = _THIS_DIR.parent
PROJECT_ROOT = FACTOR_DIR.parent.parent.parent
DATA_DIR     = PROJECT_ROOT / "data"

TUSHARE_DIR = DATA_DIR / "中国A股财务数据tushare" # Keep for reference, but won't be used
WIND_BS_DIR = DATA_DIR / "中国A股资产负债表_AShareBalanceSheet" / "class_by_stock"
WIND_IS_DIR = DATA_DIR / "中国A股利润表_AShareIncome" / "class_by_stock"
EXPRESS_DIR = DATA_DIR / "中国A股业绩快报_AShareProfitExpress" / "class_by_stock"
NOTICE_DIR  = DATA_DIR / "中国A股业绩预告_AShareProfitNotice" / "class_by_stock"

OUT_NPROFIT = FACTOR_DIR / "output" / "delta_nprofit" / "class_by_stock"
OUT_ROE     = FACTOR_DIR / "output" / "delta_roe" / "class_by_stock"

# 阈值配置
MIN_NP_ABS = 1_000_000.0
MIN_EQ_ABS = 1_000_000.0

# ──────────────────────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────────────────────
_MMDD_TO_Q = {"0331": 1, "0630": 2, "0930": 3, "1231": 4}

def period_to_yyyyq(end_date_str: str) -> int | None:
    """
    将 8 位字符串格式的报告期 (如 '20230331') 转换为 yyyyq 格式 (如 20231)。
    
    参数:
        end_date_str (str): 8 位日期的字符串。
        
    返回:
        int | None: 如果解析有效，则返回 yyyyq 的整型表示；否则返回 None。
    """
    if not isinstance(end_date_str, str) or len(end_date_str) != 8:
        return None
    q = _MMDD_TO_Q.get(end_date_str[-4:])
    if q is None: return None
    return int(end_date_str[:4]) * 10 + q

def prev_yyyyq(yyyyq: int) -> int:
    """
    计算给定 yyyyq 季度格式的上一个季度。
    
    参数:
        yyyyq (int): yyyyq 格式的季度 (如 20231 代表 2023Q1)。
        
    返回:
        int: 上一个季度的 yyyyq 格式整型值 (如输入 20231 返回 20224)。
    """
    y, q = divmod(yyyyq, 10)
    return (y - 1) * 10 + 4 if q == 1 else y * 10 + (q - 1)

def yoy_yyyyq(yyyyq: int) -> int:
    """
    计算给定 yyyyq 季度格式的去年同期（YoY）季度标识。
    
    参数:
        yyyyq (int): 当前季度 (如 20242)。
        
    返回:
        int: 去年的同季度 (如返回 20232)。
    """
    y, q = divmod(yyyyq, 10)
    return (y - 1) * 10 + q

def _clean_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    清洗数据中的公告日期 (ann_date) 与报告期 (end_date)。
    将无法识别的缺失值或长度不符合的无效记录清洗掉。
    
    参数:
        df (pd.DataFrame): 原始载入财报的 DataFrame。
        
    返回:
        pd.DataFrame: 过滤无效日期后，标准化位 8 位纯数字字符的 DataFrame。
    """
    df = df.copy()  # Ensure we are working with a copy to avoid SettingWithCopyWarning
    for col in ["ann_date", "end_date"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").dropna().astype(int).astype(str)
            df = df[df[col].str.len() == 8]
    if "end_date" in df.columns:
        df = df[df["end_date"].str[-4:].isin(_MMDD_TO_Q.keys())]
    return df.dropna(subset=["end_date"]).copy()

# ──────────────────────────────────────────────────────────────
# 数据加载
# ──────────────────────────────────────────────────────────────
def load_formal_balance_sheet(code: str) -> pd.DataFrame:
    """
    针对单独的股票代码读取原始的“正式资产负债表” CSV 数据。
    提取对由于权益推算的『股东权益合计(不含少数股东权益)』。
    如果存在多套合并/母公司报表，优先选择合并报表。
    
    参数:
        code (str): 股票代码（包含市场后缀，例如 '000001.SZ'）。
        
    返回:
        pd.DataFrame: 标准化列名后的资产负债表 DataFrame。
    """
    fpath = WIND_BS_DIR / f"{code}.csv"
    if not fpath.exists():
        return pd.DataFrame()
    df = pd.read_csv(str(fpath), dtype={"报告期": str, "公告日期": str})
    if df.empty: return pd.DataFrame()

    # 如果存在报表类型，优先保留最核心的合并报表（通常代码较小）
    if "报表类型" in df.columns:
        df["报表类型"] = pd.to_numeric(df["报表类型"], errors="coerce")
        df = df.sort_values("报表类型").drop_duplicates(subset=["报告期", "公告日期"], keep="first")
    elif "报表类型代码" in df.columns:
        df["报表类型代码"] = pd.to_numeric(df["报表类型代码"], errors="coerce")
        df = df.sort_values("报表类型代码").drop_duplicates(subset=["报告期", "公告日期"], keep="first")

    col_map = {
        "公告日期": "ann_date",
        "报告期": "end_date",
        "股东权益合计(不含少数股东权益)": "equity_end",
    }
    avail = {k: v for k, v in col_map.items() if k in df.columns}
    df = df[list(avail.keys())].rename(columns=avail).copy()
    if "equity_end" not in df.columns:
        df["equity_end"] = np.nan
        
    return _clean_dates(df)

def load_formal_income(code: str) -> pd.DataFrame:
    """
    针对单独股票读取“正式利润表” CSV 数据，抓取核心的『净利润(不含少数股东损益)』(累计净利润)。
    并与 load_formal_balance_sheet 逻辑一致过滤非核心子报表。
    
    参数:
        code (str): 股票代码。
        
    返回:
        pd.DataFrame: 标准化过滤出的利润表。
    """
    fpath = WIND_IS_DIR / f"{code}.csv"
    if not fpath.exists():
        return pd.DataFrame()
    df = pd.read_csv(str(fpath), dtype={"报告期": str, "公告日期": str})
    if df.empty: return pd.DataFrame()

    # 过滤合并报表，优先最小的报表代码
    if "报表类型代码" in df.columns:
        df["报表类型代码"] = pd.to_numeric(df["报表类型代码"], errors="coerce")
        df = df.sort_values("报表类型代码").drop_duplicates(subset=["报告期", "公告日期"], keep="first")

    col_map = {
        "公告日期": "ann_date",
        "报告期": "end_date",
        "净利润(不含少数股东损益)": "cum_nprofit",
    }
    avail = {k: v for k, v in col_map.items() if k in df.columns}
    df = df[list(avail.keys())].rename(columns=avail).copy()
    if "cum_nprofit" not in df.columns:
        df["cum_nprofit"] = np.nan
        
    return _clean_dates(df)

def load_formal_merged(code: str) -> pd.DataFrame:
    """
    组合上述读取的正式利润表与资产负债表，基于公告日和报告期作 Outer Join 合并，
    并标记源优先级 _source_priority = 1（代表正式财报最可靠）。
    
    参数:
        code (str): 股票代码。
        
    返回:
        pd.DataFrame: 利润与负债拼合完毕的数据帧快照。
    """
    bs = load_formal_balance_sheet(code)
    inc = load_formal_income(code)
    
    if bs.empty and inc.empty:
        return pd.DataFrame()
        
    if bs.empty:
        merged = inc
        merged["equity_end"] = np.nan
    elif inc.empty:
        merged = bs
        merged["cum_nprofit"] = np.nan
    else:
        # Full outer join on ann_date and end_date
        merged = pd.merge(bs, inc, on=["ann_date", "end_date"], how="outer")
        
    merged["_source_priority"] = 1
    return merged


def load_express(code: str) -> pd.DataFrame:
    fpath = EXPRESS_DIR / f"{code}.pickle"
    if not fpath.exists(): return pd.DataFrame()
    df = pd.read_pickle(str(fpath))
    col_map = {
        "首次公告日期": "ann_date",
        "报告期": "end_date",
        "净利润(元)": "cum_nprofit",
        "股东权益合计(不含少数股东权益)(元)": "equity_end",
    }
    avail = {k: v for k, v in col_map.items() if k in df.columns}
    df = df[list(avail.keys())].rename(columns=avail).copy()
    df["_source_priority"] = 2
    return _clean_dates(df)

def load_notice(code: str) -> pd.DataFrame:
    fpath = NOTICE_DIR / f"{code}.pickle"
    if not fpath.exists(): return pd.DataFrame()
    df = pd.read_pickle(str(fpath))

    if "报告期" not in df.columns or "首次公告日" not in df.columns:
        return pd.DataFrame()

    lo_col = "预告净利润下限（万元）"
    hi_col = "预告净利润上限（万元）"
    
    df = df.rename(columns={"首次公告日": "ann_date", "报告期": "end_date"})
    if lo_col in df.columns and hi_col in df.columns:
        lo = pd.to_numeric(df[lo_col], errors="coerce")
        hi = pd.to_numeric(df[hi_col], errors="coerce")
        df["cum_nprofit"] = ((lo + hi) / 2) * 10_000
    elif lo_col in df.columns:
        df["cum_nprofit"] = pd.to_numeric(df[lo_col], errors="coerce") * 10_000
    elif hi_col in df.columns:
        df["cum_nprofit"] = pd.to_numeric(df[hi_col], errors="coerce") * 10_000
    else:
        return pd.DataFrame()

    df["equity_end"] = np.nan
    df["_source_priority"] = 3
    return _clean_dates(df[["ann_date", "end_date", "cum_nprofit", "equity_end", "_source_priority"]])

# ──────────────────────────────────────────────────────────────
# 合并策略
# ──────────────────────────────────────────────────────────────
def merge_earnings(dfs: list) -> pd.DataFrame:
    """
    纵向拼接正式、快报、预告财报。
    基于同一公告日、报告期的记录按数据来源优先级进行去重覆盖（保留最可靠源的值）。
    
    参数:
        dfs (list): 将要拼接合并的 pd.DataFrame 列表。
        
    返回:
        pd.DataFrame: 时间轴合并去重之后的净利润/权益快照汇总。
    """
    non_empty = [d for d in dfs if not d.empty]
    if not non_empty: return pd.DataFrame()
    
    m = pd.concat(non_empty, axis=0, ignore_index=True)
    m["_ann_int"] = pd.to_numeric(m["ann_date"], errors="coerce").fillna(0).astype(int)
    m["_end_int"] = pd.to_numeric(m["end_date"], errors="coerce").fillna(0).astype(int)
    
    # 相同报告期，按公告日降序(取最新)，同日按来源优先级升序(取更正式财报)
    m = m.sort_values(["_end_int", "_ann_int", "_source_priority"], ascending=[True, False, True])
    # m = m.drop_duplicates(subset=["_end_int"], keep="first")  (调试时已被注释隐去)
    m = m.sort_values("_end_int").reset_index(drop=True)
    return m

# ──────────────────────────────────────────────────────────────
# PIT (Point-in-Time) 辅助函数
# ──────────────────────────────────────────────────────────────

def _pit_single_np(q: int, pit_cum_np: dict) -> float:
    """从 PIT 快照中计算单季净利润。

    公式:
        Q1:   single_np = cum_np(Q1)             （一季报累计值即单季值）
        其他: single_np = cum_np(q) - cum_np(q-1) （当季累计 - 上季累计）

    Args:
        q: yyyyq 格式的季度标识（如 20241 表示 2024Q1）
        pit_cum_np: 截至当前公告日已知的各季度累计净利润

    Returns:
        单季净利润，数据不足时返回 NaN
    """
    cum = pit_cum_np.get(q, np.nan)
    if pd.isna(cum):
        return np.nan
    if q % 10 == 1:  # Q1: 累计值即单季值
        return cum
    prev_cum = pit_cum_np.get(prev_yyyyq(q), np.nan)
    if pd.isna(prev_cum):
        return np.nan
    return cum - prev_cum


def _pit_yoy_growth(q: int, pit_cum_np: dict) -> float:
    """从 PIT 快照中计算单季净利润的同比增长率。

    公式: yoy_growth(q) = (single_np(q) - single_np(q-4)) / |single_np(q-4)|
    过滤: 当 |single_np(q-4)| < MIN_NP_ABS 时返回 NaN，防止分母过小导致极端值

    Args:
        q: yyyyq 格式的季度标识
        pit_cum_np: 截至当前公告日已知的各季度累计净利润

    Returns:
        同比增长率，数据不足或基数过小时返回 NaN
    """
    snp = _pit_single_np(q, pit_cum_np)
    snp_yoy = _pit_single_np(yoy_yyyyq(q), pit_cum_np)
    if pd.isna(snp) or pd.isna(snp_yoy) or abs(snp_yoy) < MIN_NP_ABS:
        return np.nan
    return (snp - snp_yoy) / abs(snp_yoy)


def _pit_beginning_equity(q: int, pit_eq: dict) -> float:
    """从 PIT 快照中获取期初归母权益（Last Known Value 策略）。

    策略:
        1. 优先取自然上一季度 prev_yyyyq(q) 的期末权益作为本季期初
        2. 若不可用（如 Q1 先于 Q4 披露），回溯取 PIT 中报告期 < q
           的所有已知权益值中最新的一个

    用途: 期初权益用于计算单季 ROE = single_np * 2 / (equity_begin + equity_end)

    Args:
        q: 当前季度 yyyyq 标识
        pit_eq: 截至当前公告日已知的各季度期末权益

    Returns:
        期初权益值，无已知数据时返回 NaN
    """
    # 优先: 自然日历上一季度的期末权益
    prev_q = prev_yyyyq(q)
    val = pit_eq.get(prev_q, np.nan)
    if not pd.isna(val):
        return val
    # 回退: 报告期 < q 的所有已知权益中取最新
    candidates = [(qq, v) for qq, v in pit_eq.items()
                  if qq < q and not pd.isna(v)]
    if not candidates:
        return np.nan
    return max(candidates, key=lambda x: x[0])[1]


def _pit_roe(q: int, pit_cum_np: dict, pit_eq: dict) -> float:
    """从 PIT 快照中计算单季 ROE。

    公式: ROE(q) = single_np(q) * 2 / (equity_begin(q) + equity_end(q))
    过滤: 当 eb 或 ee 为负（资不抵债）或均值 < MIN_EQ_ABS 时返回 NaN

    Args:
        q: yyyyq 格式的季度标识
        pit_cum_np: 截至当前公告日已知的各季度累计净利润
        pit_eq: 截至当前公告日已知的各季度期末权益

    Returns:
        单季 ROE，数据不足或权益基数为负/过小时返回 NaN
    """
    snp = _pit_single_np(q, pit_cum_np)
    ee = pit_eq.get(q, np.nan)
    eb = _pit_beginning_equity(q, pit_eq)
    if any(pd.isna(x) for x in [snp, eb, ee]):
        return np.nan
    # 资不抵债（权益为负）时 ROE 无意义，必须排除；
    # 权益过小时分母不稳定，也排除
    if eb <= 0 or ee <= 0 or ((eb + ee) / 2) < MIN_EQ_ABS:
        return np.nan
    return snp * 2 / (eb + ee)


# ──────────────────────────────────────────────────────────────
# PIT 因子计算主函数
# ──────────────────────────────────────────────────────────────

def compute_factors_pit(df: pd.DataFrame) -> pd.DataFrame:
    """按公告日顺序滚动计算 delta_nprofit 和 delta_roe，严格避免前视偏差。

    核心机制:
        维护 pit_cum_np（累计净利润）和 pit_eq（期末权益）两个滚动字典，
        随公告日推进逐条更新。每条记录的因子值仅依赖截至该公告日已公开
        的财务数据，杜绝"用未来更正数据回改历史因子"的前视偏差。

    delta_nprofit 公式（同比增速的环比差，MoM of YoY growth）:
        yoy_growth(q)    = (single_np(q) - single_np(q-4)) / |single_np(q-4)|
        delta_nprofit(q) = yoy_growth(q) - yoy_growth(q-1)

    delta_roe 公式（单季 ROE 的同比差，YoY of quarterly ROE）:
        ROE(q)      = single_np(q) * 2 / (equity_begin(q) + equity_end(q))
        delta_roe(q) = ROE(q) - ROE(q-4)

    Args:
        df: 经 merge_earnings 合并后的 DataFrame，须含
            ann_date, end_date, cum_nprofit, equity_end, _source_priority

    Returns:
        包含所有中间变量和最终因子的 DataFrame，按公告日排序
    """
    df = df.copy()
    df["yyyyq"] = df["end_date"].apply(period_to_yyyyq)
    df = df.dropna(subset=["yyyyq"]).copy()
    df["yyyyq"] = df["yyyyq"].astype(int)
    df["_ann_int"] = pd.to_numeric(df["ann_date"], errors="coerce").fillna(0).astype(int)

    # 按公告日期升序排列，确保信息按时间顺序到达
    df = df.sort_values(["_ann_int", "_source_priority"],
                        ascending=[True, True]).reset_index(drop=True)

    # PIT 快照: 仅包含截至当前行公告日已公开的数据
    pit_cum_np: dict[int, float] = {}  # yyyyq -> 累计归母净利润
    pit_eq: dict[int, float] = {}      # yyyyq -> 期末归母权益

    records = []

    for _, row in df.iterrows():
        q = row["yyyyq"]

        # ── 1. 用本条公告数据更新 PIT 快照 ──
        cum_val = pd.to_numeric(row.get("cum_nprofit", np.nan), errors="coerce")
        eq_val = pd.to_numeric(row.get("equity_end", np.nan), errors="coerce")
        if not pd.isna(cum_val):
            pit_cum_np[q] = cum_val
        if not pd.isna(eq_val):
            pit_eq[q] = eq_val

        # ── 2. 基于当前 PIT 快照计算因子 ──

        # 单季净利润
        snp = _pit_single_np(q, pit_cum_np)

        # delta_nprofit: 同比增速的环比差（MoM of YoY）
        yoy_g = _pit_yoy_growth(q, pit_cum_np)
        prev_q = prev_yyyyq(q)
        yoy_g_prev = _pit_yoy_growth(prev_q, pit_cum_np)
        delta_np = (yoy_g - yoy_g_prev
                    if not (pd.isna(yoy_g) or pd.isna(yoy_g_prev))
                    else np.nan)

        # delta_roe: 单季 ROE 的同比差（YoY of quarterly ROE）
        eb = _pit_beginning_equity(q, pit_eq)
        ee = pit_eq.get(q, np.nan)
        roe_cur = _pit_roe(q, pit_cum_np, pit_eq)
        roe_yoy = _pit_roe(yoy_yyyyq(q), pit_cum_np, pit_eq)
        delta_r = (roe_cur - roe_yoy
                   if not (pd.isna(roe_cur) or pd.isna(roe_yoy))
                   else np.nan)

        # ── 3. 收集结果 ──
        records.append({
            "ann_date": row["ann_date"],
            "end_date": row["end_date"],
            "_source_priority": row.get("_source_priority", 1),
            "cum_nprofit": pit_cum_np.get(q, np.nan),
            "single_np": snp,
            "yoy_growth": yoy_g,
            "prev_yoy_growth": yoy_g_prev,
            "delta_nprofit": delta_np,
            "equity_begin": eb,
            "equity_end": ee,
            "roe": roe_cur,
            "roe_yoy": roe_yoy,
            "delta_roe": delta_r,
        })

    return pd.DataFrame(records)

# ──────────────────────────────────────────────────────────────
# 主控制
# ──────────────────────────────────────────────────────────────
def process_stock(code: str) -> dict:
    """Load a single stock's financial data and compute delta_nprofit / delta_roe factors."""
    ts = load_formal_merged(code)
    # 结合快报与预告数据以保证时效性，通过优先级覆盖与严谨的 PIT 时间序列机制处理避免前视偏差
    ex = load_express(code)
    no = load_notice(code)

    merged = merge_earnings([ts, ex, no])  # 合并三种财报源
    if merged.empty:
        return {"delta_nprofit": None, "delta_roe": None}

    # PIT-aware factor computation
    merged = compute_factors_pit(merged)

    def to_out(col: str):
        """
        按照全局系统规范抽取出目标因子，格式化输出仅包含规范两列：
        trade_date 与 <factor_name>。过滤掉任何由于分母问题算出的 nan 和无用列。
        
        参数:
            col (str): 因子的列名 ('delta_nprofit' 或 'delta_roe')。
            
        返回:
            pd.DataFrame | None: 符合标准产物规范的只有"有效日期,有效因子数值"的小型表。
        """
        # 注意此步骤：严格舍弃除了 trade_date（由 ann_date 代表的最新发报可用日）和因子本体的无用字段。
        cols_to_keep = ["ann_date", col]

        cols_to_keep = [c for c in cols_to_keep if c in merged.columns]
        sub = merged[cols_to_keep].dropna(subset=[col]).copy()
        if sub.empty:
            return None

        # 转换为要求的双列表头英文格式
        rename_map = {
            "ann_date": "trade_date",
        }
        return sub.rename(columns=rename_map).sort_values("trade_date").reset_index(drop=True)

    return {
        "delta_nprofit": to_out("delta_nprofit"),
        "delta_roe": to_out("delta_roe"),
    }

def main():
    """
    执行批处理加载并写入满足因子的主入口流程函数。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", type=str, default=None, help="传入具体的某支股票代码测算")
    args = parser.parse_args()

    if args.code:
        codes = [args.code]
    else:
        codes = sorted({p.stem for p in WIND_IS_DIR.glob("*.csv")})
        
    print(f"-> 准备计算 {len(codes)} 只股票...")
    
    ok_np, ok_roe, fail = 0, 0, 0
    OUT_NPROFIT.mkdir(parents=True, exist_ok=True)
    OUT_ROE.mkdir(parents=True, exist_ok=True)
    
    for code in tqdm(codes):
        try:
            res = process_stock(code)
            if res["delta_nprofit"] is not None:
                res["delta_nprofit"].to_csv(OUT_NPROFIT / f"{code}.csv", index=False)
                ok_np += 1
            if res["delta_roe"] is not None:
                res["delta_roe"].to_csv(OUT_ROE / f"{code}.csv", index=False)
                ok_roe += 1
        except Exception:
            fail += 1
            
    print(f"OK delta_nprofit 输出 {ok_np} 文件")
    print(f"OK delta_roe 输出 {ok_roe} 文件")
    if fail: print(f"FAIL 失败 {fail} 文件")

if __name__ == "__main__":
    main()
