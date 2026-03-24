"""
build_financial_panel.py
------------------------
将 A 股五张财务数据表（利润表、资产负债表、现金流量表、业绩快报、业绩预告）
的关键字段合并后按股票代码拆分，输出每家公司一个 pickle 文件。

输出目录  : data/中国A股三表_快报_预报/  (由 data_sources.json 配置)
输出文件名 : <Wind代码>.pickle   例如 000001.SZ.pickle

用法:
  python build_financial_panel.py                         # 全量处理
  python build_financial_panel.py --codes 000001.SZ 600000.SH  # 仅处理指定股票
  python build_financial_panel.py --overwrite              # 覆盖已有文件
"""

import os
import json
import argparse
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────
# 配置加载
# ──────────────────────────────────────────────────────────────
_THIS_DIR    = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parent.parent.parent

DS_CONF = PROJECT_ROOT / "conf" / "data" / "data_sources.json"

def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)

try:
    _ds = _load_json(DS_CONF)
except FileNotFoundError:
    print(f"Warning: DS_CONF not found at {DS_CONF}. Using local fallback paths in E:\\1_basement\\quant_research\\data.")
    _ds = {
        "income_dir": r"E:\1_basement\quant_research\data\中国A股利润表_AShareIncome",
        "balance_sheet_dir": r"E:\1_basement\quant_research\data\中国A股资产负债表_AShareBalanceSheet",
        "cashflow_dir": r"E:\1_basement\quant_research\data\中国A股现金流量表_AShareCashFlow",
        "profit_express_dir": r"E:\1_basement\quant_research\data\中国A股业绩快报_AShareProfitExpress",
        "profit_notice_dir": r"E:\1_basement\quant_research\data\中国A股业绩预告_AShareProfitNotice",
        "financial_panel_dir": r"E:\1_basement\quant_research\data\中国A股三表_快报_预报"
    }

def _abs(rel: str) -> Path:
    return PROJECT_ROOT / rel

# 五张源表路径
INCOME_PICKLE   = _abs(_ds["income_dir"])          / "利润表.pickle"
BALANCE_PICKLE  = _abs(_ds["balance_sheet_dir"])   / "资产负债表.pickle"
CASHFLOW_PICKLE = _abs(_ds["cashflow_dir"])        / "现金流量表.pickle"
EXPRESS_PICKLE  = _abs(_ds["profit_express_dir"])  / "业绩快报.pickle"
NOTICE_PICKLE   = _abs(_ds["profit_notice_dir"])   / "业绩预告.pickle"

# 输出目录
OUTPUT_REL = _ds.get("financial_panel_dir", "data/中国A股三表_快报_预报")
OUTPUT_DIR = _abs(OUTPUT_REL)

MAP_PATH = r"E:\1_basement\quant_research\data\中国A股利润表_AShareIncome\报表类型.xlsx"
try:
    df_map = pd.read_excel(MAP_PATH)
    df_map = df_map[['Unnamed: 1', '报表类型']].dropna()
    cons_df = df_map[df_map['报表类型'].astype(str).str.contains('合并')]
    CONSOLIDATED_CODES = cons_df['Unnamed: 1'].astype(float).unique().tolist()
    CONSOLIDATED_CODES += [int(c) for c in CONSOLIDATED_CODES]
except Exception as e:
    print(f"Warning: Failed to load report_type mapping from {MAP_PATH}: {e}")
    CONSOLIDATED_CODES = []

# ──────────────────────────────────────────────────────────────
# 每张表只提取的列 + 列名映射
# ──────────────────────────────────────────────────────────────
# 统一列名
CODE_COL   = "ts_code"        # 股票代码
ANN_COL    = "ann_dt"         # 公告日期
PERIOD_COL = "report_period"  # 报告期
SOURCE_COL = "_source"        # 来源标记

# 各表需要提取的原始列名 → 统一列名
INCOME_COLS = {
    "Wind代码":    CODE_COL,
    "公告日期":    ANN_COL,
    "报告期":      PERIOD_COL,
    "报表类型代码": "report_type",
    "净利润(不含少数股东损益)": "net_profit_parent",
    "净利润(含少数股东损益)": "net_profit_total",
}

BALANCE_COLS = {
    "Wind代码": CODE_COL,
    "公告日期": ANN_COL,
    "报告期":   PERIOD_COL,
    "报表类型": "report_type",
    "股东权益合计(不含少数股东权益)": "equity_parent",
    "资产总计": "total_assets",
}

CASHFLOW_COLS = {
    "Wind代码": CODE_COL,
    "公告日期": ANN_COL,
    "报告期":   PERIOD_COL,
    "报表类型": "report_type",
}

EXPRESS_COLS = {
    "Wind代码":    CODE_COL,
    "首次公告日期": ANN_COL,
    "报告期":      PERIOD_COL,
}

NOTICE_COLS = {
    "Wind代码":      CODE_COL,
    "最新公告日期":   ANN_COL,
    "报告期":        PERIOD_COL,
    "业绩预告类型代码": "notice_type",
    "首次公告日":     "first_ann_dt",
    "业绩预告摘要":   "notice_summary",
    "业绩变动原因":   "notice_reason",
}

# ──────────────────────────────────────────────────────────────
# 日志
# ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────

def load_table(path: Path, col_map: dict, source_tag: str) -> pd.DataFrame:
    """
    读取 pickle，只选取 col_map 中指定的列并重命名。
    处理重复列名（先通过位置去重），添加来源标记。
    """
    log.info(f"Loading {source_tag} from {path} ...")
    df = pd.read_pickle(path)
    log.info(f"  raw shape: {df.shape}")

    # 处理重复列名：只保留每个列名的第一次出现
    df = df.loc[:, ~df.columns.duplicated()]

    # 只提取需要的列
    available = [c for c in col_map.keys() if c in df.columns]
    missing   = [c for c in col_map.keys() if c not in df.columns]
    if missing:
        log.warning(f"  [{source_tag}] 缺少以下列，将跳过：{missing}")

    df = df[available].rename(columns=col_map)
    df[SOURCE_COL] = source_tag

    # 增加过滤：只保留“合并”类报表 (防止包含母公司报表的干扰)
    if 'report_type' in df.columns:
        df['report_type_numeric'] = pd.to_numeric(df['report_type'], errors='coerce')
        if 'CONSOLIDATED_CODES' in globals() and CONSOLIDATED_CODES:
            df_is_code = df['report_type_numeric'].isin(CONSOLIDATED_CODES) | df['report_type'].astype(str).str.contains('合并', na=False)
            df = df[df_is_code]
        else:
            df = df[df['report_type'].astype(str).str.contains('合并', na=False)]
        
        if 'report_type_numeric' in df.columns:
            df = df.drop(columns=['report_type_numeric'])

    # 日期列转数字
    for dt_col in [ANN_COL, PERIOD_COL]:
        if dt_col in df.columns:
            df[dt_col] = pd.to_numeric(df[dt_col], errors="coerce")

    log.info(f"  selected shape: {df.shape}, cols={df.columns.tolist()}")
    return df


def merge_tables(tables: list[pd.DataFrame]) -> pd.DataFrame:
    """垂直拼接所有表，按 (代码, 公告日, 报告期) 排序。"""
    log.info("Concatenating tables ...")
    merged = pd.concat(tables, axis=0, ignore_index=True, sort=False)
    log.info(f"  concat shape: {merged.shape}")

    # 排序
    sort_keys = [c for c in [CODE_COL, ANN_COL, PERIOD_COL] if c in merged.columns]
    merged = merged.sort_values(sort_keys).reset_index(drop=True)
    log.info(f"  sorted shape: {merged.shape}")
    return merged


def split_and_save(
    merged: pd.DataFrame,
    output_dir: Path,
    target_codes: list[str] | None,
    overwrite: bool,
) -> None:
    """按股票代码拆分，每家公司保存为 <Code>.pickle。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    all_codes = sorted(merged[CODE_COL].dropna().unique())
    if target_codes:
        target_set = set(target_codes)
        all_codes = [c for c in all_codes if c in target_set]

    log.info(f"Saving {len(all_codes)} company files to {output_dir} ...")
    saved, skipped = 0, 0
    for code in tqdm(all_codes, desc="Saving"):
        out_path = output_dir / f"{code}.parquet"
        if out_path.exists() and not overwrite:
            skipped += 1
            continue
        sub = merged[merged[CODE_COL] == code].reset_index(drop=True)
        sub.to_parquet(out_path, index=False)
        saved += 1

    log.info(f"  saved: {saved}, skipped: {skipped}")


# ──────────────────────────────────────────────────────────────
# 主程序
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="合并 A 股三表+快报+预报关键字段 → 每公司 pickle")
    parser.add_argument("--codes", nargs="+", default=None,
                        help="仅处理指定股票代码（Wind 格式）")
    parser.add_argument("--overwrite", action="store_true",
                        help="覆盖已存在的 pickle 文件")
    parser.add_argument("--no-notice",  action="store_true", help="跳过业绩预告")
    parser.add_argument("--no-express", action="store_true", help="跳过业绩快报")
    args = parser.parse_args()

    # 1. 逐表加载（只提取关键列）
    tables = [
        load_table(INCOME_PICKLE,   INCOME_COLS,   "利润表"),
        load_table(BALANCE_PICKLE,  BALANCE_COLS,  "资产负债表"),
        load_table(CASHFLOW_PICKLE, CASHFLOW_COLS, "现金流量表"),
    ]
    if not args.no_express:
        tables.append(load_table(EXPRESS_PICKLE, EXPRESS_COLS, "业绩快报"))
    if not args.no_notice:
        tables.append(load_table(NOTICE_PICKLE,  NOTICE_COLS,  "业绩预告"))

    # 2. 合并
    merged = merge_tables(tables)

    # 3. 拆分保存
    split_and_save(merged, OUTPUT_DIR, args.codes, args.overwrite)

    log.info("All done.")
    n_files = len(list(OUTPUT_DIR.glob("*.parquet")))
    print(f"\n✓ 输出目录: {OUTPUT_DIR}")
    print(f"✓ 共 {n_files} 个公司 parquet 文件")


if __name__ == "__main__":
    main()
