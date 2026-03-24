"""
download_tushare_financials.py
------------------------------
拉取 A股全量财务数据，按股票分别保存为 pickle 文件。

数据来源 (tushare):
  1. pro.income()           利润表
  2. pro.balancesheet()     资产负债表
  3. pro.cashflow()         现金流量表
  4. pro.fina_indicator()   财务指标数据

输出:
  E:/1_basement/quant_research/data/中国A股财务数据tushare/{ts_code}.pickle

用法:
  python download_tushare_financials.py
"""

import os
import time
import json
import argparse
from pathlib import Path

import tushare as ts
import pandas as pd

# ─── 配置 ────────────────────────────────────────────────────────
_THIS_DIR    = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parent.parent.parent   # universe -> src -> quant_research -> 1_basement

BASE_CONF = PROJECT_ROOT / "conf" / "base.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "中国A股财务数据tushare"

SLEEP_SEC = 0.4  # API 调用间隔，防止触发限流 (Tushare 通常 200次/分钟以上，保守一点)
MAX_RETRIES = 5  # API 失败重试次数

# 获取 token
try:
    with open(BASE_CONF, "r", encoding="utf-8-sig") as _f:
        _base = json.load(_f)
        TOKEN = _base.get("tushare_token", "")
except Exception as e:
    # 尝试从另一个配置取
    TOKEN = "2d4f555869182905bfd48ce1fd0f649015f2bf10b3ef4a7e558573bb" # Fallback from update_ashare_eod.py

if not TOKEN:
    raise ValueError("未找到 tushare token，请检查配置。")

ts.set_token(TOKEN)
pro = ts.pro_api()


# ─── 辅助函数 ──────────────────────────────────────────────────────
def fetch_with_retry(api_func, **kwargs):
    """带重试机制的 tushare API 请求"""
    for attempt in range(MAX_RETRIES):
        try:
            df = api_func(**kwargs)
            time.sleep(SLEEP_SEC)
            return df
        except Exception as e:
            err_msg = str(e)
            if "抱歉，您每分钟最多访问" in err_msg or "频次" in err_msg:
                wait_time = (attempt + 1) * 20
                print(f"    [限流] 等待 {wait_time} 秒后重试 ({attempt+1}/{MAX_RETRIES})...")
                time.sleep(wait_time)
            elif "接口异常" in err_msg or "网络" in err_msg or "timeout" in err_msg.lower():
                wait_time = (attempt + 1) * 5
                print(f"    [网络异常] 等待 {wait_time} 秒后重试 ({attempt+1}/{MAX_RETRIES})...")
                time.sleep(wait_time)
            else:
                print(f"    [未知异常] {err_msg}。等待 5 秒后重试 ({attempt+1}/{MAX_RETRIES})...")
                time.sleep(5)
    print(f"    [ERROR] API 请求失败，已达到最大重试次数。参数: {kwargs}")
    return None


def get_stock_list() -> list[str]:
    """获取所有A股股票列表 (包含退市、暂停上市等历史股票)"""
    print("获取全市场股票列表...")
    df_l = fetch_with_retry(pro.stock_basic, exchange='', list_status='L')  # 上市
    df_d = fetch_with_retry(pro.stock_basic, exchange='', list_status='D')  # 退市
    df_p = fetch_with_retry(pro.stock_basic, exchange='', list_status='P')  # 暂停上市
    
    dfs = [df for df in [df_l, df_d, df_p] if df is not None and not df.empty]
    if not dfs:
        raise RuntimeError("无法获取股票列表")
        
    df = pd.concat(dfs, ignore_index=True)
    stocks = sorted(df["ts_code"].unique().tolist())
    print(f"共获取到 {len(stocks)} 只股票。")
    return stocks


def process_stock(ts_code: str):
    """下载并合并单只股票的三表及指标，返回 DataFrame"""
    # 1. 利润表 income
    df_inc = fetch_with_retry(pro.income, ts_code=ts_code)
    # 2. 资产负债表 balancesheet
    df_bal = fetch_with_retry(pro.balancesheet, ts_code=ts_code)
    # 3. 现金流量表 cashflow
    df_cf = fetch_with_retry(pro.cashflow, ts_code=ts_code)
    # 4. 财务指标 fina_indicator
    df_ind = fetch_with_retry(pro.fina_indicator, ts_code=ts_code)

    # 如果没有任何数据，则跳过
    dfs = []
    for df, name in [(df_inc, "inc"), (df_bal, "bal"), (df_cf, "cf"), (df_ind, "ind")]:
        if df is not None and not df.empty:
            # 数据可能包含重复行 (例如公司更正公告导致多条，tushare中update_flag会有不同)
            # 我们保留 update_flag 最大的 / 最新的
            if "update_flag" in df.columns:
                df = df.sort_values(by=["end_date", "ann_date", "update_flag"], ascending=[True, True, False])
                df = df.drop_duplicates(subset=["end_date"], keep="first")
            else:
                df = df.sort_values(by=["end_date", "ann_date"])
                df = df.drop_duplicates(subset=["end_date"], keep="last")
                
            # 重命名除关联键外的列，防止合并时列名冲突
            rename_dict = {}
            for col in df.columns:
                if col not in ["ts_code", "ann_date", "f_ann_date", "end_date", "report_type", "comp_type", "update_flag"]:
                    rename_dict[col] = f"{col}_{name}"
            df = df.rename(columns=rename_dict)
            dfs.append(df)

    if not dfs:
        return None

    # 合并（以 end_date 为主键，使用 outer join 确保不错过任何季报）
    # 注意：为了保留真实的 ann_date，合并时如果有冲突，优先保留最晚的 ann_date
    res = dfs[0]
    for df in dfs[1:]:
        res = pd.merge(res, df, on=["ts_code", "end_date"], how="outer", suffixes=("", "_drop"))
        
        # 处理 ann_date
        if "ann_date_drop" in res.columns:
            # 如果两个表都有 ann_date，取非空的。如果都非空，理论上是一致的，随便取一个
            res["ann_date"] = res["ann_date"].fillna(res["ann_date_drop"])
            res = res.drop(columns=["ann_date_drop"])
        
        # 删除其他冗余的关联列 (f_ann_date, report_type)
        drop_cols = [c for c in res.columns if c.endswith("_drop")]
        res = res.drop(columns=drop_cols)

    # 排序
    # 主要按 end_date (报告期) 排序
    res = res.sort_values("end_date").reset_index(drop=True)
    
    return res


# ─── 主模块 ────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="下载全A股 Tushare 全量财务数据")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有文件")
    parser.add_argument("--stock", type=str, help="只下载指定的单个股票代码，如 000001.SZ")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"保存目录: {OUTPUT_DIR}")

    if args.stock:
        stocks = [args.stock]
    else:
        stocks = get_stock_list()

    total = len(stocks)
    saved = 0
    skipped = 0
    failed = 0
    
    t0 = time.time()

    for i, ts_code in enumerate(stocks, 1):
        fpath = OUTPUT_DIR / f"{ts_code}.pickle"
        
        # 断点续传
        if fpath.exists() and not args.overwrite:
            skipped += 1
            if i % 100 == 0:
                print(f"[{i:04d}/{total}] {ts_code} 已存在，跳过...")
            continue

        print(f"[{i:04d}/{total}] 正在拉取 {ts_code} 财务数据 ...")
        
        try:
            df = process_stock(ts_code)
            if df is not None and not df.empty:
                df.to_pickle(fpath)
                saved += 1
            else:
                print(f"  [WARN] {ts_code} 无财务数据。")
                failed += 1
        except Exception as e:
            print(f"  [ERROR] {ts_code} 处理失败: {e}")
            failed += 1

    t1 = time.time()
    print("-" * 60)
    print("下载完成！")
    print(f"总计: {total} 只股票")
    print(f"成功: {saved} 个")
    print(f"跳过: {skipped} 个")
    print(f"失败/无数据: {failed} 个")
    print(f"耗时: {(t1 - t0) / 60:.1f} 分钟")

if __name__ == "__main__":
    main()
