#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import math
import time

import pandas as pd


def _setup_logger(log_path):
    logger = logging.getLogger("prepare_factor_A021")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def _progress(logger, i, total, start_ts, hit_count):
    elapsed = time.perf_counter() - start_ts
    rate = i / elapsed if elapsed > 0 else 0.0
    remain = (total - i) / rate if rate > 0 else math.inf
    pct = i / total * 100 if total else 100.0
    eta_txt = f"{remain:.1f}s" if math.isfinite(remain) else "N/A"
    logger.info(
        "[Match] %.2f%% (%d/%d) | elapsed=%.1fs | eta=%s | lncap_matched_rows=%d",
        pct,
        i,
        total,
        elapsed,
        eta_txt,
        hit_count,
    )


def main():
    factor_raw = r"E:\1_basement\quant_research\back_test\sigle_factor_test\function_breakdown\factor_engineering\factor_A021_raw.parquet"
    pool_path = r"E:\1_basement\quant_research\back_test\sigle_factor_test\function_breakdown\stock_pool\original_stock_pool_with_st_and_first_dates_liq_mv__listed_only.parquet"
    ind_path = r"E:\1_basement\quant_research\data\申万行业分类2021版_AShareSWNIndustriesClass\申万行业分类2021版.pickle"
    out_path = r"E:\1_basement\quant_research\back_test\sigle_factor_test\function_breakdown\factor_engineering\factor_A021_for_orchestrate.parquet"
    log_path = r"E:\1_basement\quant_research\back_test\sigle_factor_test\function_breakdown\factor_engineering\prepare_factor_A021_for_orchestrate.log"

    logger = _setup_logger(log_path)
    t0 = time.perf_counter()

    logger.info("[Prep] 开始执行")
    logger.info("[Prep] 输出日志: %s", log_path)

    t = time.perf_counter()
    logger.info("[Prep] 读取因子原始文件")
    f = pd.read_parquet(factor_raw)
    f["code"] = f["code"].astype(str).str.upper().str.replace(r"\.(SZ|SH|BJ)$", "", regex=True).str.zfill(6)
    f["date"] = pd.to_datetime(f["date"], errors="coerce").dt.strftime("%Y%m%d")
    f["date"] = pd.to_numeric(f["date"], errors="coerce").astype("Int64")
    f = f.dropna(subset=["code", "date", "Factor_A021"]).copy()
    f["date"] = f["date"].astype("int64")
    logger.info("[Prep] 因子行数=%d, 代码数=%d, 耗时=%.2fs", len(f), f["code"].nunique(), time.perf_counter() - t)

    t = time.perf_counter()
    logger.info("[Prep] 读取池子中的市值并按 code+date 向后匹配")
    p = pd.read_parquet(pool_path, columns=["code", "date", "lncap"])
    p["code"] = p["code"].astype(str).str.zfill(6)
    p["date"] = pd.to_numeric(p["date"], errors="coerce").astype("Int64")
    p = p.dropna(subset=["code", "date"]).copy()
    p["date"] = p["date"].astype("int64")
    logger.info("[Prep] 市值行数=%d, 代码数=%d, 耗时=%.2fs", len(p), p["code"].nunique(), time.perf_counter() - t)

    t = time.perf_counter()
    f = f.sort_values(["code", "date"]).reset_index(drop=True)
    p = p.sort_values(["code", "date"]).reset_index(drop=True)
    logger.info("[Prep] 排序完成，耗时=%.2fs", time.perf_counter() - t)

    # 预分组避免每轮 p[p["code"]==code] 的全表扫描，显著降低耗时
    t = time.perf_counter()
    p_groups = {k: g[["date", "lncap"]].reset_index(drop=True) for k, g in p.groupby("code", sort=False)}
    logger.info("[Prep] 市值预分组完成，组数=%d，耗时=%.2fs", len(p_groups), time.perf_counter() - t)

    merged_parts = []
    grouped_f = list(f.groupby("code", sort=False))
    total_groups = len(grouped_f)
    logger.info("[Prep] 开始按代码 merge_asof，组数=%d", total_groups)

    t_match = time.perf_counter()
    hit_rows = 0
    for i, (code, fg) in enumerate(grouped_f, start=1):
        pg = p_groups.get(code)
        if pg is None or pg.empty:
            one = fg.copy()
            one["lncap"] = pd.NA
            merged_parts.append(one)
            if i % 200 == 0 or i == total_groups:
                _progress(logger, i, total_groups, t_match, hit_rows)
            continue
        one = pd.merge_asof(
            fg.sort_values("date"),
            pg.sort_values("date"),
            on="date",
            direction="backward",
            allow_exact_matches=True,
        )
        hit_rows += int(one["lncap"].notna().sum())
        merged_parts.append(one)

        if i % 200 == 0 or i == total_groups:
            _progress(logger, i, total_groups, t_match, hit_rows)

    t = time.perf_counter()
    merged = pd.concat(merged_parts, ignore_index=True)
    logger.info("[Prep] merge_asof 拼接完成，行数=%d，耗时=%.2fs", len(merged), time.perf_counter() - t)

    t = time.perf_counter()
    logger.info("[Prep] 读取行业并映射 ind_code")
    ind = pd.read_pickle(ind_path)
    ind = ind[["S_INFO_WINDCODE", "SW_IND_CODE", "最新标志"]].copy()
    ind["code"] = ind["S_INFO_WINDCODE"].astype(str).str.upper().str.replace(r"\.(SZ|SH|BJ)$", "", regex=True).str.zfill(6)
    ind = ind[ind["最新标志"].astype(str) == "1"]
    ind = ind.drop_duplicates(subset=["code"], keep="first")
    ind = ind[["code", "SW_IND_CODE"]].rename(columns={"SW_IND_CODE": "ind_code"})

    merged = merged.merge(ind, on="code", how="left")
    logger.info("[Prep] 行业合并完成，耗时=%.2fs", time.perf_counter() - t)

    # 输出编排器所需核心列
    out = merged[["code", "date", "Factor_A021", "lncap", "ind_code"]].copy()

    t = time.perf_counter()
    out.to_parquet(out_path, index=False)
    logger.info("[Prep] 输出写盘完成，耗时=%.2fs", time.perf_counter() - t)

    logger.info("[Prep] 输出完成: %s", out_path)
    logger.info("[Prep] 行数: %d", len(out))
    logger.info("[Prep] lncap缺失率: %s", f"{out['lncap'].isna().mean():.2%}")
    logger.info("[Prep] ind_code缺失率: %s", f"{out['ind_code'].isna().mean():.2%}")
    logger.info("[Prep] 总耗时: %.2fs", time.perf_counter() - t0)


if __name__ == "__main__":
    main()
