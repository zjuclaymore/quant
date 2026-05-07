#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path(
    r"E:\1_basement\quant_research\back_test\sigle_factor_test\function_breakdown\back_test\factor_A021_real_delivery_order_with_group_adjclose.parquet"
)
DEFAULT_OUT_DIR = Path(
    r"E:\1_basement\quant_research\back_test\sigle_factor_test\function_breakdown\back_test\analysis_A021"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按时间区间计算分组净值曲线与 IC/RANKIC/IR/RANKIR"
    )
    parser.add_argument(
        "--input", type=str, default=str(DEFAULT_INPUT), help="输入 parquet"
    )
    parser.add_argument("--start", type=str, default=None, help="开始日期 YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="结束日期 YYYY-MM-DD")
    parser.add_argument("--date-col", type=str, default="buy_date", help="时间列")
    parser.add_argument("--group-col", type=str, default="group_id", help="分组列")
    parser.add_argument("--factor-col", type=str, default="factor_value", help="因子列")
    parser.add_argument("--ret-col", type=str, default="monthly_return", help="收益列")
    parser.add_argument(
        "--max-stocks-per-group",
        type=int,
        default=0,
        help="每个日期-分组最多保留股票数，<=0 表示不限制",
    )
    parser.add_argument(
        "--out-dir", type=str, default=str(DEFAULT_OUT_DIR), help="输出目录"
    )
    parser.add_argument(
        "--ic-decay-lags", type=int, default=6, help="计算 IC 衰减的期数（Lag）"
    )
    return parser.parse_args()


def _safe_ir(x: pd.Series) -> float:
    s = pd.to_numeric(x, errors="coerce").dropna()
    if len(s) < 2:
        return float("nan")
    std = s.std(ddof=1)
    if std == 0 or pd.isna(std):
        return float("nan")
    return float(s.mean() / std)


def _safe_annualized_ir(x: pd.Series, periods_per_year: int = 12) -> float:
    ir = _safe_ir(x)
    if pd.isna(ir):
        return float("nan")
    return float(ir * np.sqrt(periods_per_year))


def _safe_variance(x: pd.Series) -> float:
    s = pd.to_numeric(x, errors="coerce").dropna()
    if len(s) < 2:
        return float("nan")
    return float(s.var(ddof=1))


def main() -> int:
    args = parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)

    df = pd.read_parquet(input_path)

    required = [args.date_col, args.group_col, args.factor_col, args.ret_col]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"输入缺少必要列: {miss}")

    df = df.copy()
    df[args.date_col] = pd.to_datetime(df[args.date_col], errors="coerce")

    if args.start:
        start_dt = pd.to_datetime(args.start)
        df = df[df[args.date_col] >= start_dt]
    if args.end:
        end_dt = pd.to_datetime(args.end)
        df = df[df[args.date_col] <= end_dt]

    if df.empty:
        raise ValueError("筛选后无数据，请检查时间区间")

    df[args.group_col] = pd.to_numeric(df[args.group_col], errors="coerce").astype(
        "Int64"
    )
    df[args.ret_col] = pd.to_numeric(df[args.ret_col], errors="coerce")
    df[args.factor_col] = pd.to_numeric(df[args.factor_col], errors="coerce")

    # 分组净值可选按每组股票数做截断，专门用于单调性评估。
    # IC/IR 也使用截断后的数据计算，以反映实际持仓的 IC
    if args.max_stocks_per_group and args.max_stocks_per_group > 0:
        tie_cols = [
            c
            for c in ["stock_code", "code", "wind_code", "S_INFO_WINDCODE"]
            if c in df.columns
        ]
        sort_cols = [args.date_col, args.group_col, args.factor_col] + tie_cols
        asc = [True, True, False] + [True] * len(tie_cols)
        df = (
            df.sort_values(sort_cols, ascending=asc)
            .groupby([args.date_col, args.group_col], as_index=False, group_keys=False)
            .head(args.max_stocks_per_group)
            .reset_index(drop=True)
        )

    # 1) 每组净值曲线（按 buy_date 截面求组内等权收益），并保留因子均值
    grp = df.dropna(subset=[args.group_col, args.ret_col, args.factor_col])
    grp_ret = grp.groupby([args.date_col, args.group_col], as_index=False).agg(
        group_return=(args.ret_col, "mean"),
        group_factor_mean=(args.factor_col, "mean"),
    )

    nav = grp_ret.pivot(
        index=args.date_col, columns=args.group_col, values="group_return"
    ).sort_index()
    nav = nav.fillna(0.0)
    nav = (1.0 + nav).cumprod()
    nav.columns = [f"group_{int(c)}" for c in nav.columns]
    nav = nav.reset_index()

    # 额外输出：每组因子均值随时间变化
    factor_mean = grp_ret.pivot(
        index=args.date_col, columns=args.group_col, values="group_factor_mean"
    ).sort_index()
    factor_mean.columns = [f"factor_mean_group_{int(c)}" for c in factor_mean.columns]
    factor_mean = factor_mean.reset_index()

    # 合并净值和因子均值
    nav = pd.merge(nav, factor_mean, on=args.date_col, how="left")

    # 2) 每期 IC / RANKIC（横截面: stock_factor vs stock_return，使用截断后数据）
    ic_rows = []
    for d, g in df.groupby(args.date_col):
        x = g[[args.factor_col, args.ret_col]].dropna()
        if len(x) < 2:
            ic_rows.append({"date": d, "ic": np.nan, "rank_ic": np.nan, "n": len(x)})
            continue
        ic = x[args.factor_col].corr(x[args.ret_col], method="pearson")
        ric = x[args.factor_col].corr(x[args.ret_col], method="spearman")
        ic_rows.append({"date": d, "ic": ic, "rank_ic": ric, "n": len(x)})

    ic_ts = pd.DataFrame(ic_rows).sort_values("date").reset_index(drop=True)

    # 3) 月度先平均，再计算整体均值与 IR
    ic_ts["year_month"] = pd.to_datetime(ic_ts["date"]).dt.to_period("M").astype(str)
    ic_monthly = (
        ic_ts.groupby("year_month", as_index=False)
        .agg(
            ic=("ic", "mean"),
            rank_ic=("rank_ic", "mean"),
            month_obs=("date", "size"),
        )
        .sort_values("year_month")
        .reset_index(drop=True)
    )

    ic_mean = float(pd.to_numeric(ic_monthly["ic"], errors="coerce").mean())
    rank_ic_mean = float(pd.to_numeric(ic_monthly["rank_ic"], errors="coerce").mean())
    ic_variance = _safe_variance(ic_monthly["ic"])
    rank_ic_variance = _safe_variance(ic_monthly["rank_ic"])
    ir = _safe_ir(ic_monthly["ic"])
    rank_ir = _safe_ir(ic_monthly["rank_ic"])
    annualized_ir = _safe_annualized_ir(ic_monthly["ic"], periods_per_year=12)
    annualized_rank_ir = _safe_annualized_ir(ic_monthly["rank_ic"], periods_per_year=12)

    summary = pd.DataFrame(
        [
            {
                "start": str(df[args.date_col].min().date()),
                "end": str(df[args.date_col].max().date()),
                "periods": int(df[args.date_col].nunique()),
                "months": int(ic_monthly["year_month"].nunique()),
                "rows": int(len(df)),
                "max_stocks_per_group": int(args.max_stocks_per_group),
                "ic": ic_mean,
                "rank_ic": rank_ic_mean,
                "ic_variance": ic_variance,
                "rank_ic_variance": rank_ic_variance,
                "ir": ir,
                "rank_ir": rank_ir,
                "annualized_ir": annualized_ir,
                "annualized_rank_ir": annualized_rank_ir,
            }
        ]
    )

    # 将汇总统计信息附加到ic_ts文件最后一行（用特殊标记）
    summary_row = summary.iloc[0].to_dict()
    summary_row.update({k: None for k in ic_ts.columns if k not in summary_row})
    summary_row["date"] = "SUMMARY"
    ic_ts = pd.concat([ic_ts, pd.DataFrame([summary_row])], ignore_index=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    nav_path = out_dir / "group_nav_curve.csv"
    ic_path = out_dir / "ic_rankic_timeseries.csv"
    summary_path = out_dir / "ic_summary.csv"
    ic_monthly_path = out_dir / "ic_rankic_monthly.csv"

    nav.to_csv(nav_path, index=False, encoding="utf-8-sig")
    ic_ts.to_csv(ic_path, index=False, encoding="utf-8-sig")
    ic_monthly.to_csv(ic_monthly_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    # 4) IC 衰减（IC Decay）计算
    if args.ic_decay_lags > 0:
        print(f"[IC Decay] 正在计算未来 {args.ic_decay_lags} 期的 IC 衰减...")
        # 为了计算位移，需要全量数据（即使被截断了也按输入算，或者按截断后的算？通常按截断后的算反映持仓稳定性）
        # 这里使用 df (可能是截断后的)
        
        # 提取必要列并整理
        decay_df = df[[args.date_col, "code", args.factor_col, args.ret_col]].copy()
        # 确保按股票和时间排序
        decay_df = decay_df.sort_values(["code", args.date_col])
        
        decay_results = []
        # Lag 1 已经是普通的 ic 了，我们计算 1 到 lags
        for lag in range(1, args.ic_decay_lags + 1):
            if lag == 1:
                decay_results.append({"lag": 1, "rank_ic": rank_ic_mean})
                continue
            
            # 对收益率进行负向位移，使得 T 期的因子对应 T+lag-1 期的收益
            # 这里的 monthly_return 是 T 到 T+1 的收益。
            # Lag 2 IC 是 Factor_T 与 Return_T+1_to_T+2 的相关性。
            # 所以对 Return 向上位移 lag-1 位。
            decay_df[f"ret_lag_{lag}"] = decay_df.groupby("code")[args.ret_col].shift(-(lag - 1))
            
            # 计算全样本下的横截面平均 RankIC
            lag_ics = []
            for d, g in decay_df.groupby(args.date_col):
                x = g[[args.factor_col, f"ret_lag_{lag}"]].dropna()
                if len(x) > 10: # 样本量太小不计算
                    lag_ics.append(x[args.factor_col].corr(x[f"ret_lag_{lag}"], method="spearman"))
            
            avg_lag_ic = np.mean(lag_ics) if lag_ics else np.nan
            decay_results.append({"lag": lag, "rank_ic": avg_lag_ic})
            print(f"[IC Decay] Lag {lag}: {avg_lag_ic:.6f}")

        decay_summary = pd.DataFrame(decay_results)
        decay_path = out_dir / "ic_decay.csv"
        decay_summary.to_csv(decay_path, index=False, encoding="utf-8-sig")
        print(f"[Done] ic_decay: {decay_path}")

    print(f"[Done] rows: {len(df):,}")
    print(f"[Done] periods: {df[args.date_col].nunique():,}")
    print(f"[Done] IC: {ic_mean:.6f}")
    print(f"[Done] RANKIC: {rank_ic_mean:.6f}")
    print(f"[Done] IC Variance: {ic_variance:.6f}")
    print(f"[Done] RANKIC Variance: {rank_ic_variance:.6f}")
    print(f"[Done] IR: {ir:.6f}")
    print(f"[Done] RANKIR: {rank_ir:.6f}")
    print(f"[Done] Annualized IR: {annualized_ir:.6f}")
    print(f"[Done] Annualized RANKIR: {annualized_rank_ir:.6f}")
    print(f"[Done] nav: {nav_path}")
    print(f"[Done] ic_ts: {ic_path}")
    print(f"[Done] ic_monthly: {ic_monthly_path}")
    print(f"[Done] summary: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
