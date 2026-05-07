#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

DEFAULT_INPUT = Path(
    r"E:\1_basement\quant_research\back_test\sigle_factor_test\function_breakdown\back_test\analysis_A021_from_20080430\group_nav_curve.csv"
)
DEFAULT_OUTPUT = Path(
    r"E:\1_basement\quant_research\back_test\sigle_factor_test\function_breakdown\plot\group_nav_curve_20080430_interactive.html"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绘制分组净值曲线（交互式）")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT), help="group_nav_curve.csv 路径")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="输出 html 路径")
    parser.add_argument("--date-col", type=str, default="buy_date", help="日期列名")
    parser.add_argument(
        "--title",
        type=str,
        default="Group NAV Curve (Interactive, Rebased to 1.0)",
        help="图标题",
    )
    return parser.parse_args()


def _rebase_to_one(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    valid = x.dropna()
    if valid.empty:
        return x
    base = valid.iloc[0]
    if pd.isna(base) or base == 0:
        return x
    return x / base


def main() -> int:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    df = pd.read_csv(input_path)
    if args.date_col not in df.columns:
        raise ValueError(f"缺少日期列: {args.date_col}")

    df[args.date_col] = pd.to_datetime(df[args.date_col], errors="coerce")
    df = df.dropna(subset=[args.date_col]).sort_values(args.date_col)

    group_cols = [c for c in df.columns if c.startswith("group_")]
    if not group_cols:
        group_cols = [c for c in df.columns if c != args.date_col]

    if not group_cols:
        raise ValueError("未找到可绘制的分组列")

    fig = go.Figure()

    for c in group_cols:
        y = _rebase_to_one(df[c])
        fig.add_trace(
            go.Scatter(
                x=df[args.date_col],
                y=y,
                mode="lines",
                name=c,
                line={"width": 2},
                hovertemplate="%{x|%Y-%m-%d}<br>%{fullData.name}: %{y:.4f}<extra></extra>",
            )
        )

    fig.update_layout(
        template="plotly_white",
        title=args.title,
        xaxis_title="Date",
        yaxis_title="Rebased NAV (Start = 1)",
        legend_title="Groups",
        hovermode="x unified",
        width=1300,
        height=760,
    )
    fig.update_xaxes(rangeslider_visible=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs="cdn")

    print(f"[Done] input: {input_path}")
    print(f"[Done] output: {output_path}")
    print(f"[Done] groups: {len(group_cols)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
