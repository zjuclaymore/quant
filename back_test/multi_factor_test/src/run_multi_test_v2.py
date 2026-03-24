"""
多因子回测入口驱动 V2 (批量矩阵版)

支持多聚合方法 × 多优化方法的矩阵式批量回测。
数据加载仅执行一次，中间结果缓存为 parquet 文件，
每种策略组合输出到独立子文件夹，
最终生成合并 HTML 报告叠加所有策略净值曲线。
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm

from back_test.multi_factor_test.src.logger import setup_logger
from back_test.multi_factor_test.src.config_loader import ConfigLoader
from back_test.multi_factor_test.src.data_loader import MultiFactorDataLoader
from back_test.multi_factor_test.src.factor_aggregator import FactorAggregator
from back_test.multi_factor_test.src.stock_selector import StockSelector
from back_test.multi_factor_test.src.trading_executor import TradingExecutor
from back_test.multi_factor_test.src.evaluator import MultiFactorEvaluator
from back_test.multi_factor_test.src.attribution import PerformanceAttribution
from back_test.sigle_factor_test.src.data_loader import load_aux_data


# ──────────────────────────────────────────────────────────────────
# 中间文件工具
# ──────────────────────────────────────────────────────────────────

def _save_stock_pool_snapshots(df_group, cal, out_dir, logger):
    """
    按期导出选股池快照 xlsx 文件, 格式参照 stock_pool#YYYYMMDD.xlsx。

    每个调仓期一个文件，包含:
        S_INFO_WINDCODE, TRADE_DT, SCORE, GROUP, TARGET_WEIGHT, 各因子列

    Args:
        df_group: 分组后的因子 DataFrame (含 symbol, year_month, multi_alpha_score, group 等)
        cal: 调仓日历 DataFrame
        out_dir: stock_pool 输出目录
    """
    os.makedirs(out_dir, exist_ok=True)
    factor_cols = [c for c in df_group.columns
                   if c not in ["symbol", "year_month", "multi_alpha_score",
                                "group", "lncap", "target_weight", "_mv_pct"]]

    for _, row in cal.iterrows():
        ym = row["year_month"]
        trade_dt = row["buy_date"].strftime("%Y%m%d")
        slice_df = df_group[df_group["year_month"] == ym].copy()
        if slice_df.empty:
            continue

        out_df = pd.DataFrame({
            "S_INFO_WINDCODE": slice_df["symbol"],
            "TRADE_DT": trade_dt,
            "SCORE": slice_df.get("multi_alpha_score", np.nan),
            "GROUP": slice_df.get("group", np.nan),
        })
        # 添加因子列
        for col in factor_cols:
            if col in slice_df.columns:
                out_df[col] = slice_df[col].values

        fpath = os.path.join(out_dir, f"stock_pool#{trade_dt}.xlsx")
        out_df.to_excel(fpath, index=False, engine="openpyxl")

    logger.info(f"选股池快照已导出到: {out_dir} (共 {len(cal)} 期)")


def _build_combined_report(base_dir, test_name, logger):
    """
    从各子策略文件夹读取 eval_results.pkl，提取日频净值，
    在同一张图上叠加所有策略的净值曲线 + 基准，
    输出到 combined_Report.html。

    Args:
        base_dir: 顶层输出目录 (含各 agg_method/weight_method 子目录)
        test_name: 回测名称
        logger: 日志记录器
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import io, base64

    from back_test.sigle_factor_test.src.bt_utils import (
        REPORT_CSS, build_section, add_table_to_html
    )

    # 收集所有策略的日频净值
    strategy_navs = {}
    all_kpis_rows = []

    for root, dirs, files in os.walk(base_dir):
        if "eval_results.pkl" in files:
            pkl_path = os.path.join(root, "eval_results.pkl")
            # 从路径推断策略名称
            rel = os.path.relpath(root, base_dir)
            if rel == ".":
                strategy_label = f"{test_name} (Base)"
            else:
                parts = rel.replace("\\", "/").split("/")
                strategy_label = " / ".join(parts)

            try:
                with open(pkl_path, "rb") as f:
                    data = pickle.load(f)
                nav_df = data.get("daily_portfolio_df")
                kpis = data.get("kpis", {})
                if nav_df is not None and not nav_df.empty:
                    strategy_navs[strategy_label] = nav_df.set_index("date")["nav"]
                    print(f"DEBUG: Found Strategy={strategy_label}, NavSize={len(nav_df)}")
                    all_kpis_rows.append({
                        "策略": strategy_label,
                        "年化收益": f"{kpis.get('ann_ret', 0):.2%}",
                        "夏普比率": f"{kpis.get('sharpe', 0):.2f}",
                        "最大回撤": f"{kpis.get('max_drawdown', 0):.2%}",
                        "年化超额": f"{kpis.get('ann_excess_ret', 0):.2%}",
                        "信息比率": f"{kpis.get('info_ratio', 0):.2f}",
                        "月度胜率": f"{kpis.get('win_rate', 0):.2%}",
                    })
            except Exception as e:
                logger.warning(f"读取 {pkl_path} 失败: {e}")

    if not strategy_navs:
        logger.warning("未找到任何策略净值数据，跳过合并报告")
        return

    # 加载基准净值
    loader = MultiFactorDataLoader(logger)
    bench_info = {
        "h00300.CSI": "沪深300全收益",
        "h00922.CSI": "中证红利全收益",
        "h00816.CSI": "红利低波全收益"
    }
    first_nav = list(strategy_navs.values())[0]
    start_d = first_nav.index.min().strftime("%Y%m%d")
    end_d = first_nav.index.max().strftime("%Y%m%d")

    bench_navs = {}
    for symbol, name in bench_info.items():
        df_b = loader.load_benchmark(symbol, start_date=start_d, end_date=end_d)
        if not df_b.empty:
            bench_navs[name] = df_b.set_index("date")["nav"]

    # 绘制叠加净值图
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 130

    fig, ax = plt.subplots(figsize=(16, 7))
    fig.subplots_adjust(left=0.06, right=0.97, top=0.92, bottom=0.12)

    # 策略配色 (鲜艳色)
    strategy_colors = [
        "#2563eb", "#059669", "#d97706", "#7c3aed",
        "#dc2626", "#0891b2", "#4f46e5", "#ca8a04"
    ]
    # 基准配色 (灰系)
    bench_colors = ["#94a3b8", "#a1a1aa", "#78716c"]

    for i, (label, nav_s) in enumerate(strategy_navs.items()):
        color = strategy_colors[i % len(strategy_colors)]
        ax.plot(nav_s.index, nav_s.values, color=color, linewidth=1.8,
                label=label, zorder=5)

    for i, (label, nav_s) in enumerate(bench_navs.items()):
        color = bench_colors[i % len(bench_colors)]
        ax.plot(nav_s.index, nav_s.values, color=color, linewidth=1.0,
                linestyle="--", label=label, zorder=3)

    ax.set_title(f"多策略净值对比 — {test_name}", fontsize=14, fontweight="bold")
    ax.set_ylabel("净值")
    ax.set_facecolor("#f5f5f5")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.7)
    ax.legend(loc="upper left", fontsize=9, ncol=2, framealpha=0.95)

    # 图转 base64
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    plt.close(fig)

    chart_html = f"<div class='plot-block'><img class='plot-img' src='data:image/png;base64,{img_b64}'></div>"

    # KPI 对比表
    kpi_table_parts = []
    if all_kpis_rows:
        kpi_df = pd.DataFrame(all_kpis_rows)
        add_table_to_html(kpi_table_parts, kpi_df, "策略 KPI 对比", max_rows=None)

    section_html = build_section(
        "多策略对比",
        chart_html + "\n".join(kpi_table_parts),
        "叠加所有聚合方法 × 权重方法的策略净值曲线与核心指标",
        tab_id="tab-summary"
    ).replace("class='tab-content'", "class='tab-content active'")

    # 输出 HTML
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>{test_name} 多策略对比报告</title>
        {REPORT_CSS}
    </head>
    <body>
        <div class="dashboard">
            <div class='report-header'>
                <h1>多策略对比报告: {test_name}</h1>
                <div class='subtitle'>MULTI-STRATEGY COMPARISON REPORT</div>
            </div>
            {section_html}
        </div>
    </body>
    </html>
    """

    report_path = os.path.join(base_dir, f"{test_name}_combined_Report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_code)
    logger.info(f"合并对比报告已生成: {report_path}")


# ──────────────────────────────────────────────────────────────────
# 核心回测流程
# ──────────────────────────────────────────────────────────────────

def _run_single_strategy(
    df_alpha, cal, df_daily, bt_params, port_cfg, weight_method,
    factor_cols, out_dir, loader, limit_df, liq_df, trade_dates,
    st_df, first_dates, logger, benchmark_name
):
    """
    运行单个策略的选股 → 交易 → 评估 → 报告流程。

    Args:
        df_alpha: 聚合后的因子 DataFrame (含 multi_alpha_score)
        cal: 调仓日历
        df_daily: 全市场日行情
        bt_params: 回测核心参数
        port_cfg: 组合参数
        weight_method: 当前使用的权重方法
        factor_cols: 因子列名列表
        out_dir: 本策略的输出目录
        loader: 数据加载器
        limit_df, liq_df, trade_dates, st_df, first_dates: 交易约束数据
        logger: 日志记录器
        benchmark_name: 基准代码
    """
    os.makedirs(out_dir, exist_ok=True)
    strategy_logger = setup_logger(f"mf_{os.path.basename(out_dir)}", out_dir)
    strategy_logger.info(f"=== 策略启动: weight_method={weight_method} ===")

    # 选股与分组
    selector = StockSelector(strategy_logger)
    df_group, n_groups, sel_limit_df, sel_liq_df, sel_first_dates = selector.exclude_invalid_and_group(
        df_alpha, df_daily, trade_dates, group_size=port_cfg.get('group_size', 0)
    )

    # 交易执行
    executor = TradingExecutor(sel_limit_df, sel_liq_df, trade_dates, st_df, sel_first_dates, strategy_logger)

    # 过滤掉 build_target_portfolio 不接受的参数
    clean_port = {k: v for k, v in port_cfg.items()
                  if k not in ["weight_methods", "rebalance_freq", "group_num", "group_size"]}

    holding_period_returns = executor.execute_trades(
        df_group, cal,
        commission=bt_params.get('commission', 0.0015),
        df_daily=df_daily,
        weight_method=weight_method,
        **clean_port
    )

    if holding_period_returns.empty:
        strategy_logger.error("无交易结果")
        return

    # 评估
    test_name = os.path.basename(out_dir)
    evaluator = MultiFactorEvaluator(df_daily["symbol"].nunique(), out_dir, strategy_logger)
    df_bench_m = loader.load_benchmark(bt_params.get('benchmark', '000300.SH'), cal=cal)

    evaluator.compute_returns_and_kpis(
        holding_period_returns, df_bench_m, df_group, cal,
        factor_name=test_name,
        start_month=bt_params['start_date'],
        end_month=bt_params['end_date'],
        commission=bt_params.get('commission', 0.0015),
        executor=executor,
        df_daily=df_daily,
        log_path=os.path.join(out_dir, "backtest_debug.log"),
        factor_cols=factor_cols,
        benchmark_name=benchmark_name
    )
    strategy_logger.info(f"=== 策略完成: {test_name} ===")


def main():
    """
    多因子回测主入口。

    流程:
    1. 数据加载 (一次性) → 缓存到 intermediate/
    2. 对 aggregation_methods × weight_methods 矩阵循环
    3. 每种组合输出到独立子文件夹
    4. 生成合并对比报告
    """
    # ── 配置加载 ──
    config = ConfigLoader()
    bt_params = config.get_backtest_params()
    factor_cfg = config.get_factors_config()
    port_cfg = config.get_portfolio_params()
    filter_cfg = config.get_stock_filter_params()
    output_cfg = config.get_output_params()
    mcap_cfg = config.config.get('micro_cap_filter', {})

    test_name = bt_params.get('test_name', 'mf_test_v2')
    base_out_dir = os.path.join(output_cfg.get('base_dir'), test_name)
    intermediate_dir = os.path.join(base_out_dir, "intermediate")
    os.makedirs(intermediate_dir, exist_ok=True)

    logger = setup_logger("multi_factor", base_out_dir)

    # 聚合方法和权重方法列表
    agg_methods = factor_cfg.get('aggregation_methods',
                                  [factor_cfg.get('aggregation_method', 'equal_weight')])
    weight_methods = port_cfg.get('weight_methods',
                                   [port_cfg.get('weight_method', 'equal')])

    logger.info(f"=== 批量回测启动: {test_name} ===")
    logger.info(f"聚合方法: {agg_methods}")
    logger.info(f"权重方法: {weight_methods}")
    logger.info(f"组合矩阵: {len(agg_methods)} × {len(weight_methods)} = {len(agg_methods)*len(weight_methods)} 种策略")

    # ── 1. 数据加载 (带缓存) ──
    loader = MultiFactorDataLoader(logger)
    factors_to_load = factor_cfg.get('list', [])
    factor_cols = [name for _, name in factors_to_load]

    neu_cache = os.path.join(intermediate_dir, "df_neu.parquet")
    ret_cache = os.path.join(intermediate_dir, "df_returns.parquet")

    if os.path.exists(neu_cache) and os.path.exists(ret_cache):
        logger.info(f"发现中间文件缓存，跳过数据加载: {neu_cache}")
        df_neu = pd.read_parquet(neu_cache)
        df_returns = pd.read_parquet(ret_cache)
        # 仍需加载行情和日历
        start_dt = pd.to_datetime(bt_params['start_date']) - pd.Timedelta(days=60)
        end_dt = pd.to_datetime(bt_params['end_date']) + pd.offsets.MonthEnd(2)
        df_daily = loader.load_market_data(start_dt.strftime("%Y%m%d"), end_dt.strftime("%Y%m%d"))
        st_df, ind_df = loader.load_st_and_industry()
        full_cal = loader.load_calendar(df_daily, delay_days=bt_params.get('delay_days', 0))
    else:
        logger.info("首次运行，加载全量数据...")
        df_raw = loader.load_factors_aligned(factors_to_load)

        start_dt = pd.to_datetime(bt_params['start_date']) - pd.Timedelta(days=60)
        end_dt = pd.to_datetime(bt_params['end_date']) + pd.offsets.MonthEnd(2)
        df_daily = loader.load_market_data(start_dt.strftime("%Y%m%d"), end_dt.strftime("%Y%m%d"))

        st_df, ind_df = loader.load_st_and_industry()
        full_cal = loader.load_calendar(df_daily, delay_days=bt_params.get('delay_days', 0))

        # 时间范围过滤
        df_raw = df_raw[
            (df_raw["year_month"] >= (pd.to_datetime(bt_params['start_date']) - pd.offsets.MonthEnd(2)).to_period("M")) &
            (df_raw["year_month"] <= pd.to_datetime(bt_params['end_date']).to_period("M"))
        ].copy()
        logger.info(f"过滤后因子数据: {len(df_raw)} 行")

        # 市值数据对齐 + 微盘过滤
        df_neu = df_raw.copy()
        v_whitelist = set(df_raw["symbol"].unique())
        mv_dir = r"E:\1_basement\quant_research\factor\1_算术转换因子_ArithmeticFactors\log_mv_1\output\class_by_stock"
        mv_df = load_aux_data(mv_dir, v_whitelist)

        if mv_df is not None and not mv_df.empty:
            mv = mv_df.copy()
            for c in mv.columns:
                if "date" in c.lower() or "日期" in c.lower():
                    mv.rename(columns={c: "date"}, inplace=True)
                if "code" in c.lower() or "symbol" in c.lower():
                    mv.rename(columns={c: "symbol"}, inplace=True)
            mv["date"] = pd.to_datetime(mv["date"].astype(str))
            mv["year_month"] = mv["date"].dt.to_period("M")
            mv = mv.groupby(["symbol", "year_month"]).last().reset_index()
            mv_col = [c for c in mv.columns if c not in ["date", "symbol", "year_month"]][0]
            mv = mv[["year_month", "symbol", mv_col]].rename(columns={mv_col: "lncap"})
            df_neu = pd.merge(df_neu, mv, on=["year_month", "symbol"], how="left")

            if mcap_cfg.get('exclude_micro_cap', True):
                pct = mcap_cfg.get('micro_cap_percentile', 0.20)
                df_neu["_mv_pct"] = df_neu.groupby("year_month")["lncap"].transform(lambda x: x.rank(pct=True))
                before = len(df_neu)
                df_neu = df_neu[df_neu["_mv_pct"] > pct].drop(columns=["_mv_pct"])
                logger.info(f"微盘过滤: {before} -> {len(df_neu)} 行 (剔除市值最小 {pct:.0%})")
        else:
            logger.warning("市值数据缺失，未能筛除微盘股")

        # 计算月度收益率
        trade_dates_idx = pd.DatetimeIndex(
            full_cal["buy_date"].tolist() + full_cal["sell_date"].tolist()
        ).drop_duplicates().sort_values()

        returns_list = []
        for _, row in full_cal.iterrows():
            ym = row["year_month"]
            t_buy = row["buy_date"]
            t_sell = row["sell_date"]
            p_buy = df_daily[df_daily["date"] == t_buy][["symbol", "adj_close"]].rename(columns={"adj_close": "p_buy"})
            p_sell = df_daily[df_daily["date"] == t_sell][["symbol", "adj_close"]].rename(columns={"adj_close": "p_sell"})
            if not p_buy.empty and not p_sell.empty:
                m = pd.merge(p_buy, p_sell, on="symbol")
                m["ret"] = m["p_sell"] / m["p_buy"] - 1
                m["year_month"] = ym
                returns_list.append(m[["symbol", "year_month", "ret"]])
        df_returns = pd.concat(returns_list, ignore_index=True) if returns_list else pd.DataFrame()
        logger.info(f"已计算 {len(df_returns)} 条月度收益记录")

        # 保存中间文件
        df_neu.to_parquet(neu_cache, index=False)
        df_returns.to_parquet(ret_cache, index=False)
        logger.info(f"中间文件已缓存: {neu_cache}, {ret_cache}")

    # ── 公共数据准备 ──
    trade_dates = pd.DatetimeIndex(
        full_cal["buy_date"].tolist() + full_cal["sell_date"].tolist()
    ).drop_duplicates().sort_values()

    cal = full_cal[
        (full_cal["year_month"] >= bt_params['start_date']) &
        (full_cal["year_month"] <= bt_params['end_date'])
    ].copy()

    benchmark_name = bt_params.get('benchmark', '000300.SH')

    # ── 2. 矩阵式批量回测 ──
    agg = FactorAggregator(logger)
    ic_params = factor_cfg.get('ic_weighted_params', {})

    for agg_method in agg_methods:
        logger.info(f"\n{'='*60}")
        logger.info(f"聚合方法: {agg_method}")
        logger.info(f"{'='*60}")

        # 因子聚合
        if agg_method == 'equal_weight':
            df_alpha = agg.aggregate_equal_weight(df_neu, factor_cols)
        elif agg_method == 'ic_weighted':
            df_alpha = agg.aggregate_ic_weighted(
                df_neu, factor_cols, returns=df_returns,
                lookback=ic_params.get('lookback', 12),
                half_life=ic_params.get('half_life', 6)
            )
        elif agg_method == 'ic_ir_weighted':
            df_alpha = agg.aggregate_ic_ir_weighted(df_neu, factor_cols, returns=df_returns)
        elif agg_method == 'custom_weight':
            weights = factor_cfg.get('custom_weights', {})
            df_alpha = agg.aggregate_custom_weight(df_neu, weights)
        else:
            logger.warning(f"未知聚合方法: {agg_method}, 使用等权")
            df_alpha = agg.aggregate_equal_weight(df_neu, factor_cols)

        # 缓存聚合结果
        agg_cache = os.path.join(intermediate_dir, f"df_alpha_{agg_method}.parquet")
        df_alpha.to_parquet(agg_cache, index=False)
        logger.info(f"聚合结果已缓存: {agg_cache}")

        # 导出选股池快照 (每个聚合方法一套)
        selector = StockSelector(logger)
        df_group_snap, _, snap_limit, snap_liq, snap_first = selector.exclude_invalid_and_group(
            df_alpha, df_daily, trade_dates, group_size=port_cfg.get('group_size', 0)
        )
        pool_dir = os.path.join(intermediate_dir, "stock_pool", agg_method)
        _save_stock_pool_snapshots(df_group_snap, cal, pool_dir, logger)

        for weight_method in weight_methods:
            strategy_name = f"{agg_method}__{weight_method}"
            out_dir = os.path.join(base_out_dir, agg_method, weight_method)

            logger.info(f"\n--- 策略: {strategy_name} ---")

            try:
                _run_single_strategy(
                    df_alpha=df_alpha.copy(),
                    cal=cal,
                    df_daily=df_daily,
                    bt_params=bt_params,
                    port_cfg=port_cfg,
                    weight_method=weight_method,
                    factor_cols=factor_cols,
                    out_dir=out_dir,
                    loader=loader,
                    limit_df=snap_limit,
                    liq_df=snap_liq,
                    trade_dates=trade_dates,
                    st_df=st_df,
                    first_dates=snap_first,
                    logger=logger,
                    benchmark_name=benchmark_name
                )
            except Exception as e:
                logger.error(f"策略 {strategy_name} 执行失败: {e}", exc_info=True)

    # ── 3. 合并对比报告 ──
    logger.info("\n生成合并对比报告...")
    try:
        _build_combined_report(base_out_dir, test_name, logger)
    except Exception as e:
        logger.error(f"合并报告生成失败: {e}", exc_info=True)

    logger.info("=== 批量回测全部完成 ===")


if __name__ == "__main__":
    main()
