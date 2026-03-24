"""
评估归因模块 (Evaluator)

负责将 df_trades 的交易结果聚合为组合净值。
同时复用了单因子的核心指标计算和制图报告代码。
"""

import os
import logging
import pandas as pd
import numpy as np

try:
    from .logger import get_logger
except ImportError:
    from logger import get_logger

# 引入复用功能
from back_test.sigle_factor_test.src.bt_utils import (
    compute_backtest_kpis,
    add_card_to_html,
    add_table_to_html,
    add_log_block,
    build_section,
    build_report_header,
    write_report_html,
    write_report_html_no_trades,
    build_workflow_pipeline,
    build_strategy_context,
    plot_pre_post_distribution,
    plot_core_performance,
    plot_group_nav,
    plot_extreme_groups,
    plot_monotonicity_bar,
    plot_ic_trend,
    plot_ir_rank_ir_bar,
    plot_group_win_rates,
    plot_daily_nav,
    save_extreme_groups_csv
)

class MultiFactorEvaluator:
    def __init__(self, df_daily_symbol_count: int, out_dir: str, logger: logging.Logger = None):
        self.logger = logger or get_logger(__name__)
        self.out_dir = out_dir
        self.df_daily_symbol_count = df_daily_symbol_count

    def compute_returns_and_kpis(
        self,
        holding_period_returns: pd.DataFrame,
        df_bench_m: pd.DataFrame,
        df_alpha: pd.DataFrame,
        cal: pd.DataFrame,
        factor_name: str,
        start_month: str, 
        end_month: str,
        commission: float,
        executor=None,
        df_daily=None,
        log_path: str = None,
        factor_cols: list = None,
        benchmark_name: str = "沪深300 (000300.SH)"
    ):
        """
        根据交易日历、基准、真实交易明细，计算多空、市场净值，并计算各类KPI
        """
        self.logger.info("归因模块: 各分组算术收益聚合及全市场基准推演...")
        
        df_trades = holding_period_returns.copy()
        df_trades = df_trades.dropna(subset=["group"])
        df_trades["group"] = df_trades["group"].astype(int)

        def _weighted_mean(d):
            d = d.dropna(subset=["actual_ret"])
            if d.empty:
                return np.nan
            if "target_weight" not in d.columns:
                return d["actual_ret"].mean()
            w = d["target_weight"].fillna(0)
            ws = w.sum()
            if ws <= 0:
                return d["actual_ret"].mean()
            return (d["actual_ret"] * w).sum() / ws

        grouped_returns = (
            df_trades.groupby(["year_month", "group"])
            .apply(lambda d: _weighted_mean(d))
            .reset_index(name="actual_ret")
        )
        grouped_returns = grouped_returns.sort_values(["group", "year_month"])
        grouped_returns["净值"] = grouped_returns.groupby("group")["actual_ret"].transform(lambda x: (1 + x).cumprod())

        first_ym = grouped_returns["year_month"].min()
        init_ym = first_ym - 1
        init_rows = [
            {"year_month": init_ym, "group": int(g), "actual_ret": 0.0, "净值": 1.0}
            for g in grouped_returns["group"].unique()
        ]
        grouped_returns = pd.concat([pd.DataFrame(init_rows), grouped_returns], ignore_index=True)
        grouped_returns = grouped_returns.sort_values(["group", "year_month"]).reset_index(drop=True)

        market_returns = (
            df_trades.groupby("year_month")
            .apply(lambda d: _weighted_mean(d))
            .reset_index(name="actual_ret")
        )
        market_returns = market_returns.sort_values("year_month")
        market_returns["市场组合净值"] = (1 + market_returns["actual_ret"]).cumprod()
        market_init = pd.DataFrame([{"year_month": init_ym, "actual_ret": 0.0, "市场组合净值": 1.0}])
        market_returns = pd.concat([market_init, market_returns], ignore_index=True).sort_values("year_month").reset_index(drop=True)

        # 找最大和最小组
        all_groups = sorted(df_trades["group"].unique())
        max_group = max(all_groups)
        
        if len(all_groups) == 1:
            # 纯多方模式（如 Top50 实盘）：唯一组就是策略组合
            best_g = all_groups[0]
            worst_g = all_groups[0]
            self.logger.info(f"单组模式: 策略组合为 G{best_g}（Top50 纯多方）")
        else:
            # 多组模式：自动判断多空方向
            g1_f = (1 + grouped_returns.loc[grouped_returns["group"] == 1, "actual_ret"]).prod()
            gmax_f = (1 + grouped_returns.loc[grouped_returns["group"] == max_group, "actual_ret"]).prod()
            direction = 1 if gmax_f >= g1_f else -1
            self.logger.info(f"多空方向自动判定: {'多G' + str(max_group) if direction == 1 else '多G1'}")
            best_g, worst_g = (max_group, 1) if direction == 1 else (1, max_group)

        best_df = grouped_returns[grouped_returns["group"] == best_g].copy()
        worst_df = grouped_returns[grouped_returns["group"] == worst_g].copy()

        # 基准对齐
        best_df = self._align_benchmark(best_df, df_bench_m)
        worst_df = self._align_benchmark(worst_df, df_bench_m)

        # Multi-factor IC/RankIC (使用 multi_alpha_score)
        df_ic = self._calc_overall_ic(df_trades, cal)

        # 每日持仓收益: 只绘制最优组 (策略组合) 的日频净值
        daily_portfolio_df = None
        if executor is not None and df_daily is not None:
            self.logger.info(f"生成策略组合 (G{best_g}) 的日频净值序列...")
            best_trades = df_trades[df_trades["group"] == best_g].copy()
            daily_portfolio_df = executor.build_daily_portfolio_series(best_trades, df_daily)

        kpis = compute_backtest_kpis(
            best_df, worst_df, df_trades, df_ic, market_returns,
            cal, best_g, worst_g, df_alpha, self.df_daily_symbol_count
        )
        
        kpis = self._calc_advanced_kpis(best_df, kpis)

        # --- 新增: Top 5 最大回撤分析 ---
        top5_drawdowns_table = self._compute_top5_drawdowns(
            strategy_daily_df=daily_portfolio_df
        )

        self._build_report(
            factor_name, df_alpha, best_df, worst_df, grouped_returns,
            df_trades, market_returns, df_ic, cal, kpis,
            best_g, worst_g, max_group, commission, start_month, end_month,
            daily_portfolio_df, log_path, factor_cols, benchmark_name,
            top5_drawdowns_table = top5_drawdowns_table
        )

    # ------------------------------------------------------------------
    # 年度绩效对比表 (含夏普、超额收益、超额回撤、净值回归天数)
    # ------------------------------------------------------------------

    def _compute_top5_drawdowns(self, strategy_daily_df):
        """
        计算全样本期内的前5大回撤及其对应的回归天数 (非重叠期间)。

        Args:
            strategy_daily_df: 策略日频净值 DataFrame, 含 date 和 nav 列

        Returns:
            pd.DataFrame: 包含 5 次最大回撤详情的表格
        """
        if strategy_daily_df is None or strategy_daily_df.empty:
            return None

        nav_s = strategy_daily_df.set_index("date")["nav"].dropna().sort_index()
        if len(nav_s) < 2:
            return None

        cum_max = nav_s.cummax()
        drawdowns = nav_s / cum_max - 1

        # 找出所有回撤区间 (从破前高开始，到恢复前高结束)
        # 简化版：识别每一个局部最小值 (trough)
        
        # 为了找非重叠的最大回撤，我们需要更系统的算法：
        # 1. 找到全局最低点，这构成了最大的回撤区间 [最高点, 恢复点]
        # 2. 屏蔽这个区间，再找下一个最大的回撤
        # 3. 重复 5 次

        results = []
        temp_drawdowns = drawdowns.copy()

        for _ in range(5):
            if temp_drawdowns.min() >= 0 or len(temp_drawdowns) == 0:
                break
                
            # 当前剩余数据中的最大回撤低点
            trough_idx = temp_drawdowns.idxmin()
            
            # 找到导致此低点的前期最高点
            peak_val = cum_max.loc[trough_idx]
            # 往回找最近的等于 peak_val 的点，这就是下跌起点
            before_trough = nav_s.loc[:trough_idx]
            peak_idx = before_trough[before_trough == peak_val].index[-1]
            
            # 从低点往后找恢复点
            after_trough = nav_s.loc[trough_idx:]
            recovered = after_trough[after_trough >= peak_val]
            
            if len(recovered) > 1:
                # 存在真实的恢复点
                recovery_idx = recovered.index[1] if recovered.index[0] == trough_idx else recovered.index[0]
                status = "已恢复"
                days = (recovery_idx - trough_idx).days
                end_mask_idx = recovery_idx
            else:
                recovery_idx = nav_s.index[-1]
                status = "未恢复"
                days = "未恢复"
                end_mask_idx = nav_s.index[-1]
                
            dd_value = temp_drawdowns.loc[trough_idx]
            
            results.append({
                "排名": len(results) + 1,
                "下跌起点": peak_idx.strftime("%Y-%m-%d"),
                "回撤低点": trough_idx.strftime("%Y-%m-%d"),
                "恢复高点": recovery_idx.strftime("%Y-%m-%d") if status == "已恢复" else "至今",
                "最大回撤": f"{dd_value:.2%}",
                "修复天数(自低点)": str(days)
            })
            
            # 屏蔽此回撤区间，以便找下一个非重叠的回撤
            # 将 [peak_idx, end_mask_idx] 区间的回撤设为 0
            mask = (temp_drawdowns.index >= peak_idx) & (temp_drawdowns.index <= end_mask_idx)
            temp_drawdowns.loc[mask] = 0.0

        if not results:
            return None
            
        return pd.DataFrame(results)

    def _align_benchmark(self, df, df_bench_m):
        df = pd.merge(df, df_bench_m, on="year_month", how="left").fillna(0)
        df["excess_ret"] = df["actual_ret"] - df["bench_return"]
        df["excess_cum"] = (1 + df["excess_ret"]).cumprod()
        df["bench_nav"] = (1 + df["bench_return"]).cumprod()
        return df

    def _calc_advanced_kpis(self, strategy_df, kpis):
        """
        计算业界标准的组合指标：年化超额、跟踪误差、信息比率
        """
        # 年化超额收益 (几何年化)
        total_months = len(strategy_df)
        if total_months == 0:
            return kpis

        ann_factor = 12 / total_months
        strategy_nav = strategy_df["净值"].iloc[-1]
        bench_nav = strategy_df["bench_nav"].iloc[-1]
        
        ann_ret = strategy_nav ** ann_factor - 1
        ann_bench_ret = bench_nav ** ann_factor - 1
        
        # 简单年化超额 (算术平均之差的年化)
        avg_excess = strategy_df["excess_ret"].mean()
        ann_excess_ret_arith = avg_excess * 12
        
        # 跟踪误差 (年化)
        tracking_error = strategy_df["excess_ret"].std() * np.sqrt(12)
        
        # 信息比率 (业界标准: 年化超额 / 跟踪误差)
        info_ratio = ann_excess_ret_arith / tracking_error if tracking_error > 0 else np.nan
        
        kpis.update({
            "ann_ret": ann_ret,
            "ann_bench_ret": ann_bench_ret,
            "ann_excess_ret": ann_excess_ret_arith,
            "tracking_error": tracking_error,
            "info_ratio": info_ratio
        })
        return kpis

    def _calc_overall_ic(self, df_trades, cal):
        from scipy.stats import spearmanr, pearsonr
        ic_records = []
        for ym in cal["year_month"]:
            d = df_trades[df_trades["year_month"] == ym].copy()
            d = d.dropna(subset=["multi_alpha_score", "actual_ret"])
            if len(d) > 10:
                rank_ic, _ = spearmanr(d["multi_alpha_score"].values, d["actual_ret"].values)
                ic, _ = pearsonr(d["multi_alpha_score"].values, d["actual_ret"].values)
                ic_records.append({"year_month": ym, "ic": ic, "rank_ic": rank_ic})
        return pd.DataFrame(ic_records)

    def _build_report(self,
                      factor_name, df_alpha,
                      best_df, worst_df, grouped_returns,
                      df_trades, market_returns, df_ic, cal, kpis,
                      best_g, worst_g, max_group, commission,
                      start_month, end_month, daily_portfolio_df, log_path, factor_cols, benchmark_name,
                      top5_drawdowns_table=None):
        html_parts = []
        k = kpis

        pipeline_html = build_workflow_pipeline(
            f"多因子量化组合流程 — {factor_name}",
            [
                ("因子库构建", f"集成 {len(factor_cols) if factor_cols else 0} 个预处理信号", "cyan"),
                ("打分合成", "等权/IC加权聚合多因子 Alpha Score", "purple"),
                ("股票选择", f"按 Score 排名选 Top {max_group * 50 if max_group else 50} 只", "amber"),
                ("股票池约束", "ST/流动性/次新股过滤 / 个股权重上限 10%", "rose"),
                ("组合构建", f"股息率加权策略组合 vs 业绩基准", "emerald"),
            ]
        )
        # 顶部策略上下文
        context_html = build_strategy_context(
            test_name=f"多因子组合策略: {factor_name}",
            start_month=start_month,
            end_month=end_month,
            benchmark=benchmark_name, 
            factor_cols=factor_cols
        )
        html_parts.append(context_html)

        # KPI 卡片
        card_4 = add_card_to_html("策略核心统计", "📊", [
            ("年化收益率", f"{k['ann_ret']:.2%}"),
            ("年化超额", f"{k['ann_excess_ret']:.2%}"),
            ("年化夏普", f"{k['sharpe']:.2f}"),
            ("信息比率 IR", f"{k['info_ratio']:.2f}"),
            ("最大回撤", f"{k['max_drawdown']:.2%}"),
            ("月度胜率", f"{k['win_rate']:.2%}"),
        ], theme_color="#d97706")

        # 把用于作图的数据进行改造
        df1 = df_alpha.copy()
        if "multi_alpha_score" in df1.columns:
            df1[factor_name] = df1["multi_alpha_score"]

        # 核心逻辑：单组模式下不显示多空对冲图
        is_single_group = (best_g == worst_g)

        # Section: 收益概况
        summary_parts = []
        summary_parts.append(pipeline_html)
        summary_parts.append(card_4)
        
        if top5_drawdowns_table is not None and not top5_drawdowns_table.empty:
            add_table_to_html(summary_parts, top5_drawdowns_table, "策略历史前五大回撤及修复时间", max_rows=None)

        plot_core_performance(
            best_df, kpis["ls_cum"], kpis["drawdowns"], best_g, 
            summary_parts, delay_days=0, strategy_label="多因子策略组合",
            show_ls=not is_single_group
        )
        summary_section = build_section("策略表现", "".join(summary_parts), "策略净值、超额表现与核心 KPI 综览", tab_id="tab-summary")
        html_parts.append(summary_section)

        # Section: 因子分析 (IC走势，不含冗余交易明细)
        trade_parts = []
        # 单组模式下不显示分组净值对比; 多组模式保留
        if not is_single_group:
            plot_group_nav(grouped_returns, best_g, worst_g, trade_parts)
            plot_extreme_groups(best_df, worst_df, best_g, worst_g, trade_parts)
            plot_monotonicity_bar(df_trades, market_returns,
                                  kpis["overall_monotony"], kpis["overall_pval"], trade_parts)
            plot_group_win_rates(df_trades, market_returns, trade_parts)

        plot_ic_trend(df_ic, trade_parts, rank=False)
        plot_ic_trend(df_ic, trade_parts, rank=True)
        plot_ir_rank_ir_bar(df_ic, trade_parts)

        # 月度调仓摘要（只含汇总行，不含个股列表，篇幅小）
        trade_summary = df_trades.copy()
        trade_summary["buy_date"] = pd.to_datetime(trade_summary["buy_date"])
        trade_summary["sell_date"] = pd.to_datetime(trade_summary["sell_date"])
        trade_summary_agg = (
            trade_summary.groupby("year_month")
            .agg(
                buy_date=("buy_date", "min"),
                sell_date=("sell_date", "max"),
                n_stocks=("symbol", "nunique"),
                avg_ret=("actual_ret", "mean"),
                weighted_ret=("actual_ret", lambda x: (x * trade_summary.loc[x.index, "target_weight"]).sum() / trade_summary.loc[x.index, "target_weight"].sum() if "target_weight" in trade_summary.columns else x.mean()),
            )
            .reset_index()
        )
        trade_summary_agg["year_month"] = trade_summary_agg["year_month"].astype(str)
        trade_summary_agg["buy_date"] = trade_summary_agg["buy_date"].dt.strftime("%Y-%m-%d")
        trade_summary_agg["sell_date"] = trade_summary_agg["sell_date"].dt.strftime("%Y-%m-%d")
        trade_summary_agg["avg_ret"] = (trade_summary_agg["avg_ret"] * 100).round(2).astype(str) + "%"
        trade_summary_agg["weighted_ret"] = (trade_summary_agg["weighted_ret"] * 100).round(2).astype(str) + "%"
        trade_summary_agg = trade_summary_agg.rename(columns={
            "buy_date": "买入日", "sell_date": "卖出日",
            "n_stocks": "持仓数", "avg_ret": "等权均收益", "weighted_ret": "加权收益"
        })
        add_table_to_html(trade_parts, trade_summary_agg, "月度调仓摘要", max_rows=None)

        trade_section = build_section("因子分析 & 调仓摘要", "".join(trade_parts), "IC 走势、月度调仓与因子表现综览", tab_id="tab-trades")
        html_parts.append(trade_section)

        # Section: 每日持仓和收益
        daily_parts = []
        if daily_portfolio_df is not None and not daily_portfolio_df.empty:
            plot_daily_nav(daily_portfolio_df, daily_parts)
            daily_section = build_section("策略日频表现", "".join(daily_parts), "基于日终视角的策略组合净值走势", tab_id="tab-daily")
            html_parts.append(daily_section)

        # Section: 输出日志
        logs_parts = []
        log_path = os.path.join(self.out_dir, "backtest_debug.log")
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8") as f:
                log_content = f.read()[-30000:]
            add_log_block(logs_parts, log_content)
        logs_section = build_section("运行日志", "".join(logs_parts), "回测系统输出的详细运行日志", tab_id="tab-logs")
        html_parts.append(logs_section)

        csv_path = save_extreme_groups_csv(self.out_dir, best_df, worst_df, best_g, worst_g, factor_name)

        # 导出后端画图所需的核心数据到 eval_results.pkl
        import pickle
        pkl_path = os.path.join(self.out_dir, "eval_results.pkl")
        try:
            with open(pkl_path, "wb") as f:
                pickle.dump({
                    "kpis": kpis,
                    "best_df": best_df,
                    "worst_df": worst_df,
                    "grouped_returns": grouped_returns,
                    "market_returns": market_returns,
                    "df_ic": df_ic,
                    "daily_portfolio_df": daily_portfolio_df,
                    "factor_name": factor_name,
                    "start_month": start_month,
                    "end_month": end_month,
                    "max_group": max_group,
                    "best_g": best_g,
                    "worst_g": worst_g,
                    "factor_cols": factor_cols,
                }, f)
            self.logger.info(f"[{factor_name}] MF 图表数据已导出: {pkl_path}")
        except Exception as e:
            self.logger.error(f"导出 eval_results.pkl 失败: {e}")

        report_path = write_report_html_no_trades(self.out_dir, factor_name, html_parts)
        self.logger.info(f"[{factor_name}] MF 研报已生成! 文件: {report_path}")
        if csv_path:
            self.logger.info(f"[{factor_name}] MF 极端组月度数据已导出: {csv_path}")
