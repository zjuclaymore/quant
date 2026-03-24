"""
业绩归因模块 (Performance Attribution)

提供多因子策略的业绩归因分析功能，包括:
- Brinson归因分析：分解超额收益为行业配置效应、选股效应和交互效应
- 因子暴露分析：计算组合在各因子上的暴露度
- IC统计分析：按年度和行业维度分析因子 IC 表现
- 换手率分析：计算组合月度换手率
- 交易成本分析：估算交易佣金成本
"""

import logging
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

try:
    from .logger import get_logger
except ImportError:
    from logger import get_logger

class PerformanceAttribution:
    """
    业绩归因分析器类

    提供多因子策略的业绩归因功能，用于分析组合超额收益的来源
    以及因子、行业的贡献程度。

    Attributes:
        logger: 日志记录器，默认为 get_logger(__name__)
    """

    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or get_logger(__name__)

    def brinson_attribution(self, portfolio_df: pd.DataFrame, benchmark_df: pd.DataFrame,
                           returns_df: pd.DataFrame, industry_df: pd.DataFrame) -> pd.DataFrame:
        """
        Brinson归因分析

        将组合的超额收益率分解为三个组成部分:
        - 资产配置效应 (Allocation): 由于组合在行业权重上与基准的偏离而产生的收益差异
        - 选股效应 (Selection): 由于组合在特定行业内选股能力而产生的超额收益
        - 交互效应 (Interaction): 配置效应和选股效应的交叉影响

        参数:
            portfolio_df: 包含组合持仓的DataFrame，需要包含 year_month, symbol, weight 列
            benchmark_df: 包含基准持仓的DataFrame，需要包含 year_month, symbol, weight 列
            returns_df: 包含股票收益率的DataFrame，需要包含 year_month, symbol, ret 列
            industry_df: 包含股票行业分类的DataFrame，需要包含 symbol, industry 列

        返回:
            包含各月各行业归因结果的DataFrame，列包括 year_month, industry, allocation, selection, interaction
        """
        results = []

        for ym in portfolio_df["year_month"].unique():
            port = portfolio_df[portfolio_df["year_month"] == ym]
            bench = benchmark_df[benchmark_df["year_month"] == ym]
            rets = returns_df[returns_df["year_month"] == ym]

            # 合并行业信息
            port = pd.merge(port, industry_df[["symbol", "industry"]], on="symbol", how="left")
            bench = pd.merge(bench, industry_df[["symbol", "industry"]], on="symbol", how="left")

            # 计算行业权重和收益
            port_ind = port.groupby("industry").agg({"weight": "sum"})
            bench_ind = bench.groupby("industry").agg({"weight": "sum"})

            # 合并收益率
            port = pd.merge(port, rets[["symbol", "ret"]], on="symbol", how="left")
            bench = pd.merge(bench, rets[["symbol", "ret"]], on="symbol", how="left")

            port_ind_ret = port.groupby("industry").apply(
                lambda x: (x["weight"] * x["ret"]).sum() / x["weight"].sum()
            )
            bench_ind_ret = bench.groupby("industry").apply(
                lambda x: (x["weight"] * x["ret"]).sum() / x["weight"].sum()
            )

            # Brinson分解
            for ind in port_ind.index:
                wp = port_ind.loc[ind, "weight"] if ind in port_ind.index else 0
                wb = bench_ind.loc[ind, "weight"] if ind in bench_ind.index else 0
                rp = port_ind_ret.loc[ind] if ind in port_ind_ret.index else 0
                rb = bench_ind_ret.loc[ind] if ind in bench_ind_ret.index else 0

                allocation = (wp - wb) * rb
                selection = wb * (rp - rb)
                interaction = (wp - wb) * (rp - rb)

                results.append({
                    "year_month": ym,
                    "industry": ind,
                    "allocation": allocation,
                    "selection": selection,
                    "interaction": interaction
                })

        return pd.DataFrame(results)

    def factor_exposure_analysis(self, portfolio_df: pd.DataFrame,
                                 factor_df: pd.DataFrame) -> pd.DataFrame:
        """
        因子暴露分析

        计算组合在每个时点各因子上的加权平均暴露度，用于分析策略的风格特征。

        参数:
            portfolio_df: 包含组合持仓的DataFrame，需要包含 year_month, symbol, weight 列
            factor_df: 包含因子值的DataFrame，需要包含 year_month, symbol 以及以 _neu 结尾的因子列

        返回:
            包含各月各因子暴露度的DataFrame，列包括 year_month 以及各因子名称
        """
        results = []

        for ym in portfolio_df["year_month"].unique():
            port = portfolio_df[portfolio_df["year_month"] == ym]
            factors = factor_df[factor_df["year_month"] == ym]

            merged = pd.merge(port, factors, on="symbol", how="inner")

            factor_cols = [c for c in factors.columns if c.endswith("_neu")]

            exposures = {}
            for col in factor_cols:
                exposure = (merged["weight"] * merged[col]).sum()
                exposures[col] = exposure

            exposures["year_month"] = ym
            results.append(exposures)

        return pd.DataFrame(results)

    def ic_statistics_by_year(self, df_ic: pd.DataFrame) -> pd.DataFrame:
        """
        分年度IC统计分析

        计算每年因子的 IC 均值、标准差、胜率以及信息比率 IR。

        参数:
            df_ic: 包含 IC 值的 DataFrame，需要包含 year_month, ic, rank_ic 列

        返回:
            包含各年度 IC 统计指标的 DataFrame，列包括:
            - year: 年份
            - ic_mean: IC 均值
            - ic_std: IC 标准差
            - ic_win_rate: IC 胜率 (IC > 0 的比例)
            - rank_ic_mean: RankIC 均值
            - rank_ic_std: RankIC 标准差
            - rank_ic_win_rate: RankIC 胜率
            - ic_ir: IC 信息比率 (ic_mean / ic_std)
            - rank_ic_ir: RankIC 信息比率
        """
        df_ic["year"] = df_ic["year_month"].dt.year

        stats = df_ic.groupby("year").agg({
            "ic": ["mean", "std", lambda x: (x > 0).sum() / len(x)],
            "rank_ic": ["mean", "std", lambda x: (x > 0).sum() / len(x)]
        })

        stats.columns = ["ic_mean", "ic_std", "ic_win_rate",
                        "rank_ic_mean", "rank_ic_std", "rank_ic_win_rate"]
        stats["ic_ir"] = stats["ic_mean"] / stats["ic_std"]
        stats["rank_ic_ir"] = stats["rank_ic_mean"] / stats["rank_ic_std"]

        return stats.reset_index()

    def ic_statistics_by_industry(self, df_trades: pd.DataFrame,
                                  industry_df: pd.DataFrame) -> pd.DataFrame:
        """
        分行业IC统计分析

        计算每个行业内的因子 IC 表现，用于分析因子在不同行业的有效性。

        参数:
            df_trades: 包含交易记录的 DataFrame，需要包含 year_month, symbol, multi_alpha_score, actual_ret 列
            industry_df: 包含股票行业分类的 DataFrame，需要包含 symbol, industry 列

        返回:
            按 IC_IR 降序排列的行业 IC 统计 DataFrame，列包括:
            - industry: 行业名称
            - ic_mean: IC 均值
            - ic_std: IC 标准差
            - ic_ir: IC 信息比率
            - ic_win_rate: IC 胜率
            - n_periods: 统计的月份数
        """
        merged = pd.merge(df_trades, industry_df[["symbol", "industry"]],
                         on="symbol", how="left")

        results = []
        for ind in merged["industry"].unique():
            if pd.isna(ind):
                continue

            ind_data = merged[merged["industry"] == ind]

            ics = []
            for ym in ind_data["year_month"].unique():
                d = ind_data[ind_data["year_month"] == ym]
                if len(d) > 10:
                    ic, _ = spearmanr(d["multi_alpha_score"].dropna(),
                                     d["actual_ret"].dropna())
                    if not np.isnan(ic):
                        ics.append(ic)

            if len(ics) > 0:
                results.append({
                    "industry": ind,
                    "ic_mean": np.mean(ics),
                    "ic_std": np.std(ics),
                    "ic_ir": np.mean(ics) / (np.std(ics) + 1e-6),
                    "ic_win_rate": sum(1 for x in ics if x > 0) / len(ics),
                    "n_periods": len(ics)
                })

        return pd.DataFrame(results).sort_values("ic_ir", ascending=False)

    def turnover_analysis(self, df_trades: pd.DataFrame) -> pd.DataFrame:
        """
        换手率分析

        计算组合每月的换手率，即相邻月份持仓股票的变化比例。
        换手率 = (新进入股票数 + 退出股票数) / 总股票数

        参数:
            df_trades: 包含交易记录的 DataFrame，需要包含 year_month, symbol 列

        返回:
            包含月度换手率统计的 DataFrame，列包括:
            - year_month: 年月
            - turnver: 换手率 (0-1 之间的比例)
            - n_stocks: 当期持仓股票数量
        """
        results = []
        prev_holdings = set()

        for ym in sorted(df_trades["year_month"].unique()):
            curr = df_trades[df_trades["year_month"] == ym]
            curr_holdings = set(curr["symbol"].unique())

            if prev_holdings:
                turnover = len(curr_holdings.symmetric_difference(prev_holdings)) / len(curr_holdings)
            else:
                turnover = 1.0

            results.append({
                "year_month": ym,
                "turnover": turnover,
                "n_stocks": len(curr_holdings)
            })

            prev_holdings = curr_holdings

        return pd.DataFrame(results)

    def transaction_cost_analysis(self, df_trades: pd.DataFrame,
                                  commission: float = 0.0015) -> dict:
        """
        交易成本分析

        基于换手率和佣金费率估算策略的交易成本。

        参数:
            df_trades: 包含交易记录的 DataFrame，需要包含 year_month, symbol 列
            commission: 单边交易佣金费率，默认为 0.0015 (千1.5)

        返回:
            包含交易成本统计指标的字典:
            - avg_monthly_turnover: 平均月度换手率
            - total_transaction_cost: 总交易成本 (百分比)
            - annualized_cost: 年化交易成本 (百分比)
            - cost_per_trade: 每笔交易的成本 (双边佣金)
        """
        turnover_df = self.turnover_analysis(df_trades)

        avg_turnover = turnover_df["turnover"].mean()
        total_cost = avg_turnover * commission * 2 * len(turnover_df)  # 买卖双边

        return {
            "avg_monthly_turnover": avg_turnover,
            "total_transaction_cost": total_cost,
            "annualized_cost": total_cost / len(turnover_df) * 12,
            "cost_per_trade": commission * 2
        }
