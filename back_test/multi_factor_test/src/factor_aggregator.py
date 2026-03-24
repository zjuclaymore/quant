"""
因子聚合模块 (Factor Aggregator)

负责将多个中性化后的因子合成为单一 Alpha score。
支持等权、IC加权、IC_IR加权、半衰加权等业界主流方法。
"""

import logging
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

try:
    from .logger import get_logger
except ImportError:
    from logger import get_logger

class FactorAggregator:
    """
    因子聚合器类

    负责将多个中性化后的因子合成为一个综合 Alpha 分数 (multi_alpha_score)。
    支持多种聚合方法:
    - 等权聚合 (equal_weight): 所有因子简单平均
    - IC加权聚合 (ic_weighted): 根据历史 IC 的半衰加权
    - IC_IR加权聚合 (ic_ir_weighted): 根据 IC 信息比率加权
    - 自定义权重 (custom_weight): 用户指定权重

    Attributes:
        logger: 日志记录器
        ic_history: 存储历史 IC 值的字典
    """

    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or get_logger(__name__)
        self.ic_history = {}

    def aggregate_equal_weight(self, df_factors: pd.DataFrame, factor_cols: list) -> pd.DataFrame:
        """
        等权聚合因子

        优先使用中性化后的因子列 (<因子名>_neu)，若不存在则直接使用原始因子列。
        这支持外部已预处理好的因子数据，无需重复标准化。

        参数:
            df_factors: 包含因子数据的 DataFrame
            factor_cols: 因子名称列表（可以是原始列名或带 _neu 后缀）

        Returns:
            pd.DataFrame: 输入 DataFrame 添加 multi_alpha_score 列
        """
        # 优先用 _neu 列，否则退回到原始因子列
        neu_cols = [f"{col}_neu" for col in factor_cols if f"{col}_neu" in df_factors.columns]
        if not neu_cols:
            # 因子已外部预处理，直接使用原始列
            raw_cols = [col for col in factor_cols if col in df_factors.columns]
            if not raw_cols:
                self.logger.error("未找到任何可用的因子列（_neu 或原始列均不存在）")
                return df_factors
            use_cols = raw_cols
            self.logger.info(f"未找到 _neu 列，使用原始因子列: {raw_cols}")
        else:
            use_cols = neu_cols
        res_df = df_factors.copy()
        res_df["multi_alpha_score"] = res_df[use_cols].mean(axis=1)
        self.logger.info(f"等权聚合完成: {len(use_cols)} 个因子, 列: {use_cols}")
        return res_df

    def aggregate_ic_weighted(self, df_factors: pd.DataFrame, factor_cols: list,
                              returns: pd.DataFrame = None, lookback: int = 12,
                              half_life: int = 6) -> pd.DataFrame:
        """
        IC加权聚合 (半衰加权)

        基于历史 IC 序列计算因子权重，使用指数衰减来确定近期 IC 的更高权重。

        参数:
            df_factors: 包含因子数据的 DataFrame
            factor_cols: 因子名称列表
            returns: 收益率数据，用于计算 IC
            lookback: 回溯期月份数，默认为 12
            half_life: 半衰期月数，默认为 6

        Returns:
            pd.DataFrame: 添加了 multi_alpha_score 列的 DataFrame
        """
        if returns is None:
            self.logger.warning("无收益率数据,回退到等权")
            return self.aggregate_equal_weight(df_factors, factor_cols)

        neu_cols = [f"{col}_neu" for col in factor_cols if f"{col}_neu" in df_factors.columns]
        if not neu_cols:
            return df_factors

        ic_weights = self._calc_ic_weights(df_factors, neu_cols, returns, lookback, half_life)
        res_df = df_factors.copy()
        res_df["multi_alpha_score"] = sum(res_df[col] * ic_weights.get(col, 0) for col in neu_cols)
        self.logger.info(f"IC加权聚合完成,权重: {ic_weights}")
        return res_df

    def aggregate_ic_ir_weighted(self, df_factors: pd.DataFrame, factor_cols: list,
                                  returns: pd.DataFrame = None, lookback: int = 12) -> pd.DataFrame:
        """
        IC_IR加权聚合

        根据信息比率 (IC / IC标准差) 计算因子权重，优选稳定且高效的因子。

        参数:
            df_factors: 包含因子数据的 DataFrame
            factor_cols: 因子名称列表
            returns: 收益率数据，用于计算 IC
            lookback: 回溯期月份数，默认为 12

        Returns:
            pd.DataFrame: 添加了 multi_alpha_score 列的 DataFrame
        """
        if returns is None:
            return self.aggregate_equal_weight(df_factors, factor_cols)

        neu_cols = [f"{col}_neu" for col in factor_cols if f"{col}_neu" in df_factors.columns]
        if not neu_cols:
            return df_factors

        ic_ir_weights = self._calc_ic_ir_weights(df_factors, neu_cols, returns, lookback)
        res_df = df_factors.copy()
        res_df["multi_alpha_score"] = sum(res_df[col] * ic_ir_weights.get(col, 0) for col in neu_cols)
        self.logger.info(f"IC_IR加权聚合完成,权重: {ic_ir_weights}")
        return res_df

    def _calc_ic_weights(self, df_factors, neu_cols, returns, lookback, half_life):
        """
        计算半衰IC权重

        私有方法。根据历史 IC 序列计算半衰加权权重。

        参数:
            df_factors: 因子数据
            neu_cols: 中性化因子列名列表
            returns: 收益率数据
            lookback: 回溯期
            half_life: 半衰期

        Returns:
            Dict[str, float]: 各因子名称到权重的映射
        """
        ic_series = {}
        for col in neu_cols:
            ics = []
            for ym in sorted(df_factors["year_month"].unique())[-lookback:]:
                d = df_factors[df_factors["year_month"] == ym]
                r = returns[returns["year_month"] == ym]
                merged = pd.merge(d[["symbol", col]], r[["symbol", "ret"]], on="symbol")
                valid_merged = merged[[col, "ret"]].dropna()
                if len(valid_merged) > 10:
                    ic, _ = spearmanr(valid_merged[col], valid_merged["ret"])
                    ics.append(ic if not np.isnan(ic) else 0)
            ic_series[col] = np.array(ics) if ics else np.array([0])

        weights = {}
        decay = np.exp(-np.log(2) / half_life * np.arange(lookback)[::-1])
        for col, ics in ic_series.items():
            if len(ics) > 0:
                weights[col] = np.average(ics[-len(decay):], weights=decay[-len(ics):])

        total = sum(abs(w) for w in weights.values())
        return {k: v/total for k, v in weights.items()} if total > 0 else {k: 1/len(weights) for k in weights}

    def _calc_ic_ir_weights(self, df_factors, neu_cols, returns, lookback):
        """
        计算IC_IR权重

        私有方法。根据 IC 信息比率计算因子权重。

        参数:
            df_factors: 因子数据
            neu_cols: 中性化因子列名列表
            returns: 收益率数据
            lookback: 回溯期

        Returns:
            Dict[str, float]: 各因子名称到权重的映射
        """
        ic_ir = {}
        for col in neu_cols:
            ics = []
            for ym in sorted(df_factors["year_month"].unique())[-lookback:]:
                d = df_factors[df_factors["year_month"] == ym]
                r = returns[returns["year_month"] == ym]
                merged = pd.merge(d[["symbol", col]], r[["symbol", "ret"]], on="symbol")
                valid_merged = merged[[col, "ret"]].dropna()
                if len(valid_merged) > 10:
                    ic, _ = spearmanr(valid_merged[col], valid_merged["ret"])
                    ics.append(ic if not np.isnan(ic) else 0)
            if len(ics) > 1:
                ic_ir[col] = np.mean(ics) / (np.std(ics) + 1e-6)
            else:
                ic_ir[col] = 0

        total = sum(max(0, v) for v in ic_ir.values())
        return {k: max(0, v)/total for k, v in ic_ir.items()} if total > 0 else {k: 1/len(ic_ir) for k in ic_ir}

    def aggregate_custom_weight(self, df_factors: pd.DataFrame, weight_dict: dict) -> pd.DataFrame:
        """
        自定义权重聚合

        根据用户提供的权重字典，对各因子进行加权求和。

        参数:
            df_factors: 包含因子数据的 DataFrame
            weight_dict: 因子权重字典，键为因子基名，值为权重值

        Returns:
            pd.DataFrame: 添加了 multi_alpha_score 列的 DataFrame
        """
        res_df = df_factors.copy()
        res_df["multi_alpha_score"] = 0.0
        for base_col, w in weight_dict.items():
            neu_col = f"{base_col}_neu"
            if neu_col in res_df.columns:
                res_df["multi_alpha_score"] += res_df[neu_col] * w
        return res_df
