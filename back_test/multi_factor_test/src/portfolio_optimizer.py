"""
组合优化模块 (Portfolio Optimizer)

负责根据传入的有效选股池输出目标资产篮子及权重。
支持等权、市值加权、行业中性、风险平价等业界主流方法。
"""

import logging
import pandas as pd
import numpy as np

try:
    from .logger import get_logger
    from .optimizer import CvxPortfolioOptimizer, CVXPY_AVAILABLE
except ImportError:
    from logger import get_logger
    from optimizer import CvxPortfolioOptimizer, CVXPY_AVAILABLE

class PortfolioOptimizer:
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or get_logger(__name__)
        self.cvx_opt = CvxPortfolioOptimizer(logger=self.logger) if CVXPY_AVAILABLE else None

    def build_optimized_portfolio(
        self,
        valid_buys: pd.DataFrame,
        index_weights: pd.Series,
        prev_weights: pd.Series = None,
        style_df: pd.DataFrame = None,
        **params
    ) -> pd.DataFrame:
        """
        基于 CVXPY 的专业组合优化
        
        参数:
            valid_buys: 包含 SCORE (因子得分) 和 industry 的股票池
            index_weights: 基准权重 Series (index 为 symbol)
            prev_weights: 上期持仓权重 Series (index 为 symbol)
            style_df: 风格因子暴露 DataFrame (symbol 为 index, 列为因子名)
            params: 优化参数 (drift, turnover, etc.)
        """
        if not CVXPY_AVAILABLE or self.cvx_opt is None:
            self.logger.warning("CVXPY 不可用，回退到等权分配")
            df = valid_buys.copy()
            df["target_weight"] = 1.0 / len(df) if len(df) > 0 else 0.0
            return df

        df = valid_buys.copy()
        symbols = df["symbol"].tolist()
        n = len(symbols)
        
        # 1. 准备 Score
        score_col = params.get("score_col", "multi_alpha_score")
        if score_col not in df.columns:
            # 如果没找到指定列，试着找一个看起来像因子的列
            possible = [c for c in df.columns if c not in ["symbol", "year_month", "industry", "lncap"]]
            score_col = possible[0] if possible else None
        
        scores = df[score_col].values if score_col else np.zeros(n)
        
        # 2. 准备基准权重
        w_bench = index_weights.reindex(symbols).fillna(0).values
        # 归一化基准权重（确保在当前选股池内 sum=1，或者保持原始比例）
        # 通常在指数增强中，如果选股池是全市场，我们直接用原始权重；
        # 如果选股池是窄池，可能需要重归一化。
        if w_bench.sum() > 0:
            w_bench = w_bench / w_bench.sum()
        
        # 3. 准备漂移约束 (lb, ub)
        weight_drift_low = params.get("weight_low_drift", -0.01)
        weight_drift_high = params.get("weight_high_drift", 0.01)
        lb = np.maximum(0, w_bench + weight_drift_low)
        ub = w_bench + weight_drift_high
        
        constraints = {
            'weight_lb': lb,
            'weight_ub': ub
        }
        
        # 4. 准备行业约束
        if "industry" in df.columns:
            industry_dummies = pd.get_dummies(df["industry"].astype(str))
            X_ind = industry_dummies.values.T # (n_ind, n_stocks)
            
            # 计算基准行业分布
            ind_bench = X_ind @ w_bench
            
            ind_drift_low = params.get("industry_weight_low_drift", -0.02)
            ind_drift_high = params.get("industry_weight_high_drift", 0.02)
            
            ind_lb = np.maximum(0, ind_bench + ind_drift_low)
            ind_ub = ind_bench + ind_drift_high
            
            constraints['industry_matrix'] = X_ind
            constraints['industry_bounds'] = np.column_stack([ind_lb, ind_ub])
            
        # 5. 准备风格约束
        if style_df is not None:
            # 仅保留在当前 symbols 中的股票
            style_data = style_df.reindex(symbols).fillna(0)
            X_style = style_data.values.T # (n_style, n_stocks)
            
            style_bench = X_style @ w_bench
            
            style_drift_low = params.get("style_low_drift", -0.1)
            style_drift_high = params.get("style_high_drift", 0.1)
            
            style_lb = style_bench + style_drift_low
            style_ub = style_bench + style_drift_high
            
            constraints['factor_matrix'] = X_style
            constraints['factor_bounds'] = np.column_stack([style_lb, style_ub])
            
        # 6. 准备换手率约束
        if prev_weights is not None:
            w_prev = prev_weights.reindex(symbols).fillna(0).values
            constraints['prev_weights'] = w_prev
            constraints['max_turnover'] = params.get("turnover_limit", 1.0) # 默认不限制
            
        # 7. 准备成分股约束 (逻辑：本例中暂定 index_weight > 0 为成分股)
        X_comp = (w_bench > 0).astype(int)
        constraints['comp_matrix'] = X_comp
        constraints['comp_lb'] = params.get("composite_weight_low", 0.8)
        constraints['comp_ub'] = params.get("composite_weight_high", 1.0)

        # 执行优化
        self.logger.info(f"执行指数增强优化: 标的数={n}, 求解器={params.get('solver', 'OSQP')}")
        target_w = self.cvx_opt.optimize_index_enhanced(
            expected_returns=scores,
            benchmark_weights=w_bench,
            constraints=constraints,
            solver=params.get("solver", "OSQP")
        )
        
        df["target_weight"] = target_w
        
        # 最后的微小权重剔除与重归一化
        min_w = params.get("min_weight_threshold", 0.0001)
        df.loc[df["target_weight"] < min_w, "target_weight"] = 0.0
        if df["target_weight"].sum() > 0:
            df["target_weight"] = df["target_weight"] / df["target_weight"].sum()
            
        return df

    def build_target_portfolio(
        self,
        valid_buys: pd.DataFrame,
        weight_method: str = "equal",
        mv_df: pd.DataFrame = None,
        ind_df: pd.DataFrame = None,
        industry_neutral: bool = False,
        max_industry_weight: float = 0.3,
        max_single_weight: float = 0.05
    ) -> pd.DataFrame:
        """
        根据给定的候选池分配权重

        参数:
            valid_buys: 候选股票池
            weight_method: 'equal', 'mc_weight', 'score_weight', 'risk_parity'
            industry_neutral: 是否行业中性
            max_industry_weight: 单行业最大权重
            max_single_weight: 单股票最大权重
        """
        df = valid_buys.copy()

        if weight_method == "equal":
            df["target_weight"] = 1.0 / len(df) if len(df) > 0 else 0.0

        elif weight_method == "mc_weight":
            if mv_df is not None and not mv_df.empty:
                df = pd.merge(df, mv_df[["symbol", "lncap"]], on="symbol", how="left")
                df["target_weight"] = np.exp(df["lncap"]) / np.exp(df["lncap"]).sum()
            else:
                self.logger.warning("市值数据缺失,回退到等权")
                df["target_weight"] = 1.0 / len(df) if len(df) > 0 else 0.0

        elif weight_method == "score_weight":
            if "multi_alpha_score" in df.columns:
                scores = df["multi_alpha_score"].fillna(0)
                scores = scores - scores.min() + 1e-6
                df["target_weight"] = scores / scores.sum()
            else:
                df["target_weight"] = 1.0 / len(df) if len(df) > 0 else 0.0

        elif weight_method.endswith("_weight") and weight_method.replace("_weight", "") in df.columns:
            # 支持指定列加权 (如 dividend_yield_processed_parquet_weight)
            col = weight_method.replace("_weight", "")
            self.logger.info(f"使用列 [{col}] 进行加权分配. 可用列: {df.columns.tolist()}")
            weights = df[col].fillna(0)
            weights = weights.clip(lower=0) # 仅支持非负权重
            if weights.sum() > 0:
                df["target_weight"] = weights / weights.sum()
            else:
                self.logger.warning(f"权重列 {col} 全为0或缺失, 回退到等权")
                df["target_weight"] = 1.0 / len(df) if len(df) > 0 else 0.0

        elif weight_method == "risk_parity":
            df["target_weight"] = self._risk_parity_weights(df, mv_df)
        else:
            raise ValueError(f"未知权重方法: {weight_method}")

        # 行业中性约束
        if industry_neutral and "industry" in df.columns:
            df = self._apply_industry_neutral(df, max_industry_weight)

        # 单股票权重上限 (采用迭代重平衡确保严格不超限)
        if max_single_weight < 1.0:
            df["target_weight"] = self._apply_max_weight_constraint(df["target_weight"], max_single_weight)

        return df

    def _apply_max_weight_constraint(self, weights: pd.Series, max_w: float, max_iter: int = 20) -> pd.Series:
        """迭代对权重进行上限约束"""
        w = weights.copy()
        for _ in range(max_iter):
            if w.max() <= max_w + 1e-8:
                break
            # 标记超限部分
            is_over = w > max_w
            over_weight = w[is_over].sum() - is_over.sum() * max_w
            w[is_over] = max_w
            # 将多出的权重按比例分配给未超限的股票
            is_under = w < max_w
            if is_under.any():
                w[is_under] += over_weight * (w[is_under] / w[is_under].sum())
            else:
                # 如果全部都超限(理论上不可能，除非 N < 1/max_w)，则只能等权平分
                w = pd.Series(1.0 / len(w), index=w.index)
                break
        return w

    def _risk_parity_weights(self, df, mv_df):
        """风险平价权重(简化版:市值倒数)"""
        if mv_df is not None and "lncap" in df.columns:
            inv_vol = 1.0 / np.exp(df["lncap"])
            return inv_vol / inv_vol.sum()
        return pd.Series(1.0 / len(df), index=df.index)

    def _apply_industry_neutral(self, df, max_ind_weight):
        """行业中性约束"""
        ind_weights = df.groupby("industry")["target_weight"].sum()
        for ind, w in ind_weights.items():
            if w > max_ind_weight:
                mask = df["industry"] == ind
                df.loc[mask, "target_weight"] *= max_ind_weight / w
        df["target_weight"] = df["target_weight"] / df["target_weight"].sum()
        return df
