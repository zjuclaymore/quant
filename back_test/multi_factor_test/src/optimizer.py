"""
组合优化求解器 (Portfolio Optimizer with CVXPY)

使用cvxpy进行带约束的组合优化,支持:
- 最小化跟踪误差
- 最大化预期收益
- 风险平价
- 行业/风格中性约束
- 权重上下限约束
"""

import logging
import pandas as pd
import numpy as np

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False

try:
    from .logger import get_logger
except ImportError:
    from logger import get_logger

class CvxPortfolioOptimizer:
    """
    基于CVXPY的投资组合优化器类

    使用凸优化方法求解投资组合权重，支持:
    - 最小化跟踪误差
    - 最大化预期收益 (Sharpe比率优化)
    - 风险平价
    - 行业/风格中性约束
    - 权重上下限约束

    Attributes:
        logger: 日志记录器
    """

    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or get_logger(__name__)
        if not CVXPY_AVAILABLE:
            self.logger.warning("cvxpy未安装,仅支持简单权重方法")

    def optimize_min_tracking_error(self, expected_returns: np.ndarray,
                                    cov_matrix: np.ndarray,
                                    benchmark_weights: np.ndarray,
                                    constraints: dict = None) -> np.ndarray:
        """最小化跟踪误差"""
        if not CVXPY_AVAILABLE:
            raise ImportError("需要安装cvxpy: pip install cvxpy")

        n = len(expected_returns)
        w = cp.Variable(n)

        # 目标: 最小化跟踪误差
        tracking_error = cp.quad_form(w - benchmark_weights, cov_matrix)
        objective = cp.Minimize(tracking_error)

        # 基础约束
        cons = [cp.sum(w) == 1, w >= 0]

        # 添加自定义约束
        if constraints:
            cons.extend(self._build_constraints(w, constraints, n))

        prob = cp.Problem(objective, cons)
        prob.solve(solver=cp.OSQP)

        if prob.status not in ["optimal", "optimal_inaccurate"]:
            self.logger.warning(f"优化未收敛: {prob.status}")
            return benchmark_weights

        return w.value

    def optimize_max_return(self, expected_returns: np.ndarray,
                           cov_matrix: np.ndarray,
                           risk_aversion: float = 1.0,
                           constraints: dict = None) -> np.ndarray:
        """最大化风险调整后收益 (均值-方差优化)"""
        if not CVXPY_AVAILABLE:
            raise ImportError("需要安装cvxpy")

        n = len(expected_returns)
        w = cp.Variable(n)

        # 目标: 最大化 E[r] - λ*Var[r]
        ret = expected_returns @ w
        risk = cp.quad_form(w, cov_matrix)
        objective = cp.Maximize(ret - risk_aversion * risk)

        cons = [cp.sum(w) == 1, w >= 0]
        if constraints:
            cons.extend(self._build_constraints(w, constraints, n))

        prob = cp.Problem(objective, cons)
        prob.solve(solver=cp.OSQP)

        if prob.status not in ["optimal", "optimal_inaccurate"]:
            self.logger.warning(f"优化未收敛: {prob.status}")
            return np.ones(n) / n

        return w.value

    def optimize_risk_parity(self, cov_matrix: np.ndarray,
                            constraints: dict = None) -> np.ndarray:
        """风险平价优化"""
        if not CVXPY_AVAILABLE:
            # 简化版: 波动率倒数加权
            vols = np.sqrt(np.diag(cov_matrix))
            w = 1.0 / vols
            return w / w.sum()

        n = cov_matrix.shape[0]
        w = cp.Variable(n)

        # 风险平价近似: 最小化风险贡献的方差
        risk_contrib = cp.multiply(w, cov_matrix @ w)
        objective = cp.Minimize(cp.sum_squares(risk_contrib - cp.sum(risk_contrib) / n))

        cons = [cp.sum(w) == 1, w >= 0.001]
        if constraints:
            cons.extend(self._build_constraints(w, constraints, n))

        prob = cp.Problem(objective, cons)
        prob.solve(solver=cp.OSQP)

        if prob.status not in ["optimal", "optimal_inaccurate"]:
            vols = np.sqrt(np.diag(cov_matrix))
            w = 1.0 / vols
            return w / w.sum()

        return w.value

    def optimize_index_enhanced(self, 
                               expected_returns: np.ndarray,
                               benchmark_weights: np.ndarray,
                               constraints: dict = None,
                               solver: str = "OSQP") -> np.ndarray:
        """
        指数增强优化 (Index Enhanced Optimization)
        
        目标: 最大化 Alpha 得分 (Maximize Expected Returns)
        约束: 
        1. 个股权重在基准权重附近的漂移限制 (Drift Constraints)
        2. 行业/风格暴露在基准暴露附近的漂移限制
        3. 换手率限制
        4. 投资组合成分股比例限制
        
        参数:
            expected_returns: 预期收益/Alpha得分 (n,)
            benchmark_weights: 基准权重 (n,)
            constraints: 约束配置字典
            solver: 求解器名称 (OSQP, ECOS, CBC等)
        """
        if not CVXPY_AVAILABLE:
            raise ImportError("需要安装cvxpy")

        n = len(expected_returns)
        w = cp.Variable(n)
        
        # 目标函数: 最大化 Alpha
        objective = cp.Maximize(expected_returns @ w)
        
        # 基础约束: 权重归一化 (Sum=1)
        cons = [cp.sum(w) == 1]
        
        # 1. 个股漂移约束 (Drift)
        # 这里的 lb, ub 通常是 max(0, w_bench + drift_low) 和 w_bench + drift_high
        if 'weight_lb' in constraints and 'weight_ub' in constraints:
            cons.append(w >= constraints['weight_lb'])
            cons.append(w <= constraints['weight_ub'])
        else:
            cons.append(w >= 0)
            
        # 2. 其它通用约束 (行业、风格、换手率)
        if constraints:
            cons.extend(self._build_constraints(w, constraints, n))
            
        # 3. 成分股比例约束 (X_comp @ w >= composite_weight_low)
        if 'comp_matrix' in constraints:
            X_comp = constraints['comp_matrix']
            if 'comp_lb' in constraints:
                cons.append(X_comp @ w >= constraints['comp_lb'])
            if 'comp_ub' in constraints:
                cons.append(X_comp @ w <= constraints['comp_ub'])

        prob = cp.Problem(objective, cons)
        
        # 尝试求解
        try:
            # 默认使用 OSQP，如果指定了 CBC 且环境支持也可以
            prob.solve(solver=solver)
        except Exception as e:
            self.logger.error(f"优化器求解异常: {e}")
            return benchmark_weights

        if prob.status not in ["optimal", "optimal_inaccurate"]:
            self.logger.warning(f"指数增强优化未找到最优解: {prob.status}")
            return benchmark_weights

        return w.value

    def optimize_industry_neutral(self, expected_returns: np.ndarray,
                                  industry_matrix: np.ndarray,
                                  benchmark_industry: np.ndarray,
                                  constraints: dict = None) -> np.ndarray:
        """行业中性优化"""
        if not CVXPY_AVAILABLE:
            raise ImportError("需要安装cvxpy")

        n = len(expected_returns)
        w = cp.Variable(n)

        objective = cp.Maximize(expected_returns @ w)

        # 行业中性约束: Aw = b (A是行业矩阵, b是基准行业权重)
        cons = [
            cp.sum(w) == 1,
            w >= 0,
            industry_matrix @ w == benchmark_industry
        ]

        if constraints:
            cons.extend(self._build_constraints(w, constraints, n))

        prob = cp.Problem(objective, cons)
        prob.solve(solver=cp.OSQP)

        if prob.status not in ["optimal", "optimal_inaccurate"]:
            self.logger.warning(f"优化未收敛: {prob.status}")
            return np.ones(n) / n

        return w.value

    def _build_constraints(self, w, constraints: dict, n: int) -> list:
        """构建约束条件"""
        cons = []

        # 权重上下限
        if 'min_weight' in constraints:
            cons.append(w >= constraints['min_weight'])
        if 'max_weight' in constraints:
            cons.append(w <= constraints['max_weight'])

        # 行业暴露约束
        if 'industry_matrix' in constraints and 'industry_bounds' in constraints:
            A = constraints['industry_matrix']
            bounds = constraints['industry_bounds']
            industry_exp = A @ w
            cons.append(industry_exp >= bounds[:, 0])
            cons.append(industry_exp <= bounds[:, 1])

        # 换手率约束
        if 'prev_weights' in constraints and 'max_turnover' in constraints:
            w_prev = constraints['prev_weights']
            turnover = cp.norm(w - w_prev, 1)
            cons.append(turnover <= constraints['max_turnover'])

        # 因子暴露约束
        if 'factor_matrix' in constraints and 'factor_bounds' in constraints:
            F = constraints['factor_matrix']
            bounds = constraints['factor_bounds']
            factor_exp = F @ w
            cons.append(factor_exp >= bounds[:, 0])
            cons.append(factor_exp <= bounds[:, 1])

        return cons

    def simple_weights(self, n: int, method: str = "equal") -> np.ndarray:
        """简单权重方法(无需cvxpy)"""
        if method == "equal":
            return np.ones(n) / n
        elif method == "random":
            w = np.random.dirichlet(np.ones(n))
            return w
        else:
            return np.ones(n) / n
