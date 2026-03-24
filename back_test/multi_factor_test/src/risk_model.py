"""
风险模型模块 (Risk Model)

负责计算资产协方差矩阵及各项风险指标。
支持：
- 样本协方差矩阵计算
- 压缩估计 (Shrinkage) 协方差
- 风险归因预处理
"""

import pandas as pd
import numpy as np
import logging

try:
    from .logger import get_logger
except ImportError:
    from logger import get_logger

class RiskModel:
    """
    风险模型类

    用于从历史行情数据中提取风险特征，特别是协方差矩阵。
    """

    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or get_logger(__name__)

    def compute_covariance(self, df_daily: pd.DataFrame, 
                           symbol_list: list, 
                           current_date: pd.Timestamp, 
                           lookback: int = 120,
                           min_periods: int = 30,
                           shrinkage: float = 0.01) -> np.ndarray:
        """
        计算选定证券池的协方差矩阵

        参数:
            df_daily: 包含 date, symbol, adj_close 的日线数据
            symbol_list: 当前选出的 Top N 股票列表
            current_date: 当前调仓基准日(信号日)
            lookback: 回看天数 (交易日)
            min_periods: 最少样本数
            shrinkage: 压缩系数，用于修正非正定矩阵

        Returns:
            np.ndarray: 对齐后的协方差矩阵 (dim: N x N)
        """
        # 1. 提取回看窗口数据 (排除成交当日 current_date, 避免未来函数)
        start_date = current_date - pd.Timedelta(days=lookback * 1.6) 
        mask = (df_daily["symbol"].isin(symbol_list)) & \
               (df_daily["date"] < current_date) & \
               (df_daily["date"] >= start_date)
        
        window_data = df_daily[mask].copy()
        
        # 2. 计算收益率矩阵 (Pivot)
        # 注意：这里需要确保日期对齐
        prices = window_data.pivot(index="date", columns="symbol", values="adj_close")
        
        # 按照 symbol_list 排序，确保返回的矩阵顺序正确
        # 补全缺失的 symbol (如果有加载失败的情况)
        for s in symbol_list:
            if s not in prices.columns:
                prices[s] = np.nan
        prices = prices[symbol_list]
        
        # 取最近 lookback 条数据
        prices = prices.tail(lookback)
        
        # 计算日收益率 (对数收益率更稳健)
        returns = np.log(prices / prices.shift(1)).dropna(how='all')
        
        if len(returns) < min_periods:
            self.logger.warning(f"[{current_date}] 风险模型样本不足: {len(returns)} < {min_periods}")
            # 样本不足时返回单位阵作为占位
            return np.eye(len(symbol_list)) * 0.0001 

        # 3. 计算样本协方差 (年化: * 252)
        cov = returns.cov().values * 252
        
        # 4. 填充缺失值 (某些新股或停牌股可能有 NaN)
        cov = np.nan_to_num(cov, nan=0.0)
        
        # 5. 压缩处理 (Ledoit-Wolf 简化版: (1-s)*S + s*I*trace(S)/n)
        # 确保矩阵正定，避免优化器报错
        if shrinkage > 0:
            n = cov.shape[0]
            trace_avg = np.trace(cov) / n if n > 0 else 0
            cov = (1 - shrinkage) * cov + shrinkage * np.eye(n) * trace_avg
            
        return cov

    def get_stock_vols(self, df_daily, symbol_list, current_date, lookback=60):
        """获取个股年化波动率"""
        start_date = current_date - pd.Timedelta(days=lookback * 1.6)
        prices = df_daily[(df_daily["symbol"].isin(symbol_list)) & 
                          (df_daily["date"] <= current_date) & 
                          (df_daily["date"] >= start_date)].pivot(index="date", columns="symbol", values="adj_close")
        prices = prices.tail(lookback)
        returns = prices.pct_change().dropna(how='all')
        vols = returns.std() * np.sqrt(252)
        return vols.reindex(symbol_list).fillna(vols.mean()).values
