"""
选股模块 (Stock Selector)

负责根据市场过滤条件 (ST/停牌/次新/流动性差) 清洗标的池，
并根据最后的综合得分分配多空分组。
"""

import logging
import pandas as pd
import numpy as np

try:
    from .logger import get_logger
except ImportError:
    from logger import get_logger

class StockSelector:
    """
    股票选择器类

    负责:
    1. 根据市场条件过滤股票 (ST、停牌、次新股、流动性差的股票)
    2. 根据因子得分对股票进行分组

    Attributes:
        logger: 日志记录器
    """

    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or get_logger(__name__)

    def exclude_invalid_and_group(
        self, 
        df_alpha: pd.DataFrame, 
        df_daily: pd.DataFrame, 
        trade_dates: pd.DatetimeIndex,
        df_st: pd.DataFrame = None,
        df_ipo: pd.DataFrame = None,
        group_size: int = 0
    ) -> tuple:
        """
        在横截面上剔除不可交易标的，并按照 Alpha Score 进行分组排位。
        
        参数:
            df_alpha: 预处理+聚合后的全量 signal 集合，包含 multi_alpha_score
            df_daily: 原始日频行情（用于提前拦截 ST、计算流动性窗口过滤）
            trade_dates: 交易日历
            df_ipo: 上市日期数据
            group_size: 0表示10等分，大于0表示固定支数
            
        返回:
            tuple: (打好 group 标签的信号表，分组数，涨跌停掩码表)
        """
        self.logger.info("开始选股与分组排位...")
        
        # 1. 构建全市场的限制面板
        limit_df = self._build_limit_flags(df_daily)
        liq_df = self._build_liquidity(df_daily, trade_dates)
        first_dates = df_daily.groupby("symbol", as_index=False)["date"].min().rename(columns={"date": "first_date"})

        result_df = df_alpha.copy()

        # --- 新增筛选逻辑 ---
        # 1. 上市满三个月 (约 90 天)
        if df_ipo is not None:
            result_df = pd.merge(result_df, df_ipo, on="symbol", how="left")
            result_df["temp_date"] = result_df["year_month"].dt.to_timestamp(how='end')
            result_df = result_df[(result_df["temp_date"] - result_df["list_date"]).dt.days >= 90].copy()
            result_df.drop(columns=["temp_date", "list_date"], inplace=True)
            self.logger.info(f"上市满三月过滤后剩余: {len(result_df)} 条")

        # 1.5 非 ST 过滤
        if df_st is not None and not df_st.empty:
            st_intervals = df_st[["S_INFO_WINDCODE", "ENTRY_DT", "REMOVE_DT"]].copy()
            st_intervals.rename(columns={"S_INFO_WINDCODE": "symbol"}, inplace=True)
            st_intervals["ENTRY_DT"] = pd.to_datetime(st_intervals["ENTRY_DT"].astype(str), errors="coerce")
            st_intervals["REMOVE_DT"] = pd.to_datetime(st_intervals["REMOVE_DT"].astype(str), errors="coerce").fillna(pd.Timestamp("2099-12-31"))
            
            # 判定基准点：每期月末 (year_month.end_date)
            result_df["base_date"] = result_df["year_month"].dt.to_timestamp(how='end')
            merged_st = pd.merge(result_df[["symbol", "year_month", "base_date"]], st_intervals, on="symbol", how="inner")
            st_hits = merged_st[(merged_st["base_date"] >= merged_st["ENTRY_DT"]) & (merged_st["base_date"] <= merged_st["REMOVE_DT"])]
            
            if not st_hits.empty:
                hit_keys = st_hits.set_index(["symbol", "year_month"]).index
                result_df = result_df[~result_df.set_index(["symbol", "year_month"]).index.isin(hit_keys)].copy()
                self.logger.info(f"非 ST 过滤后剩余: {len(result_df)} 条")
            if "base_date" in result_df.columns:
                result_df.drop(columns=["base_date"], inplace=True)

        # 2. 股息率 > 0
        div_cols = [c for c in result_df.columns if "dividend_yield" in c]
        if div_cols:
            result_df = result_df[result_df[div_cols[0]] > 0]
            self.logger.info(f"股息率 > 0 过滤后剩余: {len(result_df)} 条 (使用列: {div_cols[0]})")

        # 3. 盈余公告开盘跳空超额 > 0
        gap_cols = [c for c in result_df.columns if "announcement_gap" in c]
        if gap_cols:
            result_df = result_df[result_df[gap_cols[0]] > 0]
            self.logger.info(f"盈余公告跳空 > 0 过滤后剩余: {len(result_df)} 条 (使用列: {gap_cols[0]})")

        # --- 由于 df_alpha 只有 year_month 级别，具体的买卖日期
        # 需要在 trading executor 中按历月切片。在此只做基于 factor 的单纯分层排位
        
        # --- 分组逻辑 ---
        if group_size > 0:
            # 实盘选股模式: 按 Alpha Score 降序排名，仅选 Top group_size 只，统一标记为 group=1
            # 其余股票标记为 group=0，排除在撮合之外，避免形成多组导致评估器混乱
            def _assign_top_group(x):
                """对每个截面期: Top N 只 -> group=1, 其余 -> group=0"""
                rank = x.rank(method="first", ascending=True)  # ascending=True: 最大值 rank 最高
                n = len(x)
                threshold = n - group_size  # rank > threshold 为 Top N
                return (rank > threshold).astype(int)

            result_df["group"] = result_df.groupby("year_month")["multi_alpha_score"].transform(
                _assign_top_group
            )
            result_df["group"] = result_df["group"].fillna(0).astype(int)
            n_groups = 1  # 只有一个策略组 (group=1)
            selected_counts = result_df[result_df["group"] == 1].groupby("year_month")["symbol"].count()
            self.logger.info(
                f"Top{group_size} 实盘选股模式: 每期平均入选 {selected_counts.mean():.1f} 只, "
                f"min={selected_counts.min()}, max={selected_counts.max()}"
            )
        else:
            result_df["group"] = result_df.groupby("year_month")["multi_alpha_score"].transform(
                lambda x: pd.qcut(
                    x.rank(method="first"), 10, labels=False, duplicates="drop"
                ) + 1 if x.notna().sum() >= 10 else pd.Series(np.nan, index=x.index)
            )
            result_df["group"] = result_df["group"].fillna(0).astype(int)
            n_groups = int(result_df["group"].max())
            self.logger.info(f"按固定 10 组均分, 实际有效分组数: {n_groups}")
            
        return result_df, n_groups, limit_df, liq_df, first_dates

    def _build_limit_flags(self, df_daily):
        df = df_daily[
            ["date", "symbol", "open", "high", "low", "up_limit", 
             "down_limit", "close", "adj_close", "vwap", "vol"]
        ].copy()
        df["is_limit_up"] = ((df["open"] == df["up_limit"]) & (df["high"] == df["up_limit"]) & df["up_limit"].notna())
        df["is_limit_down"] = ((df["open"] == df["down_limit"]) & (df["low"] == df["down_limit"]) & df["down_limit"].notna())
        return df

    def _build_liquidity(self, df_daily, trade_dates):
        self.logger.info("预计算近期交易流动性掩码...")
        df = df_daily[["date", "symbol", "vol"]].copy()
        df["vol"] = df["vol"].fillna(0)
        
        df = df.sort_values(["symbol", "date"])
        results = []
        for sym, grp in df.groupby("symbol"):
            grp_aligned = grp.set_index("date").reindex(trade_dates)
            grp_aligned["vol"] = grp_aligned["vol"].fillna(0)
            grp_aligned["is_traded"] = (grp_aligned["vol"] > 0).astype("int8")
            grp_aligned["valid_days_20"] = grp_aligned["is_traded"].rolling(20, min_periods=1).sum()
            grp_aligned["avg_vol_20"] = grp_aligned["vol"].rolling(20, min_periods=1).mean()
            grp_aligned["symbol"] = sym
            
            orig_dates = grp["date"].values
            grp_aligned = grp_aligned.loc[grp_aligned.index.isin(orig_dates)]
            grp_aligned = grp_aligned.reset_index().rename(columns={"index": "date"})
            results.append(grp_aligned[["date", "symbol", "valid_days_20", "avg_vol_20"]])
            
        return pd.concat(results, ignore_index=True)
