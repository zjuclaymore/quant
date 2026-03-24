"""
交割模块 (Trading Executor)

负责根据每个月的信号表，结合具体真实发车日（考虑涨跌停与延迟卖出），
计算组合实际能成交的价格、扣减摩擦成本并最终生成持仓收益序列。
"""

import logging
import pandas as pd
import numpy as np
from tqdm import tqdm

try:
    from .logger import get_logger
except ImportError:
    from logger import get_logger
from .portfolio_optimizer import PortfolioOptimizer
from .optimizer import CvxPortfolioOptimizer, CVXPY_AVAILABLE
from .risk_model import RiskModel


class TradingExecutor:
    """
    交易执行器类

    负责模拟真实的交易执行过程，包括:
    - 基于涨跌停限制的可交易股票筛选
    - 考虑流动性过滤
    - ST股票和次新股过滤
    - 延迟卖出处理 (遇到跌停时)
    - 日频组合净值的构建

    Attributes:
        limit_df: 涨跌停数据
        liq_df: 流动性数据
        trade_dates: 交易日期列表
        st_df: ST股票数据
        first_dates: 股票上市日期数据
        logger: 日志记录器
    """

    def __init__(self, limit_df: pd.DataFrame, liq_df: pd.DataFrame, trade_dates: pd.DatetimeIndex, st_df: pd.DataFrame, first_dates: pd.DataFrame, logger: logging.Logger = None):
        self.logger = logger or get_logger(__name__)
        self.limit_df = limit_df
        self.liq_df = liq_df
        self.trade_dates = trade_dates
        self.st_df = st_df
        self.first_dates = first_dates
        self.optimizer = PortfolioOptimizer(logger=self.logger)
        self.cvx_optimizer = CvxPortfolioOptimizer(logger=self.logger) if CVXPY_AVAILABLE else None
        self.risk_model = RiskModel(logger=self.logger)

    def _get_trade_mask(self, target_dates):
        """
        复用前人的可交易掩码面板判定。
        排除 停牌、ST、流动性差（前期处理过了，这只是安全兜底）。
        """
        df_d = self.limit_df[self.limit_df["date"].isin(target_dates)].copy()
        
        if "vol" in df_d.columns:
            df_d = df_d[(df_d["vol"].notna()) & (df_d["vol"] > 0)]

        if self.st_df is not None and not self.st_df.empty:
            st_cols = self.st_df.columns.tolist()
            if "S_INFO_WINDCODE" in st_cols and "ENTRY_DT" in st_cols:
                st_intervals = self.st_df[["S_INFO_WINDCODE", "ENTRY_DT", "REMOVE_DT"]].copy()
                st_intervals.rename(columns={"S_INFO_WINDCODE": "symbol"}, inplace=True)
                st_intervals["ENTRY_DT"] = pd.to_datetime(st_intervals["ENTRY_DT"].astype(str), errors="coerce")
                st_intervals["REMOVE_DT"] = pd.to_datetime(st_intervals["REMOVE_DT"].astype(str), errors="coerce").fillna(pd.Timestamp("2099-12-31"))

                target_df = df_d[["date", "symbol"]].copy()
                merged = pd.merge(target_df, st_intervals, on="symbol", how="inner")
                st_hits = merged[(merged["date"] >= merged["ENTRY_DT"]) & (merged["date"] <= merged["REMOVE_DT"])]

                if not st_hits.empty:
                    hit_keys = st_hits.set_index(["date", "symbol"]).index
                    df_d = df_d[~df_d.set_index(["date", "symbol"]).index.isin(hit_keys)]

        # 次新过滤
        df_d = pd.merge(df_d, self.first_dates, on="symbol", how="left")
        df_d = df_d[df_d["date"] >= df_d["first_date"] + pd.Timedelta(days=60)]
        df_d = df_d.drop(columns=["first_date"])

        return df_d[
            ["date", "symbol", "close", "adj_close", "vwap", "up_limit", 
             "down_limit", "open", "high", "low", "is_limit_up", "is_limit_down"]
        ]

    def _handle_delayed_sells(self, trade2, delayed_symbols, t_sell, ym, max_delay_days=20):
        trade2["exit_reason"] = trade2.get("exit_reason", "")
        if delayed_symbols:
            future_dates = self.trade_dates[self.trade_dates > t_sell][:max_delay_days]
            if len(future_dates) > 0:
                future_df = self.limit_df[
                    (self.limit_df["symbol"].isin(delayed_symbols)) & 
                    (self.limit_df["date"].isin(future_dates))
                ].copy()
                if not future_df.empty:
                    future_df["can_sell"] = (~future_df["is_limit_down"]) & (future_df["vol"] > 0)
                    future_df["adj_vwap"] = future_df["vwap"] * (future_df["adj_close"] / future_df["close"])
                    
                    first_sell = (
                        future_df[future_df["can_sell"]]
                        .sort_values("date").groupby("symbol").first().reset_index()
                    )
                    if not first_sell.empty:
                        fsu = first_sell[["symbol", "vwap", "adj_vwap", "date"]].rename(
                            columns={"vwap": "sell_price", "adj_vwap": "sell_adj_price", "date": "sell_date"}
                        )
                        fsu.set_index("symbol", inplace=True)
                        trade2.set_index("symbol", inplace=True)
                        trade2.update(fsu)
                        trade2.reset_index(inplace=True)

        # 仍无法卖出：用最后一个可用交易日价格做强制退出，并标记原因
        drop_mask = trade2["sell_adj_price"].isna()
        if drop_mask.sum() > 0:
            delayed_ext = trade2[drop_mask]["symbol"].tolist()
            future_ext = self.trade_dates[self.trade_dates > t_sell][:max_delay_days]
            if len(future_ext) > 0:
                last_date = future_ext[-1]
                # Fallback directly to df_daily state is tricky without df_daily reference here, 
                # passing limit_df instead which has required columns
                me = self.limit_df[
                    (self.limit_df["symbol"].isin(delayed_ext)) & 
                    (self.limit_df["date"] == last_date)
                ].copy()
                if not me.empty:
                    me["adj_vwap"] = me["vwap"] * (me["adj_close"] / me["close"])
                    eu = me[["symbol", "vwap", "adj_vwap", "date"]].rename(
                        columns={"vwap": "sell_price", "adj_vwap": "sell_adj_price", "date": "sell_date"}
                    )
                    eu.set_index("symbol", inplace=True)
                    trade2.set_index("symbol", inplace=True)
                    trade2.update(eu)
                    trade2.reset_index(inplace=True)
                    trade2.loc[trade2["symbol"].isin(delayed_ext), "exit_reason"] = "max_delay_forced"

        return trade2

    def execute_trades(self, df_alpha: pd.DataFrame, cal: pd.DataFrame, 
                       commission=0.0015, weight_method="equal", **portfolio_params):
        self.logger.info(f"开启撮合循环 (共 {len(cal)} 期), 权重方法: {weight_method}, 成交价: 均价(VWAP)")
        
        holding_period_returns = []
        last_target_weights = None # 用于换手率约束

        for _, row in tqdm(cal.iterrows(), total=len(cal), desc="撮合交割"):
            ym = row["year_month"]
            t_buy = row["buy_date"]
            t_sell = row["sell_date"]

            signal_df = df_alpha[df_alpha["year_month"] == ym].copy()
            if signal_df.empty:
                continue

            # 只交割 group=1（Top50 实盘组），group=0 为被排除的股票，直接跳过
            if "group" in signal_df.columns:
                # 如果是指数增强优化，可能需要全量股票池作为输入，不需要提前过滤 group
                if weight_method != "cvx_enhanced":
                    signal_df = signal_df[signal_df["group"] == 1]
            
            if signal_df.empty:
                continue
                
            # 流动性过滤: 在买入期验证
            liq_snap = self.liq_df[self.liq_df["date"] == t_buy]
            if not liq_snap.empty:
                liq_snap = liq_snap[liq_snap["valid_days_20"] >= 15]
                if not liq_snap.empty:
                    vol_th = liq_snap["avg_vol_20"].quantile(0.10)
                    valid_liq = liq_snap[liq_snap["avg_vol_20"] >= vol_th]["symbol"].tolist()
                    signal_df = signal_df[signal_df["symbol"].isin(valid_liq)]

            buy_mask = self._get_trade_mask([t_buy])
            valid_buys = buy_mask[~buy_mask["is_limit_up"]].copy()
            valid_buys["adj_vwap"] = valid_buys["vwap"] * (valid_buys["adj_close"] / valid_buys["close"])

            # 与组合优化器对接, 得到目标权重
            if weight_method == "cvx_enhanced":
                # 指数增强优化逻辑 (集成了示例代码中的所有约束)
                self.logger.info(f"[{ym}] 执行 Enhanced CVX 优化...")
                
                # 准备基准权重 (从数据中寻找 INDEX_WEIGHT 列，否则尝试市值加权作为基准代理)
                if "INDEX_WEIGHT" in signal_df.columns:
                    idx_w = signal_df.set_index("symbol")["INDEX_WEIGHT"]
                elif "lncap" in signal_df.columns:
                    # 使用当前池子的市值比例作为基准代理
                    signal_df["_tmp_w"] = np.exp(signal_df["lncap"])
                    signal_df["_tmp_w"] = signal_df["_tmp_w"] / signal_df["_tmp_w"].sum()
                    idx_w = signal_df.set_index("symbol")["_tmp_w"]
                else:
                    # 彻底没有权重信息，则使用等权作为基准代理
                    idx_w = pd.Series(1.0 / len(signal_df), index=signal_df["symbol"])

                # 准备风格因子暴露 (除 ID, YM, Industry 外的所有列)
                exclude = ["symbol", "year_month", "industry", "INDEX_WEIGHT", "lncap", "multi_alpha_score", "group", "target_weight"]
                style_cols = [c for c in signal_df.columns if c not in exclude]
                style_df = signal_df.set_index("symbol")[style_cols] if style_cols else None

                # 执行优化
                signal_df = self.optimizer.build_optimized_portfolio(
                    signal_df,
                    index_weights=idx_w,
                    prev_weights=last_target_weights,
                    style_df=style_df,
                    **portfolio_params
                )
                # 记录本期权重供下期换手率约束使用
                last_target_weights = signal_df.set_index("symbol")["target_weight"]

            elif weight_method == "mvo" and self.cvx_optimizer:
                # MVO 优化逻辑
                self.logger.info(f"[{ym}] 执行 MVO 优化...")
                symbols = signal_df["symbol"].tolist()
                expected_returns = signal_df["multi_alpha_score"].values
                
                # 计算协方差矩阵 (需要从外部传入 df_daily)
                df_daily = portfolio_params.get("df_daily")
                if df_daily is not None:
                    cov_matrix = self.risk_model.compute_covariance(df_daily, symbols, t_buy)
                    
                    # 执行优化
                    constraints = {
                        'min_weight': 0.0,
                        'max_weight': portfolio_params.get("max_single_weight", 0.1)
                    }
                    
                    # 使用均值-方差优化
                    best_w = self.cvx_optimizer.optimize_max_return(
                        expected_returns, cov_matrix, 
                        risk_aversion=portfolio_params.get("risk_aversion", 1.0),
                        constraints=constraints
                    )
                    signal_df["target_weight"] = best_w
                else:
                    self.logger.warning("MVO 优化缺少 df_daily, 回退到 score_weight")
                    signal_df = self.optimizer.build_target_portfolio(
                        signal_df, weight_method="score_weight", **portfolio_params
                    )
            else:
                # 仅传递 build_target_portfolio 接受的参数
                _accepted_keys = {
                    "mv_df", "ind_df", "industry_neutral",
                    "max_industry_weight", "max_single_weight"
                }
                filtered_params = {k: v for k, v in portfolio_params.items() if k in _accepted_keys}
                signal_df = self.optimizer.build_target_portfolio(
                    signal_df, 
                    weight_method=weight_method,
                    **filtered_params
                )

            trade1 = pd.merge(
                signal_df,
                valid_buys[["symbol", "vwap", "adj_vwap"]],
                on="symbol", how="inner",
            )
            trade1 = trade1.rename(columns={"vwap": "buy_price", "adj_vwap": "buy_adj_price"})
            trade1["buy_date"] = t_buy
            if trade1.empty:
                continue
            
            # 在可交易股票上重新归一化权重
            # 如果是分组回测且不是特定的组合优化方法，通常按组内等权
            if weight_method == "equal":
                if "group" in trade1.columns:
                    trade1["target_weight"] = trade1.groupby("group")["symbol"].transform(
                        lambda x: 1.0 / len(x) if len(x) > 0 else 0.0
                    )
                else:
                    trade1["target_weight"] = 1.0 / len(trade1) if len(trade1) > 0 else 0.0
            else:
                # 保留 optimizer 计算出的权重并重归一化（因为部分因涨停无法买入）
                if "group" in trade1.columns:
                    trade1["target_weight"] = trade1.groupby("group")["target_weight"].transform(
                        lambda x: x / x.sum() if x.sum() > 0 else 0.0
                    )
                else:
                    trade1["target_weight"] = trade1["target_weight"] / trade1["target_weight"].sum()

            sell_mask = self._get_trade_mask([t_sell])
            t_sell_actual = sell_mask[~sell_mask["is_limit_down"]].copy()
            t_sell_actual["adj_vwap"] = t_sell_actual["vwap"] * (t_sell_actual["adj_close"] / t_sell_actual["close"])
            delayed_symbols = sell_mask[sell_mask["is_limit_down"]]["symbol"].tolist()

            trade2 = pd.merge(
                trade1,
                t_sell_actual[["symbol", "vwap", "adj_vwap"]],
                on="symbol", how="left",
            )
            trade2 = trade2.rename(columns={"vwap": "sell_price", "adj_vwap": "sell_adj_price"})
            trade2["sell_date"] = t_sell

            trade2 = self._handle_delayed_sells(trade2, delayed_symbols, t_sell, ym)
            
            trade2["actual_ret"] = (trade2["sell_adj_price"] / trade2["buy_adj_price"]) * (1 - commission) - 1
            holding_period_returns.append(trade2)

        if not holding_period_returns:
            return pd.DataFrame()
        return pd.concat(holding_period_returns, ignore_index=True)

    def build_daily_portfolio_series(self, df_trades: pd.DataFrame, df_daily: pd.DataFrame):
        """
        基于撮合结果 df_trades, 重建日终视角下的持仓股票及市值.
        假设初始分配给每个标的的资金为其对应 target_weight 比例,
        在真实日行情的 adj_close 序列上计算价值增长情况.
        """
        if df_trades is None or df_trades.empty:
            return pd.DataFrame()

        dft = df_trades.dropna(subset=["buy_adj_price", "target_weight", "buy_date", "sell_date", "actual_ret"]).copy()
        if dft.empty:
            return pd.DataFrame()

        dft["buy_date"] = pd.to_datetime(dft["buy_date"])
        dft["sell_date"] = pd.to_datetime(dft["sell_date"])
        dft["period"] = dft["year_month"].astype(str)

        def _norm(w):
            s = w.sum()
            return w / s if s and s > 0 else w

        # 每期的目标权重归一化 (总和=1)
        dft["target_weight"] = dft.groupby("period")["target_weight"].transform(_norm)
        
        # 卖出后变现锁定的价值
        dft["final_pos_value"] = dft["target_weight"] * (1 + dft["actual_ret"])

        # 定义每期的追踪区间 [period_start, period_end]
        dft["period_start"] = dft.groupby("period")["buy_date"].transform("min")
        dft["period_end"] = dft.groupby("period")["sell_date"].transform("max")

        all_dates = pd.Series(df_daily["date"].unique()).sort_values().reset_index(drop=True)
        
        # 将每一笔交易扩展成完整的日频面板
        idx_dfs = []
        for _, row in dft.iterrows():
            mask = (all_dates >= row["period_start"]) & (all_dates <= row["period_end"])
            dates_in_period = all_dates[mask]
            if dates_in_period.empty: continue
            
            df_sym = pd.DataFrame({
                "symbol": row["symbol"],
                "period": row["period"],
                "date": dates_in_period,
                "buy_date": row["buy_date"],
                "sell_date": row["sell_date"],
                "buy_adj_price": row["buy_adj_price"],
                "target_weight": row["target_weight"],
                "final_pos_value": row["final_pos_value"]
            })
            idx_dfs.append(df_sym)
            
        if not idx_dfs:
            return pd.DataFrame()
            
        merged = pd.concat(idx_dfs, ignore_index=True)
        
        # 关联真实日行情
        merged = pd.merge(merged, df_daily[["symbol", "date", "adj_close"]], on=["symbol", "date"], how="left")
        
        # 针对停牌/空缺数据(可能发生在持仓期中间)进行日度插值前填
        merged = merged.sort_values(["period", "symbol", "date"])
        merged["adj_close"] = merged.groupby(["period", "symbol"])["adj_close"].ffill()
        
        # 计算每一天的资产价值 (核心逻辑：未买入视为等额现金，卖出后锁定收益)
        merged["pos_value"] = merged["target_weight"]  # 初始化为现金
        
        is_active = (merged["date"] >= merged["buy_date"]) & (merged["date"] <= merged["sell_date"])
        is_after_sell = merged["date"] > merged["sell_date"]
        
        has_price = is_active & merged["adj_close"].notna()
        merged.loc[has_price, "pos_value"] = merged.loc[has_price, "target_weight"] * (merged.loc[has_price, "adj_close"] / merged.loc[has_price, "buy_adj_price"])
        
        merged.loc[is_after_sell, "pos_value"] = merged.loc[is_after_sell, "final_pos_value"]
        
        # 各组每日净值加总
        daily = merged.groupby(["period", "date"], as_index=False).agg({"pos_value": "sum", "symbol": "count"})
        daily.rename(columns={"symbol": "holdings"}, inplace=True)
        daily = daily.sort_values(["period", "date"])

        def _normalize(group):
            if group.empty: return group
            first = group["pos_value"].iloc[0]
            if not first or first == 0:
                group["nav_period"] = 1.0
            else:
                group["nav_period"] = group["pos_value"] / first
            return group

        daily = daily.groupby("period", group_keys=False).apply(_normalize)
        period_order = daily.groupby("period")["date"].min().sort_values()
        
        frames = []
        cum = 1.0
        for p in period_order.index:
            sub = daily[daily["period"] == p].copy()
            if sub.empty: continue
            
            sub["nav"] = sub["nav_period"] * cum
            cum = sub["nav"].iloc[-1]
            frames.append(sub)

        if not frames:
            return pd.DataFrame()
        result = pd.concat(frames, ignore_index=True)
        
        # 处理重叠的换仓日: 去重保留最后一个，确保不同期的平滑连接
        result = result.drop_duplicates(subset=["date"], keep="last").sort_values("date")
        
        # 日频收益最后结算计算
        result["daily_ret"] = result["nav"].pct_change().fillna(0)
        
        return result[["date", "nav", "daily_ret", "holdings"]].reset_index(drop=True)
