"""
单因子回测引擎 V2 核心实现 (Single Factor Backtester V2 Engine)

本模块定义了 `SingleFactorBacktesterV2` 类，它是整个回测框架的运行时核心。
通过继承 `BacktestCoreMixin`，该引擎整合了行情加载、全局状态预计算、以及高仿真的模拟交易逻辑。

主要职责:
    - 环境初始化: 配置日志系统，建立输出目录，完成原始行情字段的标准化映射。
    - 全局数据预计算: 在回测开始前，利用全量数据预计算股票首次上市日、20日流动性均值以及全历史涨跌停标记。
    - 状态管理: 维护 ST 数据、行业分类数据以及基准指数的缓存，加速截面检索。
    - 接口外接: 为 `BacktestCoreMixin` 提供底层行情数据源 (`self.df_daily`) 和交易日历 (`self.df_calendar`)。
"""

import os
import json
import logging

import numpy as np
import pandas as pd
from .data_loader import load_calendar, load_benchmark_component

from back_test.sigle_factor_test.src.bt_core import BacktestCoreMixin


WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


class SingleFactorBacktesterV2(BacktestCoreMixin):
    """
    单因子回测引擎 V2.5 核心类

    该类是回测系统的“主机盘”，负责管理内存中的行情资产、全局配置及日志记录。
    它采用模块化设计，将具体的算法逻辑混入（Mixin）自 `BacktestCoreMixin`，
    自身则专注于数据准备、预计算优化以及外部接口调用。

    核心执行链:
        Init (行情清洗) -> Precompute (全局加速) -> Run Backtest (主流程) -> Report (结果生成)

    属性说明:
        df_daily (pd.DataFrame): 标准化后的日行情主表。
        out_dir (str): 结果与日志的存储路径。
        st_df / ind_df (pd.DataFrame): 辅助风险数据映射表。
        liq_df / limit_df (pd.DataFrame): 预计算得到的流动性与涨跌停状态。
    """

    STOCK_POOL_CACHE_VERSION = "mid_file_v1"
    NEW_STOCK_DAYS = 60

    def __init__(
        self,
        df_daily_price: pd.DataFrame,
        out_dir: str,
        st_df: pd.DataFrame = None,
        ind_df: pd.DataFrame = None,
        delay_days: int = 0,
        rebalance_month_start: bool = False,
    ):
        """
        初始化回测引擎并激活全局预处理管线

        参数说明:
            df_daily_price: 
                全量日行情 DataFrame，要求包含 'Wind代码', '交易日期' 及 OHLC、成交量等核心列。
            out_dir: 
                回测报告输出的主目录。若不存在则自动递归创建。
            st_df (pd.DataFrame, 可选): 
                ST 状态区间记录。用于在 `_get_trade_mask` 阶段进行标的剔除。
            ind_df (pd.DataFrame, 必填): 
                申万行业分类数据，用于因子行业中性化。不允许为空。
            delay_days (int, 默认 0): 
                交易执行延迟。0 代表信号产生当日即确定次日买入；>0 用于模拟信号计算耗时导致的延迟开仓。

        调用示例:
            >>> engine = SingleFactorBacktesterV2(
            ...     df_daily_price=df_market,
            ...     out_dir="./backtest_results",
            ...     st_df=df_st,
            ...     ind_df=df_industry
            ... )
        """
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.delay_days = int(delay_days)
        self.rebalance_month_start = bool(rebalance_month_start)
        # 基准指数已移除，不再维护 index_daily_dir 和 bench_cache

        self._setup_logger()
        self.logger.info("=== SingleFactorBacktesterV2 (V2.5) 初始化 ===")

        # --- 列名映射 ---
        col_map = {
            "Wind代码": "symbol",
            "交易日期": "date",
            "昨收盘价(元)": "preclose",
            "开盘价(元)": "open",
            "最高价(元)": "high",
            "最低价(元)": "low",
            "收盘价(元)": "close",
            "涨跌幅(%)": "pct_chg",
            "成交量(手)": "vol",
            "均价(VWAP)": "vwap",
            "交易状态": "trade_status",
            "涨停价(元)": "up_limit",
            "跌停价(元)": "down_limit",
            "复权收盘价(元)": "adj_close",
        }
        df = df_daily_price.rename(columns=col_map)
        df["date"] = pd.to_datetime(df["date"].astype(str))

        # 补全必须列
        for c in [
            "symbol",
            "date",
            "open",
            "high",
            "low",
            "close",
            "vol",
            "up_limit",
            "down_limit",
            "adj_close",
        ]:
            if c not in df.columns:
                df[c] = np.nan

        # adj_close 容错：先兜底价格列，再落盘到对象，再输出数据概览用于排查数据质量
        if df["adj_close"].isna().all() and "close" in df.columns:
            df["adj_close"] = df["close"]
            self.logger.warning("adj_close 全缺失, 以 close 替代")

        self.df_daily = df
        self.logger.info(
            f"日行情加载完成: {len(self.df_daily)} 条, "
            f"标的数: {self.df_daily['symbol'].nunique()}, "
            f"日期范围: {self.df_daily['date'].min()} ~ {self.df_daily['date'].max()}"
        )

        # --- ST 数据 ---
        self._load_st_data(st_df)

        # --- 行业数据（强制） ---
        if ind_df is None or ind_df.empty:
            raise ValueError("行业数据为必需输入，禁止空值或缺失")
        self.ind_df = ind_df.copy()
        self.logger.info(f"行业数据已加载: {len(self.ind_df)} 条")

        # --- 全局预处理 ---
        self._build_calendar(delay_days)
        self._precompute_first_dates()
        self._precompute_liquidity()
        self._precompute_limit_flags()
        self._build_stock_pool_cache()

        self.logger.info("=== 初始化及预处理全部完成 ===")

    # ==========================================
    # 日志配置
    # ==========================================

    def _setup_logger(self):
        """
        配置标准化的双回路日志系统 (Dual-stream Logger)

        - 通道1 (StreamHandler): 实时将 INFO 级别及以上日志输出至终端，方便开发者监控进度。
        - 通道2 (FileHandler): 将全部 DEBUG 级别细节写入 `out_dir/backtest_debug.log`，供复盘故障使用。
        """
        self.logger = logging.getLogger(f"SFB_V2_{id(self)}")
        self.logger.setLevel(logging.DEBUG)

        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
            self.logger.addHandler(ch)

            log_file = os.path.join(self.out_dir, "backtest_debug.log")
            fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(
                logging.Formatter(
                    "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
                )
            )
            self.logger.addHandler(fh)

    # ==========================================
    # ST 数据加载
    # ==========================================

    def _load_st_data(self, st_df):
        """
        同步本地 ST 数据资产

        逻辑说明:
            若输入非空，则将其挂载至 `self.st_df`。该数据在 `_get_trade_mask` 阶段用于
            通过区间重叠算法剔除处于 ST/退市状态的风险标的。
        """
        if st_df is None or st_df.empty:
            raise ValueError("ST数据为必需输入，禁止空值或缺失")
        self.st_df = st_df
        self.logger.info(f"ST数据已加载: {len(st_df)} 条区间记录")

    # ==========================================
    # 调仓日历构建
    # ==========================================


    def _build_calendar(self, delay_days=0):
        """
        构建回测专属调仓日历

        逻辑细节:
            通过 `data_loader.load_calendar` 读取缓存并注入发车延迟参数。
            生成的 `self.df_calendar` 决定了信号触发与买卖撮合的精确时间轴。
        """
        self.df_calendar, self.trade_dates = load_calendar(
            self.df_daily,
            delay_days,
            self.rebalance_month_start,
            self.logger,
        )

    # ==========================================
    # 全局预处理
    # ==========================================

    def _precompute_first_dates(self):
        """
        全局预处理 1: 首次上市日计算 (IPO Date Tracking)

        逻辑说明:
            从全量行情中提取每个标的的最小日期。此资产用于后续过滤“次新股” (上市不足 60 天)，
            以防止新股上市初期的剧烈波动和不合理定价干扰因子表现。
        """
        self.logger.info("开始全局预处理: 计算股票首次出现日期...")
        self.first_dates = self.df_daily.groupby("symbol", as_index=False)["date"].min()
        self.first_dates.rename(columns={"date": "first_date"}, inplace=True)
        self.logger.info(
            f"股票首次出现日期预先计算完成 (标的数: {len(self.first_dates)})"
        )

    def _precompute_liquidity(self):
        """
        全局预处理 2: 滚动流动性指标计算 (Rolling Liquidity Metrics)

        逻辑说明:
            1. 向量化滚动: 对每只股票在全背景日历下计算 `rolling(20)` 的有效交易天数和平均成交量。
            2. 数据对齐: 关键修复了停牌期间的空值处理，将其填充为 0 vol，确保滑动窗口的物理时间跨度准确，
               有效消除了“复牌即交易”的未来函数偏差。
            3. 实战约束: 只有当标的在过去一个月内流动性充足时，回测引擎才允许其被选入组合。
        """
        self.logger.info("开始全局预处理: 计算 rolling(20) 流动性指标 (日历对齐)...")

        df = self.df_daily[["date", "symbol", "vol"]].copy()
        df["vol"] = df["vol"].fillna(0)

        # 构建完整交易日历索引
        full_dates = self.trade_dates

        # 按股票分组, 逐只对齐日历后滚动
        df = df.sort_values(["symbol", "date"])
        results = []
        for sym, grp in df.groupby("symbol"):
            grp = grp.copy()
            grp_aligned = grp.set_index("date").reindex(full_dates)
            grp_aligned["vol"] = grp_aligned["vol"].fillna(0)
            grp_aligned["is_traded"] = (grp_aligned["vol"] > 0).astype("int8")
            grp_aligned["valid_days_20"] = (
                grp_aligned["is_traded"].rolling(20, min_periods=1).sum()
            )
            grp_aligned["avg_vol_20"] = (
                grp_aligned["vol"].rolling(20, min_periods=1).mean()
            )
            # 仅保留原始有行情的行 (symbol 非空)
            grp_aligned["symbol"] = sym
            # 只保留 df_daily 中实际存在的日期
            original_dates = grp["date"].values
            grp_aligned = grp_aligned.loc[grp_aligned.index.isin(original_dates)]
            grp_aligned = grp_aligned.reset_index().rename(columns={"index": "date"})
            results.append(grp_aligned[["date", "symbol", "valid_days_20", "avg_vol_20"]])

        self.liq_df = pd.concat(results, ignore_index=True)
        self.logger.info(f"流动性预处理完成 (日历对齐), 共 {len(self.liq_df)} 条记录")

        # 清理临时内存
        del df, results

    def _precompute_limit_flags(self):
        """
        全局预处理 3: 涨跌停极端状态标记 (Limit Price Flagging)

        逻辑说明:
            利用 `open`, `high`, `low` 与 `up_limit`, `down_limit` 的价格关系，预先标记每一天的：
            - 一字涨停 (`is_limit_up`): 模拟实盘中无法挂单买入。
            - 一字跌停 (`is_limit_down`): 模拟实盘中无法挂单卖出，触发延迟交易逻辑。
        """
        self.logger.info("开始全局预处理: 标记涨跌停状态...")

        df = self.df_daily[
            [
                "date", "symbol", "open", "high", "low",
                "up_limit", "down_limit", "close", "adj_close", "vwap", "vol",
            ]
        ].copy()

        df["is_limit_up"] = (
            (df["open"] == df["up_limit"])
            & (df["high"] == df["up_limit"])
            & df["up_limit"].notna()
        )
        df["is_limit_down"] = (
            (df["open"] == df["down_limit"])
            & (df["low"] == df["down_limit"])
            & df["down_limit"].notna()
        )

        self.limit_df = df
        self.logger.info("涨跌停标记预处理完成")

    def _build_stock_pool_cache(self):
        """
        全局预处理 4: 生成并落盘可交易股票池表 (Stock Pool Cache)

        产物:
            1) mid_file/stock_trade_pool.parquet
               字段包含代码、时间、量价信息及交易掩码(可买/可卖)。
            2) mid_file/filter_rules.json
               描述股票池筛选规则与样本统计，便于复盘口径。
        """
        self.logger.info("开始全局预处理: 生成股票池交易表缓存...")

        # 股票池作为全局复用资产，落盘到项目根目录的 mid_file 目录，供后续同口径回测直接读取
        cache_dir = os.path.join(WORKSPACE_ROOT, "mid_file")
        os.makedirs(cache_dir, exist_ok=True)
        pool_path = os.path.join(cache_dir, "stock_trade_pool.parquet")
        explain_path = os.path.join(cache_dir, "filter_rules.json")

        expected_meta = self._build_stock_pool_meta(stats=None)
        cache_hit, cached_df, reason = self._try_load_stock_pool_cache(pool_path, explain_path, expected_meta)
        # if cache_hit:
        #     self.stock_pool_df = cached_df
        #     self.logger.info(f"股票池缓存命中并复用: {pool_path}")
        #     return
        self.logger.info(f"股票池缓存未命中，重建原因: {reason}")

        df = self.limit_df.copy()

        # 统一基础可交易掩码：非停牌 + 非北交所 + 非次新 + 非ST
        df["is_not_suspended"] = df["vol"].notna() & (df["vol"] > 0)
        df["is_not_bj"] = ~df["symbol"].astype(str).str.endswith(".BJ")

        # 次新股过滤
        df = pd.merge(df, self.first_dates, on="symbol", how="left")
        df["is_not_new"] = df["date"] >= (df["first_date"] + pd.Timedelta(days=self.NEW_STOCK_DAYS))

        # ST 过滤（向量化）
        df["is_not_st"] = True
        if not self.st_df.empty:
            try:
                st_cols = self.st_df.columns.tolist()
                if "S_INFO_WINDCODE" in st_cols and "ENTRY_DT" in st_cols:
                    st_intervals = self.st_df[["S_INFO_WINDCODE", "ENTRY_DT", "REMOVE_DT"]].copy()
                    st_intervals = st_intervals.rename(columns={"S_INFO_WINDCODE": "symbol"})
                    st_intervals["ENTRY_DT"] = pd.to_datetime(st_intervals["ENTRY_DT"].astype(str), errors="coerce")
                    st_intervals["REMOVE_DT"] = pd.to_datetime(
                        st_intervals["REMOVE_DT"].astype(str), errors="coerce"
                    ).fillna(pd.Timestamp("2099-12-31"))

                    target_df = df[["date", "symbol"]].copy()
                    merged = pd.merge(target_df, st_intervals, on="symbol", how="inner")
                    st_hits = merged[
                        (merged["date"] >= merged["ENTRY_DT"]) & (merged["date"] <= merged["REMOVE_DT"])
                    ]
                    if not st_hits.empty:
                        hit_keys = st_hits[["date", "symbol"]].drop_duplicates().set_index(["date", "symbol"]).index
                        df["is_not_st"] = ~df.set_index(["date", "symbol"]).index.isin(hit_keys)
            except Exception as e:
                self.logger.debug(f"股票池 ST 过滤预处理异常 (非致命): {e}")

        df["tradable_base"] = df["is_not_suspended"] & df["is_not_bj"] & df["is_not_new"] & df["is_not_st"]
        df["can_buy"] = df["tradable_base"] & (~df["is_limit_up"])
        df["can_sell"] = df["tradable_base"] & (~df["is_limit_down"])

        keep_cols = [
            "date", "symbol", "open", "high", "low", "close", "adj_close", "vwap", "vol",
            "up_limit", "down_limit", "is_limit_up", "is_limit_down",
            "tradable_base", "can_buy", "can_sell",
        ]
        self.stock_pool_df = df[keep_cols].copy()

        self.stock_pool_df.to_parquet(pool_path, index=False)

        stats = {
            "rows_total": int(len(self.stock_pool_df)),
            "rows_tradable_base": int(self.stock_pool_df["tradable_base"].sum()),
            "rows_can_buy": int(self.stock_pool_df["can_buy"].sum()),
            "rows_can_sell": int(self.stock_pool_df["can_sell"].sum()),
            "unique_symbols": int(self.stock_pool_df["symbol"].nunique()),
            "date_start": str(self.stock_pool_df["date"].min()),
            "date_end": str(self.stock_pool_df["date"].max()),
        }
        explain = self._build_stock_pool_meta(stats=stats)
        with open(explain_path, "w", encoding="utf-8") as f:
            json.dump(explain, f, ensure_ascii=False, indent=2)

        self.logger.info(f"股票池缓存已生成: {pool_path}")
        self.logger.info(f"股票池筛选说明已生成: {explain_path}")

    def _st_signature(self):
        if self.st_df is None or self.st_df.empty:
            return {"enabled": False}

        cols = self.st_df.columns.tolist()
        if "S_INFO_WINDCODE" not in cols or "ENTRY_DT" not in cols:
            return {
                "enabled": True,
                "format": "unknown",
                "rows": int(len(self.st_df)),
                "columns": cols,
            }

        tmp = self.st_df[["S_INFO_WINDCODE", "ENTRY_DT", "REMOVE_DT"]].copy()
        tmp["ENTRY_DT"] = pd.to_datetime(tmp["ENTRY_DT"].astype(str), errors="coerce")
        tmp["REMOVE_DT"] = pd.to_datetime(tmp["REMOVE_DT"].astype(str), errors="coerce")

        return {
            "enabled": True,
            "format": "wind_interval",
            "rows": int(len(tmp)),
            "symbols": int(tmp["S_INFO_WINDCODE"].nunique()),
            "entry_min": str(tmp["ENTRY_DT"].min()),
            "entry_max": str(tmp["ENTRY_DT"].max()),
            "remove_min": str(tmp["REMOVE_DT"].min()),
            "remove_max": str(tmp["REMOVE_DT"].max()),
        }

    def _calendar_signature(self):
        if getattr(self, "trade_dates", None) is None or len(self.trade_dates) == 0:
            return {"count": 0}
        return {
            "count": int(len(self.trade_dates)),
            "min": str(min(self.trade_dates)),
            "max": str(max(self.trade_dates)),
            "delay_days": int(getattr(self, "delay_days", 0)),
        }

    def _build_stock_pool_meta(self, stats=None):
        filters = [
            "停牌过滤: vol > 0 且非空",
            "北交所过滤: symbol 不以 .BJ 结尾",
            f"次新股过滤: date >= first_date + {self.NEW_STOCK_DAYS}天",
            "ST过滤: 不在 [ENTRY_DT, REMOVE_DT] 区间",
        ]
        return {
            "version": self.STOCK_POOL_CACHE_VERSION,
            "generated_at": pd.Timestamp.now().isoformat(),
            "source": "SingleFactorBacktesterV2._build_stock_pool_cache",
            "filters": filters,
            "mask_definitions": {
                "tradable_base": "通过全部基础过滤",
                "can_buy": "tradable_base 且非一字涨停",
                "can_sell": "tradable_base 且非一字跌停",
            },
            "spec": {
                "new_stock_days": int(self.NEW_STOCK_DAYS),
                "exclude_bj": True,
                "suspend_rule": "vol_notna_and_gt0",
                "st_rule": "wind_interval_active",
                "buy_sell_rule": "limit_up_down_guard",
            },
            "validation": {
                "daily_rows": int(len(self.limit_df)),
                "daily_symbols": int(self.limit_df["symbol"].nunique()),
                "daily_date_min": str(self.limit_df["date"].min()),
                "daily_date_max": str(self.limit_df["date"].max()),
                "calendar": self._calendar_signature(),
                "st_signature": self._st_signature(),
            },
            "stats": stats or {},
            "files": {
                "trade_table": "stock_trade_pool.parquet",
                "explain": "filter_rules.json",
            },
        }

    def _try_load_stock_pool_cache(self, pool_path, explain_path, expected_meta):
        if not (os.path.exists(pool_path) and os.path.exists(explain_path)):
            return False, None, "缓存文件不存在"

        try:
            with open(explain_path, "r", encoding="utf-8") as f:
                cache_meta = json.load(f)
        except Exception as e:
            return False, None, f"读取筛选说明失败: {e}"

        mismatches = []
        if cache_meta.get("version") != expected_meta.get("version"):
            mismatches.append("version")
        if cache_meta.get("spec") != expected_meta.get("spec"):
            mismatches.append("spec")
        if cache_meta.get("filters") != expected_meta.get("filters"):
            mismatches.append("filters")
        if cache_meta.get("validation") != expected_meta.get("validation"):
            mismatches.append("validation")

        if mismatches:
            return False, None, f"元信息不匹配: {','.join(mismatches)}"

        try:
            cache_df = pd.read_parquet(pool_path)
        except Exception as e:
            return False, None, f"读取缓存表失败: {e}"

        required_cols = {
            "date", "symbol", "open", "high", "low", "close", "adj_close", "vwap", "vol",
            "up_limit", "down_limit", "is_limit_up", "is_limit_down",
            "tradable_base", "can_buy", "can_sell",
        }
        if not required_cols.issubset(set(cache_df.columns)):
            return False, None, "缓存表字段不完整"

        cache_df["date"] = pd.to_datetime(cache_df["date"], errors="coerce")
        if cache_df["date"].isna().any():
            return False, None, "缓存表日期字段异常"

        return True, cache_df, "ok"

    # ==========================================
    # 截面数据检索
    # ==========================================

    def _get_trade_mask(self, target_dates):
        """
        获取动态可交易股票池掩码 (Dynamic Tradable Universe Mask)

        这是回测引擎中最重要的过滤接口，确保入选组合的标的在逻辑上和制度上都是可成交的。

        过滤准则:
            1. 停牌过滤: 剔除成交量 (vol) 为 0 或 Nan 的标的。
            2. ST 过滤: 实时校验股票在 `target_dates` 是否处于 ST 或退市整理期。
            3. 次新股过滤: 剔除上市时间不满 60 个自然日的标的。
            4. 交易限制: 暴露涨跌停标记，供调仓循环决定具体撮合逻辑。

        参数:
            target_dates (list): 需要获取掩码的交易日期集合。

        返回:
            pd.DataFrame: 合格标的的截面属性表。
        """
        # 复用预处理缓存，避免每个调仓日重复执行过滤逻辑
        if hasattr(self, "stock_pool_df") and self.stock_pool_df is not None:
            df_d = self.stock_pool_df[
                self.stock_pool_df["date"].isin(target_dates) & self.stock_pool_df["tradable_base"]
            ].copy()
        else:
            df_d = self.limit_df[self.limit_df["date"].isin(target_dates)].copy()
            if "vol" in df_d.columns:
                df_d = df_d[(df_d["vol"].notna()) & (df_d["vol"] > 0)]
            df_d["can_buy"] = ~df_d["is_limit_up"]
            df_d["can_sell"] = ~df_d["is_limit_down"]

        return df_d[
            [
                "date", "symbol", "close", "adj_close", "vwap",
                "up_limit", "down_limit", "open", "high", "low",
                "is_limit_up", "is_limit_down", "can_buy", "can_sell",
            ]
        ]


    # _load_benchmark_returns 已移除：内部基准不再需要加载指数数据。
    # 收益对齐由 BacktestCoreMixin._align_benchmark 以 bench_return=0 展开。
