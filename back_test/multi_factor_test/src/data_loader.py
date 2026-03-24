"""
数据读取模块 (Data Loader)

负责并发/批量读取行情、辅助特征、ST标记、日历及多个因子源。
并且统一为多因子矩阵结构供后续预处理使用。
"""

import os
import glob
import json
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm

try:
    from .logger import get_logger
except ImportError:
    from logger import get_logger

class MultiFactorDataLoader:
    """
    多因子数据加载器类

    负责加载多因子回测所需的各种数据:
    - 因子数据: 从因子库读取因子值，优先使用 Parquet 格式
    - 市场数据: 加载日线行情数据（收盘价、复权价、涨跌停价等）
    - 基准数据: 加载指数收益率作为业绩基准
    - ST 和行业数据: 加载股票 ST 状态和行业分类
    - 日历数据: 生成调仓日历

    Attributes:
        logger: 日志记录器
    """

    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or get_logger(__name__)

    def _get_valid_factor_path(self, factor_group, factor_name, prefer_parquet=True):
        """
        查找因子数据的有效存储路径

        在因子目录下搜索可用的数据文件，优先级为:
        1. 直接在 factor_group/factor_name 目录下查找 (如 factor_package)
        2. Parquet 格式 (class_by_date_parquet)
        3. CSV 格式 (class_by_stock_csv)
        4. 其他兼容格式

        参数:
            factor_group: 因子所属组别名称，如 "1_算术转换因子_ArithmeticFactors" 或 "factor_package"
            factor_name: 因子名称，如 "log_mv_1" 或 "dividend_yield_processed_parquet"
            prefer_parquet: 是否优先使用 Parquet 格式，默认为 True

        Returns:
            Tuple[str, str]: 返回 (数据路径, 格式类型) 元组，如 ("/path/to/data", "parquet")
            如果未找到有效路径则返回 (None, None)
        """
        # 1. 优先尝试直接路径 (对应 factor_package 等扁平化存储)
        direct_path = rf"E:\1_basement\quant_research\factor\{factor_group}\{factor_name}"
        if os.path.isdir(direct_path):
            pq_files = glob.glob(os.path.join(direct_path, "*.parquet"))
            if pq_files:
                return direct_path, "parquet"
            csv_files = glob.glob(os.path.join(direct_path, "*.csv"))
            if csv_files:
                return direct_path, "csv"

        # 2. 尝试标准输出路径
        base_path = rf"E:\1_basement\quant_research\factor\{factor_group}\{factor_name}\output"

        if prefer_parquet:
            parquet_path = os.path.join(base_path, "class_by_date_parquet")
            if os.path.isdir(parquet_path):
                return parquet_path, "parquet"
            
            # 搜索内嵌的一层
            nested_dirs = glob.glob(os.path.join(base_path, "*", "class_by_date_parquet"))
            for p in nested_dirs:
                return p, "parquet"

        candidates = [
            (os.path.join(base_path, "class_by_date_parquet"), "parquet"),
            (os.path.join(base_path, "class_by_stock_csv"), "csv"),
            (os.path.join(base_path, "class_by_stock"), "csv"),
        ]
        
        # 兼容补充
        nested_parquet = glob.glob(os.path.join(base_path, "*", "class_by_date_parquet"))
        for p in nested_parquet:
            candidates.append((p, "parquet"))
            
        for path, fmt in candidates:
            if os.path.isdir(path):
                if fmt == "parquet" or glob.glob(os.path.join(path, "*.csv")):
                    return path, fmt
                    
        raise FileNotFoundError(f"未能找到因子: {factor_group}/{factor_name}")

    def _load_single_factor_parquet(self, factor_dir):
        """
        加载 Parquet 格式的截面因子数据

        从因子目录读取按日期存储的 Parquet 文件，构建因子时间序列。

        参数:
            factor_dir: 因子数据目录路径，需包含按日期命名的 .parquet 文件

        Returns:
            Tuple[pd.DataFrame, str]: 返回 (因子数据DataFrame, 因子名称)
            - DataFrame 包含列: date, symbol, <factor_col>
            - factor_col 为实际因子列名

        Raises:
            FileNotFoundError: 如果未找到 Parquet 文件
            RuntimeError: 如果因子数据为空或读取失败
        """
        meta_file = os.path.join(factor_dir, "_meta.json")
        factor_col = None
        if os.path.exists(meta_file):
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            factor_col = meta.get("factor_name")

        files = sorted(glob.glob(os.path.join(factor_dir, "*.parquet")))
        if not files:
            raise FileNotFoundError(f"未找到Parquet文件: {factor_dir}")

        all_dfs = []
        for f in tqdm(files, desc=f"加载 Parquet {os.path.basename(factor_dir)[:15]}..."):
            try:
                df = pd.read_parquet(f)
                if df.empty:
                    continue
                date_str = os.path.basename(f).replace(".parquet", "")
                df["date"] = pd.to_datetime(date_str, format="%Y%m%d")
                
                if factor_col is None:
                    potential = [c for c in df.columns if c not in ["date", "symbol"]]
                    factor_col = potential[0] if potential else None
                
                # 统一确保列为 factor_col
                if factor_col and factor_col in df.columns:
                    all_dfs.append(df[["date", "symbol", factor_col]])
                else:
                    self.logger.warning(f"因子列未找到: {factor_dir} -> {f}")
            except Exception as e:
                self.logger.warning(f"读取 Parquet 失败: {f} ({e})", exc_info=True)
                
        if not all_dfs:
            raise RuntimeError(f"因子数据为空或读取失败: {factor_dir}")
        full_df = pd.concat(all_dfs, ignore_index=True)
        return full_df, factor_col

    def _load_single_factor_csv(self, factor_dir):
        """
        加载 CSV 格式的时序因子数据

        从因子目录读取按股票存储的 CSV 文件，构建因子时间序列。
        CSV 文件名为股票代码，内容包含日期列和因子值列。

        参数:
            factor_dir: 因子数据目录路径，需包含按股票代码命名的 .csv 文件

        Returns:
            Tuple[pd.DataFrame, str]: 返回 (因子数据DataFrame, 因子名称)
            - DataFrame 包含列: date, symbol, <factor_col>
            - factor_col 为自动识别的因子列名

        Raises:
            RuntimeError: 如果 CSV 数据为空或格式错误
        """
        files = glob.glob(os.path.join(factor_dir, "*.csv"))
        factor_dfs = []
        for f in tqdm(files, desc=f"加载 CSV {os.path.basename(factor_dir)[:15]}..."):
            symbol = os.path.basename(f).replace(".csv", "")
            try:
                df = pd.read_csv(f)
                if df.empty:
                    continue
                date_col = next((c for c in df.columns if "date" in c.lower() or "日期" in c), None)
                if date_col is None:
                    self.logger.warning(f"CSV 缺少日期列: {f}")
                    continue
                df.rename(columns={date_col: "date"}, inplace=True)
                df["date"] = pd.to_datetime(df["date"].astype(str))
                df["symbol"] = symbol
                factor_dfs.append(df)
            except Exception as e:
                self.logger.warning(f"读取 CSV 失败: {f} ({e})", exc_info=True)
                
        if not factor_dfs:
            raise RuntimeError(f"因子数据为空或读取失败: {factor_dir}")
        full_df = pd.concat(factor_dfs, ignore_index=True)
        potential = [c for c in full_df.columns if c not in ["date", "symbol"]]
        factor_col = potential[0] if potential else None
        
        if factor_col:
            full_df = full_df[["date", "symbol", factor_col]]
            
        return full_df, factor_col

    def load_factors_aligned(self, factor_list, prefer_parquet=True):
        """
        加载多个因子并按股票和日期对齐合并

        依次加载每个因子，并根据 symbol 和 date 进行内连接合并，
        生成包含所有因子值的宽表结构。

        参数:
            factor_list: 因子列表，每个元素为 (因子组, 因子名) 的元组
                例如: [('ArithmeticFactors', 'turnover_rate'), ('RegressionFactors', 'beta')]
            prefer_parquet: 是否优先使用 Parquet 格式，默认为 True

        Returns:
            pd.DataFrame: 包含 date, symbol 及各因子的宽表
                -date: 日期
                -symbol: 股票代码
                -<factor_name>: 各因子的值

        Example:
            >>> factors = [('ArithmeticFactors', 'turnover_rate'), ('ValuationFactors', 'pe_ttm')]
            >>> df = loader.load_factors_aligned(factors)
        """
        self.logger.info(f"开始加载 {len(factor_list)} 个因子...")
        
        combined_df = None
        
        for group, name in factor_list:
            try:
                f_dir, fmt = self._get_valid_factor_path(group, name, prefer_parquet)
                self.logger.debug(f"加载因子 {name} 来源: {fmt} - {f_dir}")
                
                if fmt == "parquet":
                    df, col_name = self._load_single_factor_parquet(f_dir)
                else:
                    df, col_name = self._load_single_factor_csv(f_dir)
                    
                if df.empty or col_name is None:
                    self.logger.warning(f"因子 {name} 数据为空，跳过")
                    continue
                    
                # 统一重命名因子列为 name，方便后续调用
                df = df.rename(columns={col_name: name})
                
                # 转换为月频进行粗侧对齐 (单因子的截面对齐逻辑)
                # 后期可以在聚合或截面抽取时再细粒度对齐，这里我们统一到月末。
                df["year_month"] = df["date"].dt.to_period("M")
                df = df.groupby(["symbol", "year_month"]).last().reset_index()
                df = df.drop(columns=["date"])
                
                if combined_df is None:
                    combined_df = df
                else:
                    # 使用 outer join 防止部分因子存在空缺导致整体丢弃
                    combined_df = pd.merge(combined_df, df, on=["symbol", "year_month"], how="outer")
                    
                self.logger.info(f"因子 [{name}] 加载成功.")
                
            except Exception as e:
                self.logger.error(f"加载因子 [{name}] 失败: {e}", exc_info=True)
                
        if combined_df is not None:
            self.logger.info(f"多因子对齐合并完成，数据总行数: {len(combined_df)}")
        return combined_df

    def load_market_data(self, start_date=None, end_date=None):
        """
        加载全市场日行情数据 (EOD Prices)

        从本地数据目录加载 A 股日线行情数据，包含:
        - 基础行情: 开盘价、收盘价、最高价、最低价
        - 交易信息: 成交量、涨跌幅
        - 特殊价格: 均价(VWAP)、涨停价、跌停价、复权收盘价

        参数:
            start_date: 开始日期，格式为 YYYYMMDD 的字符串，默认为 None
            end_date: 结束日期，格式为 YYYYMMDD 的字符串，默认为 None

        Returns:
            pd.DataFrame: 包含日行情数据的 DataFrame，列包括:
                - date: 交易日期
                - symbol: 股票代码
                - open: 开盘价
                - close: 收盘价
                - high: 最高价
                - low: 最低价
                - vol: 成交量
                - adj_close: 复权收盘价
                - vwap: 均价
                - up_limit: 涨停价
                - down_limit: 跌停价
                - is_limit_up: 是否涨停
                - is_limit_down: 是否跌停

        Note:
            数据来源: E:\\1_basement\\quant_research\\data\\中国A股日行情_AShareEODPrices
        """
        data_dir = r"E:\1_basement\quant_research\data\中国A股日行情_AShareEODPrices"
        files = sorted(glob.glob(os.path.join(data_dir, "*.pickle")))

        if start_date or end_date:
            filtered = []
            for f in files:
                fname = os.path.basename(f).replace(".pickle", "")
                if start_date and fname < start_date:
                    continue
                if end_date and fname > end_date:
                    continue
                filtered.append(f)
            files = filtered

        self.logger.info(f"将加载 {len(files)} 个日行情文件...")
        
        keep_cols = [
            "Wind代码", "交易日期", "昨收盘价(元)", "开盘价(元)", 
            "最高价(元)", "最低价(元)", "收盘价(元)", "涨跌幅(%)", 
            "成交量(手)", "均价(VWAP)", "涨停价(元)", "跌停价(元)", "复权收盘价(元)"
        ]
        
        all_dfs = []
        for f in tqdm(files, desc="加载行情"):
            try:
                df = pd.read_pickle(f)
                cols_exist = [c for c in keep_cols if c in df.columns]
                if cols_exist:
                    all_dfs.append(df[cols_exist])
                else:
                    self.logger.warning(f"行情文件缺少期望列: {f}")
            except Exception as e:
                self.logger.warning(f"读取行情失败: {f} ({e})", exc_info=True)

        if not all_dfs:
            raise RuntimeError("未能加载任何行情数据")
            
        result = pd.concat(all_dfs, ignore_index=True)
        col_map = {
            "Wind代码": "symbol", "交易日期": "date", "昨收盘价(元)": "preclose",
            "开盘价(元)": "open", "最高价(元)": "high", "最低价(元)": "low",
            "收盘价(元)": "close", "涨跌幅(%)": "pct_chg", "成交量(手)": "vol",
            "均价(VWAP)": "vwap", "涨停价(元)": "up_limit", "跌停价(元)": "down_limit",
            "复权收盘价(元)": "adj_close"
        }
        result = result.rename(columns=col_map)
        result["date"] = pd.to_datetime(result["date"].astype(str))
        
        # 兼容补充
        if result["adj_close"].isna().all() and "close" in result.columns:
            result["adj_close"] = result["close"]
            
        self.logger.info(f"市场日行情加载完成: {len(result)} 条")
        return result

    def load_benchmark(self, benchmark_symbol, start_date=None, end_date=None, cal=None):
        """
        加载指数收益率作为业绩基准
        If cal is provided: returns monthly returns.
        If cal is NOT provided: returns daily NAV series.
        """
        idx_dir = r"E:\1_basement\quant_research\data\中国A股指数日行情_AIndexEODPrices"
        self.logger.info(f"加载基准数据: {benchmark_symbol}")
        
        bench_prices = {}
        needed_dates = set()
        
        if cal is not None and not cal.empty:
            for _, row in cal.iterrows():
                needed_dates.add(row["buy_date"])
                needed_dates.add(row["sell_date"])
        else:
            # 加载整个日期范围
            if start_date and end_date:
                s = pd.to_datetime(start_date)
                e = pd.to_datetime(end_date)
                # 获取目录下所有日期
                all_files = [f for f in os.listdir(idx_dir) if f.endswith(".pickle")]
                for f in all_files:
                    d_str = f.replace(".pickle", "")
                    try:
                        d_dt = pd.to_datetime(d_str)
                        if s <= d_dt <= e:
                            needed_dates.add(d_dt)
                    except:
                        continue
                
        for d in sorted(needed_dates):
            date_str = d.strftime("%Y%m%d")
            fpath = os.path.join(idx_dir, f"{date_str}.pickle")
            if not os.path.exists(fpath):
                continue
            try:
                df_idx = pd.read_pickle(fpath)
                code_col = [c for c in df_idx.columns if "Wind" in c or "code" in c.lower()][0]
                close_col = [c for c in df_idx.columns if "收盘" in c][0]
                # 不区分大小写匹配
                row_match = df_idx[df_idx[code_col].str.lower() == benchmark_symbol.lower()]
                if not row_match.empty:
                    val = float(row_match.iloc[0][close_col])
                    if pd.notna(val) and val > 0:
                        bench_prices[d] = val
            except Exception:
                pass
                
        bench_series = pd.Series(bench_prices).sort_index()
        
        if cal is not None and not cal.empty:
            b_rets = []
            for _, row in cal.iterrows():
                ym, buy_d, sell_d = row["year_month"], row["buy_date"], row["sell_date"]
                p_b = bench_series.get(buy_d, np.nan)
                p_s = bench_series.get(sell_d, np.nan)
                
                # 回退寻找前一个可用日期
                if np.isnan(p_b):
                    prior = bench_series[bench_series.index <= buy_d]
                    p_b = prior.iloc[-1] if len(prior) > 0 else np.nan
                if np.isnan(p_s):
                    prior = bench_series[bench_series.index <= sell_d]
                    p_s = prior.iloc[-1] if len(prior) > 0 else np.nan
                    
                if not np.isnan(p_b) and not np.isnan(p_s) and p_b > 0:
                    b_rets.append({"year_month": ym, "bench_return": p_s / p_b - 1})
            return pd.DataFrame(b_rets)
        else:
            df_b = bench_series.reset_index()
            if df_b.empty:
                return pd.DataFrame(columns=["date", "daily_ret", "nav"])
            df_b.columns = ["date", "close"]
            df_b["daily_ret"] = df_b["close"].pct_change().fillna(0)
            df_b["nav"] = (1 + df_b["daily_ret"]).cumprod()
            return df_b

    def load_ipo_data(self):
        """
        加载股票上市日期数据
        """
        ipo_path = r"E:\1_basement\quant_research\data\中国A股基本资料_AShareDescription\A股基本资料.pickle"
        if os.path.exists(ipo_path):
            df_ipo = pd.read_pickle(ipo_path)
            # 找到上市公司代码列和上市日期列
            # 根据之前的探索: 'Wind代码' 为 symbol, index 9 为上市日期
            cols = df_ipo.columns.tolist()
            code_col = [c for c in cols if "Wind" in c or "code" in c.lower()][0]
            # 寻找包含 "上市" 和 "日期" 的列
            list_date_col = [c for c in cols if "上市" in c and "日期" in c][0]
            
            df_ipo = df_ipo[[code_col, list_date_col]].copy()
            df_ipo.columns = ["symbol", "list_date"]
            df_ipo["list_date"] = pd.to_datetime(df_ipo["list_date"].astype(str), errors="coerce")
            self.logger.info(f"加载上市日期数据: {len(df_ipo)} 条")
            return df_ipo
        else:
            self.logger.warning("未找到基本资料文件")
            return None

    def load_st_and_industry(self, load_st=True, load_ind=True):
        """
        加载股票 ST 标记和行业分类数据

        加载用于过滤问题股票和进行行业中性分析的数据。

        参数:
            load_st: 是否加载 ST 股票标记数据，默认为 True
            load_ind: 是否加载行业分类数据，默认为 True

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (ST数据DataFrame, 行业数据DataFrame)
            - ST数据: 包含 S_INFO_WINDCODE (股票代码)、ENTRY_DT (开始日期)、REMOVE_DT (解除日期)
            - 行业数据: 包含 symbol, industry (申万行业分类)

        Note:
            ST数据来源: E:\\1_basement\\quant_research\\data\\中国A股特别处理_AShareST
            行业数据来源: E:\\1_basement\\quant_research\\data\\申万行业分类2021版_AShareSWNIndustriesClass
        """
        st_df, ind_df = None, None
        
        if load_st:
            st_path = r"E:\1_basement\quant_research\data\中国A股特别处理_AShareST\ST.pickle"
            if os.path.exists(st_path):
                st_df = pd.read_pickle(st_path)
                # 归一化列名 (避免重复映射)
                target_names = {
                    "S_INFO_WINDCODE": ["Wind代码", "Wind", "S_INFO_WINDCODE"],
                    "ENTRY_DT": ["实施日期", "ENTRY_DT"],
                    "REMOVE_DT": ["撤销日期", "REMOVE_DT"]
                }
                rename_map = {}
                for target, potential_names in target_names.items():
                    for col in st_df.columns:
                        if any(pn in col for pn in potential_names):
                            rename_map[col] = target
                            break # 只匹配一个
                if rename_map:
                    st_df.rename(columns=rename_map, inplace=True)
                self.logger.info(f"加载ST数据: {len(st_df)}")
            else:
                self.logger.warning("未找到ST数据缓存文件")
                
        if load_ind:
            ind_path = r"E:\1_basement\quant_research\data\申万行业分类2021版_AShareSWNIndustriesClass\申万行业分类2021版.pickle"
            if os.path.exists(ind_path):
                ind_df = pd.read_pickle(ind_path)
                self.logger.info(f"加载行业数据: {len(ind_df)}")
            else:
                self.logger.warning("未找到行业分类数据文件")
                
        return st_df, ind_df
        
    def load_calendar(self, df_daily, delay_days=0):
        """
        加载并生成调仓日历

        从预计算的缓存文件中读取交易日历，并根据延迟天数生成每月的调仓时点。
        调仓日历定义了每个月的:
        - 调仓月 (year_month)
        - 买入日期 (buy_date): 信号发布日期或下一个交易日
        - 卖出日期 (sell_date): 持有期结束后的卖出日期

        参数:
            df_daily: 日行情数据 DataFrame，用于确定数据覆盖范围
            delay_days: 信号到实际买入的延迟天数，默认为 0

        Returns:
            pd.DataFrame: 调仓日历，包含列:
                - year_month: 调仓月份
                - buy_date: 买入日期
                - sell_date: 卖出日期

        Raises:
            FileNotFoundError: 如果日历缓存文件不存在

        Note:
            日历来源: E:\\1_basement\\quant_research\\data\\交易日历\\rebalance_calendar_cache.parquet
        """
        cal_path = r"E:\1_basement\quant_research\data\交易日历\rebalance_calendar_cache.parquet"
        
        data_min = pd.to_datetime(df_daily["date"].min()) - pd.Timedelta(days=30)
        data_max = pd.to_datetime(df_daily["date"].max()) + pd.Timedelta(days=30)
        
        if not os.path.exists(cal_path):
            self.logger.error("交易日历不存在, 需要先运行 generate_calendar_cache.py")
            raise FileNotFoundError(cal_path)

        cache = pd.read_parquet(cal_path)
        cache["date"] = pd.to_datetime(cache["date"])
        cache = cache.sort_values("date").reset_index(drop=True)

        cmin, cmax = cache["date"].min(), cache["date"].max()
        if (cmin <= data_min) and (cmax >= data_max):
            cal_raw = cache[(cache["date"] >= data_min) & (cache["date"] <= data_max)].copy()
            trade_dates = pd.DatetimeIndex(cal_raw["date"].values)

            cal_raw["ym"] = cal_raw["date"].dt.to_period("M")
            first_days = cal_raw.groupby("ym")["date"].min()
            last_days = cal_raw.groupby("ym")["date"].max()

            cal_list = []
            yms = sorted(cal_raw["ym"].unique())
            for i in range(len(yms) - 1):
                cur_ym = yms[i]
                next_ym = yms[i + 1]
                
                base_buy = first_days[next_ym]
                base_sell = last_days[next_ym]
                
                buy_idx = trade_dates.get_loc(base_buy)
                sell_idx = trade_dates.get_loc(base_sell)
                
                adj_buy_idx = min(buy_idx + delay_days, len(trade_dates) - 1)
                adj_sell_idx = min(sell_idx + delay_days, len(trade_dates) - 1)
                
                cal_list.append(
                    {
                        "year_month": cur_ym,
                        "signal_date": last_days[cur_ym],
                        "buy_date": trade_dates[adj_buy_idx],
                        "sell_date": trade_dates[adj_sell_idx],
                    }
                )
            df_calendar = pd.DataFrame(cal_list)
            self.logger.info(f"生成包含 {len(df_calendar)} 期的调仓日历")
            return df_calendar
        else:
            raise ValueError("Calendar cache date range is insufficient.")

if __name__ == "__main__":
    # Test DataLoader
    loader = MultiFactorDataLoader()
    # Mock parameters
    factors = [("1_算术转换因子_ArithmeticFactors", "turnover_rate")]
    df_f = loader.load_factors_aligned(factors)
    print(df_f.head())
