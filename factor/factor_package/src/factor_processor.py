import pandas as pd
import numpy as np
import os
import json
import pickle
from scipy.stats.mstats import winsorize
import statsmodels.api as sm

class FactorProcessor:
    """
    因子加工类：实现行业中值填充、去极值、中性化和去极值标准化。
    
    Attributes:
        config (dict): 包含路径和列名定义的配置。
        industry_df (pd.DataFrame): 行业分类数据。
        log_mv_df (pd.DataFrame): 对数市值数据（按需缓存）。
    """

    def __init__(self, config_path: str):
        """
        初始化 FactorProcessor。
        
        Args:
            config_path (str): 配置文件路径。
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.industry_df = self._load_industry_data()
        self.log_mv_cache = {} # date -> Series(stock: log_mv)

    def _load_industry_data(self) -> pd.DataFrame:
        """
        加载并预处理行业分类数据。
        
        Returns:
            pd.DataFrame: 处理后的行业分类数据。
        """
        path = self.config['paths']['industry']
        cols = self.config['logic_mappings']['industry']
        
        df = pd.read_pickle(path)
        # 仅保留必要列并重命名
        df = df[[cols['stock'], cols['industry'], cols['start_date'], cols['end_date']]]
        df.columns = ['stock_code', 'industry_code', 'start_date', 'end_date']
        
        # 处理日期格式，确保对比准确
        df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce').dt.strftime('%Y%m%d')
        # 99991231 会导致 OutOfBoundsDatetime，改为 20991231
        df['end_date'] = df['end_date'].fillna('20991231')
        # 尝试转换并处理异常
        df['end_date'] = pd.to_datetime(df['end_date'], errors='coerce').dt.strftime('%Y%m%d')
        df['end_date'] = df['end_date'].fillna('20991231')
        
        return df

    def get_industry_on_date(self, trade_date: str) -> pd.Series:
        """
        获取指定日期的各股行业分类。
        
        Args:
            trade_date (str): 交易日期 'YYYYMMDD'。
            
        Returns:
            pd.Series: 索引为 stock_code，值为 industry_code。
        """
        mask = (self.industry_df['start_date'] <= trade_date) & (self.industry_df['end_date'] >= trade_date)
        daily_industry = self.industry_df[mask].drop_duplicates(subset='stock_code', keep='last')
        return daily_industry.set_index('stock_code')['industry_code']

    def load_log_mv_on_date(self, trade_date: str, stock_pool: list) -> pd.Series:
        """
        加载指定日期的对数市值数据。由于原始数据按股票存储，这里会按需读取并缓存。
        若存在按日期存储的 Parquet（log_mv_processed），优先读取以提升性能。
        
        Args:
            trade_date (str): 交易日期 'YYYYMMDD'。
            stock_pool (list): 股票代码列表。
            
        Returns:
            pd.Series: 索引为股票代码，值为对数市值。
        """
        if trade_date in self.log_mv_cache:
            return self.log_mv_cache[trade_date]

        val_col = self.config['logic_mappings']['log_mv']['value']

        # 优先尝试按日期 Parquet（处理后 log_mv）
        log_mv_processed_dir = self.config['paths'].get('log_mv_processed')
        if log_mv_processed_dir and os.path.isdir(log_mv_processed_dir):
            parquet_path = os.path.join(log_mv_processed_dir, f"{trade_date}.parquet")
            if os.path.exists(parquet_path):
                try:
                    df = pd.read_parquet(parquet_path)
                    if "symbol" not in df.columns:
                        sym_col = next(
                            (c for c in df.columns if "code" in c.lower() or "symbol" in c.lower()),
                            None,
                        )
                        if sym_col:
                            df = df.rename(columns={sym_col: "symbol"})
                    if val_col not in df.columns:
                        candidate_cols = [c for c in df.columns if c != "symbol"]
                        if candidate_cols:
                            val_col = candidate_cols[0]
                    if "symbol" in df.columns and val_col in df.columns:
                        if stock_pool is not None:
                            df = df[df["symbol"].isin(stock_pool)]
                        res_series = df.set_index("symbol")[val_col]
                        self.log_mv_cache[trade_date] = res_series
                        return res_series
                except Exception:
                    pass
        
        log_mv_dir = self.config['paths']['log_mv']
        date_col = self.config['logic_mappings']['log_mv']['date']
        
        results = {}
        for stock in stock_pool:
            file_path = os.path.join(log_mv_dir, f"{stock}.csv")
            if os.path.exists(file_path):
                # 优化读取：如果文件很大，考虑只读取匹配行，但目前 CSV 较小，直接读取
                df = pd.read_csv(file_path)
                match = df[df[date_col].astype(str) == str(trade_date)]
                if not match.empty:
                    results[stock] = match[val_col].iloc[0]
        
        res_series = pd.Series(results)
        self.log_mv_cache[trade_date] = res_series
        return res_series

    @staticmethod
    def winsorize_3mad(series: pd.Series, n: float = 3.0) -> pd.Series:
        """
        3MAD 去极值法。
        公式：[median - n * MAD, median + n * MAD]
        其中 MAD = Median(|x - median|)
        
        Args:
            series (pd.Series): 待处理序列。
            n (float): MAD 倍数。
            
        Returns:
            pd.Series: 去极值后的序列。
        """
        if series is None or len(series) == 0:
            return series
        valid = series.dropna()
        if valid.empty:
            return series

        median = valid.median()
        mad = (valid - median).abs().median()
        if pd.isna(mad) or mad == 0:
            # 无有效波动时不做截断，避免产生大量无意义告警
            return series
        threshold_low = median - n * mad
        threshold_high = median + n * mad
        return series.clip(lower=threshold_low, upper=threshold_high)

    def process_cross_section(self, df: pd.DataFrame, factor_col: str, mv_col: str = None, ind_col: str = None, 
                              do_neutralization: bool = True, do_standardization: bool = True, do_imputes: bool = True) -> pd.DataFrame:
        """
        截面处理核心逻辑：中值填充 -> 去极值 -> 中性化 -> 标准化。
        
        Args:
            df (pd.DataFrame): 包含因子的数据框。
            factor_col (str): 因子列名。
            mv_col (str): 市值列名（中性化用）。
            ind_col (str): 行业列名（填充和中性化用）。
            do_neutralization (bool): 是否中性化。
            do_standardization (bool): 是否标准化。
            do_imputes (bool): 是否行业中值填充。
            
        Returns:
            pd.DataFrame: 处理后的数据框。
        """
        work_df = df.copy()
        if factor_col not in work_df.columns:
            return work_df.iloc[0:0]
        if work_df[factor_col].dropna().empty:
            return work_df.iloc[0:0]
        
        # 1. 行业中值填充
        if do_imputes and ind_col:
            work_df[factor_col] = work_df.groupby(ind_col)[factor_col].transform(lambda x: x.fillna(x.median()))
            work_df[factor_col] = work_df[factor_col].fillna(work_df[factor_col].median())
        
        # 2. 去极值 (3MAD)
        work_df[factor_col] = self.winsorize_3mad(work_df[factor_col])
        if mv_col and mv_col in work_df.columns:
            work_df[mv_col] = self.winsorize_3mad(work_df[mv_col])
            
        # 3. 中性化
        if do_neutralization:
            # 必须有市值和行业
            drop_cols = [factor_col]
            if mv_col: drop_cols.append(mv_col)
            if ind_col: drop_cols.append(ind_col)
            
            work_df = work_df.dropna(subset=drop_cols, how='any')
            if work_df.empty: return work_df
            
            y = work_df[factor_col]
            X_list = []
            if ind_col:
                X_ind = pd.get_dummies(work_df[ind_col], drop_first=True).astype(float)
                X_list.append(X_ind)
            if mv_col:
                X_list.append(work_df[mv_col])
            
            if X_list:
                X = pd.concat(X_list, axis=1)
                X = sm.add_constant(X)
                model = sm.OLS(y, X).fit()
                work_df[factor_col] = model.resid
        else:
            # 如果不中性化，也要确保因子值有效
            work_df = work_df.dropna(subset=[factor_col])

        # 4. 标准化 (Z-Score)
        if do_standardization and not work_df.empty:
            std = work_df[factor_col].std()
            if std > 0:
                work_df[factor_col] = (work_df[factor_col] - work_df[factor_col].mean()) / std
            else:
                work_df[factor_col] = work_df[factor_col] - work_df[factor_col].mean()
                
        return work_df

    def process_day(self, factor_series: pd.Series, trade_date: str, fill_na: bool = True) -> pd.Series:
        """
        处理单日因子数据：行业中值填充(可选) -> 3MAD去极值 -> 市值行业中性化 -> 标准化。
        
        Args:
            factor_series (pd.Series): 原始因子数据，索引为股票代码。
            trade_date (str): 交易日期 'YYYYMMDD'。
            fill_na (bool): 是否进行行业中值填充。针对某些因子（如股息率）建议设为 False。
            
        Returns:
            pd.Series: 处理后的因子。
        """
        if factor_series is None or len(factor_series) == 0:
            return pd.Series(dtype=float)

        stock_pool = factor_series.index.tolist()
        
        # 1. 获取行业和市值数据
        industry = self.get_industry_on_date(trade_date)
        log_mv = self.load_log_mv_on_date(trade_date, stock_pool)
        
        # 合并数据
        df = pd.DataFrame({
            'factor': factor_series,
            'industry': industry,
            'log_mv': log_mv
        })
        
        # 使用通用的截面处理逻辑
        processed_df = self.process_cross_section(
            df, 
            factor_col='factor', 
            mv_col='log_mv', 
            ind_col='industry',
            do_neutralization=True,
            do_standardization=True,
            do_imputes=fill_na
        )
        
        if processed_df.empty or 'factor' not in processed_df.columns:
            return pd.Series(dtype=float)
        return processed_df['factor']

if __name__ == "__main__":
    # 简单的内部测试，逻辑主要由 smoke_test.py 验证
    pass
