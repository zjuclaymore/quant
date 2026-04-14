"""
数据加载与 I/O 管理模块 (Data Loader & I/O Management)

本模块负责回测过程中所有外部信号、行情、辅助资产及缓存数据的统一加载。它封装了底层文件系统的复杂度，
提供统一的 DataFrame 接口，并支持多种工业级数据格式（Parquet/CSV/Pickle）。

主要工作内容:
    - 智能路径发现: 多层级搜索因子文件夹，优先匹配高性能 Parquet 格式。
    - 异构数据解析: 兼容截面式数据 (Cross-section) 与长表式数据 (Long-form)。
    - 环境依赖检索: 加载并对齐 ST、行业、市值等风险因子。
    - 调仓引擎支持: 读取并验证交易日历缓存，支持实战化的信号延迟发车逻辑。

设计原则:
    - 内存友好: 在加载大规模行情数据时，通过显式指定 `keep_cols` 优化内存占用。
    - 容错性: 对异常数据文件进行静默跳过并记录告警，确保回测链路的稳定性。
"""

import os
import glob
import json
import hashlib
import pandas as pd
import numpy as np
from tqdm import tqdm

try:
    from back_test.path_utils import (
        default_data_config_path,
        load_data_paths,
        get_path_from_config,
    )
except ImportError:
    from pathlib import Path

    def default_data_config_path():
        return Path(__file__).resolve().parents[3] / "factor" / "factor_package" / "src" / "data_config.json"

    def load_data_paths(config_path=None):
        return {}

    def get_path_from_config(paths, key, fallback_relative):
        base = Path(__file__).resolve().parents[3]
        if key in paths and paths[key]:
            p = Path(paths[key])
            return str(p if p.is_absolute() else (base / p).resolve())
        return str((base / fallback_relative).resolve())


DEFAULT_DATA_CONFIG_PATH = str(default_data_config_path())


def _get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _get_factor_cache_dir():
    cache_dir = os.path.join(_get_project_root(), "mid_file", "factor_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _build_source_signature(files, factor_dir, fmt):
    parts = [os.path.abspath(factor_dir), fmt, str(len(files))]
    for path in files:
        try:
            stat = os.stat(path)
            parts.append(f"{os.path.basename(path)}|{stat.st_size}|{int(stat.st_mtime_ns)}")
        except OSError:
            parts.append(f"{os.path.basename(path)}|missing")
    digest = hashlib.sha1("::".join(parts).encode("utf-8")).hexdigest()
    return digest


def _factor_cache_paths(cache_key):
    cache_dir = _get_factor_cache_dir()
    data_path = os.path.join(cache_dir, f"{cache_key}.parquet")
    meta_path = os.path.join(cache_dir, f"{cache_key}.json")
    return data_path, meta_path


def _try_load_factor_cache(factor_dir, fmt, files):
    cache_key = _build_source_signature(files, factor_dir, fmt)
    cache_path, meta_path = _factor_cache_paths(cache_key)
    if not (os.path.exists(cache_path) and os.path.exists(meta_path)):
        return False, None, None, None, cache_path, meta_path

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("source_dir") != os.path.abspath(factor_dir):
            return False, None, None, None, cache_path, meta_path
        if meta.get("fmt") != fmt:
            return False, None, None, None, cache_path, meta_path
        if meta.get("cache_key") != cache_key:
            return False, None, None, None, cache_path, meta_path

        cached_df = pd.read_parquet(cache_path)
        factor_col = meta.get("factor_col")
        symbols_whitelist = set(cached_df["symbol"].astype(str).tolist()) if "symbol" in cached_df.columns else set()
        return True, cached_df, factor_col, symbols_whitelist, cache_path, meta_path
    except Exception as e:
        print(f"[Error] 因子缓存读取失败({cache_path}): {type(e).__name__}: {e}")
        return False, None, None, None, cache_path, meta_path


def _write_factor_cache(cache_path, meta_path, df, factor_col, factor_dir, fmt, cache_key):
    df.to_parquet(cache_path, index=False)
    meta = {
        "source_dir": os.path.abspath(factor_dir),
        "fmt": fmt,
        "cache_key": cache_key,
        "factor_col": factor_col,
        "row_count": int(len(df)),
        "symbol_count": int(df["symbol"].nunique()) if "symbol" in df.columns else 0,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_data_config(config_path=None):
    """
    读取统一数据配置 data_config.json

    参数:
        config_path (str, 可选): 配置文件路径。不传则使用默认路径。
        当配置文件不存在时返回空字典，让调用的地方使用默认路径。
    """
    if config_path is None:
        config_path = DEFAULT_DATA_CONFIG_PATH
    if not os.path.exists(config_path):
        print(f"[Info] data_config.json not found ({config_path}), 使用默认路径")
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_valid_factor_path(factor_group, factor_name, prefer_parquet=False):
    """
    因子数据自动探测器 (Factor Path Auto-Discovery)

    在标准因子存储结构中递归搜索有效的数据目录。支持从因子大类到具体因子名的多层路径解析。

    逻辑优先级:
        1. 若 `prefer_parquet=True`，优先搜索 `class_by_date_parquet` 目录。
        2. 依次搜索 CSV 格式目录: `class_by_stock_csv`, `class_by_stock`。
        3. 同时支持搜索嵌套子目录中的 Parquet 资产。

    参数:
        factor_group (str): 因子大类文件夹名 (如 '1_算术转换因子')。
        factor_name (str): 因子具体名称。
        prefer_parquet (bool): 是否强制优先使用 Parquet (推荐，因其 IO 速度远快于 CSV)。

    返回:
        tuple: (有效目录的绝对路径, 格式字符串 'csv'|'parquet')。

    异常:
        FileNotFoundError: 如果在所有预定义路径下均未找到数据。
    """
    data_paths = load_data_paths()
    factor_root = get_path_from_config(data_paths, "factor_root", "factor")

    # 优先支持直接路径 (Direct Path support for factor_package structure)
    direct_path = os.path.join(factor_root, factor_group, factor_name)
    if os.path.isdir(direct_path):
        import glob
        parquet_files = glob.glob(os.path.join(direct_path, "*.parquet"))
        if parquet_files:
            return direct_path, "parquet"

    base_path = os.path.join(factor_root, factor_group, factor_name, "output")

    if prefer_parquet:
        parquet_path = os.path.join(base_path, "class_by_date_parquet")
        if os.path.isdir(parquet_path):
            meta_file = os.path.join(parquet_path, "_meta.json")
            if os.path.exists(meta_file):
                return parquet_path, "parquet"
        
        # Check subdirectories for nested parquet (e.g. output/factor_name/class_by_date_parquet)
        nested_dirs = glob.glob(os.path.join(base_path, "*", "class_by_date_parquet"))
        for p in nested_dirs:
            meta_file = os.path.join(p, "_meta.json")
            if os.path.exists(meta_file):
                return p, "parquet"

    candidates = [
        (os.path.join(base_path, "class_by_date_parquet"), "parquet"),
        (os.path.join(base_path, "class_by_stock_csv"), "csv"),
        (os.path.join(base_path, "class_by_stock"), "csv"),
    ]
    # Add nested candidates
    nested_parquet = glob.glob(os.path.join(base_path, "*", "class_by_date_parquet"))
    for p in nested_parquet:
        candidates.append((p, "parquet"))
    for path, fmt in candidates:
        if os.path.isdir(path):
            if fmt == "parquet":
                return path, fmt
            files = glob.glob(os.path.join(path, "*.csv"))
            if files:
                return path, fmt
    raise FileNotFoundError(
        f"[Phase 1] 错误: 未找到因子数据目录 (搜索路径: {base_path})"
    )


def load_main_factor_data(factor_dir, fmt="csv"):
    """
    通用因子加载分发器 (Universal Factor Data Loader)

    根据指定的格式协议，从本地磁盘加载全量因子数据并进行初步清洗。

    处理流程:
        1. 分流: 若为 Parquet 格式，则调用 `load_factor_parquet`。
        2. 遍历: 若为 CSV 格式，则按股票代码逐个读取文件。
        3. 对齐: 统一日期列名为 'date'，并将代码注入 'symbol' 列。
        4. 聚合: 将各个标的的序列合并为一个包含全历史的长表。

    调用示例:
        >>> df, col, whitelist = load_main_factor_data("path/to/factor", fmt="parquet")

    参数:
        factor_dir (str): 因子存储根目录。
        fmt (str): 存储格式 ('csv' 或 'parquet')。

    返回:
        tuple: (full_factor_df, factor_col, symbols_whitelist)。
    """
    source_files = sorted(glob.glob(os.path.join(factor_dir, "*.parquet" if fmt == "parquet" else "*.csv")))
    if not source_files:
        raise FileNotFoundError(f"未找到因子文件: {factor_dir}")

    cache_hit, cached_df, cached_col, cached_whitelist, cache_path, meta_path = _try_load_factor_cache(
        factor_dir, fmt, source_files
    )
    # if cache_hit:
    #     print(f"[Phase 1] 因子缓存命中: {cache_path}")
    #     return cached_df, cached_col, cached_whitelist

    print(f"[Phase 1] 因子缓存未命中，开始重新加载并写入缓存: {cache_path}")

    if fmt == "parquet":
        full_factor_df, factor_col, symbols_whitelist = load_factor_parquet(factor_dir)
        _write_factor_cache(cache_path, meta_path, full_factor_df, factor_col, factor_dir, fmt, _build_source_signature(source_files, factor_dir, fmt))
        return full_factor_df, factor_col, symbols_whitelist

    files = source_files
    factor_dfs = []
    symbols_whitelist = set()
    for f in tqdm(files, desc="[Phase 1] 加载因子数据"):
        symbol = os.path.basename(f).replace(".csv", "")
        try:
            df = pd.read_csv(f)
            if df.empty:
                continue
            date_col = next(
                (c for c in df.columns if "date" in c.lower() or "日期" in c), None
            )
            if date_col is None:
                continue
            df.rename(columns={date_col: "date"}, inplace=True)
            df["date"] = pd.to_datetime(df["date"].astype(str))
            df["symbol"] = symbol
            symbols_whitelist.add(symbol)
            factor_dfs.append(df)
        except Exception as e:
            print(f"[Error] 因子CSV加载失败({f}): {type(e).__name__}: {e}")
    full_factor_df = pd.concat(factor_dfs, ignore_index=True)
    potential_cols = [c for c in full_factor_df.columns if c not in ["date", "symbol"]]
    factor_col = potential_cols[0]
    _write_factor_cache(cache_path, meta_path, full_factor_df, factor_col, factor_dir, fmt, _build_source_signature(source_files, factor_dir, fmt))
    return full_factor_df, factor_col, symbols_whitelist


def load_factor_parquet(factor_dir):
    """
    高性能 Parquet 因子加载引擎 (Optimized Parquet Loader)

    针对按日期存储的 Parquet 文件进行快速扫描。每个文件通常代表一个交易日的全市场截面。

    逻辑细节:
        - 信号自发现: 优先从 `_meta.json` 中读取 `factor_name`。若不存在，则猜测首个非索引列。
        - 自动日期注入: 从文件名 YYYYMMDD 提取日期信息，确保数据的时序准确性。
        - 并发安全: 采用顺序读取配合 `pd.concat`，在大规模数据集下保持性能优势。

    参数:
        factor_dir (str): 包含多个日期 Parquet 文件的目录。

    返回:
        tuple: (合并后的因子 DataFrame, 因子列名, 股票名单集合)。
    """
    meta_file = os.path.join(factor_dir, "_meta.json")
    if os.path.exists(meta_file):
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        factor_col = meta.get("factor_name")
    else:
        factor_col = None

    files = sorted(glob.glob(os.path.join(factor_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"未找到Parquet文件: {factor_dir}")

    all_dfs = []
    symbols_whitelist = set()

    for f in tqdm(files, desc="[Phase 1] 加载Parquet因子数据"):
        try:
            df = pd.read_parquet(f)
            if df.empty:
                continue

            date_str = os.path.basename(f).replace(".parquet", "")
            df["date"] = pd.to_datetime(date_str, format="%Y%m%d")

            if factor_col is None:
                potential_cols = [c for c in df.columns if c not in ["date", "symbol"]]
                factor_col = potential_cols[0] if potential_cols else None

            symbols_whitelist.update(df["symbol"].tolist())
            all_dfs.append(df)
        except Exception as e:
            print(f"[Error] 因子Parquet加载失败({f}): {type(e).__name__}: {e}")

    full_factor_df = pd.concat(all_dfs, ignore_index=True)
    print(
        f"[Phase 1] Parquet因子加载完成: {len(full_factor_df)} 条记录, "
        f"股票数: {len(symbols_whitelist)}, 因子: {factor_col}"
    )

    return full_factor_df, factor_col, symbols_whitelist


def load_aux_data(aux_dir, whitelist):
    """
    加载辅助数据 (如市值因子), 格式与因子数据相同

    参数:
        aux_dir: 辅助数据目录
        whitelist: 股票白名单集合, 仅加载白名单内的股票

    返回:
        DataFrame 或 None (目录不存在时)
    """
    if not os.path.isdir(aux_dir):
        return None
    meta_file = os.path.join(aux_dir, "_meta.json")
    parquet_files = glob.glob(os.path.join(aux_dir, "*.parquet"))
    if os.path.exists(meta_file) or parquet_files:
        return load_aux_parquet(aux_dir, whitelist)

    files = glob.glob(os.path.join(aux_dir, "*.csv"))
    aux_dfs = []
    for f in files:
        symbol = os.path.basename(f).replace(".csv", "")
        if symbol not in whitelist:
            continue
        try:
            df = pd.read_csv(f)
            for c in df.columns:
                if "date" in c.lower() or "日期" in c:
                    df.rename(columns={c: "date"}, inplace=True)
                    break
            df["date"] = pd.to_datetime(df["date"].astype(str))
            df["symbol"] = symbol
            aux_dfs.append(df)
        except Exception as e:
            print(f"[Error] 辅助因子CSV加载失败({f}): {type(e).__name__}: {e}")
    return pd.concat(aux_dfs, ignore_index=True) if aux_dfs else None


def load_aux_parquet(aux_dir, whitelist):
    """
    加载按日期存储的 Parquet 辅助数据 (如 log_mv_processed_parquet)

    参数:
        aux_dir: Parquet 目录
        whitelist: 股票白名单集合, 仅加载白名单内的股票
    """
    meta_file = os.path.join(aux_dir, "_meta.json")
    value_col = None
    if os.path.exists(meta_file):
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            value_col = meta.get("factor_name") or meta.get("value_col")
        except Exception as e:
            print(f"[Error] 辅助因子元数据读取失败({meta_file}): {type(e).__name__}: {e}")

    files = sorted(glob.glob(os.path.join(aux_dir, "*.parquet")))
    if not files:
        return None

    wl = set(whitelist) if whitelist is not None else None
    aux_dfs = []
    for f in tqdm(files, desc="[Phase 1b] 加载辅助因子(Parquet)"):
        try:
            df = pd.read_parquet(f)
            if df.empty:
                continue

            if "symbol" not in df.columns:
                sym_col = next(
                    (c for c in df.columns if "code" in c.lower() or "symbol" in c.lower()),
                    None,
                )
                if sym_col:
                    df = df.rename(columns={sym_col: "symbol"})

            if "symbol" not in df.columns:
                continue

            if wl is not None:
                df = df[df["symbol"].isin(wl)]
                if df.empty:
                    continue

            date_str = os.path.basename(f).replace(".parquet", "")
            date_val = pd.to_datetime(date_str, format="%Y%m%d", errors="coerce")
            if pd.isna(date_val):
                continue
            df["date"] = date_val

            if value_col is None or value_col not in df.columns:
                potential_cols = [c for c in df.columns if c not in ["date", "symbol"]]
                if potential_cols:
                    value_col = potential_cols[0]

            if value_col and value_col in df.columns:
                df = df[["date", "symbol", value_col]]

            aux_dfs.append(df)
        except Exception as e:
            print(f"[Error] 辅助因子Parquet加载失败({f}): {type(e).__name__}: {e}")

    return pd.concat(aux_dfs, ignore_index=True) if aux_dfs else None


def load_daily_prices_cross_section(data_dir, start_date=None, end_date=None):
    """
    截面日行情加载器 (Cross-sectional Market Data Loader)

    从全市场截面目录中加载日线级别的量价资产。这是回测引擎中计算收益的基础。

    核心优化:
        1. 日期预筛选: 仅加载处于 [start_date, end_date] 时间窗内的 Pickle 文件，大幅削减 IO 负载。
        2. 列名投影: 强制仅保留如 'Wind代码'、'VWAP'、'涨停价' 等 14 个核心列，防止无关数据挤占内存。
        3. 内存汇报: 实时输出加载后的 memory_usage，便于在资源受限环境下排查性能瓶颈。

    调用示例:
        >>> df_market = load_daily_prices_cross_section(
        ...     "path/to/prices", start_date="20200101", end_date="20231231"
        ... )

    参数:
        data_dir (str): 存放按日 pickle 文件的目录。
        start_date/end_date (str, 可选): 过滤区间，格式为 'YYYYMMDD'。

    返回:
        pd.DataFrame: 合并后的极简行情长表。
    """
    files = sorted(glob.glob(os.path.join(data_dir, "*.pickle")))

    # 按日期过滤文件, 避免加载无用数据
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

    print(f"[Phase 2] 将加载 {len(files)} 个截面日行情文件...")

    # 只保留必要列, 减少内存
    keep_cols = [
        "Wind代码",
        "交易日期",
        "昨收盘价(元)",
        "开盘价(元)",
        "最高价(元)",
        "最低价(元)",
        "收盘价(元)",
        "涨跌幅(%)",
        "成交量(手)",
        "均价(VWAP)",
        "交易状态",
        "涨停价(元)",
        "跌停价(元)",
        "复权收盘价(元)",
    ]

    all_dfs = []
    for f in tqdm(files, desc="[Phase 2] 加载截面日行情"):
        try:
            df = pd.read_pickle(f)
            # 只保留存在的列
            cols_exist = [c for c in keep_cols if c in df.columns]
            df = df[cols_exist]
            all_dfs.append(df)
        except Exception as e:
            print(f"[Error] 日行情文件加载失败({f}): {type(e).__name__}: {e}")

    if not all_dfs:
        raise RuntimeError("未能加载任何日行情数据!")

    result = pd.concat(all_dfs, ignore_index=True)
    print(
        f"[Phase 2] 日行情加载完成: {len(result)} 条记录, "
        f"内存: {result.memory_usage(deep=True).sum() / 1024**2:.1f} MB"
    )
    return result


def load_st_data(st_path=None):
    """
    加载本地 ST 标记数据缓存 (ST Status Loader)

    逻辑说明:
        1. 默认路径指向 E 盘量化研究中心标准数据目录。
        2. 读取 Pickle 格式的 ST 状态表，包含股票代码及 ST/退市预警的起止日期。
        3. 该数据用于在回测各阶段剔除具有非正常交易风险的标的。

    参数:
        st_path (str, 可选): ST 数据文件的绝对路径。

    返回:
        pd.DataFrame 或 None: 加载成功的状态表。
    """
    if st_path is None:
        data_paths = load_data_paths()
        st_base = get_path_from_config(data_paths, "AShareST", "data/中国A股特别处理_AShareST")
        st_path = os.path.join(st_base, "ST.pickle")
    else:
        if os.path.isdir(st_path):
            st_path = os.path.join(st_path, "ST.pickle")
    if os.path.exists(st_path):
        try:
            return pd.read_pickle(st_path)
        except Exception as e:
            print(f"[Warning] 加载 ST 数据失败: {e}")
    return None

def load_industry_data(ind_path=None):
    """
    加载申万行业分类 2021 版 (Industry Classification Loader)

    获取 A 股市场的行业标签资产，这是执行“行业中性化”的前提。

    参数:
        ind_path (str, 可选): 行业分类 Pickle 文件的绝对路径。

    返回:
        pd.DataFrame: 包含股票各时期行业分类映射的数据表。
    """
    if ind_path is None:
        data_paths = load_data_paths()
        ind_base = get_path_from_config(
            data_paths,
            "SWIndustryClass2021",
            "data/申万行业分类2021版_AShareSWNIndustriesClass",
        )
        ind_path = os.path.join(ind_base, "申万行业分类2021版.pickle")
    else:
        if os.path.isdir(ind_path):
            ind_path = os.path.join(ind_path, "申万行业分类2021版.pickle")
    if os.path.exists(ind_path):
        try:
            return pd.read_pickle(ind_path)
        except Exception as e:
            print(f"[Warning] 加载行业分类数据失败: {e}")
    return None

def load_calendar(df_daily, delay_days=0, rebalance_month_start=False, logger=None):
    """
    生成严格的调仓日历 (Rebalance Calendar Engine)

    根据官方交易日历和用户的信号延迟配置，计算每个月的信号发出日、买入日及卖出日。

    关键逻辑:
        1. 缓存依赖: 必须先运行 `generate_calendar_cache.py` 生成 Parquet 缓存，以保证回测的日期全局一致。
          2. 信号延迟 (`delay_days`): 支持实战仿真的“发车延迟”。若 `delay_days > 0`，
              所有的买入和卖出执行日将相对于标准计划日顺延对应个交易日。
          3. 月初调仓模式 (`rebalance_month_start`): 若开启，则当期信号对应“次月初买入、再下月初卖出”，
              让卖旧与买新在月初同一调仓点发生。
          4. 状态闭环: 确保回测区间完全被缓存覆盖，否则抛出异常。

    参数:
        df_daily (pd.DataFrame): 基础行情数据。
        delay_days (int): 交易延迟天数。
        rebalance_month_start (bool): 是否采用月初调仓口径。
        logger (logging.Logger): 日志对象。

    返回:
        tuple: (调仓日历 DataFrame, 全量交易日序列)。
    """
    data_paths = load_data_paths()
    calendar_dir = get_path_from_config(data_paths, "TradeCalendar", "data/交易日历")
    calendar_cache_path = os.path.join(calendar_dir, "rebalance_calendar_cache.parquet")

    data_min = pd.to_datetime(df_daily["date"].min()) - pd.Timedelta(days=30)
    data_max = pd.to_datetime(df_daily["date"].max()) + pd.Timedelta(days=30)

    if not os.path.exists(calendar_cache_path):
        msg = f"交易日历缓存文件不存在，请先运行 generate_calendar_cache.py: {calendar_cache_path}"
        if logger: logger.error(msg)
        raise FileNotFoundError(msg)

    try:
        cache = pd.read_parquet(calendar_cache_path)
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
            max_i = len(yms) - (2 if rebalance_month_start else 1)
            if max_i <= 0:
                raise ValueError("交易日历月份数量不足，无法构建调仓序列")

            for i in range(max_i):
                cur_ym = yms[i]
                next_ym = yms[i + 1]
                
                base_buy = first_days[next_ym]
                if rebalance_month_start:
                    base_sell = first_days[yms[i + 2]]
                else:
                    base_sell = last_days[next_ym]
                
                # Apply delay_days
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
            if logger:
                logger.info(
                    f"已从本地读取调仓日历: {len(df_calendar)} 期 "
                    f"(本地路径: {calendar_cache_path})"
                )
            return df_calendar, trade_dates
        else:
            msg = (f"本地区间不足。缓存区间: {cmin.date()}~{cmax.date()}，"
                   f"需求区间: {data_min.date()}~{data_max.date()}。"
                   f"请重新运行 generate_calendar_cache.py 扩充区间。")
            if logger: logger.error(msg)
            raise ValueError("Calendar cache date range is insufficient for the underlying data.")
    except Exception as e:
        if logger: logger.error(f"读取交易日历缓存失败: {e}")
        raise

def load_benchmark_component(index_daily_dir, needed_dates, logger=None):
    """
    加载基准指数成分股及行情 (Benchmark Component Loader)

    从指定的本地 Pickle 目录中提取回测所需日期的指数成分信息。

    参数:
        index_daily_dir (str): 基准描述文件存放目录。
        needed_dates (list): 需要加载的日期集合。

    返回:
        dict: 以日期为 Key，对应成分股 DataFrame 为 Value 的映射字典。
    """
    bench_prices = {}
    if index_daily_dir is None or not os.path.isdir(index_daily_dir):
        msg = f"未提供有效基准指数目录: {index_daily_dir}"
        if logger:
            logger.error(msg)
        raise FileNotFoundError(msg)

    for d in sorted(needed_dates):
        date_str = d.strftime("%Y%m%d")
        fpath = os.path.join(index_daily_dir, f"{date_str}.pickle")
        if not os.path.exists(fpath):
            continue
        try:
            df_idx = pd.read_pickle(fpath)
            bench_prices[d] = df_idx
        except Exception as e:
            if logger:
                logger.warning(
                    f"读取基准日频成分股失败: {fpath} ({type(e).__name__}: {e})"
                )
    return bench_prices
