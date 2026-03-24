"""
单因子回测 CLI 驱动程序 (Single Factor Backtest CLI Runner)

本脚本是整个回测系统的主要入口点。它集成了命令行接口 (CLI)、数据加载流以及回测引擎的初始化逻辑。
用户可以通过传递不同的参数来快速启动针对特定因子的多维度测试。

主要阶段 (Phases):
    Phase 1: 扫描磁盘并加载目标因子数据 (支持 CSV/Parquet)。
    Phase 2: 加载高频截面日行情、ST 数据及行业分类数据。
    Phase 3: 初始化 `SingleFactorBacktesterV2` 引擎并执行回测。
    Phase 4: 生成可视化 HTML 研报并持久化结果。

使用方法:
    python run_test_v2.py --factor-group "1_算术转换因子" --factor-name "log_mv_1" --start "2010-01" --enable-ind-neu
"""

import argparse
import os
import sys

# 稳健调用：将项目根目录加入 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
from back_test.sigle_factor_test.src.single_factor_bt_v2 import SingleFactorBacktesterV2
from back_test.sigle_factor_test.src.data_loader import (
    get_valid_factor_path,
    load_main_factor_data,
    load_aux_data,
    load_data_config,
    load_daily_prices_cross_section,
    load_st_data,
    load_industry_data,
)


def parse_args():
    """
    解析命令行输入参数 (CLI Argument Parser)

    定义了回测任务的所有可配置项，包括路径配置、算法参数、风控开关等。
    """
    parser = argparse.ArgumentParser(
        description="单因子回测驱动脚本 V2.5 (优化版)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--factor-group",
        type=str,
        default="3_回归类因子_RegressionFactors",
        help="因子大类目录名",
    )
    parser.add_argument(
        "--factor-name", type=str, default="1_minus_r2_30", help="因子名"
    )
    parser.add_argument(
        "--start", type=str, default="2004-01", help="回测起始月份 (YYYY-MM), 默认2004-01"
    )
    parser.add_argument(
        "--end", type=str, default="2024-12", help="回测结束月份 (YYYY-MM), 默认2024-12, 2025年预留测试集"
    )
    parser.add_argument(
        "--benchmark", type=str, default="000905.SH", help="基准指数代码"
    )
    parser.add_argument(
        "--disable-st-filter",
        action="store_true",
        default=False,
        help="关闭 ST 股票过滤 (默认开启)",
    )
    parser.add_argument(
        "--enable-st-filter",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--st-path", type=str, default=None, help="自定义 ST 数据目录或文件路径"
    )
    parser.add_argument(
        "--enable-ind-neutral",
        action="store_true",
        default=False,
        help="启用行业中性化 (默认关闭)",
    )
    parser.add_argument(
        "--ind-path", type=str, default=None, help="自定义行业分类数据路径"
    )
    parser.add_argument("--commission", type=float, default=0.0015, help="单边手续费率")
    parser.add_argument(
        "--group-size",
        type=int,
        default=0,
        help="每组股票数量 (默认0表示按10组均分, >0则按指定数量分组)",
    )
    parser.add_argument(
        "--use-parquet",
        action="store_true",
        default=False,
        help="使用Parquet格式因子数据 (按日期存储，加载更快)",
    )
    parser.add_argument(
        "--delay-days",
        type=int,
        default=0,
        help="信号建仓延迟天数 (默认: 0, 设为 1 即次日建仓, 2 即 T+2 漂移)",
    )
    parser.add_argument(
        "--factor-path",
        type=str,
        default=None,
        help="手动指定因子数据目录 (跳过自动探测)",
    )
    parser.add_argument(
        "--data-config",
        type=str,
        default=None,
        help="data_config.json 路径 (默认使用量化因子配置)",
    )
    return parser.parse_args()


def main():
    """
    回测任务执行主逻辑 (Task Orchestration)

    流程逻辑:
        1. 参数验证: 检查输入月份的合法性。
        2. 数据发现: 调用 `data_loader` 寻找因子所在的物理路径。
        3. 状态并行化: 根据配置并行加载行情资产、ST 资产和行业资产。
        4. 引擎耦合: 将内存中的数据资产注入回测引擎并启动 `run_backtest`。
    """
    args = parse_args()

    factor_group = args.factor_group
    factor_name = args.factor_name
    data_config = load_data_config(args.data_config)
    data_paths = data_config.get("paths", {})

    # --- Phase 1: 加载因子数据 ---
    if args.factor_path:
        factor_dir = args.factor_path
        fmt = "parquet" if any(f.endswith('.parquet') for f in os.listdir(factor_dir)) else "csv"
        print(f"[Phase 1] 手动指定因子目录: {factor_dir} (格式: {fmt})")
    else:
        factor_dir, fmt = get_valid_factor_path(
            factor_group, factor_name, prefer_parquet=args.use_parquet
        )
        print(f"[Phase 1] 因子数据自动探测: {fmt}, 目录: {factor_dir}")

    factor_df, factor_col, v_whitelist = load_main_factor_data(factor_dir, fmt)
    factor_df.rename(columns={factor_col: factor_name}, inplace=True)

    # --- Phase 1b: 加载市值数据 ---
    mv_dir = data_paths.get("log_mv_processed") or data_paths.get("log_mv")
    if not mv_dir:
        mv_dir = r"E:\1_basement\quant_research\factor\1_算术转换因子_ArithmeticFactors\log_mv_1\output\class_by_stock"
    mv_df = load_aux_data(mv_dir, v_whitelist)

    # --- Phase 2: 加载截面日行情 ---
    daily_dir = data_paths.get(
        "AShareEODPrices",
        r"E:\1_basement\quant_research\data\中国A股日行情_AShareEODPrices",
    )

    # 将月份转换为日期范围, 提前20天加载 (流动性 rolling 需要前置窗口)
    start_dt = pd.to_datetime(args.start) - pd.Timedelta(days=60)
    end_dt = pd.to_datetime(args.end) + pd.offsets.MonthEnd(2)

    df_daily = load_daily_prices_cross_section(
        daily_dir,
        start_date=start_dt.strftime("%Y%m%d"),
        end_date=end_dt.strftime("%Y%m%d"),
    )

    # --- Phase 2b: 加载 ST 与行业数据 (可选) ---
    st_enabled = not args.disable_st_filter
    if args.enable_st_filter:
        st_enabled = True
    st_df = load_st_data(args.st_path) if st_enabled else None
    if not st_enabled:
        print("[Phase 2b] ST 过滤已跳过 (使用 --disable-st-filter 关闭)")
    else:
        print(f"[Phase 2b] 已加载 ST 数据 (提供行数: {len(st_df) if st_df is not None else 0})")

    ind_df = load_industry_data(args.ind_path) if args.enable_ind_neutral else None
    if not args.enable_ind_neutral:
        print("[Phase 2b] 行业中性化已跳过 (使用 --enable-ind-neutral 开启)")
    else:
        print(f"[Phase 2b] 已加载行业分类数据 (提供行数: {len(ind_df) if ind_df is not None else 0})")

    # --- Phase 2c: 指数数据目录 ---
    index_daily_dir = data_paths.get(
        "AIndexEODPrices",
        r"E:\1_basement\quant_research\data\中国A股指数日行情_AIndexEODPrices",
    )

    # 确定输出目录
    out_dir = os.path.join(project_root, "back_test", "sigle_factor_test", "output", factor_name)
    os.makedirs(out_dir, exist_ok=True)

    # --- Phase 3: 初始化引擎并执行回测 ---
    tester = SingleFactorBacktesterV2(
        df_daily_price=df_daily,
        out_dir=out_dir,
        st_df=st_df,
        ind_df=ind_df,
        index_daily_dir=index_daily_dir,
        delay_days=args.delay_days,
    )
    tester.run_backtest(
        factor_name=factor_name,
        factor_df=factor_df,
        mv_df=mv_df,
        commission=args.commission,
        start_month=args.start,
        end_month=args.end,
        benchmark_symbol=args.benchmark,
        group_size=args.group_size,
    )


if __name__ == "__main__":
    main()
