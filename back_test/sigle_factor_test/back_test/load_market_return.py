"""
市场收益基准数据加载器 (Market Revenue Benchmark Loader)
用途: 强制加载CSI 985指数月度收益率
"""
import os
import pandas as pd
import argparse
import sys


def load_market_revenue_benchmark(benchmark_path=None):
    """
    强制加载市场收益基准数据（CSI 985指数）
    
    参数:
        benchmark_path (str): 基准数据文件路径 (XLSX格式)。
                            若为None，默认使用硬编码路径。
    
    返回:
        pd.DataFrame: 包含columns=['date', 'monthly_return'] 的月度收益率数据框。
    
    异常:
        FileNotFoundError: 文件不存在
        ValueError: 数据格式不正确
    """
    # 默认路径
    if benchmark_path is None:
        benchmark_path = r'E:\1_basement\quant_research\data\市场收益_MarketRevenue\000985.CSI.xlsx'
    
    # 验证文件存在
    if not os.path.exists(benchmark_path):
        raise FileNotFoundError(
            f"[Benchmark Error] 基准数据文件不存在: {benchmark_path}\n"
            f"请确保文件位置正确或通过CLI参数 --benchmark 指定。"
        )
    
    try:
        # 读取Excel文件
        df = pd.read_excel(benchmark_path)
        print(f"[Benchmark Loader] 成功加载: {benchmark_path}")
        print(f"[Benchmark Loader] 数据形状: {df.shape}, 列名: {list(df.columns)}")
        
        # 验证必要列
        required_cols = ['date', 'monthly_return']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(
                f"[Benchmark Error] 缺少必要列。期望: {required_cols}, 实际: {list(df.columns)}"
            )
        
        # 数据类型转换
        df['date'] = pd.to_datetime(df['date'])
        df['monthly_return'] = pd.to_numeric(df['monthly_return'], errors='coerce')
        
        # 排序
        df = df.sort_values('date').reset_index(drop=True)
        
        print(f"[Benchmark Loader] 数据范围: {df['date'].min()} 至 {df['date'].max()}")
        
        return df
        
    except pd.errors.ParserError as e:
        raise ValueError(f"[Benchmark Error] Excel文件解析失败: {e}")
    except Exception as e:
        raise ValueError(f"[Benchmark Error] 加载失败: {e}")


def main():
    """
    CLI入口：支持指定自定义基准数据路径
    """
    parser = argparse.ArgumentParser(
        description="市场收益基准数据加载器 (CSI 985指数)",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        '--benchmark',
        type=str,
        default=None,
        help=(
            '基准数据文件路径 (XLSX格式)\n'
            '  默认: E:\\1_basement\\quant_research\\data\\市场收益_MarketRevenue\\000985.CSI.xlsx\n'
            '  示例: python script.py --benchmark "E:\\path\\to\\benchmark.xlsx"'
        )
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='输出文件路径 (可选，用于保存处理后的Parquet或CSV)'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        choices=['parquet', 'csv'],
        default='parquet',
        help='输出文件格式 (默认: parquet)'
    )
    
    args = parser.parse_args()
    
    try:
        # 加载基准数据
        df_benchmark = load_market_revenue_benchmark(benchmark_path=args.benchmark)
        
        print("\n[Benchmark Loader] 数据预览:")
        print(df_benchmark.head())
        print(f"\n[Benchmark Loader] 总行数: {len(df_benchmark)}")
        
        # 若指定输出路径，保存数据
        if args.output:
            os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
            if args.format == 'parquet':
                df_benchmark.to_parquet(args.output, index=False)
            else:
                df_benchmark.to_csv(args.output, index=False)
            print(f"\n[Benchmark Loader] 已保存到: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
