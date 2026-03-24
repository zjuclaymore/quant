import json
import os
import pandas as pd
from factor_processor import FactorProcessor
  """
    批量处理原始股息率数据，生成标准化、去极值、去行业和市值暴露后的因子数据。
    处理流程：
    1. 读取配置文件 data_config.json 中的路径与参数
    2. 遍历原始数据目录下的所有 parquet 文件（按文件名排序）
    3. 对每个交易日的数据：
       - 格式化股票代码为 Wind 标准（6位补零 + .SH/.SZ 后缀）
       - 提取股息率因子列
       - 使用 FactorProcessor 进行以下处理：
         - 缺失值剔除（不填充）
         - 3倍MAD去极值（因子值与对数市值）
         - 行业与对数市值中性化
         - Z-Score标准化
       - 保存处理后的因子数据为 parquet 文件，仅保留 symbol 和 dividend_yield 列
       - 记录每日处理股票数量、均值、标准差至汇总日志
    4. 输出元信息 _meta.json 与处理摘要 processing_summary.json
    目录结构：
    - 输入：配置中的 dividend_yield_raw 路径，存放每日原始因子 parquet 文件
    - 输出：dividend_yield_processed 路径，生成：
        - {trade_date}.parquet: 处理后因子数据
        - _meta.json: 因子元信息
        - processing_summary.json: 处理过程统计
    异常处理：
    - 跳过不含 'symbol' 列的文件
    - 捕获单日处理异常，不影响整体流程
    - 打印详细处理日志
    依赖：
    - factor_processor.FactorProcessor: 提供标准化处理逻辑
    - pandas, pyarrow: parquet 读写支持
    """
def batch_process():
    config_path = 'data_config.json'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    raw_dir = config['paths']['dividend_yield_raw']
    output_dir = config['paths']['dividend_yield_processed']
    os.makedirs(output_dir, exist_ok=True)
    
    # Save _meta.json
    meta = {
        'factor_name': 'dividend_yield',
        'description': 'Processed dividend yield factor (Standardized, Winsorized, Neutralized)',
        'processing_steps': [
            'Drop missing values (fill_na=False)',
            '3MAD Winsorization (Factor & Log Market Value)',
            'Neutralization (Industry + Log Market Value)',
            'Z-Score Standardization'
        ],
        'columns': ['symbol', 'dividend_yield'],
        'format': 'parquet'
    }
    with open(os.path.join(output_dir, '_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    
    processor = FactorProcessor(config_path)
    
    files = [f for f in os.listdir(raw_dir) if f.endswith('.parquet')]
    files.sort()
    
    print(f'Found {len(files)} files to process.')
    
    summary = []
    
    for filename in files:
        trade_date = filename.split('.')[0]
        file_path = os.path.join(raw_dir, filename)
        
        try:
            df = pd.read_parquet(file_path)
            
            if 'symbol' not in df.columns:
                print(f'[{trade_date}] symbol column not found.')
                continue
            
            def format_symbol(s):
                s = str(s).zfill(6)
                if '.' in s: return s
                if s.startswith('6'): return s + '.SH'
                else: return s + '.SZ'
            
            df['symbol'] = df['symbol'].apply(format_symbol)
            
            # Identify factor column
            factor_cols = [c for c in df.columns if c != 'symbol']
            if not factor_cols: continue
            
            factor_name = factor_cols[0]
            factor_series = df.set_index('symbol')[factor_name]
            
            # Process (fill_na=False)
            processed_series = processor.process_day(factor_series, trade_date, fill_na=False)
            
            if not processed_series.empty:
                # Save as Parquet
                output_path = os.path.join(output_dir, f'{trade_date}.parquet')
                processed_df = processed_series.reset_index()
                processed_df.columns = ['symbol', 'dividend_yield']
                processed_df.to_parquet(output_path, index=False)
                
                mean = processed_series.mean()
                std = processed_series.std()
                print(f'[{trade_date}] Processed {len(processed_series)} stocks. Mean={mean:.4f}, Std={std:.4f}')
                summary.append({'date': trade_date, 'count': len(processed_series), 'mean': mean, 'std': std})
                
        except Exception as e:
            print(f'[{trade_date}] Error: {e}')

    # Save summary
    with open(os.path.join(output_dir, 'processing_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

if __name__ == '__main__':
    batch_process()
