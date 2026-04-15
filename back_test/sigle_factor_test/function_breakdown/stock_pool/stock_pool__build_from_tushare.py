#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
╔═══════════════════════════════════════════════════════════════════╗
║  股票池构建器 (Stock Pool Builder from Tushare)                   ║
║  阶段: p00-数据准备 / 功能: 从tushare构建截面股票池               ║
╚═══════════════════════════════════════════════════════════════════╝

功能:
  从tushare获取所有A股股票代码，与交易日历合并
  生成截面数据：每个交易日 × 所有有效股票代码
  数据包含：交易日期、股票代码、股票名称
  **默认从市场最早数据时间开始爬取**

CLI 用法:
  python p00_stock_pool__build_from_tushare.py \
    [--start-date <YYYYMMDD|earliest>] # 开始日期，默认 earliest (从市场最早)
    [--end-date <YYYYMMDD>]         # 结束日期，默认 今天
    [--output <path>]               # 输出路径，默认本文件夹下的 stock_pool.parquet
    [--exchange {SSE|SZSE|all}]     # 交易所，默认 all（所有A股）
    [--remove-st {yes|no}]          # 是否移除ST股，默认 no
    [--preview-rows <int>]          # 预览行数，默认 10

数据格式:
  截面结构：(date, code) 为唯一键
  - date: 交易日期 (YYYYMMDD 格式)
  - code: 股票代码 (例如 000001)
  - name: 股票名称 (例如 平安银行)

示例:
  # 全量数据：从市场最早时间至今 (推荐)
  python p00_stock_pool__build_from_tushare.py

  # 指定具体日期范围
  python p00_stock_pool__build_from_tushare.py \
    --start-date 20200101 \
    --end-date 20231231

  # 排除ST股
  python p00_stock_pool__build_from_tushare.py --remove-st yes

  # 自定义输出路径
  python p00_stock_pool__build_from_tushare.py \
    --output "E:\\my_stock_pool.parquet"

输出文件: stock_pool.parquet
  形状: (总交易日数 × 股票数, 3列)
  全量数据通常约 2000+ 万行 (1990年左右至今)
"""

import pandas as pd
import numpy as np
import tushare as ts
import argparse
import os
from datetime import datetime
from pathlib import Path
import logging

# 配置Tushare Token
TUSHARE_TOKEN = '2d4f555869182905bfd48ce1fd0f649015f2bf10b3ef4a7e558573bb'
ts.set_token(TUSHARE_TOKEN)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_trade_dates(start_date, end_date):
    """
    获取交易日历
    
    参数:
        start_date (str): 开始日期 YYYYMMDD，如果为 'earliest' 则自动找最早日期
        end_date (str): 结束日期 YYYYMMDD
    返回:
        list: 交易日期列表
    """
    pro = ts.pro_api()  # 需要本地配置tushare token
    
    # 如果指定了 'earliest'，则找最早的交易日期
    if start_date.lower() == 'earliest':
        logger.info("正在查找市场最早交易日期...")
        try:
            # 从1990年开始查询（中国A股开放之初）
            earliest_cal = pro.trade_cal(
                exchange='SSE',
                start_date='19900101',
                end_date=end_date,
                is_open=1
            )
            if len(earliest_cal) > 0:
                start_date = earliest_cal['cal_date'].iloc[0]
                logger.info(f"找到最早交易日: {start_date}")
            else:
                start_date = '19900101'
        except Exception as e:
            logger.warning(f"查询最早交易日失败 ({e})，使用默认日期 19900101")
            start_date = '19900101'
    
    logger.info(f"正在获取交易日历 {start_date} - {end_date}")
    
    try:
        # 获取交易日历
        calendar = pro.trade_cal(
            exchange='SSE',
            start_date=start_date,
            end_date=end_date,
            is_open=1  # 只获取交易日
        )
        
        trade_dates = calendar['cal_date'].tolist()
        logger.info(f"获取到 {len(trade_dates)} 个交易日")
        return sorted(trade_dates)
    
    except Exception as e:
        logger.error(f"获取交易日历失败: {e}")
        raise


def get_all_stocks():
    """
    获取所有A股股票列表
    
    返回:
        pd.DataFrame: 包含 ts_code, symbol, name 的数据框
    """
    logger.info("正在获取A股股票列表...")
    
    pro = ts.pro_api()
    
    try:
        # 获取所有上市公司基本信息
        stocks = pro.stock_basic(
            exchange='',  # 空表示全部交易所
            list_status='L',  # L:上市 D:退市 P:暂停上市
            fields='ts_code,symbol,name,area,industry'
        )
        
        # 提取标准代码 (去掉后缀，例如 000001.SZ -> 000001)
        stocks['code'] = stocks['ts_code'].str.replace('.SZ', '').str.replace('.SH', '')
        
        logger.info(f"获取到 {len(stocks)} 只A股")
        return stocks[['code', 'name']]
    
    except Exception as e:
        logger.error(f"获取股票列表失败: {e}")
        raise


def build_stock_pool(trade_dates, stocks_df, remove_st=False):
    """
    构建截面股票池数据
    
    参数:
        trade_dates (list): 交易日期列表 [YYYYMMDD, ...]
        stocks_df (pd.DataFrame): 股票信息 (code, name)
        remove_st (bool): 是否移除ST股
    返回:
        pd.DataFrame: 截面股票池 (date, code, name)
    """
    logger.info(f"正在构建股票池 ({len(trade_dates)} 个交易日 × {len(stocks_df)} 只股票)...")
    
    if remove_st:
        stocks_df = stocks_df[~stocks_df['name'].str.contains('ST', na=False)]
        logger.info(f"移除ST股后剩余 {len(stocks_df)} 只股票")
    
    # 创建截面数据：每个交易日 × 所有股票
    dates_repeated = []
    codes_repeated = []
    names_repeated = []
    
    for date in trade_dates:
        dates_repeated.extend([date] * len(stocks_df))
        codes_repeated.extend(stocks_df['code'].tolist())
        names_repeated.extend(stocks_df['name'].tolist())
    
    stock_pool = pd.DataFrame({
        'date': dates_repeated,
        'code': codes_repeated,
        'name': names_repeated
    })
    
    logger.info(f"股票池数据形状: {stock_pool.shape}")
    logger.info(f"\n数据预览:\n{stock_pool.head(10)}")
    
    return stock_pool


def save_stock_pool(stock_pool, output_path):
    """
    保存股票池数据为parquet格式
    
    参数:
        stock_pool (pd.DataFrame): 股票池数据
        output_path (str): 输出文件路径
    """
    logger.info(f"正在保存数据到 {output_path}...")
    
    # 创建目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存为parquet格式
    stock_pool.to_parquet(output_path, compression='snappy', index=False)
    
    # 获取文件大小
    file_size_mb = os.path.getsize(output_path) / (1024 ** 2)
    logger.info(f"✓ 保存成功! 文件大小: {file_size_mb:.2f} MB")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="从tushare构建A股股票池截面数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基础用法
  python p00_stock_pool__build_from_tushare.py

  # 指定日期范围
  python p00_stock_pool__build_from_tushare.py --start-date 20200101 --end-date 20231231

  # 排除ST股
  python p00_stock_pool__build_from_tushare.py --remove-st yes
        """
    )
    
    parser.add_argument('--start-date', type=str, default='earliest',
                       help='开始日期 YYYYMMDD，或 "earliest" 自动从市场最早日期开始 (默认: earliest)')
    parser.add_argument('--end-date', type=str, 
                       default=datetime.now().strftime('%Y%m%d'),
                       help='结束日期 YYYYMMDD (默认: 今天)')
    parser.add_argument('--output', type=str, 
                       default=os.path.join(os.path.dirname(__file__), 'stock_pool.parquet'),
                       help='输出文件路径')
    parser.add_argument('--exchange', type=str, choices=['SSE', 'SZSE', 'all'], 
                       default='all', help='交易所选择')
    parser.add_argument('--remove-st', type=str, choices=['yes', 'no'], 
                       default='no', help='是否移除ST股')
    parser.add_argument('--preview-rows', type=int, default=10,
                       help='预览行数')
    
    args = parser.parse_args()
    
    # 参数验证
    start_date = args.start_date
    end_date = args.end_date
    remove_st = args.remove_st == 'yes'
    
    # 检查日期格式，除非 start_date 是 'earliest'
    if start_date.lower() != 'earliest':
        if len(start_date) != 8:
            logger.error("日期格式错误，应为 YYYYMMDD 或 'earliest'")
            return
    
    if len(end_date) != 8:
        logger.error("日期格式错误，应为 YYYYMMDD")
        return
    
    if start_date.lower() != 'earliest' and start_date > end_date:
        logger.error(f"起始日期 {start_date} 大于结束日期 {end_date}")
        return
    
    logger.info(f"参数: start_date={start_date}, end_date={end_date}, remove_st={remove_st}")
    logger.info(f"输出路径: {args.output}")
    
    try:
        # 步骤1: 获取交易日历
        trade_dates = get_trade_dates(start_date, end_date)
        
        # 步骤2: 获取股票列表
        stocks_df = get_all_stocks()
        
        # 步骤3: 构建截面股票池
        stock_pool = build_stock_pool(trade_dates, stocks_df, remove_st=remove_st)
        
        # 步骤4: 保存数据
        output_path = save_stock_pool(stock_pool, args.output)
        
        logger.info("=" * 60)
        logger.info("✓ 股票池构建完成!")
        logger.info("=" * 60)
        logger.info(f"文件位置: {output_path}")
        logger.info(f"数据统计:")
        logger.info(f"  - 交易日数: {len(trade_dates)}")
        logger.info(f"  - 股票数: {len(stocks_df)}")
        logger.info(f"  - 总记录数: {len(stock_pool):,}")
        
    except Exception as e:
        logger.error(f"构建过程出错: {e}")
        raise


if __name__ == '__main__':
    main()
