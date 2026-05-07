#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于截面的量价状态过滤：剔除次新股、停牌股、当日涨跌停封死股，产生 allow_flag
"""
import pandas as pd
import numpy as np
import tushare as ts
import argparse
import os
from pathlib import Path
import logging

# 配置Tushare Token
TUSHARE_TOKEN = os.getenv('TUSHARE_TOKEN')
if not TUSHARE_TOKEN:
    raise RuntimeError('Missing TUSHARE_TOKEN environment variable.')
ts.set_token(TUSHARE_TOKEN)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_all_stocks_list_date():
    pro = ts.pro_api()
    # 尽可能包括全维度生命周期的股票
    stocks = pro.stock_basic(
        exchange='',
        list_status='L,D,P',
        fields='ts_code,list_date'
    )
    stocks['code'] = stocks['ts_code'].str.replace('.SZ', '').str.replace('.SH', '').str.replace('.BJ', '')
    stocks['list_date'] = pd.to_datetime(stocks['list_date'], format='%Y%m%d', errors='coerce')
    return stocks[['code', 'list_date']]

def code_to_wind(code: str) -> str:
    c = str(code).zfill(6)
    if c.startswith(("8", "4")) or c.startswith("92"):
        return f"{c}.BJ"
    if c.startswith(("5", "6", "9")):
        return f"{c}.SH"
    return f"{c}.SZ"

def main():
    parser = argparse.ArgumentParser("Filter stock pool with price/volume and new-stock rules")
    
    default_input = Path(__file__).parent / "stock_pool.parquet"
    default_output = Path(__file__).parent / "stock_pool_base.parquet"
    
    parser.add_argument('--input', default=str(default_input), help='输入的纯股票池路径')
    parser.add_argument('--output', default=str(default_output), help='输出附加allow_flag后的股票池路径')
    parser.add_argument('--price-dir', default=r"E:\1_basement\quant_research\data\中国A股日行情_AShareEODPrices", help='日线级高频文件夹')
    parser.add_argument('--subnew-days', type=int, default=250, help='低于此上市自然天数的列为次新股 (默认 250 天)')
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    price_dir = Path(args.price_dir).resolve()

    if not input_path.exists():
        logger.error(f"Input file {input_path} not found.")
        return

    logger.info(f"--> [1/5] 载入股票池基座: {input_path}")
    pool = pd.read_parquet(input_path)
    
    # 转换日期格式进行计算
    pool['date_dt'] = pd.to_datetime(pool['date'].astype(str), format='%Y%m%d', errors='coerce')
    pool['Wind代码'] = pool['code'].map(code_to_wind)

    logger.info("--> [2/5] 正在通过 Tushare 获取全量股票上市日期...")
    df_list_date = get_all_stocks_list_date()
    pool = pool.merge(df_list_date, on='code', how='left')
    
    # 次新股判定
    pool['is_subnew'] = (pool['date_dt'] - pool['list_date']).dt.days < args.subnew_days

    # 提取所有包含的独特日期
    unique_dates = pool['date'].dropna().unique()
    unique_dates = sorted([str(int(d)) for d in unique_dates])

    logger.info(f"--> [3/5] 从日线行情库拉取跨界 {len(unique_dates)} 个交易日的 [收盘/高低/成交量] 数据...")
    price_parts = []
    
    needed = ['Wind代码', '成交量(手)', '收盘价(元)', '涨停价(元)', '跌停价(元)']
    
    # 由于不确定是否有tqdm依赖，手动计时汇报进度
    total = len(unique_dates)
    for i, dt_str in enumerate(unique_dates):
        if (i + 1) % 1000 == 0:
            logger.info(f"   [进度] 已读取 {i+1} / {total} 天...")
        
        p_file = price_dir / f"{dt_str}.pickle"
        if p_file.exists():
            try:
                df_p = pd.read_pickle(p_file)
                if all(c in df_p.columns for c in needed):
                    df_ext = df_p[needed].copy()
                    df_ext['date'] = int(dt_str)
                    price_parts.append(df_ext)
            except Exception as e:
                pass
                
    if price_parts:
        all_prices = pd.concat(price_parts, ignore_index=True)
    else:
        all_prices = pd.DataFrame(columns=needed + ['date'])
        
    logger.info(f"--> [4/5] 量价特征数据切片构建完成，合体后行数: {len(all_prices):,}")
    
    pool['date'] = pool['date'].astype(int)
    all_prices['date'] = all_prices['date'].astype(int)
    
    pool = pool.merge(all_prices, on=['date', 'Wind代码'], how='left')
    
    logger.info("--> [5/5] 执行条件拦截运算：停牌 / 涨跌停留板...")
    # 停牌判定: 没有价格数据或者当天成交量<=0
    is_suspended = pool['成交量(手)'].isna() | (pool['成交量(手)'] <= 0)
    
    # 涨跌停封板判定 (对标精度加减 0.01 以防浮点陷阱)
    is_limit_up = (pool['收盘价(元)'] >= pool['涨停价(元)'] - 0.01)
    is_limit_down = (pool['收盘价(元)'] <= pool['跌停价(元)'] + 0.01)
    
    # 综合合并判定为安全标的才允许 `allow_flag = 1`
    pool['allow_flag'] = 1
    
    # 基础过滤：仅针对 次新股 和 停牌（无流动性） 设为 allow_flag=0
    # 涨跌停过滤交给下游交易掩码处理（因为涨跌停并不影响反方向交易）
    reject_mask = pool['is_subnew'] | is_suspended
    pool.loc[reject_mask, 'allow_flag'] = 0
    
    # 状态汇集统计
    total_len = len(pool)
    sub_count = pool['is_subnew'].sum()
    susp_count = is_suspended.sum()
    liup_count = is_limit_up.sum()  # 记录用于展示，不计入 allow_flag 拦截
    lidw_count = is_limit_down.sum() # 记录用于展示，不计入 allow_flag 拦截
    allow_count = pool['allow_flag'].sum()
    
    logger.info("================== 过滤截面积分榜 ==================")
    logger.info(f"总载发股数: {total_len:,}")
    logger.info(f"  [X] 因 未满 250 天次新 遭阻截: {sub_count:,} ({sub_count/total_len:.1%})")
    logger.info(f"  [X] 因 成交低迷或查无报价 遭阻截: {susp_count:,} ({susp_count/total_len:.1%})")
    logger.info(f"  [i] 指标统计 - 涨停板数量 (未拦截): {liup_count:,} ({liup_count/total_len:.1%})")
    logger.info(f"  [i] 指标统计 - 跌停板数量 (未拦截): {lidw_count:,} ({lidw_count/total_len:.1%})")
    logger.info(f"  [√] 最终放行通过 (allow_flag=1): {allow_count:,} ({allow_count/total_len:.1%})")
    logger.info("====================================================")
    
    # 最终出表清理格式
    if 'name' not in pool.columns:
        pool['name'] = ""
        
    out_cols = ['date', 'code', 'name', 'allow_flag']
    out_df = pool[out_cols]
    
    logger.info(f"----> Saving output to: {output_path}")
    out_df.to_parquet(output_path, compression='snappy', index=False)
    logger.info("All done.")

if __name__ == '__main__':
    main()
