from datetime import datetime
from pathlib import Path

import akshare as ak
import pandas as pd
import plotly.graph_objects as go

"""
市场风险与连板质量分析 (Market Risk & Quality Analysis)
数据源: ak.stock_zt_pool_dtgc_em (跌停), ak.stock_zt_pool_zbgc_em (炸板)
功能: 
1. 深度剖析今日跌停个股成份（杀跌动能）。
2. 计算全市场炸板率，分析连板接力质量。
"""

BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "special_data_analysis" / "visual_output" / "risk"


def analyze_market_risk(target_date=None, output_dir=None):
    target_date = target_date or datetime.now().strftime("%Y%m%d")
    print(f"正在进行 {target_date} 市场风险深度扫描...")

    try:
        df_dt = ak.stock_zt_pool_dtgc_em(date=target_date)

        out = Path(output_dir) if output_dir else OUTPUT_DIR
        out.mkdir(parents=True, exist_ok=True)

        if df_dt is not None and len(df_dt) > 0:
            dt_details = pd.DataFrame({
                '代码': df_dt.iloc[:, 1],
                '名称': df_dt.iloc[:, 2],
                '跌幅': df_dt.iloc[:, 3].round(2),
                '成交额(亿)': (df_dt.iloc[:, 5] / 1e8).round(2),
                '行业': df_dt.iloc[:, 15]
            })
            dt_details = dt_details[dt_details['成交额(亿)'] > 0].sort_values('成交额(亿)', ascending=False)
        else:
            dt_details = pd.DataFrame(columns=['代码', '名称', '跌幅', '成交额(亿)', '行业'])

        fig = go.Figure(go.Table(
            header=dict(values=list(dt_details.columns), fill_color='#2e7d32', font=dict(color='white')),
            cells=dict(values=[dt_details[c] for c in dt_details.columns], fill_color='#e8f5e9')
        ))
        fig.update_layout(height=800, title_text="<b>今日跌停个股详单</b>", template='plotly_white')

        table_path = out / "risk_details_tables.html"
        fig.write_html(table_path)

        print(f"跌停详单已生成: {table_path}")
        print(f"今日跌停家数: {len(dt_details)}")
        if len(dt_details) > 0:
            print(f"跌停压力板块: {dt_details['行业'].value_counts().index[0]}")
        return table_path

    except Exception as e:
        print(f"分析失败: {e}")
        raise

if __name__ == "__main__":
    import sys
    analyze_market_risk(sys.argv[1] if len(sys.argv) > 1 else None)
