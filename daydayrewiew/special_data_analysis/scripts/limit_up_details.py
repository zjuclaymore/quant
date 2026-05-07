from datetime import datetime
from pathlib import Path

import akshare as ak
import pandas as pd
import plotly.graph_objects as go

"""
今日涨停详情分析 (Limit Up Details)
数据源: ak.stock_zt_pool_em
功能: 详细列出今日连板梯队中每一只个股的详细信息（名称、行业、成交额、连板数）。
"""

BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "special_data_analysis" / "visual_output" / "ladder"


def analyze_today_limit_up(target_date=None, output_dir=None):
    target_date = target_date or datetime.now().strftime("%Y%m%d")
    print(f"正在获取 {target_date} 涨停板详细名录...")

    try:
        df = ak.stock_zt_pool_em(date=target_date)
        if df is None or df.empty:
            print("今日暂无涨停数据或尚未收盘。")
            return
            
        df_clean = pd.DataFrame({
            '代码': df.iloc[:, 1],
            '名称': df.iloc[:, 2],
            '连板数': pd.to_numeric(df.iloc[:, 14], errors='coerce').fillna(1).astype(int),
            '成交额(亿)': (df.iloc[:, 5] / 1e8).round(2),
            '所属行业': df.iloc[:, 15]
        })
        
        df_clean = df_clean.sort_values(['连板数', '成交额(亿)'], ascending=[False, False]).reset_index(drop=True)

        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['<b>连板高度</b>', '<b>股票名称</b>', '<b>代码</b>', '<b>所属行业</b>', '<b>成交额 (亿)</b>'],
                fill_color='#C62828',
                align='center',
                font=dict(color='white', size=14),
                height=35
            ),
            cells=dict(
                values=[
                    df_clean['连板数'].apply(lambda x: f"{x}连板" if x > 1 else "首板"),
                    df_clean['名称'],
                    df_clean['代码'],
                    df_clean['所属行业'],
                    df_clean['成交额(亿)']
                ],
                fill_color=[['#FFEBEE', '#FFFFFF'] * len(df_clean)],
                align='center',
                font=dict(color='#37474f', size=12),
                height=30
            )
        )])

        fig.update_layout(
            title=dict(
                text=f"<b>今日涨停连板梯队详单 ({target_date})</b>",
                x=0.5, font=dict(size=22)
            ),
            autosize=True,
            height=min(1200, 400 + len(df_clean) * 35),
            template='plotly_white'
        )

        out = Path(output_dir) if output_dir else OUTPUT_DIR
        out.mkdir(parents=True, exist_ok=True)
        output_path = out / "limit_up_details.html"
        fig.write_html(output_path)
        print(f"今日涨停详单已生成: {output_path}")

        print(f"\n[今日连板梯队概览]")
        for board in sorted(df_clean['连板数'].unique(), reverse=True):
            sub = df_clean[df_clean['连板数'] == board]
            print(f"\n【{board}连板】({len(sub)}家):")
            print(sub[['名称', '所属行业', '成交额(亿)']].to_string(index=False))
        return output_path

    except Exception as e:
        print(f"分析失败: {e}")
        raise

if __name__ == "__main__":
    import sys
    analyze_today_limit_up(sys.argv[1] if len(sys.argv) > 1 else None)
