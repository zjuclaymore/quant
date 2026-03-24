import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 配置全局中文字体（Windows 下一般有微軟雅黑）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

parquet_path = r"E:\1_basement\quant_research\data\AIndexEODPrices_all.parquet"
output_image = r"E:\1_basement\quant_research\shanghai_index_2004_2024.png"

try:
    df = pd.read_parquet(parquet_path)
    # 过滤上证指数
    sub = df[df['symbol'] == '000001.SH'].copy()
    sub['date'] = pd.to_datetime(sub['date'])
    sub = sub.sort_values('date')
    
    # 过滤时间 2004-01-01 到 2024-12-31
    mask = (sub['date'] >= '2004-01-01') & (sub['date'] <= '2024-12-31')
    sub_filtered = sub[mask].copy()
    
    if sub_filtered.empty:
        print("Error: No data found in the specified date range 2004-2024.")
    else:
        # 归一化：以第一天的收盘价为 1
        sub_filtered = sub_filtered.reset_index(drop=True)
        base_close = sub_filtered['close'].iloc[0]
        sub_filtered['net_value'] = sub_filtered['close'] / base_close
        
        # 绘图
        plt.figure(figsize=(14, 7), dpi=100)
        plt.style.use('seaborn-v0_8-whitegrid') # 选用一个现代且清爽的样式
        
        # 渐变填充效果
        plt.fill_between(sub_filtered['date'], sub_filtered['net_value'], 
                         color='skyblue', alpha=0.3)
        
        # 绘制主线
        plt.plot(sub_filtered['date'], sub_filtered['net_value'], 
                 color='#1f77b4', linewidth=1.8, label='上证指数净值 (以 2004 为基 1)')
        
        # 加点修饰
        plt.title('上证指数净值曲线 (000001.SH, 2004 - 2024)', fontsize=15, fontweight='bold', pad=15)
        plt.xlabel('日期', fontsize=12)
        plt.ylabel('净值', fontsize=12)
        
        # 强调一个最高点和关键时间跨度
        max_idx = sub_filtered['net_value'].idxmax()
        plt.scatter(sub_filtered['date'].iloc[max_idx], sub_filtered['net_value'].iloc[max_idx], 
                    color='red', s=50, zorder=5)
        plt.annotate(f"牛市顶点\n({sub_filtered['date'].iloc[max_idx].strftime('%Y-%m')})", 
                     xy=(sub_filtered['date'].iloc[max_idx], sub_filtered['net_value'].iloc[max_idx]),
                     xytext=(10, 10), textcoords='offset points', arrowprops=dict(arrowstyle="->", color='red'))

        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.legend(loc='upper left', fontsize=11)
        
        plt.savefig(output_image)
        plt.close()
        print(f"SUCCESS: Plot saved to {output_image}")
except Exception as e:
    print(f"Error executing plot: {e}")
