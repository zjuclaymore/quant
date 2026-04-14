"""
回测辅助工具模块 (Backtest Utilities Module) - Basic version
"""

import io
import os
import base64
import json
import logging
import warnings
import datetime
import re
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy import stats
import statsmodels.api as sm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick

# 基础样式重置
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["axes.grid"] = True
plt.rcParams["axes.axisbelow"] = True
plt.rcParams["grid.color"] = "#e0e6ee"
plt.rcParams["grid.linewidth"] = 0.7
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["legend.framealpha"] = 0.92
plt.rcParams["legend.edgecolor"] = "#d1d9e6"
plt.rcParams["legend.fontsize"] = 9

_BAR_ABS_COLOR = "#4da6d9"   # 绝对收益-蓝色
_BAR_EXC_COLOR = "#f5a01e"   # 超额收益-橙色

CHART_TITLE_1 = "1.  因子分布处理流程（四子图）"
CHART_TITLE_1_COV = "2.  因子覆盖度（四子图）"
CHART_TITLE_2 = "2.  策略净值与基准对比"
CHART_TITLE_3 = "3.  策略对冲组合净值"
CHART_TITLE_4 = "4.  因子分组分层测试"
CHART_TITLE_5 = "5.  因子 IC / Rank IC 测算走势"
CHART_TITLE_6 = "6.  因子单调性与分组超额"
CHART_TITLE_DAILY = "日频策略净值走势"


class ReportChartContext:
    """管理报告图表的本地导出顺序和路径。"""

    def __init__(self, out_dir, sub_dir="charts"):
        self.out_dir = out_dir
        self.sub_dir = sub_dir
        self.chart_dir = os.path.join(out_dir, sub_dir)
        self.counter = 0
        os.makedirs(self.chart_dir, exist_ok=True)

    def next_relpath(self, title):
        self.counter += 1
        safe = re.sub(r"[^\w\-\u4e00-\u9fff]+", "_", str(title)).strip("_")
        if not safe:
            safe = f"chart_{self.counter}"
        fname = f"{self.counter:02d}_{safe}.png"
        return os.path.join(self.sub_dir, fname).replace("\\", "/")


def _apply_report_caption(fig, factor_name):
    return


def _year_locator_from_dates(dates):
    if len(dates) < 2:
        return mdates.YearLocator(base=1)
    years = pd.Series(pd.to_datetime(dates)).dt.year
    span = int(years.max() - years.min()) if len(years) else 1
    if span >= 16:
        return mdates.YearLocator(base=4)
    if span >= 8:
        return mdates.YearLocator(base=2)
    return mdates.YearLocator(base=1)


def _style_time_axis(ax, dates, ylabel=None, percent=False):
    ax.set_facecolor("#f5f5f5")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.7)
    ax.grid(axis="x", color="#ebebeb", linewidth=0.55, alpha=0.8)
    ax.spines["left"].set_color("#7a7a7a")
    ax.spines["bottom"].set_color("#7a7a7a")
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.xaxis.set_major_locator(_year_locator_from_dates(dates))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", rotation=0)
    if ylabel:
        ax.set_ylabel(ylabel)
    if percent:
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v*100:.2f}%"))


def _style_legend(legend):
    if legend is None:
        return
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_edgecolor("#c7ccd4")
    frame.set_linewidth(0.8)
    frame.set_alpha(0.95)


# ==========================================
# HTML 报告渲染核心组件 (基础占位版)
# ==========================================

REPORT_CSS = """
<style>
  :root {
        --bg-primary: #f8fafc;
        --bg-card: #ffffff;
        --bg-card-hover: #f1f5f9;
        --border-main: #e2e8f0;
        --border-light: #f1f5f9;
        --text-primary: #1e293b;
        --text-secondary: #475569;
        --text-muted: #94a3b8;
        --accent-blue: #2563eb;
        --accent-purple: #7c3aed;
        --accent-amber: #d97706;
        --accent-emerald: #059669;
        --accent-rose: #e11d48;
        --shadow-sm: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
        --shadow-md: 0 4px 12px rgba(0,0,0,0.07), 0 2px 4px rgba(0,0,0,0.04);
  }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(16px); }
        to { opacity: 1; transform: translateY(0); }
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
        background: var(--bg-primary);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
        color: var(--text-primary);
        margin: 0;
        padding: 32px 24px;
        min-height: 100vh;
        -webkit-font-smoothing: antialiased;
    line-height: 1.6;
  }
    h1, h2, h3, h4 { margin-top: 0; }
    .dashboard { max-width: 1440px; margin: 0 auto; }

    .report-header {
        text-align: center;
        margin-bottom: 26px;
        padding-bottom: 20px;
        border-bottom: 2px solid var(--border-main);
        animation: fadeInUp 0.5s ease-out;
    }
    .report-header h1 {
        font-size: 2.1em;
        font-weight: 800;
        letter-spacing: 0.5px;
        color: var(--text-primary);
        margin-bottom: 4px;
    }
    .report-header .subtitle {
        font-size: 13px;
        color: var(--text-muted);
        font-weight: 400;
        letter-spacing: 0.8px;
  }

    .summary-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
        gap: 16px;
        margin-bottom: 18px;
    }

    .kpi-section {
        margin-bottom: 0;
        background: var(--bg-card);
        border: 1px solid var(--border-main);
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: var(--shadow-sm);
        animation: fadeInUp 0.5s ease-out 0.1s both;
    }
    .kpi-section-header {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 14px;
        font-weight: 700;
        letter-spacing: 0.5px;
        margin-bottom: 14px;
        padding-bottom: 10px;
        border-bottom: 1px solid var(--border-light);
        color: var(--text-primary);
    }
    .kpi-section-badge {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--accent, var(--accent-blue));
    }
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
        gap: 10px;
    }
    .kpi-card {
        background: var(--bg-primary);
        border: 1px solid var(--border-light);
        border-radius: 8px;
        padding: 12px 10px;
        text-align: center;
        transition: all 0.2s ease;
    }
    .kpi-card:hover {
        background: var(--bg-card-hover);
        transform: translateY(-1px);
    }
    .kpi-label {
        color: var(--text-secondary);
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 6px;
        display: block;
    }
    .kpi-value {
        color: var(--text-primary);
        font-size: 15px;
        font-weight: 800;
        letter-spacing: 0.1px;
    }

    .pipeline-card {
        background: var(--bg-card);
        border: 1px solid var(--border-main);
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: var(--shadow-sm);
        margin-bottom: 20px;
    }
    .pipeline-card h3 {
        font-size: 14px;
        font-weight: 700;
        letter-spacing: 0.4px;
        margin-bottom: 12px;
        color: var(--text-primary);
    }
    .pipeline-list {
        margin: 0;
        padding-left: 18px;
    }
    .pipeline-list li {
        margin: 6px 0;
        color: var(--text-secondary);
        font-size: 13px;
    }
  
  /* Tabs CSS */
  .tabs {
    display: flex;
        gap: 8px;
    margin-bottom: 20px;
        border-bottom: none;
        flex-wrap: wrap;
  }
  .tab-btn {
        background: #eef2ff;
        border: 1px solid #dbe5ff;
        border-radius: 8px;
        padding: 10px 16px;
        font-size: 13px;
        font-weight: 600;
        color: var(--text-secondary);
    cursor: pointer;
        transition: all 0.2s ease;
  }
    .tab-btn:hover {
        color: var(--accent-blue);
        border-color: #bfd0ff;
    }
  .tab-btn.active {
        color: #ffffff;
        border-color: var(--accent-blue);
        background: var(--accent-blue);
        box-shadow: var(--shadow-sm);
  }
  
  .tab-content { display: none; }
  .tab-content.active { display: block; }
  
  /* general section css */
  .section {
        background-color: var(--bg-card);
        border: 1px solid var(--border-main);
        border-radius: 12px;
    padding: 24px;
    margin-bottom: 20px;
        box-shadow: var(--shadow-sm);
        animation: fadeInUp 0.5s ease-out 0.15s both;
    }
    .section h2 {
        color: var(--text-primary);
        margin-bottom: 8px;
        font-size: 22px;
        font-weight: 700;
    }
    .section h4 {
        color: var(--text-primary);
        margin-top: 18px;
        margin-bottom: 10px;
        font-size: 15px;
        font-weight: 700;
  }
  .plot-img { max-width: 100%; height: auto; display: block; margin: 0 auto; }
      .plot-block { margin: 10px 0 22px; }
      .echarts-wrap { margin: 10px 0 22px; }
      .echarts-canvas { width: 100%; height: 420px; border: 1px solid var(--border-light); border-radius: 8px; background: #ffffff; }
      /* 第4图排版修正：外层已有章节标题，隐藏重复块标题并抬高图面高度 */
    #tab-summary .plot-block:nth-of-type(3) > h4 { display: none; }
    #tab-summary .plot-block:nth-of-type(3) .plot-img {
        height: 470px;
        width: 100%;
        object-fit: contain;
    }
  
  /* table formatting */
    .table-wrapper { overflow-y: auto; overflow-x: auto; margin-top: 15px; max-height: 600px; }
  table {
    border-collapse: collapse;
    width: 100%;
    font-size: 13px;
    white-space: nowrap;
  }
    th, td { border: 1px solid var(--border-main); padding: 10px 12px; text-align: left; }
    th { background-color: #f8fafc; font-weight: 700; color: var(--text-primary); }
    tr:nth-child(even) { background-color: #fcfdff; }
  
  pre { background-color: #1e1e1e; color: #d4d4d4; padding: 15px; border-radius: 6px; overflow-x: auto; font-family: monospace; font-size: 13px; max-height: 600px; overflow-y: auto;}
</style>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
<script>
  function openTab(evt, tabName) {
    var i, tabcontent, tablinks;
    tabcontent = document.getElementsByClassName("tab-content");
    for (i = 0; i < tabcontent.length; i++) {
      tabcontent[i].style.display = "none";
      tabcontent[i].classList.remove("active");
    }
    tablinks = document.getElementsByClassName("tab-btn");
    for (i = 0; i < tablinks.length; i++) {
      tablinks[i].className = tablinks[i].className.replace(" active", "");
    }
    document.getElementById(tabName).style.display = "block";
    document.getElementById(tabName).classList.add("active");
    evt.currentTarget.className += " active";

        // 图表可能在隐藏容器中初始化，切换到可见后统一触发 resize 修正坐标轴。
        setTimeout(function () {
            if (window.__reportCharts && window.__reportCharts.length) {
                for (var j = 0; j < window.__reportCharts.length; j++) {
                    try { window.__reportCharts[j].resize(); } catch (e) {}
                }
            }
        }, 80);
  }

    // 修复第4图标题重叠：隐藏与图内标题重复的外层 h4
    document.addEventListener("DOMContentLoaded", function () {
        var hs = document.querySelectorAll("#tab-summary .plot-block > h4");
        for (var i = 0; i < hs.length; i++) {
            var t = (hs[i].textContent || "").trim();
            if (t.indexOf("Pearson IC 与 Rank IC 测算走势") !== -1) {
                hs[i].style.display = "none";
                break;
            }
        }
    });
</script>
"""


def build_workflow_pipeline(title, steps):
    items = "".join([f"<li><b>{s[0]}</b>: {s[1]}</li>" for s in steps])
    return f"<div class='pipeline-card'><h3>{title}</h3><ul class='pipeline-list'>{items}</ul></div>"


def build_report_header(factor_name, subtitle=""):
    return f"<div class='report-header'><h1>回测报告: {factor_name}</h1><div class='subtitle'>SINGLE FACTOR BACKTEST REPORT</div></div><div class='summary-grid'>{subtitle}</div>"


def add_card_to_html(title, icon, items, theme_color=None):
    color = theme_color or "#2563eb"
    rows = "".join([
        f"<div class='kpi-card'><span class='kpi-label'>{k}</span><span class='kpi-value'>{v}</span></div>"
        for k, v in items
    ])
    return (
        f"<div class='kpi-section' style='--accent:{color};'>"
        f"<div class='kpi-section-header'><span class='kpi-section-badge'></span>{icon} {title}</div>"
        f"<div class='kpi-grid'>{rows}</div>"
        f"</div>"
    )


def add_plot_to_html(html_parts, fig, title, chart_ctx=None):
    if chart_ctx is not None:
        rel_path = chart_ctx.next_relpath(title)
        abs_path = os.path.join(chart_ctx.out_dir, rel_path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        fig.savefig(abs_path, format="png", bbox_inches="tight")
        plt.close(fig)
        html_parts.append(f"<div class='plot-block'><h4>{title}</h4><img class='plot-img' src='{rel_path}'></div>")
        return html_parts

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    plt.close(fig)
    html_parts.append(f"<div class='plot-block'><h4>{title}</h4><img class='plot-img' src='data:image/png;base64,{img_base64}'></div>")
    return html_parts


def build_section(title, content_html, desc=None, tab_id=None):
    desc_html = f"<p style='color:#64748b;'>{desc}</p>" if desc else ""
    active_cls = " active" if tab_id == "tab-summary" else ""
    return f"<div id='{tab_id}' class='tab-content{active_cls}'><div class='section'><h2>{title}</h2>{desc_html}{content_html}</div></div>"


def add_table_to_html(html_parts, df, title, max_rows=60):
    if df is None or df.empty: return html_parts
    if max_rows is not None:
        table_html = df.head(max_rows).to_html(index=False, classes="report-table")
    else:
        table_html = df.to_html(index=False, classes="report-table")
    html_parts.append(f"<div><h4>{title}</h4><div class='table-wrapper'>{table_html}</div></div>")
    return html_parts


def add_log_block(html_parts, log_text, title="日志"):
    if not log_text: return html_parts
    html_parts.append(f"<div><h4>{title}</h4><pre>{log_text}</pre></div>")
    return html_parts


def write_report_html(out_dir, factor_name, html_parts):
    # html_parts 预计包含 context 头部以及各个 tab 的内容
    # We will wrap it all in .dashboard

    # 当前模式仅保留“收益概况”，不渲染页签栏。
    tabs_html = ""
    script_html = ""
    
    # assume the first part in html_parts is the context_html header
    context_part = html_parts[0] if len(html_parts) > 0 else ""
    sections_html = "".join(html_parts[1:]) if len(html_parts) > 1 else ""
    
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>{factor_name} 回测报告</title>
        <script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
        {REPORT_CSS}
    </head>
    <body>
        <div class="dashboard">
            {context_part}
            {tabs_html}
            {sections_html}
        </div>
        {script_html}
    </body>
    </html>
    """
    
    report_path = os.path.join(out_dir, f"{factor_name}_Report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_code)
    return report_path


def write_report_html_no_trades(out_dir, factor_name, html_parts):
    # 与 write_report_html 类似，但去掉交易详情页签，解决数据量大时的页面卡顿问题
    tabs_html = """
    <div class="tabs">
      <button class="tab-btn active" onclick="openTab(event, 'tab-summary')">策略绩效</button>
      <button class="tab-btn" onclick="openTab(event, 'tab-daily')">日频表现</button>
      <button class="tab-btn" onclick="openTab(event, 'tab-logs')">运行日志</button>
    </div>
    """
    
    script_html = """
    <script>
      document.addEventListener("DOMContentLoaded", function() {
        document.querySelector('.tab-btn').click();
      });
    </script>
    """
    
    context_part = html_parts[0] if len(html_parts) > 0 else ""
    sections_html = "".join(html_parts[1:]) if len(html_parts) > 1 else ""
    
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>{factor_name} 回测报告 (精简版)</title>
        <script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
        {REPORT_CSS}
    </head>
    <body>
        <div class="dashboard">
            {context_part}
            {tabs_html}
            {sections_html}
        </div>
        {script_html}
    </body>
    </html>
    """
    
    report_path = os.path.join(out_dir, f"{factor_name}_Report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_code)
    return report_path


# ==========================================
# 核心计算逻辑
# ==========================================

def cross_section_mad_normalize(series, n=3.0):
    s = series.copy()
    median = s.median()
    mad = (s - median).abs().median()
    if mad == 0: return (s - s.mean()) / s.std() if s.std() > 0 else s - s.mean()
    upper, lower = median + n * mad, median - n * mad
    s = s.clip(lower=lower, upper=upper)
    mu, sigma = s.mean(), s.std()
    return (s - mu) / sigma if sigma > 0 else s - mu


def neutralize_factor(df, factor_col, mv_col="lncap", ind_col=None, logger=None, return_stats=False):
    resid_series = pd.Series(index=df.index, dtype=float)
    stats = {
        "rows_total": int(len(df)),
        "rows_missing_factor": int(df[factor_col].isna().sum()) if factor_col in df.columns else int(len(df)),
        "rows_missing_mv": int(df[mv_col].isna().sum()) if mv_col in df.columns else int(len(df)),
        "rows_missing_ind": int(df[ind_col].isna().sum()) if ind_col and ind_col in df.columns else 0,
        "rows_valid_neutralized": 0,
        "months_processed": 0,
        "months_skipped_small_sample": 0,
        "months_failed": 0,
    }
    for ym, group in df.groupby("year_month"):
        valid_idx = group[factor_col].notna() & group[mv_col].notna()
        if ind_col:
            valid_idx &= group[ind_col].notna()
        subset = group[valid_idx]
        if len(subset) < 10:
            stats["months_skipped_small_sample"] += 1
            continue
        Y = subset[factor_col].values
        X_mv = subset[mv_col].values
        if ind_col:
            ind_dummies = pd.get_dummies(subset[ind_col], drop_first=True).values.astype(float)
            X_mat = np.column_stack([np.ones_like(X_mv), X_mv, ind_dummies])
        else:
            X_mat = np.column_stack([np.ones_like(X_mv), X_mv])
        try:
            beta, _, _, _ = np.linalg.lstsq(X_mat, Y, rcond=None)
            resid_series.loc[subset.index] = Y - X_mat.dot(beta)
            stats["rows_valid_neutralized"] += int(len(subset))
            stats["months_processed"] += 1
        except Exception as e:
            stats["months_failed"] += 1
            warnings.warn(f"Neutralization failed at {ym}: {e}")
    if logger is not None:
        logger.info(
            "中性化统计: 总行=%s, 缺失因子=%s, 缺失市值=%s, 缺失行业=%s, 有效中性化=%s, 处理月份=%s, 小样本跳过=%s, 失败月份=%s",
            stats["rows_total"],
            stats["rows_missing_factor"],
            stats["rows_missing_mv"],
            stats["rows_missing_ind"],
            stats["rows_valid_neutralized"],
            stats["months_processed"],
            stats["months_skipped_small_sample"],
            stats["months_failed"],
        )
    if return_stats:
        return resid_series, stats
    return resid_series


def compute_backtest_kpis(
    best_df,
    worst_df,
    df_trades,
    df_ic,
    market_returns,
    cal,
    best_g,
    worst_g,
    df1,
    df_daily_nunique,
):
    total_months = len(cal)
    ann_factor = 12 / total_months if total_months > 0 else 1
    best_ret_ann = (best_df["净值"].iloc[-1]) ** ann_factor - 1 if len(best_df) > 0 else 0
    best_vol_ann = best_df["actual_ret"].std() * np.sqrt(12)
    sharpe = best_ret_ann / best_vol_ann if best_vol_ann != 0 else np.nan
    roll_max = best_df["净值"].rolling(window=len(best_df), min_periods=1).max()
    drawdowns = (best_df["净值"] / roll_max - 1).astype(float)
    max_drawdown = drawdowns.min()

    max_drawdown_period = "N/A"
    if len(best_df) > 0 and not np.isnan(max_drawdown):
        dd_series = drawdowns.reset_index(drop=True)
        nav_series = best_df["净值"].astype(float).reset_index(drop=True)
        trough_idx = int(dd_series.idxmin())
        peak_idx = int(nav_series.iloc[: trough_idx + 1].idxmax())
        peak_ym = str(best_df.iloc[peak_idx]["year_month"]) if "year_month" in best_df.columns else str(peak_idx)
        trough_ym = str(best_df.iloc[trough_idx]["year_month"]) if "year_month" in best_df.columns else str(trough_idx)
        max_drawdown_period = f"{peak_ym} -> {trough_ym}"
    ic_mean = df_ic["ic"].mean() if not df_ic.empty else 0
    ic_std = df_ic["ic"].std() if not df_ic.empty else 0
    ric_mean = df_ic["rank_ic"].mean() if not df_ic.empty else 0
    ric_std = df_ic["rank_ic"].std() if not df_ic.empty else 0
    ir = ic_mean / ic_std if ic_std and not np.isnan(ic_std) else 0
    rank_ir = ric_mean / ric_std if ric_std and not np.isnan(ric_std) else 0

    overall_grp_ret = df_trades.groupby("group")["actual_ret"].mean()
    overall_monotony, overall_pval = (spearmanr(overall_grp_ret.index, overall_grp_ret.values) if len(overall_grp_ret) > 5 else (0, 0))

    # 业内口径：先按月计算截面单调性，再对时间序列做显著性检验
    monthly_monotony = []
    if df_trades is not None and not df_trades.empty and {"year_month", "group", "actual_ret"}.issubset(df_trades.columns):
        grouped_month = (
            df_trades.groupby(["year_month", "group"], as_index=False)["actual_ret"]
            .mean()
            .sort_values(["year_month", "group"])
        )
        for ym, gdf in grouped_month.groupby("year_month"):
            gser = gdf.set_index("group")["actual_ret"].dropna().sort_index()
            if len(gser) < 6:
                continue
            rho, pval = spearmanr(gser.index.values, gser.values)
            if np.isfinite(rho):
                monthly_monotony.append({"year_month": ym, "rho": float(rho), "pval": float(pval)})

    monthly_monotony_df = pd.DataFrame(monthly_monotony)
    mono_mean_ts = 0.0
    mono_t_ols = 0.0
    mono_p_ols = 1.0
    mono_t_nw = 0.0
    mono_p_nw = 1.0
    mono_positive_ratio = 0.0
    mono_n = 0

    if not monthly_monotony_df.empty:
        rho_series = monthly_monotony_df["rho"].dropna().astype(float)
        mono_n = int(len(rho_series))
        if mono_n > 0:
            mono_mean_ts = float(rho_series.mean())
            mono_positive_ratio = float((rho_series > 0).mean())
            if mono_n >= 3:
                # 基础 t 检验（均值是否显著偏离 0）
                try:
                    t_res = stats.ttest_1samp(rho_series.values, popmean=0.0, nan_policy="omit")
                    mono_t_ols = float(t_res.statistic) if np.isfinite(t_res.statistic) else 0.0
                    mono_p_ols = float(t_res.pvalue) if np.isfinite(t_res.pvalue) else 1.0
                except Exception as e:
                    logging.getLogger("bt_utils").error(
                        "单调性t检验失败: %s: %s", type(e).__name__, e
                    )
                    mono_t_ols, mono_p_ols = 0.0, 1.0

                # Newey-West(HAC) 稳健检验：OLS(常数项) + HAC 协方差
                try:
                    y = rho_series.values
                    X = np.ones((len(y), 1))
                    nw_lags = max(1, int(4 * (len(y) / 100.0) ** (2.0 / 9.0)))
                    nw_lags = min(nw_lags, max(1, len(y) - 1))
                    nw_model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags})
                    mono_t_nw = float(nw_model.tvalues[0]) if np.isfinite(nw_model.tvalues[0]) else 0.0
                    mono_p_nw = float(nw_model.pvalues[0]) if np.isfinite(nw_model.pvalues[0]) else 1.0
                except Exception as e:
                    logging.getLogger("bt_utils").error(
                        "单调性Newey-West检验失败: %s: %s", type(e).__name__, e
                    )
                    mono_t_nw, mono_p_nw = mono_t_ols, mono_p_ols
    # 月度胜率：基准已移除， bench_return=0，因此等价于「正收益月份比例」。
    best_wins = (best_df["actual_ret"] > best_df["bench_return"]).sum()
    win_rate = best_wins / len(best_df) if len(best_df) > 0 else 0
    calmar = best_ret_ann / abs(max_drawdown) if max_drawdown != 0 else np.nan
    
    # 多空净值计算
    ls_ret = (best_df.set_index("year_month")["actual_ret"] - worst_df.set_index("year_month")["actual_ret"]).fillna(0)
    ls_cum = (1 + ls_ret).cumprod()
    
    return {
        "best_ret_ann": best_ret_ann, "sharpe": sharpe, "max_drawdown": max_drawdown,
        "ls_cum": ls_cum, "drawdowns": drawdowns,
        "ic_mean": ic_mean, "ic_std": ic_std, "ric_mean": ric_mean, "ric_std": ric_std,
        "ir": ir, "rank_ir": rank_ir,
        "overall_monotony": overall_monotony, "overall_pval": overall_pval,
        "mono_ts_mean": mono_mean_ts,
        "mono_ts_t_ols": mono_t_ols,
        "mono_ts_p_ols": mono_p_ols,
        "mono_ts_t_nw": mono_t_nw,
        "mono_ts_p_nw": mono_p_nw,
        "mono_ts_positive_ratio": mono_positive_ratio,
        "mono_ts_n": mono_n,
        "win_rate": win_rate, "max_drawdown_period": max_drawdown_period, "calmar": calmar
    }


def plot_core_performance(best_df, ls_cum, drawdowns, best_g, html_parts, delay_days=0, factor_name="", strategy_label=None, show_ls=True, df_coverage=None, chart_ctx=None):
    has_cov = df_coverage is not None and not df_coverage.empty

    # 按需求移除图2和图3，避免在报告中展示策略净值与对冲净值曲线。
    ts = best_df["year_month"].dt.to_timestamp()

    # --- 可选：覆盖度单独保留在图2之后的独立图块（如果有） ---
    if has_cov:
        fig_cov, ax3 = plt.subplots(1, 1, figsize=(10.5, 4.8))
        fig_cov.subplots_adjust(left=0.08, right=0.96, top=0.88)
        _apply_report_caption(fig_cov, factor_name)
        cov_df = df_coverage if df_coverage is not None else pd.DataFrame(columns=["year_month", "coverage"])
        cov_ts = cov_df["year_month"].dt.to_timestamp()
        cov_vals = cov_df["coverage"].fillna(0)
        ax3.bar(cov_ts, cov_vals, color="#0ea5e9", width=25, alpha=0.82, zorder=3)
        avg_cov = cov_vals.mean()
        ax3.axhline(avg_cov, color="#d97706", linewidth=1.0, linestyle="--",
                    label=f"均值  {avg_cov*100:.1f}%", zorder=4)
        ax3.set_title("1.3  因子覆盖度测算走势", loc="center")
        _style_time_axis(ax3, cov_ts, ylabel="覆盖度频次", percent=True)
        ax3.xaxis.grid(False)
        ax3.set_ylim(0, 1.05)
        _style_legend(ax3.legend(loc="lower right", framealpha=0.92))
        add_plot_to_html(html_parts, fig_cov, "1.3  因子覆盖度测算走势", chart_ctx=chart_ctx)

def plot_pre_post_distribution(df_none, df1, factor_name, html_parts, chart_ctx=None):
    if df1 is None or factor_name not in df1.columns:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.subplots_adjust(top=0.88)
    _apply_report_caption(fig, factor_name)
    data = df1[factor_name].dropna()
    ax.hist(data, bins=60, color="#3b82f6", edgecolor="white", linewidth=0.3, alpha=0.85)
    mean_val = data.mean()
    ax.axvline(mean_val, color="#dc2626", linewidth=1.3, linestyle="--",
               label=f"均值  {mean_val:.4f}")
    ax.set_title("1.2  因子暴露/打分分布", loc="center")
    ax.set_xlabel("因子值")
    ax.set_ylabel("频次")
    ax.set_facecolor("#f5f5f5")
    ax.xaxis.grid(False)
    _style_legend(ax.legend(loc="upper right", framealpha=0.92))
    add_plot_to_html(html_parts, fig, "1.2  因子暴露/打分分布", chart_ctx=chart_ctx)


def plot_coverage_and_exposure(
    df_coverage_all,
    df_coverage_tradeable,
    df_coverage_all_cs,
    df_coverage_tradeable_cs,
    df1,
    factor_name,
    html_parts,
    chart_ctx=None,
    source_factor_col=None,
    df1_all=None,
):
    """图1恢复旧版分布图；图2使用四子图覆盖度（按原始/最终 × 全样本/可交易）。"""

    df1_all = df1 if df1_all is None else df1_all

    def _pick_col(df, candidates):
        if df is None:
            return None
        for col in candidates:
            if col and col in df.columns:
                return col
        return None

    all_raw_col = _pick_col(df1_all, [source_factor_col, factor_name])
    all_final_col = _pick_col(
        df1_all,
        [
            f"{source_factor_col}_neu" if source_factor_col else None,
            f"{factor_name}_neu",
            factor_name,
        ],
    )
    trade_raw_col = _pick_col(df1, [source_factor_col, factor_name])
    trade_final_col = _pick_col(
        df1,
        [
            f"{source_factor_col}_neu" if source_factor_col else None,
            f"{factor_name}_neu",
            factor_name,
        ],
    )

    # 图1：因子分布四子图（原始 -> 3MAD去极值 -> 去极值中性化 -> 去极值中性化标准化）
    base_df = df1_all if df1_all is not None else df1
    base_raw_col = _pick_col(base_df, [source_factor_col, factor_name])
    if base_df is not None and base_raw_col and "year_month" in base_df.columns:
        dist_df = base_df[["year_month", base_raw_col, "lncap"] + (["industry"] if "industry" in base_df.columns else [])].copy()
        dist_df.rename(columns={base_raw_col: "raw_factor"}, inplace=True)
        dist_df["raw_factor"] = pd.to_numeric(dist_df["raw_factor"], errors="coerce")

        raw_series = dist_df["raw_factor"].dropna()
        raw_median = raw_series.median() if not raw_series.empty else np.nan
        raw_mad = (raw_series - raw_median).abs().median() if not raw_series.empty else np.nan
        mad_lower = raw_median - 3.0 * raw_mad if pd.notna(raw_mad) else np.nan
        mad_upper = raw_median + 3.0 * raw_mad if pd.notna(raw_mad) else np.nan

        def _mad_winsorize_series(s):
            x = pd.to_numeric(s, errors="coerce")
            x_valid = x.dropna()
            if x_valid.empty:
                return x
            m = x_valid.median()
            mad = (x_valid - m).abs().median()
            if pd.isna(mad) or mad <= 0:
                return x
            lo = m - 3.0 * mad
            hi = m + 3.0 * mad
            return x.clip(lower=lo, upper=hi)

        dist_df["winsor_factor"] = dist_df.groupby("year_month")["raw_factor"].transform(_mad_winsorize_series)

        neu_col = "winsor_neu_factor"
        ind_col = "industry" if "industry" in dist_df.columns else None
        dist_df[neu_col] = neutralize_factor(dist_df, "winsor_factor", mv_col="lncap", ind_col=ind_col)
        dist_df["winsor_neu_z_factor"] = dist_df.groupby("year_month")[neu_col].transform(
            lambda s: (s - s.mean()) / s.std() if s.std() and s.std() > 0 else s - s.mean()
        )

        fig_dist, axes_dist = plt.subplots(2, 2, figsize=(18, 10))
        fig_dist.subplots_adjust(top=0.88, left=0.05, right=0.98, bottom=0.17, wspace=0.22, hspace=0.34)
        _apply_report_caption(fig_dist, factor_name)

        panels = [
            (axes_dist[0, 0], dist_df["raw_factor"], "1.1  原始因子分布", "#3b82f6", True),
            (axes_dist[0, 1], dist_df["winsor_factor"], "1.2  去极值后因子分布", "#06b6d4", True),
            (axes_dist[1, 0], dist_df[neu_col], "1.3  去极值中性化处理后因子分布", "#8b5cf6", False),
            (axes_dist[1, 1], dist_df["winsor_neu_z_factor"], "1.4  去极值中性化标准化因子分布", "#f43f5e", False),
        ]

        for ax, series, title, color, show_mad in panels:
            s = pd.to_numeric(series, errors="coerce").dropna()
            ax.set_facecolor("#f5f5f5")
            ax.set_title(title, loc="center")
            ax.set_xlabel("因子值")
            ax.set_ylabel("频次")
            ax.xaxis.grid(False)
            if s.empty:
                ax.text(0.5, 0.5, "无有效数据", ha="center", va="center", transform=ax.transAxes)
                continue
            ax.hist(s, bins=60, color=color, edgecolor="white", linewidth=0.25, alpha=0.88, label="样本分布")
            mu = s.mean()
            sd = s.std()
            ax.axvline(mu, color="#dc2626", linewidth=1.1, linestyle="--", label=f"均值 {mu:.4f}")
            if show_mad and pd.notna(mad_lower) and pd.notna(mad_upper):
                ax.axvline(mad_lower, color="#f59e0b", linewidth=1.1, linestyle=":", label=f"3MAD下界 {mad_lower:.4f}")
                ax.axvline(mad_upper, color="#f59e0b", linewidth=1.1, linestyle=":", label=f"3MAD上界 {mad_upper:.4f}")
            ax.text(
                0.02,
                0.95,
                f"样本数={len(s):,}  标准差={sd:.4f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8.5,
                color="#475569",
                bbox=dict(boxstyle="round,pad=0.26", facecolor="white", edgecolor="#cbd5e1", alpha=0.95),
            )
            _style_legend(ax.legend(loc="upper right", framealpha=0.92))

        fig_dist.text(
            0.05,
            0.04,
            "说明: 处理流程为 原始 -> 3MAD去极值 -> 行业+市值中性化 -> 截面标准化；1.1/1.2 已恢复 3MAD 垂直参考线。",
            fontsize=9,
            ha="left",
            va="bottom",
            color="#334155",
        )
        add_plot_to_html(html_parts, fig_dist, CHART_TITLE_1, chart_ctx=chart_ctx)

    def _prep_cov_df(df_cov):
        """
        标准化覆盖度 DataFrame，确保 year_month 为 PeriodIndex，
        coverage/factor_count/market_count 为数值类型。

        公式: coverage = factor_count / market_count（已在 bt_core 计算好）

        参数:
            df_cov (pd.DataFrame): 来自 bt_core._build_coverage_records 的原始覆盖度表。
        返回:
            pd.DataFrame | None: 标准化后的 DataFrame，字段含 year_month, coverage,
                factor_count, market_count；或 None（输入无效时）。
        """
        if df_cov is None or df_cov.empty or "year_month" not in df_cov.columns:
            return None
        cov = df_cov.copy()
        cov["year_month"] = pd.PeriodIndex(cov["year_month"].astype(str), freq="M")
        # 保证 coverage 已计算
        if "coverage" not in cov.columns:
            denom_col = next((c for c in ["market_count", "stock_pool_count", "candidate_count"] if c in cov.columns), None)
            if denom_col and "factor_count" in cov.columns:
                cov["coverage"] = np.where(
                    pd.to_numeric(cov[denom_col], errors="coerce") > 0,
                    pd.to_numeric(cov["factor_count"], errors="coerce") / pd.to_numeric(cov[denom_col], errors="coerce"),
                    np.nan,
                )
        for col in ["coverage", "factor_count", "market_count", "stock_pool_count"]:
            if col in cov.columns:
                cov[col] = pd.to_numeric(cov[col], errors="coerce")
        return cov

    def _plot_cov_rate(ax, cov_df, title, color, note):
        """
        绘制覆盖率百分比柱状图（左列）。

        公式: coverage_rate = factor_count / denom_count（在 bt_core 预算好）

        参数:
            ax: matplotlib Axes 对象。
            cov_df: 标准化覆盖度 DataFrame（含 year_month, coverage 列）。
            title: 图标题。
            color: 柱颜色（hex 字符串）。
            note: 显示在图左上角的数据口径说明文字。
        """
        ax.set_facecolor("#f5f5f5")
        if cov_df is None or cov_df.empty:
            ax.text(0.5, 0.5, "无覆盖度数据", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title, loc="center")
            return
        ts = cov_df["year_month"].dt.to_timestamp()
        vals = cov_df["coverage"].fillna(0)
        ax.bar(ts, vals, color=color, width=25, alpha=0.82, zorder=3)
        avg_cov = vals.mean()
        ax.axhline(avg_cov, color="#d97706", linewidth=1.0, linestyle="--",
                   label=f"均值  {avg_cov*100:.1f}%", zorder=4)
        _style_time_axis(ax, ts, ylabel="覆盖率", percent=True)
        ax.xaxis.grid(False)
        ax.set_ylim(0, 1.05)
        ax.set_title(title, loc="center")
        ax.text(
            0.02, 0.95, note,
            transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color="#475569",
            bbox=dict(boxstyle="round,pad=0.26", facecolor="white", edgecolor="#cbd5e1", alpha=0.95),
        )
        _style_legend(ax.legend(loc="lower right", framealpha=0.92))

    def _plot_cov_count(ax, cov_df, title, color_n, color_d, note, denom_label):
        """
        绘制绝对数量对比柱状图（右列）：因子覆盖数 vs 分母总数。

        设计目的: 让用户同时看到覆盖率的分子和分母的量级，避免仅看比例时忽略
        分母本身随时间的变化（如市场扩容）。

        参数:
            ax: matplotlib Axes 对象。
            cov_df: 标准化覆盖度 DataFrame（含 year_month, factor_count, market_count/stock_pool_count 列）。
            title: 图标题。
            color_n: 分子（因子覆盖数）的柱颜色。
            color_d: 分母（总数）的折线颜色。
            note: 数据口径说明文字。
            denom_label: 分母折线的图例标签（如"全 A 可交易总数"）。
        """
        ax.set_facecolor("#f5f5f5")
        if cov_df is None or cov_df.empty:
            ax.text(0.5, 0.5, "无覆盖度数据", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title, loc="center")
            return
        ts = cov_df["year_month"].dt.to_timestamp()
        # 分子：factor_count
        fc = cov_df["factor_count"].fillna(0).astype(int)
        # 分母：market_count 或 stock_pool_count
        mc_col = "market_count" if "market_count" in cov_df.columns else "stock_pool_count"
        mc = cov_df[mc_col].fillna(0).astype(int) if mc_col in cov_df.columns else pd.Series(0, index=cov_df.index)

        ax.bar(ts, mc, color=color_d, width=25, alpha=0.35, zorder=2, label=denom_label)
        ax.bar(ts, fc, color=color_n, width=25, alpha=0.85, zorder=3, label="因子覆盖数")
        _style_time_axis(ax, ts, ylabel="股票数量（只）", percent=False)
        ax.xaxis.grid(False)
        ax.set_title(title, loc="center")
        ax.text(
            0.02, 0.95, note,
            transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color="#475569",
            bbox=dict(boxstyle="round,pad=0.26", facecolor="white", edgecolor="#cbd5e1", alpha=0.95),
        )
        _style_legend(ax.legend(loc="upper left", framealpha=0.92))

    cov_all = _prep_cov_df(df_coverage_all)
    cov_trade = _prep_cov_df(df_coverage_tradeable)

    # 图2: 覆盖度四子图
    # 布局:
    #   [2.1 全市场覆盖率%]  | [2.2 全市场绝对数量: 因子覆盖 vs 全 A 可交易]
    #   [2.3 股票池覆盖率%]  | [2.4 股票池绝对数量: 因子覆盖 vs 可交易池]
    fig_cov, axes_cov = plt.subplots(2, 2, figsize=(18, 10))
    fig_cov.subplots_adjust(top=0.88, left=0.05, right=0.98, bottom=0.18, wspace=0.22, hspace=0.34)
    _apply_report_caption(fig_cov, factor_name)

    _plot_cov_rate(
        axes_cov[0, 0],
        cov_all,
        "2.1  全市场覆盖率（因子库 / 全 A 可交易）",
        "#3b82f6",
        "分子=因子库有值标的数（未限流动性），分母=当期全 A 可交易池（剔除 BJ/ST/次新/停牌）。",
    )
    _plot_cov_count(
        axes_cov[0, 1],
        cov_all,
        "2.2  全市场绝对数量（因子覆盖 vs 全 A 可交易）",
        "#3b82f6",
        "#94a3b8",
        "蓝色柱=因子有值标的数；灰色背景=全 A 可交易总数（真实 A 股分母）。",
        "全 A 可交易总数",
    )
    _plot_cov_rate(
        axes_cov[1, 0],
        cov_trade,
        "2.3  股票池覆盖率（因子库 / 可交易池）",
        "#8b5cf6",
        "分子=经流动性筛选后因子有值标的数，分母=同期可交易池（流动性口径一致）。",
    )
    _plot_cov_count(
        axes_cov[1, 1],
        cov_trade,
        "2.4  股票池绝对数量（因子覆盖 vs 可交易池）",
        "#8b5cf6",
        "#c4b5fd",
        "紫色柱=流动性筛选后因子有值数；浅紫背景=可交易池总数（同口径分母）。",
        "可交易池总数",
    )

    fig_cov.text(
        0.05, 0.04,
        "说明: 左列为覆盖率（%），右列为绝对数量对比；"
        "全市场口径（上行）分母为全 A 可交易总数，股票池口径（下行）分母为流动性筛选后的可交易池。",
        fontsize=9, ha="left", va="bottom", color="#334155",
    )
    add_plot_to_html(html_parts, fig_cov, CHART_TITLE_1_COV, chart_ctx=chart_ctx)


def plot_group_nav(grouped_returns, best_g, worst_g, html_parts, factor_name="", chart_ctx=None):
    if grouped_returns is None or grouped_returns.empty:
        return
    groups = sorted(grouped_returns["group"].unique())
    n = len(groups)
    cmap = plt.get_cmap("RdYlGn")
    x_axis = grouped_returns["year_month"].dt.to_timestamp()
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.subplots_adjust(top=0.88, left=0.06, right=0.97)
    _apply_report_caption(fig, factor_name)
    for i, g in enumerate(groups):
        g_df = grouped_returns[grouped_returns["group"] == g].sort_values("year_month")
        x = g_df["year_month"].dt.to_timestamp()
        y = g_df["净值"]
        if g == best_g:
            color = "#0f8a4b"
            lw = 2.0
            ls = "-"
            zord = 6
        elif g == worst_g:
            color = "#d62828"
            lw = 1.8
            ls = "-"
            zord = 5
        else:
            color = cmap(1.0 - i / max(n - 1, 1))
            lw = 0.95
            ls = "--"
            zord = 2
        ax.plot(x, y, color=color, linewidth=lw, linestyle=ls, zorder=zord, label=f"G{g}")
    ax.set_title(CHART_TITLE_4, loc="center")
    _style_time_axis(ax, x_axis, ylabel="净值")
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))
    _style_legend(ax.legend(
        ncol=5, loc="upper left",
        framealpha=0.92, edgecolor="#d1d9e6",
        fontsize=9, handlelength=1.8, columnspacing=1.2,
    ))
    add_plot_to_html(html_parts, fig, CHART_TITLE_4, chart_ctx=chart_ctx)


def build_group_nav_echarts_block(grouped_returns, best_g=None, worst_g=None, title="4.  因子分组分层测试（交互）"):
    """生成可交互的分组净值 ECharts 图块。"""
    if grouped_returns is None or grouped_returns.empty:
        return ""

    gdf = grouped_returns.copy()
    if "year_month" not in gdf.columns or "group" not in gdf.columns or "净值" not in gdf.columns:
        return ""

    gdf["ym"] = gdf["year_month"].astype(str)
    pivot = gdf.pivot_table(index="ym", columns="group", values="净值", aggfunc="last").sort_index()
    if pivot.empty:
        return ""

    pivot_cols = sorted([c for c in pivot.columns if pd.notna(c)])
    x_data = pivot.index.tolist()
    series = []
    for g in pivot_cols:
        try:
            g_name = f"G{int(g)}"
        except Exception:
            g_name = f"G{g}"
        vals = pivot[g].astype(float).round(6).tolist()
        series.append(
            {
                "name": g_name,
                "type": "line",
                "showSymbol": False,
                "smooth": True,
                "lineStyle": {"width": 2.2 if g in [best_g, worst_g] else 1.3},
                "data": vals,
            }
        )

    chart_id = f"group_nav_chart_{abs(hash(tuple(x_data))) % 10000000}"
    payload = {"x": x_data, "series": series}

    return f"""
<div class='echarts-wrap'>
  <h4>{title}</h4>
  <div id='{chart_id}' class='echarts-canvas'></div>
  <script>
    (function() {{
      const payload = {json.dumps(payload, ensure_ascii=False)};
      const el = document.getElementById('{chart_id}');
      if (!el || typeof echarts === 'undefined') return;
      const chart = echarts.init(el);
            if (!window.__reportCharts) window.__reportCharts = [];
            window.__reportCharts.push(chart);
      chart.setOption({{
        tooltip: {{ trigger: 'axis' }},
        legend: {{ top: 4 }},
        grid: {{ left: 54, right: 20, top: 42, bottom: 48 }},
        dataZoom: [
          {{ type: 'inside', start: 0, end: 100 }},
          {{ type: 'slider', bottom: 10, start: 0, end: 100 }}
        ],
        xAxis: {{ type: 'category', data: payload.x, axisLabel: {{ color: '#475569' }} }},
        yAxis: {{ type: 'value', axisLabel: {{ color: '#475569' }} }},
        series: payload.series
      }});
            setTimeout(function() {{ chart.resize(); }}, 100);
      window.addEventListener('resize', function() {{ chart.resize(); }});
    }})();
  </script>
</div>
"""


def save_group_nav_csv(out_dir, grouped_returns, factor_name):
    """导出分组净值透视表，方便学习图表渲染数据结构。"""
    if grouped_returns is None or grouped_returns.empty:
        return None
    gdf = grouped_returns.copy()
    if "year_month" not in gdf.columns or "group" not in gdf.columns or "净值" not in gdf.columns:
        return None
    gdf["year_month"] = gdf["year_month"].astype(str)
    wide = gdf.pivot_table(index="year_month", columns="group", values="净值", aggfunc="last").sort_index()
    wide.columns = [f"G{int(c)}" if pd.notna(c) else str(c) for c in wide.columns]
    wide = wide.reset_index().rename(columns={"year_month": "date"})
    out_path = os.path.join(out_dir, f"{factor_name}_group_nav_curve.csv")
    wide.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path

def plot_extreme_groups(best_df, worst_df, best_g, worst_g, html_parts):
    pass # covered by others

def plot_monotonicity_bar(grouped_returns, market_returns, overall_monotony, overall_pval, html_parts, factor_name="", chart_ctx=None, mono_ts_mean=0.0, mono_ts_t_nw=0.0, mono_ts_p_nw=1.0, mono_ts_positive_ratio=0.0, mono_ts_n=0):
    groups = sorted(grouped_returns["group"].unique())
    grp_abs = grouped_returns.groupby("group")["actual_ret"].mean().reindex(groups)
    # 计算相对市场超额收益
    if "actual_ret" in market_returns.columns:
        merged = grouped_returns.merge(
            market_returns[["year_month", "actual_ret"]].rename(columns={"actual_ret": "_mkt"}),
            on="year_month", how="left"
        )
        merged["_exc"] = merged["actual_ret"] - merged["_mkt"].fillna(0)
        grp_exc = merged.groupby("group")["_exc"].mean().reindex(groups)
    else:
        grp_exc = grp_abs - grp_abs.mean()
    x = np.arange(len(groups), dtype=float)
    w = 0.38
    fig, ax = plt.subplots(figsize=(12, 5))
    # 在底部预留说明文字区域
    fig.subplots_adjust(top=0.88, left=0.08, right=0.97, bottom=0.23)
    _apply_report_caption(fig, factor_name)
    ax.bar(x - w / 2, grp_abs.values, width=w, color=_BAR_ABS_COLOR,
           label="平均月绝对收益", edgecolor="white", linewidth=0.5, zorder=3)
    ax.bar(x + w / 2, grp_exc.values, width=w, color=_BAR_EXC_COLOR,
           label="平均月超额收益", edgecolor="white", linewidth=0.5, zorder=3)
    ax.axhline(0, color="#94a3b8", linewidth=0.8, zorder=2)
    ax.set_facecolor("#f5f5f5")
    ax.set_xticks(x)
    ax.set_xticklabels([f"G{g}" for g in groups])
    ax.set_title(CHART_TITLE_6, loc="center")
    ax.set_ylabel("月均收益")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v*100:.2f}%"))
    ax.xaxis.grid(False)
    pval_str = f"{overall_pval:.4e}" if overall_pval != 0 and abs(overall_pval) < 0.001 else f"{overall_pval:.4f}"
    pval_nw_str = f"{mono_ts_p_nw:.4e}" if mono_ts_p_nw != 0 and abs(mono_ts_p_nw) < 0.001 else f"{mono_ts_p_nw:.4f}"
    stats_text = (
        f"Legacy Spearman:  {overall_monotony:.4f}  (p={pval_str})\n"
        f"TS Mean Spearman: {mono_ts_mean:.4f}  (N={mono_ts_n}, pos={mono_ts_positive_ratio:.1%})\n"
        f"NW t-stat / p-value: {mono_ts_t_nw:.3f} / {pval_nw_str}"
    )
    fig.text(
        0.08,
        0.04,
        f"说明: {stats_text.replace(chr(10), ' | ')}",
        fontsize=9,
        ha="left",
        va="bottom",
        color="#334155",
    )
    _style_legend(ax.legend(loc="upper right", framealpha=0.92, edgecolor="#d1d9e6"))
    add_plot_to_html(html_parts, fig, CHART_TITLE_6, chart_ctx=chart_ctx)

def plot_ic_trend(df_ic, html_parts, rank=False, factor_name="", combined=False, chart_ctx=None):
    if df_ic is None or df_ic.empty:
        return
    if combined:
        # 合并 IC 和 Rank IC 成双列子图
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        fig.subplots_adjust(left=0.06, right=0.97, top=0.88, bottom=0.24, wspace=0.32)
        _apply_report_caption(fig, factor_name)
        for ax_i, (col, lbl, num, pos_c) in enumerate([
            ("ic", "IC", "4a", "#3b82f6"),
            ("rank_ic", "Rank IC", "4b", "#059669")
        ]):
            ax = axes[ax_i]
            if col not in df_ic.columns:
                ax.set_visible(False)
                continue
            ts = df_ic["year_month"].dt.to_timestamp()
            vals = df_ic[col].fillna(0)
            bar_colors = [pos_c if v >= 0 else "#ef4444" for v in vals]
            ax.bar(ts, vals, color=bar_colors, width=25, alpha=0.82, zorder=3)
            roll = pd.Series(vals.values).rolling(window=6, min_periods=1).mean()
            ax.plot(ts, np.asarray(roll.values, dtype=float), color="#1e293b", linewidth=1.2, label="6期滚动均值", zorder=4)
            ax.axhline(0, color="#94a3b8", linewidth=0.7, zorder=2)
            nz = vals[vals != 0]
            mean_v = nz.mean() if len(nz) > 0 else 0
            ax.axhline(mean_v, color="#d97706", linewidth=1.0, linestyle="--",
                       label=f"均值  {mean_v:.4f}", zorder=4)
            ax.set_title("5.1  Pearson IC 测算走势" if ax_i == 0 else "5.2  Rank IC 测算走势", loc="center")
            _style_time_axis(ax, ts, ylabel=lbl)
            ax.xaxis.grid(False)
            _style_legend(ax.legend(loc="upper right", fontsize=8, framealpha=0.92))

            total_n = len(vals)
            if total_n > 0:
                pos_ratio = (vals > 0).sum() / total_n
                neg_ratio = (vals < 0).sum() / total_n
                ax.text(
                    0.5,
                    -0.24,
                    f"正值占比: {pos_ratio:.2%}   负值占比: {neg_ratio:.2%}   (总期数={total_n})",
                    transform=ax.transAxes,
                    ha="center",
                    va="top",
                    fontsize=9,
                    color="#475569",
                    clip_on=False,
                )
        add_plot_to_html(html_parts, fig, CHART_TITLE_5, chart_ctx=chart_ctx)
    else:
        # 单图模式（向后兼容）
        y_col = "rank_ic" if rank else "ic"
        if y_col not in df_ic.columns:
            return
        fig, ax = plt.subplots(figsize=(12, 4))
        fig.subplots_adjust(top=0.88, bottom=0.24)
        _apply_report_caption(fig, factor_name)
        ts = df_ic["year_month"].dt.to_timestamp()
        vals = df_ic[y_col].fillna(0)
        pos_c = "#3b82f6" if not rank else "#059669"
        ax.bar(ts, vals, color=[pos_c if v >= 0 else "#ef4444" for v in vals],
               width=25, alpha=0.82, zorder=3)
        roll = pd.Series(vals.values).rolling(window=6, min_periods=1).mean()
        ax.plot(ts, np.asarray(roll.values, dtype=float), color="#1e293b", linewidth=1.2, label="6期滚动均值", zorder=4)
        ax.axhline(0, color="#94a3b8", linewidth=0.7, zorder=2)
        lbl = "Rank IC" if rank else "IC"
        ax.set_title(CHART_TITLE_5 if not rank else "5.2  Rank IC 测算走势", loc="center")
        _style_time_axis(ax, ts, ylabel=lbl)
        ax.xaxis.grid(False)
        _style_legend(ax.legend(loc="upper right", framealpha=0.92))

        total_n = len(vals)
        if total_n > 0:
            pos_ratio = (vals > 0).sum() / total_n
            neg_ratio = (vals < 0).sum() / total_n
            ax.text(
                0.5,
                -0.24,
                f"正值占比: {pos_ratio:.2%}   负值占比: {neg_ratio:.2%}   (总期数={total_n})",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=9,
                color="#475569",
                clip_on=False,
            )
        add_plot_to_html(html_parts, fig, f"5.  {lbl} 时序走势", chart_ctx=chart_ctx)

def plot_ir_rank_ir_bar(df_ic, html_parts):
    pass

def plot_group_win_rates(df_trades, market_returns, html_parts):
    pass

def plot_daily_nav(daily_portfolio_df, html_parts, chart_ctx=None):
    if daily_portfolio_df is not None and not daily_portfolio_df.empty:
        fig, ax = plt.subplots(figsize=(12, 4.6))
        fig.subplots_adjust(left=0.07, right=0.97, top=0.90)
        ts = pd.to_datetime(daily_portfolio_df["date"])
        ax.plot(ts, daily_portfolio_df["nav"], color="#2563eb", linewidth=1.4, label="日频净值")
        _style_time_axis(ax, ts, ylabel="净值")
        ax.set_title(CHART_TITLE_DAILY, loc="center")
        _style_legend(ax.legend(loc="upper left"))
        add_plot_to_html(html_parts, fig, CHART_TITLE_DAILY, chart_ctx=chart_ctx)

def build_strategy_context(test_name, start_month, end_month, benchmark, factor_cols):
    cols_str = ", ".join(factor_cols) if factor_cols else "N/A"
    return f"<div class='header-card'><h2>{test_name}</h2><p><b>回测区间:</b> {start_month} 到 {end_month} &nbsp;|&nbsp; <b>基准:</b> {benchmark} &nbsp;|&nbsp; <b>使用特征:</b> {cols_str}</p></div>"

def save_extreme_groups_csv(out_dir, best_df, worst_df, best_g, worst_g, factor_name):
    if best_df is None or best_df.empty or worst_df is None or worst_df.empty:
        return None

    export_df = pd.merge(
        best_df[["year_month", "actual_ret", "净值", "excess_ret", "excess_cum"]].rename(
            columns={
                "actual_ret": f"G{best_g}_ret",
                "净值": f"G{best_g}_nav",
                "excess_ret": f"G{best_g}_excess_ret",
                "excess_cum": f"G{best_g}_excess_nav",
            }
        ),
        worst_df[["year_month", "actual_ret", "净值", "excess_ret", "excess_cum"]].rename(
            columns={
                "actual_ret": f"G{worst_g}_ret",
                "净值": f"G{worst_g}_nav",
                "excess_ret": f"G{worst_g}_excess_ret",
                "excess_cum": f"G{worst_g}_excess_nav",
            }
        ),
        on="year_month",
        how="outer",
    ).sort_values("year_month")

    out_path = os.path.join(out_dir, f"{factor_name}_extreme_groups_monthly.csv")
    export_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


def plot_factor_coverage(df_coverage, html_parts, factor_name=""):
    if df_coverage is None or df_coverage.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.subplots_adjust(top=0.88, left=0.08, right=0.97)
    _apply_report_caption(fig, factor_name)
    
    ts = df_coverage["year_month"].dt.to_timestamp()
    vals = df_coverage["coverage"].fillna(0)
    
    ax.bar(ts, vals, color="#0ea5e9", width=25, alpha=0.82, zorder=3)
    avg_cov = vals.mean()
    ax.axhline(avg_cov, color="#d97706", linewidth=1.0, linestyle="--",
               label=f"均值  {avg_cov*100:.1f}%", zorder=4)
               
    ax.set_title("7.  因子覆盖度测算走势", loc="center")
    _style_time_axis(ax, ts, ylabel="覆盖度频次", percent=True)
    ax.xaxis.grid(False)
    ax.set_ylim(0, 1.05)
    _style_legend(ax.legend(loc="lower right", framealpha=0.92))
    
    add_plot_to_html(html_parts, fig, "7.  因子覆盖度走势")


def generate_all_charts(
    html_parts,
    df1,
    factor_df,
    factor_name,
    best_df,
    worst_df,
    grouped_returns,
    df_trades,
    market_returns,
    df_ic,
    best_g,
    worst_g,
    kpis,
    out_dir,
    delay_days=0,
    df_coverage_all=None,
    df_coverage_tradeable=None,
    df_coverage_all_cs=None,
    df_coverage_tradeable_cs=None,
    df1_all=None,
    source_factor_col=None,
):
    # 收益概况页：图表优先
    chart_ctx = ReportChartContext(out_dir)
    summary_blocks = []
    # 1) 第1块：因子覆盖度 + 因子暴露
    plot_coverage_and_exposure(
        df_coverage_all,
        df_coverage_tradeable,
        df_coverage_all_cs,
        df_coverage_tradeable_cs,
        df1,
        factor_name,
        summary_blocks,
        chart_ctx=chart_ctx,
        df1_all=df1_all,
        source_factor_col=source_factor_col,
    )

    # 2) 第2块：核心绩效（净值与对冲）
    plot_core_performance(
        best_df,
        kpis.get("ls_cum"),
        kpis.get("drawdowns"),
        best_g,
        summary_blocks,
        delay_days=delay_days,
        factor_name=factor_name,
        df_coverage=None,
        chart_ctx=chart_ctx,
    )

    # 3+) 后续图顺延
    interactive_group_nav_html = build_group_nav_echarts_block(grouped_returns, best_g=best_g, worst_g=worst_g)
    if interactive_group_nav_html:
        summary_blocks.append(interactive_group_nav_html)

    plot_ic_trend(df_ic, summary_blocks, combined=True, factor_name=factor_name, chart_ctx=chart_ctx)
    plot_monotonicity_bar(
        grouped_returns,
        market_returns,
        kpis.get("overall_monotony", 0),
        kpis.get("overall_pval", 0),
        summary_blocks,
        factor_name=factor_name,
        chart_ctx=chart_ctx,
        mono_ts_mean=kpis.get("mono_ts_mean", 0),
        mono_ts_t_nw=kpis.get("mono_ts_t_nw", 0),
        mono_ts_p_nw=kpis.get("mono_ts_p_nw", 1),
        mono_ts_positive_ratio=kpis.get("mono_ts_positive_ratio", 0),
        mono_ts_n=kpis.get("mono_ts_n", 0),
    )

    summary_html = "".join(summary_blocks)
    html_parts.append(
        build_section(
            title="收益概况",
            content_html=summary_html,
            desc=f"关键收益与风险图表（Delay={delay_days}）",
            tab_id="tab-summary",
        )
    )

    # 当前模式：只保留收益概况，不追加交易详情/每日表现/运行日志章节。
    save_group_nav_csv(out_dir, grouped_returns, factor_name)
    # 不需要生成极值分组月度对比CSV，已有各分组净值曲线
    # return save_extreme_groups_csv(out_dir, best_df, worst_df, best_g, worst_g, factor_name)
