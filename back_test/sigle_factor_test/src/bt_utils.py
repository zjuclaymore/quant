"""
回测辅助工具模块 (Backtest Utilities Module) - Basic version
"""

import io
import os
import base64
import warnings
import datetime
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

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

CHART_TITLE_1 = "1.  策略净值与基准对比"
CHART_TITLE_2 = "2.  策略对冲组合净值"
CHART_TITLE_3 = "3.  因子分组分层测试"
CHART_TITLE_4 = "4.  因子 IC / Rank IC 测算走势"
CHART_TITLE_5 = "5.  因子单调性与分组超额"
CHART_TITLE_6 = "6.  因子暴露/打分分布"
CHART_TITLE_DAILY = "日频策略净值走势"


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


def add_plot_to_html(html_parts, fig, title):
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
    return f"<div id='{tab_id}' class='tab-content'><div class='section'><h2>{title}</h2>{desc_html}{content_html}</div></div>"


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
    
    # 提取 tabs html
    tabs_html = """
    <div class="tabs">
      <button class="tab-btn active" onclick="openTab(event, 'tab-summary')">收益概况</button>
      <button class="tab-btn" onclick="openTab(event, 'tab-trades')">交易详情</button>
      <button class="tab-btn" onclick="openTab(event, 'tab-daily')">每日表现</button>
      <button class="tab-btn" onclick="openTab(event, 'tab-logs')">运行日志</button>
    </div>
    """
    
    # default script to click the first tab on load, since inline style=block is set but we want active state initialized completely
    script_html = """
    <script>
      document.addEventListener("DOMContentLoaded", function() {
        document.querySelector('.tab-btn').click();
      });
    </script>
    """
    
    # assume the first part in html_parts is the context_html header
    context_part = html_parts[0] if len(html_parts) > 0 else ""
    sections_html = "".join(html_parts[1:]) if len(html_parts) > 1 else ""
    
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>{factor_name} 回测报告</title>
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


def neutralize_factor(df, factor_col, mv_col="lncap", ind_col=None):
    resid_series = pd.Series(index=df.index, dtype=float)
    for ym, group in df.groupby("year_month"):
        valid_idx = group[factor_col].notna() & group[mv_col].notna()
        if ind_col: valid_idx &= group[ind_col].notna()
        subset = group[valid_idx]
        if len(subset) < 10: continue
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
        except Exception as e:
            warnings.warn(f"Neutralization failed at {ym}: {e}")
    return resid_series


def compute_backtest_kpis(best_df, worst_df, df_trades, df_ic, market_returns, cal, best_g, worst_g, df1, df_daily_nunique):
    total_months = len(cal)
    ann_factor = 12 / total_months if total_months > 0 else 1
    best_ret_ann = (best_df["净值"].iloc[-1]) ** ann_factor - 1 if len(best_df) > 0 else 0
    best_vol_ann = best_df["actual_ret"].std() * np.sqrt(12)
    sharpe = best_ret_ann / best_vol_ann if best_vol_ann != 0 else np.nan
    roll_max = best_df["净值"].rolling(window=len(best_df), min_periods=1).max()
    drawdowns = (best_df["净值"] / roll_max - 1).astype(float)
    max_drawdown = drawdowns.min()
    ic_mean = df_ic["ic"].mean() if not df_ic.empty else 0
    ic_std = df_ic["ic"].std() if not df_ic.empty else 0
    ric_mean = df_ic["rank_ic"].mean() if not df_ic.empty else 0
    ric_std = df_ic["rank_ic"].std() if not df_ic.empty else 0
    ir = ic_mean / ic_std if ic_std and not np.isnan(ic_std) else 0
    rank_ir = ric_mean / ric_std if ric_std and not np.isnan(ric_std) else 0
    overall_grp_ret = df_trades.groupby("group")["actual_ret"].mean()
    overall_monotony, overall_pval = (spearmanr(overall_grp_ret.index, overall_grp_ret.values) if len(overall_grp_ret) > 5 else (0, 0))
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
        "win_rate": win_rate, "max_drawdown_period": "N/A", "calmar": calmar
    }


def plot_core_performance(best_df, ls_cum, drawdowns, best_g, html_parts, delay_days=0, factor_name="", strategy_label=None, show_ls=True):
    if show_ls:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.subplots_adjust(left=0.06, right=0.97, top=0.88, wspace=0.32)
    else:
        fig, ax_single = plt.subplots(1, 1, figsize=(10, 5))
        fig.subplots_adjust(left=0.1, right=0.95, top=0.88)
        axes = [ax_single, None]
        
    _apply_report_caption(fig, factor_name)
    # --- 左图：策略组合净值 ---
    ax = axes[0]
    ts = best_df["year_month"].dt.to_timestamp()
    label = strategy_label if strategy_label else f"G{best_g} 策略组合"
    ax.plot(ts, best_df["净值"], color="#16a34a", linewidth=1.6, label=label)
    if "bench_nav" in best_df.columns:
        ax.plot(ts, best_df["bench_nav"], color="#94a3b8", linewidth=1.0, linestyle="--", label="基准")
    ax.set_title(CHART_TITLE_1, loc="center")
    _style_time_axis(ax, ts, ylabel="净值")
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f"))
    _style_legend(ax.legend(loc="upper left", ncol=2))
    # --- 右图：多空/对冲净值 ---
    if show_ls:
        ax2 = axes[1]
        if ls_cum is not None and len(ls_cum) > 0:
            if hasattr(ls_cum.index, "to_timestamp"):
                lt = ls_cum.index.to_timestamp()
            else:
                lt = pd.to_datetime(ls_cum.index.astype(str))
            ax2.plot(lt, ls_cum.values, color="#7c3aed", linewidth=1.6, label="对冲组合/多空净值")
            ax2.axhline(1.0, color="#94a3b8", linewidth=0.8, linestyle="--")
        ax2.set_title(CHART_TITLE_2, loc="center")
        _style_time_axis(ax2, lt if ls_cum is not None and len(ls_cum) > 0 else ts, ylabel="净值")
        _style_legend(ax2.legend(loc="upper left"))
    add_plot_to_html(html_parts, fig, "核心绩效指标报告")

def plot_pre_post_distribution(df_none, df1, factor_name, html_parts):
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
    ax.set_title(CHART_TITLE_6, loc="center")
    ax.set_xlabel("因子值")
    ax.set_ylabel("频次")
    ax.set_facecolor("#f5f5f5")
    ax.xaxis.grid(False)
    _style_legend(ax.legend(loc="upper right", framealpha=0.92))
    add_plot_to_html(html_parts, fig, CHART_TITLE_6)

def plot_group_nav(grouped_returns, best_g, worst_g, html_parts, factor_name=""):
    groups = sorted(grouped_returns["group"].unique())
    n = len(groups)
    cmap = plt.cm.RdYlGn
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
    ax.set_title(CHART_TITLE_3, loc="center")
    _style_time_axis(ax, x, ylabel="净值")
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))
    _style_legend(ax.legend(
        ncol=5, loc="upper left",
        framealpha=0.92, edgecolor="#d1d9e6",
        fontsize=9, handlelength=1.8, columnspacing=1.2,
    ))
    add_plot_to_html(html_parts, fig, CHART_TITLE_3)

def plot_extreme_groups(best_df, worst_df, best_g, worst_g, html_parts):
    pass # covered by others

def plot_monotonicity_bar(grouped_returns, market_returns, overall_monotony, overall_pval, html_parts, factor_name=""):
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
    x = np.arange(len(groups))
    w = 0.38
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.subplots_adjust(top=0.88, left=0.08, right=0.97)
    _apply_report_caption(fig, factor_name)
    ax.bar(x - w / 2, grp_abs.values, width=w, color=_BAR_ABS_COLOR,
           label="平均月绝对收益", edgecolor="white", linewidth=0.5, zorder=3)
    ax.bar(x + w / 2, grp_exc.values, width=w, color=_BAR_EXC_COLOR,
           label="平均月超额收益", edgecolor="white", linewidth=0.5, zorder=3)
    ax.axhline(0, color="#94a3b8", linewidth=0.8, zorder=2)
    ax.set_facecolor("#f5f5f5")
    ax.set_xticks(x)
    ax.set_xticklabels([f"G{g}" for g in groups])
    ax.set_title(CHART_TITLE_5, loc="center")
    ax.set_ylabel("月均收益")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v*100:.2f}%"))
    ax.xaxis.grid(False)
    pval_str = f"{overall_pval:.4e}" if overall_pval != 0 and abs(overall_pval) < 0.001 else f"{overall_pval:.4f}"
    stats_text = f"Spearman Monotony:  {overall_monotony:.4f}\nP-value:  {pval_str}"
    ax.text(0.02, 0.97, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="#c7d2dc", alpha=0.95))
    _style_legend(ax.legend(loc="upper right", framealpha=0.92, edgecolor="#d1d9e6"))
    add_plot_to_html(html_parts, fig, CHART_TITLE_5)

def plot_ic_trend(df_ic, html_parts, rank=False, factor_name="", combined=False):
    if df_ic is None or df_ic.empty:
        return
    if combined:
        # 合并 IC 和 Rank IC 成双列子图
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        fig.subplots_adjust(left=0.06, right=0.97, top=0.88, wspace=0.32)
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
            ax.plot(ts, roll.values, color="#1e293b", linewidth=1.2, label="6期滚动均值", zorder=4)
            ax.axhline(0, color="#94a3b8", linewidth=0.7, zorder=2)
            nz = vals[vals != 0]
            mean_v = nz.mean() if len(nz) > 0 else 0
            ax.axhline(mean_v, color="#d97706", linewidth=1.0, linestyle="--",
                       label=f"均值  {mean_v:.4f}", zorder=4)
            ax.set_title("4.1  Pearson IC 测算走势" if ax_i == 0 else "4.2  Rank IC 测算走势", loc="center")
            _style_time_axis(ax, ts, ylabel=lbl)
            ax.xaxis.grid(False)
            _style_legend(ax.legend(loc="upper right", fontsize=8, framealpha=0.92))
        add_plot_to_html(html_parts, fig, CHART_TITLE_4)
    else:
        # 单图模式（向后兼容）
        y_col = "rank_ic" if rank else "ic"
        if y_col not in df_ic.columns:
            return
        fig, ax = plt.subplots(figsize=(12, 4))
        fig.subplots_adjust(top=0.88)
        _apply_report_caption(fig, factor_name)
        ts = df_ic["year_month"].dt.to_timestamp()
        vals = df_ic[y_col].fillna(0)
        pos_c = "#3b82f6" if not rank else "#059669"
        ax.bar(ts, vals, color=[pos_c if v >= 0 else "#ef4444" for v in vals],
               width=25, alpha=0.82, zorder=3)
        roll = pd.Series(vals.values).rolling(window=6, min_periods=1).mean()
        ax.plot(ts, roll.values, color="#1e293b", linewidth=1.2, label="6期滚动均值", zorder=4)
        ax.axhline(0, color="#94a3b8", linewidth=0.7, zorder=2)
        lbl = "Rank IC" if rank else "IC"
        ax.set_title(CHART_TITLE_4 if not rank else "4.2  Rank IC 测算走势", loc="center")
        _style_time_axis(ax, ts, ylabel=lbl)
        ax.xaxis.grid(False)
        _style_legend(ax.legend(loc="upper right", framealpha=0.92))
        add_plot_to_html(html_parts, fig, f"4.  {lbl} 时序走势")

def plot_ir_rank_ir_bar(df_ic, html_parts):
    pass

def plot_group_win_rates(df_trades, market_returns, html_parts):
    pass

def plot_daily_nav(daily_portfolio_df, html_parts):
    if daily_portfolio_df is not None and not daily_portfolio_df.empty:
        fig, ax = plt.subplots(figsize=(12, 4.6))
        fig.subplots_adjust(left=0.07, right=0.97, top=0.90)
        ts = pd.to_datetime(daily_portfolio_df["date"])
        ax.plot(ts, daily_portfolio_df["nav"], color="#2563eb", linewidth=1.4, label="日频净值")
        _style_time_axis(ax, ts, ylabel="净值")
        ax.set_title(CHART_TITLE_DAILY, loc="center")
        _style_legend(ax.legend(loc="upper left"))
        add_plot_to_html(html_parts, fig, CHART_TITLE_DAILY)

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
    df_coverage=None,
):
    # 收益概况页：图表优先
    summary_blocks = []
    # 按顺序 1→2→3→4→5→6→7 生成各图表
    plot_core_performance(best_df, kpis.get("ls_cum"), kpis.get("drawdowns"), best_g, summary_blocks, delay_days=delay_days, factor_name=factor_name)
    plot_group_nav(grouped_returns, best_g, worst_g, summary_blocks, factor_name=factor_name)
    plot_ic_trend(df_ic, summary_blocks, combined=True, factor_name=factor_name)
    plot_monotonicity_bar(grouped_returns, market_returns, kpis.get("overall_monotony", 0), kpis.get("overall_pval", 0), summary_blocks, factor_name=factor_name)
    plot_pre_post_distribution(None, df1, factor_name, summary_blocks)
    
    # 插入新的因子覆盖度走势图
    plot_factor_coverage(df_coverage, summary_blocks, factor_name=factor_name)

    summary_html = "".join(summary_blocks)
    html_parts.append(
        build_section(
            title="收益概况",
            content_html=summary_html,
            desc=f"关键收益与风险图表（Delay={delay_days}）",
            tab_id="tab-summary",
        )
    )

    # 交易详情页：分组收益与交易样本表
    trade_blocks = []
    grouped_table = grouped_returns.copy()
    if "year_month" in grouped_table.columns:
        grouped_table["year_month"] = grouped_table["year_month"].astype(str)
    add_table_to_html(trade_blocks, grouped_table, "分组月度收益与净值", max_rows=200)

    trades_table = df_trades.copy()
    if "year_month" in trades_table.columns:
        trades_table["year_month"] = trades_table["year_month"].astype(str)
    add_table_to_html(trade_blocks, trades_table, "交易明细样本", max_rows=200)

    html_parts.append(
        build_section(
            title="交易详情",
            content_html="".join(trade_blocks),
            desc="分层收益与逐笔交易样本",
            tab_id="tab-trades",
        )
    )

    # 每日持仓和收益页：市场组合及 IC 数据
    daily_blocks = []
    market_table = market_returns.copy()
    if "year_month" in market_table.columns:
        market_table["year_month"] = market_table["year_month"].astype(str)
    add_table_to_html(daily_blocks, market_table, "市场组合月度收益", max_rows=200)

    ic_table = df_ic.copy()
    if not ic_table.empty and "year_month" in ic_table.columns:
        ic_table["year_month"] = ic_table["year_month"].astype(str)
    add_table_to_html(daily_blocks, ic_table, "IC / RankIC 序列", max_rows=240)

    html_parts.append(
        build_section(
            title="每日持仓和收益",
            content_html="".join(daily_blocks),
            desc="日频/序列指标汇总",
            tab_id="tab-daily",
        )
    )

    # 日志页
    log_blocks = []
    add_log_block(
        log_blocks,
        f"因子: {factor_name}\n最优组: G{best_g}, 最差组: G{worst_g}\n"
        f"IC={kpis.get('ic_mean', 0):.4f}, IR={kpis.get('ir', 0):.4f}, "
        f"RankIC={kpis.get('ric_mean', 0):.4f}, RankIR={kpis.get('rank_ir', 0):.4f}\n"
        f"Sharpe={kpis.get('sharpe', 0):.4f}, Calmar={kpis.get('calmar', 0):.4f}",
        title="回测摘要日志",
    )
    html_parts.append(
        build_section(
            title="输出日志",
            content_html="".join(log_blocks),
            desc="运行摘要与关键指标日志",
            tab_id="tab-logs",
        )
    )

    return save_extreme_groups_csv(out_dir, best_df, worst_df, best_g, worst_g, factor_name)
