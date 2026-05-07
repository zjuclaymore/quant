import time
from datetime import datetime, timedelta

import akshare as ak
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

"""
涨跌停情绪看板 (30-Day Plotly Version)
数据源: ak.stock_zt_pool_em, ak.stock_zt_pool_dtgc_em
功能: 扩展至 30 个交易日，展示中长期情绪周期。
"""

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "special_data_analysis" / "data"
OUTPUT_DIR = BASE_DIR / "special_data_analysis" / "visual_output" / "sentiment"

TARGET_HISTORY_DAYS = 30
MAX_SCAN_BUSINESS_DAYS = 120
REMOTE_LIMIT_HINT = "最近 30 个交易日"
FAIL_STREAK_LIMIT = 5


def _iter_business_dates(target_date, limit=MAX_SCAN_BUSINESS_DAYS):
    current = datetime.strptime(target_date, "%Y%m%d")
    yielded = 0
    while yielded < limit:
        if current.weekday() < 5:
            yield current.strftime("%Y%m%d")
            yielded += 1
        current -= timedelta(days=1)


def _normalize_cache_frame(df):
    required_columns = ["日期", "涨停家数", "跌停家数", "净涨停"]
    if df is None or df.empty:
        return pd.DataFrame(columns=required_columns)

    normalized = df.copy()
    normalized.columns = [str(column).strip() for column in normalized.columns]
    for column in ("日期", "涨停家数", "跌停家数"):
        if column not in normalized.columns:
            return pd.DataFrame(columns=required_columns)

    normalized["日期"] = normalized["日期"].astype(str).str.extract(r"(\d{8})", expand=False)
    normalized = normalized.dropna(subset=["日期"])
    normalized["涨停家数"] = pd.to_numeric(normalized["涨停家数"], errors="coerce").fillna(0).astype(int)
    normalized["跌停家数"] = pd.to_numeric(normalized["跌停家数"], errors="coerce").fillna(0).astype(int)
    if "净涨停" not in normalized.columns:
        normalized["净涨停"] = normalized["涨停家数"] - normalized["跌停家数"]
    normalized["净涨停"] = pd.to_numeric(normalized["净涨停"], errors="coerce").fillna(
        normalized["涨停家数"] - normalized["跌停家数"]
    ).astype(int)
    normalized = normalized[required_columns].drop_duplicates(subset=["日期"], keep="last")
    return normalized.sort_values("日期").reset_index(drop=True)


def _load_stats_cache():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_frames = []
    for cache_path in sorted(DATA_DIR.glob("limit_stats*.csv")):
        try:
            cache_frames.append(_normalize_cache_frame(pd.read_csv(cache_path)))
        except Exception as exc:
            print(f"[WARN] 读取缓存失败 {cache_path.name}: {exc}")
    if not cache_frames:
        return pd.DataFrame(columns=["日期", "涨停家数", "跌停家数", "净涨停"])
    combined = pd.concat(cache_frames, ignore_index=True)
    return _normalize_cache_frame(combined)


def _save_stats_cache(df):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = DATA_DIR / "limit_stats_cache.csv"
    _normalize_cache_frame(df).to_csv(cache_path, index=False, encoding="utf_8_sig")
    return cache_path


def get_limit_stats_plotly(target_date=None, output_dir=None):
    target_date = target_date or datetime.now().strftime("%Y%m%d")
    print(f"正在获取截至 {target_date} 的 30 日涨跌停统计数据...")
    cache_df = _load_stats_cache()
    cache_map = {
        row["日期"]: {
            "日期": row["日期"],
            "涨停家数": int(row["涨停家数"]),
            "跌停家数": int(row["跌停家数"]),
            "净涨停": int(row["净涨停"]),
        }
        for _, row in cache_df.iterrows()
    }

    results = {}
    new_rows = []
    failures = []
    remote_disabled = False
    remote_boundary_date = None
    failure_streak = 0

    for trade_date in _iter_business_dates(target_date):
        if trade_date in cache_map:
            results[trade_date] = cache_map[trade_date]
            failure_streak = 0
        elif not remote_disabled:
            try:
                df_zt = ak.stock_zt_pool_em(date=trade_date)
                zt_count = len(df_zt) if df_zt is not None else 0
                df_dt = ak.stock_zt_pool_dtgc_em(date=trade_date)
                dt_count = len(df_dt) if df_dt is not None else 0
                failure_streak = 0

                if zt_count > 0 or dt_count > 0:
                    row = {
                        "日期": trade_date,
                        "涨停家数": zt_count,
                        "跌停家数": dt_count,
                        "净涨停": zt_count - dt_count,
                    }
                    results[trade_date] = row
                    cache_map[trade_date] = row
                    new_rows.append(row)
                time.sleep(0.05)
            except Exception as exc:
                message = str(exc)
                if REMOTE_LIMIT_HINT in message:
                    remote_disabled = True
                    remote_boundary_date = trade_date
                    print(f"[INFO] 涨跌停统计命中历史边界 {trade_date}，后续仅使用本地缓存。")
                else:
                    failure_streak += 1
                    failures.append((trade_date, message))
                    print(f"[WARN] 涨跌停统计抓取失败 {trade_date}: {message}")
                    if failure_streak >= FAIL_STREAK_LIMIT:
                        remote_disabled = True
                        remote_boundary_date = trade_date
                        print(f"[INFO] 涨跌停统计连续 {FAIL_STREAK_LIMIT} 个日期远端失败，后续仅使用本地缓存。")

        if len(results) >= TARGET_HISTORY_DAYS:
            break

    if not results:
        raise RuntimeError("Limit stats analysis produced no rows. No dashboard was generated.")

    if new_rows:
        cache_df = pd.concat([cache_df, pd.DataFrame(new_rows)], ignore_index=True)
        cache_path = _save_stats_cache(cache_df)
        print(f"[INFO] 涨跌停统计缓存已更新: {cache_path}")

    res_df = pd.DataFrame(results.values()).sort_values("日期").tail(TARGET_HISTORY_DAYS)
    res_df["display_date"] = res_df["日期"].apply(lambda value: f"{value[4:6]}/{value[6:8]}")

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=res_df["display_date"], y=res_df["涨停家数"],
        name="涨停家数", marker_color="#ef5350", opacity=0.8,
    ))

    fig.add_trace(go.Bar(
        x=res_df["display_date"], y=-res_df["跌停家数"],
        name="跌停家数", marker_color="#66bb6a", opacity=0.8,
    ))

    fig.add_trace(go.Scatter(
        x=res_df["display_date"], y=res_df["净涨停"],
        name="情绪差值 (净涨停)",
        line=dict(color="#ffca28", width=3),
        mode="lines+markers",
    ))

    fig.update_layout(
        title={
            'text': f"<b>市场情绪月度极值看板 (涨跌停对比)</b>",
            'x': 0.5, 'y': 0.95, 'xanchor': 'center',
            'font': dict(size=24, color='#37474f')
        },
        annotations=[dict(
            text=f"统计区间: {res_df['日期'].iloc[0]} 至 {res_df['日期'].iloc[-1]}",
            showarrow=False, xref="paper", yref="paper", x=0.5, y=1.05, font=dict(color='gray')
        )],
        xaxis=dict(title="交易日期", type='category', tickangle=-45),
        yaxis=dict(title="家数", zeroline=True, zerolinewidth=2, zerolinecolor='black', gridcolor='#eceff1'),
        barmode='relative',
        template='plotly_white',
        height=650,
        margin=dict(l=50, r=50, t=100, b=120),
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5)
    )

    out = Path(output_dir) if output_dir else OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    output_path = out / "limit_stats_dashboard.html"
    fig.write_html(output_path)
    if len(res_df) < TARGET_HISTORY_DAYS:
        print(f"[WARN] 涨跌停统计仅拿到 {len(res_df)} 个交易日，未满 {TARGET_HISTORY_DAYS} 个。")
    if remote_boundary_date:
        print(f"[INFO] 涨跌停统计远端抓取在 {remote_boundary_date} 后停止，历史补齐依赖缓存。")
    if failures:
        print(f"[WARN] 涨跌停统计存在 {len(failures)} 个失败日期。")
    print(f"30日情绪看板已生成: {output_path}")
    return output_path

if __name__ == "__main__":
    import sys
    get_limit_stats_plotly(sys.argv[1] if len(sys.argv) > 1 else None)
