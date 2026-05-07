import html
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "output"
LOG_DIR = OUTPUT_ROOT / "logs"
SPECIAL_SCRIPTS_DIR = PROJECT_ROOT / "special_data_analysis" / "scripts"
INDEX_SRC_DIR = PROJECT_ROOT / "index_distortion_analysis" / "src"

for import_dir in (SPECIAL_SCRIPTS_DIR, INDEX_SRC_DIR):
    import_path = str(import_dir)
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from fetch_index_1min_akshare import fetch_index_1min_sina
from plot_cross_section_mins import visualize_cross_section_mins
from divergence_detector import detect_divergence
import limit_stats_analysis
import limit_ladder_analysis
import limit_up_details
import risk_analysis


def _normalize_target_dates(target_date=None):
    if not target_date:
        compact = datetime.now().strftime("%Y%m%d")
    elif len(target_date) == 8 and target_date.isdigit():
        compact = target_date
    elif len(target_date) == 10 and target_date[4] == "-" and target_date[7] == "-":
        compact = target_date.replace("-", "")
    else:
        raise ValueError("target_date must be in YYYYMMDD or YYYY-MM-DD format.")
    dashed = f"{compact[:4]}-{compact[4:6]}-{compact[6:8]}"
    return compact, dashed


def _write_log(compact_date, results):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"pipeline_{compact_date}.log"
    lines = [f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Pipeline summary for {compact_date}"]
    for r in results:
        lines.append(f"  [{r['status'].upper()}] {r['name']} | artifact: {r.get('artifact', '-')} | {r.get('detail', '')}")
    log_path.write_text("\n".join(lines), encoding="utf-8")
    return log_path


def _render_dashboard(target_date):
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>量化复盘中心 - {target_date}</title>
    <link href="https://cdn.bootcdn.net/ajax/libs/twitter-bootstrap/5.1.3/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {{ background: linear-gradient(180deg, #f3f6f4 0%, #eef2f7 100%); font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }}
        .glass-card {{ background: rgba(255,255,255,0.82); backdrop-filter: blur(10px); border-radius: 15px; border: 1px solid rgba(255,255,255,0.3); box-shadow: 0 8px 32px 0 rgba(31,38,135,0.15); margin-bottom: 30px; padding: 20px; }}
        .section-title {{ border-left: 5px solid #C62828; padding-left: 15px; margin-bottom: 25px; color: #263238; font-weight: bold; }}
        iframe {{ border: none; width: 100%; border-radius: 10px; }}
    </style>
</head>
<body>
    <nav class="navbar navbar-dark sticky-top bg-dark flex-md-nowrap p-2 shadow">
        <a class="navbar-brand col-md-3 col-lg-2 me-0 px-3" href="#">Quant Master Dashboard</a>
        <div class="navbar-nav">
            <div class="nav-item text-nowrap px-3 text-white">日期: {target_date}</div>
        </div>
    </nav>
    <div class="container-fluid">
        <div class="row">
            <main class="col-md-12 ms-sm-auto col-lg-12 px-md-4">
                <div class="d-flex justify-content-between flex-wrap flex-md-nowrap align-items-center pt-3 pb-2 mb-3 border-bottom">
                    <h1 class="h2">市场全景量化复盘</h1>
                    <div class="text-muted">生成时间: {generated_at}</div>
                </div>

                <h3 class="section-title">0. 宽基指数日内路径 (失真度观察)</h3>
                <div class="glass-card">
                    <h5>指数归一化走势对比</h5>
                    <iframe src="index/index_distortion_dashboard.html" style="height: 600px;"></iframe>
                </div>
                <div class="glass-card">
                    <h5>指数质心背离检测</h5>
                    <iframe src="index/divergence_dashboard.html" style="height: 750px;"></iframe>
                </div>

                <h3 class="section-title">I. 市场短线情绪与接力高度</h3>
                <div class="glass-card">
                    <h5>1. 市场情绪月度极值 (涨跌停对比)</h5>
                    <iframe src="sentiment/limit_stats_dashboard.html" style="height: 700px;"></iframe>
                </div>
                <div class="glass-card">
                    <h5>2. 连板梯度月度演变趋势</h5>
                    <iframe src="ladder/limit_ladder_dashboard.html" style="height: 750px;"></iframe>
                </div>
                <div class="glass-card">
                    <h5>3. 今日涨停梯队详单</h5>
                    <iframe src="ladder/limit_up_details.html" style="height: 800px;"></iframe>
                </div>
                <div class="glass-card">
                    <h5>4. 市场风险与负反馈监控</h5>
                    <iframe src="risk/risk_details_tables.html" style="height: 800px;"></iframe>
                </div>

                <footer class="pt-5 my-5 text-muted border-top">
                    Antigravity Quant System &middot; &copy; 2026
                </footer>
            </main>
        </div>
    </div>
</body>
</html>"""


def generate_master_dashboard(target_date=None):
    compact_date, dashed_date = _normalize_target_dates(target_date)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 启动全市场复盘流水线: {compact_date}")

    data_1min_dir = PROJECT_ROOT / "index_distortion_analysis" / "data_1min"
    index_out = OUTPUT_ROOT / "index"
    sentiment_out = OUTPUT_ROOT / "sentiment"
    ladder_out = OUTPUT_ROOT / "ladder"
    risk_out = OUTPUT_ROOT / "risk"
    steps = [
        ("index_fetch", lambda: fetch_index_1min_sina(data_1min_dir)),
        ("index_dashboard", lambda: visualize_cross_section_mins(data_1min_dir, OUTPUT_ROOT, dashed_date)),
        ("index_divergence", lambda: detect_divergence(data_1min_dir, OUTPUT_ROOT, dashed_date)),
        ("limit_stats", lambda: limit_stats_analysis.get_limit_stats_plotly(compact_date, sentiment_out)),
        ("limit_ladder", lambda: limit_ladder_analysis.get_ladder_stats_plotly(compact_date, ladder_out)),
        ("limit_up_details", lambda: limit_up_details.analyze_today_limit_up(compact_date, ladder_out)),
        ("market_risk", lambda: risk_analysis.analyze_market_risk(compact_date, risk_out)),
    ]

    results = []
    for step_name, runner in steps:
        print(f"[RUN] {step_name}")
        try:
            artifact = runner()
            results.append(
                {
                    "name": step_name,
                    "status": "success",
                    "artifact": artifact if artifact is not None else "-",
                    "detail": "step completed",
                }
            )
        except Exception as exc:
            print(f"[ERROR] {step_name} failed: {exc}")
            results.append(
                {
                    "name": step_name,
                    "status": "failed",
                    "artifact": "-",
                    "detail": str(exc),
                }
            )

    log_path = _write_log(compact_date, results)
    print(f"[LOG] 执行摘要已写入: {log_path}")

    output_path = OUTPUT_ROOT / f"Daily_Market_Review_{compact_date}.html"
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_render_dashboard(compact_date), encoding="utf-8")

    failed_count = sum(1 for result in results if result["status"] != "success")
    if failed_count:
        print(f"[WARN] 流水线完成，但有 {failed_count} 个步骤失败。")
    else:
        print("[SUCCESS] 全部步骤执行成功。")
    print(f"最终结果: {output_path}")
    return output_path, results


if __name__ == "__main__":
    cli_target = sys.argv[1] if len(sys.argv) > 1 else None
    generate_master_dashboard(cli_target)
