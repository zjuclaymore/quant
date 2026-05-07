#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import subprocess
import sys
from pathlib import Path
import os

def safe_run(cmd):
    print(f"[Orchestrate] Running CMD: {' '.join(str(x) for x in cmd)}")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    res = subprocess.run(cmd, env=env)
    if res.returncode != 0:
        raise RuntimeError(f"Step failed with exit code {res.returncode}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--factor-col", required=True)
    parser.add_argument("--factor-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--enable-mad3", default="no")
    parser.add_argument("--enable-zscore", default="no")
    parser.add_argument("--zscore-method", default="cross_sectional")
    parser.add_argument("--enable-neutral", default="no")
    parser.add_argument("--mv-col", default="lncap")
    parser.add_argument("--ind-col", default="ind_code")
    parser.add_argument("--keep-intermediate", default="no")
    args = parser.parse_args()

    ENG_DIR = Path(__file__).resolve().parent
    out_dir = Path(args.output_dir).resolve()
    current_input = args.input

    tags = []
    
    # 1. MAD3
    if args.enable_mad3 == "yes":
        tags.append("mad3")
        tag_str = "_".join(tags)
        out_path = out_dir / f"factor_{args.factor_name}_{tag_str}.parquet"
        safe_run([sys.executable, str(ENG_DIR / "factor_mad3_winsorization.py"), 
                  "--input", current_input, 
                  "--factor-col", args.factor_col, 
                  "--output", str(out_path)])
        current_input = str(out_path)

    # 2. ZSCORE
    if args.enable_zscore == "yes":
        tags.append("zscore")
        tag_str = "_".join(tags)
        out_path = out_dir / f"factor_{args.factor_name}_{tag_str}.parquet"
        safe_run([sys.executable, str(ENG_DIR / "factor_zscore_normalization.py"), 
                  "--input", current_input, 
                  "--factor-col", args.factor_col, 
                  "--method", args.zscore_method,
                  "--output", str(out_path)])
        current_input = str(out_path)

    # 3. NEUTRALIZATION
    if args.enable_neutral == "yes":
        tags.append("neutral")
        tag_str = "_".join(tags)
        out_path = out_dir / f"factor_{args.factor_name}_{tag_str}.parquet"
        safe_run([sys.executable, str(ENG_DIR / "factor_market_cap_industry_neutralization.py"), 
                  "--input", current_input, 
                  "--factor-col", args.factor_col, 
                  "--mv-col", args.mv_col, 
                  "--ind-col", args.ind_col, 
                  "--output", str(out_path)])
        current_input = str(out_path)
        
    print(f"\n[Orchestrate] Final output produced at: {current_input}")

if __name__ == "__main__":
    main()
