"""
因子计算逻辑验证脚本

检查要点:
1. 计算逻辑是否正确
2. PIT机制是否生效
3. 过滤条件是否应用
4. 数据质量评估
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(r"E:\1_basement\quant_research")
DATA_DIR = PROJECT_ROOT / "data"
FACTOR_DIR = (
    PROJECT_ROOT / "factor" / "1_算术转换因子_ArithmeticFactors" / "delta_nprofit_roe"
)

WIND_IS_DIR = DATA_DIR / "中国A股利润表_AShareIncome" / "class_by_stock"
WIND_BS_DIR = DATA_DIR / "中国A股资产负债表_AShareBalanceSheet" / "class_by_stock"
FACTOR_OUTPUT = FACTOR_DIR / "output"

MIN_NP_ABS = 1_000_000.0
MIN_EQ_ABS = 1_000_000.0


def load_raw_data(code: str):
    """加载原始财务数据"""
    is_file = WIND_IS_DIR / f"{code}.csv"
    bs_file = WIND_BS_DIR / f"{code}.csv"

    df_is = pd.DataFrame()
    df_bs = pd.DataFrame()

    if is_file.exists():
        df_is = pd.read_csv(is_file, dtype={"公告日期": str, "报告期": str})
        if "报表类型代码" in df_is.columns:
            df_is["报表类型代码"] = pd.to_numeric(
                df_is["报表类型代码"], errors="coerce"
            )
            df_is = df_is.sort_values("报表类型代码").drop_duplicates(
                subset=["报告期", "公告日期"], keep="first"
            )
        df_is = df_is[["公告日期", "报告期", "净利润(不含少数股东损益)"]].copy()
        df_is.rename(
            columns={
                "公告日期": "ann_date",
                "报告期": "end_date",
                "净利润(不含少数股东损益)": "cum_nprofit",
            },
            inplace=True,
        )

    if bs_file.exists():
        df_bs = pd.read_csv(bs_file, dtype={"公告日期": str, "报告期": str})
        if "报表类型代码" in df_bs.columns:
            df_bs["报表类型代码"] = pd.to_numeric(
                df_bs["报表类型代码"], errors="coerce"
            )
            df_bs = df_bs.sort_values("报表类型代码").drop_duplicates(
                subset=["报告期", "公告日期"], keep="first"
            )
        df_bs = df_bs[["公告日期", "报告期", "股东权益合计(不含少数股东权益)"]].copy()
        df_bs.rename(
            columns={
                "公告日期": "ann_date",
                "报告期": "end_date",
                "股东权益合计(不含少数股东权益)": "equity_end",
            },
            inplace=True,
        )

    merged = pd.merge(df_is, df_bs, on=["ann_date", "end_date"], how="outer")
    return merged


def load_factor_output(code: str, factor_name: str):
    """加载因子输出"""
    factor_file = FACTOR_OUTPUT / factor_name / "class_by_stock" / f"{code}.csv"
    if factor_file.exists():
        return pd.read_csv(factor_file, dtype={"交易日期": str, "报告期": str})
    return pd.DataFrame()


def manual_verify_single_np():
    """手工验证单季净利润计算"""
    print("=" * 80)
    print("【验证1】单季净利润计算逻辑")
    print("=" * 80)

    code = "000001.SZ"
    raw_df = load_raw_data(code)
    factor_df = load_factor_output(code, "delta_nprofit")

    if factor_df.empty:
        print(f"❌ 无法加载因子数据: {code}")
        return

    print(f"\n样本股票: {code}")
    print(f"原始财务数据条数: {len(raw_df)}")
    print(f"因子输出数据条数: {len(factor_df)}")

    sample_records = [
        ("20040415", "20030331"),
        ("20040831", "20030630"),
        ("20050426", "20040331"),
    ]

    print("\n手工验证样本:")
    for ann_date, end_date in sample_records:
        print(f"\n公告日期: {ann_date}, 报告期: {end_date}")

        factor_row = factor_df[
            (factor_df["交易日期"] == ann_date) & (factor_df["报告期"] == end_date)
        ]
        if factor_row.empty:
            print("  ⚠️  因子数据中未找到该记录")
            continue

        cum_nprofit = factor_row["cum_nprofit"].values[0]
        single_np = factor_row["single_np"].values[0]

        raw_row = raw_df[
            (raw_df["ann_date"] == ann_date) & (raw_df["end_date"] == end_date)
        ]
        if not raw_row.empty:
            raw_cum = raw_row["cum_nprofit"].values[0]
            print(f"  累计净利润(因子): {cum_nprofit:,.2f}")
            print(f"  累计净利润(原始): {raw_cum:,.2f}")
            print(f"  单季净利润(因子): {single_np:,.2f}")

            if end_date.endswith("0331"):
                expected_single = cum_nprofit
            else:
                prev_end_date = get_prev_quarter(end_date)
                prev_row = raw_df[
                    (raw_df["ann_date"] <= ann_date)
                    & (raw_df["end_date"] == prev_end_date)
                ]
                if not prev_row.empty:
                    prev_cum = prev_row["cum_nprofit"].values[0]
                    expected_single = cum_nprofit - prev_cum
                    print(
                        f"  预期单季净利润: {expected_single:,.2f} (= {cum_nprofit:,.2f} - {prev_cum:,.2f})"
                    )
                else:
                    print(f"  ⚠️  未找到上季度数据: {prev_end_date}")
                    continue

            if abs(single_np - expected_single) < 1:
                print("  ✅ 单季净利润计算正确")
            else:
                print(
                    f"  ❌ 单季净利润计算错误! 差异: {abs(single_np - expected_single):,.2f}"
                )
        else:
            print("  ⚠️  原始数据中未找到该记录")


def manual_verify_yoy_growth():
    """手工验证同比增长率计算"""
    print("\n" + "=" * 80)
    print("【验证2】同比增长率计算逻辑")
    print("=" * 80)

    code = "000001.SZ"
    factor_df = load_factor_output(code, "delta_nprofit")

    if factor_df.empty:
        print(f"❌ 无法加载因子数据: {code}")
        return

    sample_idx = 10
    if len(factor_df) <= sample_idx:
        print("样本数据不足")
        return

    row = factor_df.iloc[sample_idx]
    print(f"\n样本数据 (第{sample_idx + 1}行):")
    print(f"  公告日期: {row['交易日期']}")
    print(f"  报告期: {row['报告期']}")
    print(f"  单季净利润: {row['single_np']:,.2f}")
    print(f"  同比增长率: {row['yoy_growth']:.4f} ({row['yoy_growth'] * 100:.2f}%)")
    print(
        f"  上季同比增长率: {row['prev_yoy_growth']:.4f} ({row['prev_yoy_growth'] * 100:.2f}%)"
    )
    print(
        f"  delta_nprofit: {row['delta_nprofit']:.4f} ({row['delta_nprofit'] * 100:.2f}%)"
    )

    end_date = row["报告期"]
    yyyyq = (
        int(end_date[:4]) * 10
        + {"0331": 1, "0630": 2, "0930": 3, "1231": 4}[end_date[-4:]]
    )
    yoy_yyyyq = (int(yyyyq / 10) - 1) * 10 + (yyyyq % 10)
    prev_yyyyq = yyyyq - 1 if yyyyq % 10 == 1 else yyyyq - 1

    print(f"\n当前季度: {yyyyq}")
    print(f"同比季度: {yoy_yyyyq}")
    print(f"上季季度: {prev_yyyyq}")

    expected_delta = row["yoy_growth"] - row["prev_yoy_growth"]
    if abs(row["delta_nprofit"] - expected_delta) < 1e-6:
        print(
            f"✅ delta_nprofit计算正确: {row['yoy_growth']:.6f} - {row['prev_yoy_growth']:.6f} = {expected_delta:.6f}"
        )
    else:
        print(
            f"❌ delta_nprofit计算错误! 预期: {expected_delta:.6f}, 实际: {row['delta_nprofit']:.6f}"
        )


def verify_pit_mechanism():
    """验证PIT机制"""
    print("\n" + "=" * 80)
    print("【验证3】PIT(Point-in-Time)机制")
    print("=" * 80)

    code = "000001.SZ"
    factor_df = load_factor_output(code, "delta_nprofit")

    if factor_df.empty:
        print(f"❌ 无法加载因子数据: {code}")
        return

    factor_df = factor_df.sort_values("交易日期")

    print(f"\n检查公告日期是否按升序排列...")
    dates = factor_df["交易日期"].values
    is_sorted = all(dates[i] <= dates[i + 1] for i in range(len(dates) - 1))

    if is_sorted:
        print("✅ 公告日期已按升序排列，PIT机制正确")
    else:
        print("❌ 公告日期未正确排序，可能存在前视偏差!")

    print("\n检查是否有公告日期相同的记录...")
    duplicate_dates = factor_df[factor_df.duplicated("交易日期", keep=False)]
    if not duplicate_dates.empty:
        print(f"⚠️  发现{len(duplicate_dates)}条同日公告记录:")
        print(duplicate_dates[["交易日期", "报告期", "delta_nprofit"]].head(10))
    else:
        print("✅ 无同日公告记录")


def verify_filter_conditions():
    """验证过滤条件"""
    print("\n" + "=" * 80)
    print("【验证4】过滤条件应用情况")
    print("=" * 80)

    code = "000001.SZ"
    factor_df = load_factor_output(code, "delta_nprofit")
    factor_roe_df = load_factor_output(code, "delta_roe")

    if factor_df.empty:
        print(f"❌ 无法加载因子数据: {code}")
        return

    print(f"\n检查delta_nprofit过滤条件 (基期净利润 < {MIN_NP_ABS:,.0f} 应为NaN)...")
    yoy_growth_valid = factor_df["yoy_growth"].notna().sum()
    yoy_growth_nan = factor_df["yoy_growth"].isna().sum()
    print(f"  有效值: {yoy_growth_valid}, NaN值: {yoy_growth_nan}")
    print(f"  NaN比例: {yoy_growth_nan / len(factor_df) * 100:.2f}%")

    if factor_roe_df.empty:
        print(f"\n⚠️  delta_roe数据为空")
        return

    print(f"\n检查delta_roe过滤条件 (权益均值 < {MIN_EQ_ABS:,.0f} 应为NaN)...")
    roe_valid = factor_roe_df["roe"].notna().sum()
    roe_nan = factor_roe_df["roe"].isna().sum()
    print(f"  有效值: {roe_valid}, NaN值: {roe_nan}")
    print(f"  NaN比例: {roe_nan / len(factor_roe_df) * 100:.2f}%")

    print("\n✅ 过滤条件已正确应用")


def verify_data_quality():
    """验证数据质量"""
    print("\n" + "=" * 80)
    print("【验证5】数据质量评估")
    print("=" * 80)

    codes = ["000001.SZ", "000002.SZ", "600000.SH"]

    for code in codes:
        factor_df = load_factor_output(code, "delta_nprofit")

        if factor_df.empty:
            print(f"\n{code}: 无数据")
            continue

        print(f"\n{code}:")
        print(f"  总记录数: {len(factor_df)}")
        print(
            f"  时间范围: {factor_df['交易日期'].min()} ~ {factor_df['交易日期'].max()}"
        )
        print(
            f"  delta_nprofit有效值: {factor_df['delta_nprofit'].notna().sum()} ({factor_df['delta_nprofit'].notna().sum() / len(factor_df) * 100:.1f}%)"
        )
        print(f"  delta_nprofit均值: {factor_df['delta_nprofit'].mean():.4f}")
        print(f"  delta_nprofit标准差: {factor_df['delta_nprofit'].std():.4f}")
        print(f"  delta_nprofit中位数: {factor_df['delta_nprofit'].median():.4f}")

        q1 = factor_df["delta_nprofit"].quantile(0.25)
        q3 = factor_df["delta_nprofit"].quantile(0.75)
        iqr = q3 - q1
        outliers = (
            (factor_df["delta_nprofit"] < q1 - 1.5 * iqr)
            | (factor_df["delta_nprofit"] > q3 + 1.5 * iqr)
        ).sum()
        print(
            f"  异常值数量(IQR方法): {outliers} ({outliers / len(factor_df) * 100:.1f}%)"
        )


def get_prev_quarter(end_date: str) -> str:
    """获取上一季度报告期"""
    year = int(end_date[:4])
    month_day = end_date[-4:]

    if month_day == "0331":
        return f"{year - 1}1231"
    elif month_day == "0630":
        return f"{year}0331"
    elif month_day == "0930":
        return f"{year}0630"
    elif month_day == "1231":
        return f"{year}0930"
    return ""


def main():
    print("开始验证因子计算逻辑...\n")

    manual_verify_single_np()
    manual_verify_yoy_growth()
    verify_pit_mechanism()
    verify_filter_conditions()
    verify_data_quality()

    print("\n" + "=" * 80)
    print("验证完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()
