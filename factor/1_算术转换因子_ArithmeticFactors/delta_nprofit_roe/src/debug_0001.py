import sys
sys.path.insert(0, r'E:\1_basement\quant_research\factor\1_算术转换因子_ArithmeticFactors\delta_nprofit_roe\src')
from compute_factors import *

code = '000001.SZ'

inc = load_income_csv(code)
exp = load_express(code)
notc = load_notice(code)
bal = load_balance_csv(code)

merged_np = merge_earnings(inc, exp, notc)
if not merged_np.empty:
    merged_np = add_single_quarter_np(merged_np)
    merged_np = calc_delta_nprofit(merged_np)
    
    print("--- delta_nprofit dataframe ---")
    print(merged_np[['ann_date', 'end_date', 'yyyyq', 'cum_nprofit', 'single_np', 'yoy_np', 'delta_nprofit']].dropna(subset=['cum_nprofit']).tail(20).to_string())


merged_roe_np = merge_earnings(inc, exp)
if not merged_roe_np.empty:
    merged_roe_np = add_single_quarter_np(merged_roe_np)
    roe_df = calc_delta_roe(merged_roe_np, bal)
    print("\n--- delta_roe dataframe ---")
    print(roe_df[['ann_date', 'yyyyq', 'single_np', 'equity_begin', 'equity_end', 'roe', 'delta_roe']].dropna(subset=['equity_end']).tail(20).to_string())
