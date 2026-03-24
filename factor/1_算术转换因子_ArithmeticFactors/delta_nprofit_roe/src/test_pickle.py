import pickle
import pandas as pd
import sys

# Make pandas 2.x compatible with older pandas Pickles
class CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'pandas.core.indexes.numeric':
            if name == 'Int64Index':
                return pd.Index
            if name == 'Float64Index':
                return pd.Index
        if module == 'pandas.core.indexes.base':
            if name == '_new_Index':
                # This could be pd.Index.__new__
                pass
        return super().find_class(module, name)

def compat_read_pickle(filepath):
    with open(filepath, 'rb') as f:
        unpickler = CompatUnpickler(f)
        return unpickler.load()

try:
    df = compat_read_pickle(r'E:\1_basement\quant_research\data\中国A股财务数据tushare\000001.SZ.pickle')
    print('Shape:', df.shape)
    print('Head:')
    print(df[['ann_date', 'end_date', 'n_income_inc', 'total_hldr_eqy_exc_min_int_bal']].head())
except Exception as e:
    import traceback
    traceback.print_exc()
