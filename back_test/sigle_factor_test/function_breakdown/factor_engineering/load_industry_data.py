"""
自动拆分视图文件（仅用于阅读理解，不参与运行）。
来源文件: back_test/sigle_factor_test/src/data_loader.py
函数: load_industry_data
类型: module_function
行号: 547-575
签名: def load_industry_data(ind_path=None)
作用概述: 加载申万行业分类 2021 版 (Industry Classification Loader)
"""
def load_industry_data(ind_path=None):
    """
    加载申万行业分类 2021 版 (Industry Classification Loader)

    获取 A 股市场的行业标签资产，这是执行“行业中性化”的前提。

    参数:
        ind_path (str, 可选): 行业分类 Pickle 文件的绝对路径。

    返回:
        pd.DataFrame: 包含股票各时期行业分类映射的数据表。
    """
    if ind_path is None:
        data_paths = load_data_paths()
        ind_base = get_path_from_config(
            data_paths,
            "SWIndustryClass2021",
            "data/申万行业分类2021版_AShareSWNIndustriesClass",
        )
        ind_path = os.path.join(ind_base, "申万行业分类2021版.pickle")
    else:
        if os.path.isdir(ind_path):
            ind_path = os.path.join(ind_path, "申万行业分类2021版.pickle")
    if os.path.exists(ind_path):
        try:
            return pd.read_pickle(ind_path)
        except Exception as e:
            print(f"[Warning] 加载行业分类数据失败: {e}")
    return None
