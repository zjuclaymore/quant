"""
配置加载模块 (Config Loader)

负责从YAML配置文件加载回测参数，提供默认值和验证功能。
支持通过点号路径访问嵌套配置，并提供各模块参数的便捷获取方法。

配置文件结构示例:
    backtest:
        start_date: "2020-01-01"
        end_date: "2023-12-31"
        commission: 0.0015
    factors:
        list: [[factor1, name1], [factor2, name2]]
        aggregation_method: "equal_weight"
    preprocessing:
        outlier_method: "winsorize"
        orthogonalize: false
    portfolio:
        group_size: 10
    stock_filter:
        exclude_st: true
        min_market_cap: 1000000000
    output:
        base_dir: "./output"
        generate_attribution: true
"""

import os
import yaml
import logging
from typing import Dict, Any

try:
    from .logger import get_logger
except ImportError:
    from logger import get_logger

class ConfigLoader:
    """
    配置加载器类

    从YAML配置文件加载多因子回测的所有配置参数。
    提供便捷的方法获取不同模块的配置，并支持默认值回退。

    Attributes:
        config_path: 配置文件路径，默认为自动查找 config.yaml
        logger: 日志记录器
        config: 加载后的配置字典
    """

    def __init__(self, config_path: str = None, logger: logging.Logger = None):
        self.logger = logger or get_logger(__name__)
        self.config_path = config_path or self._find_default_config()
        self.config = self._load_config()

    def _find_default_config(self):
        """
        查找默认配置文件

        在以下位置搜索配置文件:
        1. 当前目录的 config.yaml
        2. 上级目录的 config.yaml
        3. 与脚本同级的 config.yaml

        Returns:
            str: 找到的配置文件路径

        Raises:
            FileNotFoundError: 如果未找到任何配置文件
        """
        candidates = [
            "config.yaml",
            "../config.yaml",
            os.path.join(os.path.dirname(__file__), "..", "config.yaml")
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        raise FileNotFoundError("未找到配置文件config.yaml")

    def _load_config(self) -> Dict[str, Any]:
        """
        加载YAML配置文件

        Returns:
            Dict[str, Any]: 解析后的配置字典

        Raises:
            Exception: 如果配置文件加载失败则重新抛出异常
        """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            self.logger.info(f"配置加载成功: {self.config_path}")
            return config
        except Exception as e:
            self.logger.error(f"配置加载失败: {e}")
            raise

    def get(self, key_path: str, default=None):
        """
        获取配置值，支持点号路径访问嵌套配置

        例如: get('backtest.start_date')

        参数:
            key_path: 配置路径，使用点号分隔嵌套键
            default: 默认值，当配置不存在时返回

        返回:
            配置值，如果路径不存在则返回 default
        """
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def get_backtest_params(self):
        """
        获取回测参数

        返回配置中 backtest 部分的字典，包含回测时间段、佣金设置等。

        Returns:
            Dict: 回测参数字典，键值对包含 start_date, end_date, commission 等
        """
        return self.config.get('backtest', {})

    def get_factors_config(self):
        """
        获取因子配置

        返回配置中 factors 部分的字典，包含因子列表、聚合方法等。

        Returns:
            Dict: 因子配置字典，键值对包含 list, aggregation_method 等
        """
        return self.config.get('factors', {})

    def get_preprocessing_params(self):
        """
        获取预处理参数

        返回配置中 preprocessing 部分的字典，包含去极值、中性化等设置。

        Returns:
            Dict: 预处理参数字典，键值对包含 outlier_method, orthogonalize 等
        """
        return self.config.get('preprocessing', {})

    def get_portfolio_params(self):
        """
        获取组合参数

        返回配置中 portfolio 部分的字典，包含分组数量、权重方法等。

        Returns:
            Dict: 组合参数字典，键值对包含 group_size, weight_method 等
        """
        return self.config.get('portfolio', {})

    def get_stock_filter_params(self):
        """
        获取股票池过滤参数

        返回配置中 stock_filter 部分的字典，包含ST股过滤、最小市值等设置。

        Returns:
            Dict: 选股过滤参数字典，键值对包含 exclude_st, min_market_cap 等
        """
        return self.config.get('stock_filter', {})

    def get_output_params(self):
        """
        获取输出参数

        返回配置中 output 部分的字典，包含输出目录、报告生成等设置。

        Returns:
            Dict: 输出参数字典，键值对包含 base_dir, generate_attribution 等
        """
        return self.config.get('output', {})
