# quant_research

本项目用于 A 股因子构建与单因子/多因子回测。

## 目录概览

- `back_test/sigle_factor_test/src`: 单因子回测主流程
- `back_test/multi_factor_test/src`: 多因子组合回测主流程
- `factor/`: 因子计算与处理产物
- `data/`: 原始与中间数据

## 路径配置说明（已优化）

项目已支持通过统一配置和环境变量定位数据路径，避免写死本机绝对路径。

优先级如下：

1. 环境变量 `QR_ROOT`（可选）
2. 项目根目录自动推断
3. `factor/factor_package/src/data_config.json` 中的 `paths`

建议在新机器上只做两步：

1. 保持仓库目录结构不变。
2. 根据本机情况更新 `factor/factor_package/src/data_config.json` 的 `paths`。

## 运行示例

单因子回测：

```bash
python back_test/sigle_factor_test/src/run_test_v2.py --factor-group "3_回归类因子_RegressionFactors" --factor-name "1_minus_r2_30" --start "2015-01" --end "2024-12"
```

多因子回测：

```bash
python back_test/multi_factor_test/src/run_multi_test_v2.py
```