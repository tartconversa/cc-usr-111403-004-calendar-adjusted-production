# 工业月度序列的日历调整、季调与修订稳定性分析

本项目提供移动节日及交易日设计相关资料的业务摘要，并在此基础上提供**可复算的季节与日历
调整**：区分长期趋势水平、季节项、春节移动窗口、交易日效应、事前已知干预与异常观测，
而非把春节低点一律当作噪声抹平。方法学细节见 [季节调整设计](docs/seasonal-adjustment.md)，
数据字段与读取语义见 [资料契约](docs/data-contract.md)。

## 模块一览

- `app/catalog.py` 按集合和记录键解析修订，拒绝同修订冲突；
- `app/baseline.py` 承载既有业务计算（接口保持兼容）；
- `app/service.py` 输出摘要、调整、实时修订分析及输入摘要哈希；
- `app/calendar.py` 日历/春节/季节/干预设计矩阵（跨序列复用）；
- `app/transform.py` 对数与恒等变换、非正值明确退路；
- `app/arima.py` 日历回归与 AR(p) 误差的 FGLS 联合估计；
- `app/selection.py` BIC 模型选择（只用信息截止时点可见数据）；
- `app/outlier.py` 无前瞻滚动外部学生化异常检测；
- `app/adjust.py` 单 vintage 调整主管线、三序列与诊断；
- `app/endpoint.py` 显式标注、与观测隔离的端点预测延拓；
- `app/realtime.py` 逐历史发布的修订三效应分解与固定口径对照；
- `app/freeze.py` 冻结结果的重放与逐项核对。

运行环境为 Python 3.11+，仅使用标准库。`fixtures/sample.json` 是既有合成小数据；
`fixtures/sample_long.json` 由 `fixtures/generate_sample.py`（固定种子）生成，
含已知 DGP、干预、异常、修订与非正值，不对应真实企业或个人。

## 用法

既有摘要（语义不变）：

    python -m app.service fixtures/sample.json

季节与日历调整（同时输出未调整、日历调整、季调三序列与三类诊断）：

    python -m app.service fixtures/sample_long.json adjust

逐历史发布的实时修订分析（新增月/历史修订/重新选模三效应分解 + 固定口径对照）：

    python -m app.service fixtures/sample_long.json realtime

编程入口：

```python
from app.service import adjust, realtime, replay_frozen

# 单时点调整；forecast_horizon 可选，启用端点延拓（不混入观测序列）
result = adjust(payload, forecast_horizon=6)

# 实时修订分析；min_obs 过滤样本不足的早期 vintage
report = realtime(payload, min_obs=24)

# 冻结一次结果后脱离原始快照重放，逐项核对 β/φ/BIC
check = replay_frozen(result['adjustments'][0]['freeze'])
```

## 关键方法学约束

- 日历回归与序列模型在同一 FGLS 中**联合估计**，不做先平滑再回归的两步法；
- 异常检测严格无前视：第 t 月判定只用 ≤t 的数据，设计外推点不标记为异常；
- 模型选择、变换选择均只基于该 cutoff 可见数据；
- 非正值导致对数不可用时明确退回恒等（加法）分解并记录原因；
- 修订三效应精确加总，固定口径轨道重新选模效应恒为零；
- 系列间共用日历设计但参数独立；冻结结果可追溯到实际月份、设计矩阵与模型。

## 测试与编译

    python -m unittest discover -s tests -v
    python -m compileall -q app tests

重新生成确定性合成夹具：

    python fixtures/generate_sample.py
