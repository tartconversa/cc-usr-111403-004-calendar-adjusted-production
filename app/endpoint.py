"""端点延拓：基于已估模型的前向预测，仅在未来日历资料存在时可用。

模型法（回归 + AR 误差）不使用 X-11 式对称滤波器，因此不存在对称滤波器固有的
末端偏差；本模块仍按需求提供显式延拓，用于评估末端不确定性。所有延拓点明确
标注为预测（forecast=True），绝不与观测月的季调值混在同一条序列里。
"""
import math


def extend_endpoint(snapshot, series, cutoff, frame, vintage, transform, chosen,
                    pulses, fit, horizon=None):
    """从 cutoff 末端向前预测，返回独立的延拓记录列表。

    延拓使用：估计的系统成分（截距/交易日/春节/季节/干预）+ AR 递归预测。
    未来月份必须在日历集合中有资料，否则该月不可延拓。
    """
    calendar_rows, _ = _calendar(snapshot)
    future_months = [m for m in sorted(calendar_rows) if m > cutoff]
    if horizon is not None:
        future_months = future_months[:horizon]
    if not future_months:
        return {'available': False, 'reason': 'cutoff 之后无日历资料，无法延拓',
                'forecasts': []}
    # 用含未来月的扩展 frame 构造设计行（干预按其 active_as_of 自然进入）
    all_months = list(vintage.months) + future_months
    ext_frame = frame.__class__(all_months, frame._calendar, frame._interventions_meta)
    ext_design = ext_frame.build(chosen['seasonal'], cutoff, pulses)
    pos = {m: i for i, m in enumerate(ext_design.months)}
    resid_hist = list(fit.resid)
    forecasts = []
    for m in future_months:
        row = ext_design.matrix[pos[m]]
        systematic = sum(row[j] * fit.beta[j] for j in range(fit.n_obs and len(fit.beta)))
        ar_part = sum(fit.phi[h] * resid_hist[-h - 1] for h in range(len(fit.phi)))
        z_hat = systematic + ar_part
        back = transform.invert(z_hat, residual_variance=fit.sigma2)
        # 预测标准误随视野扩大（AR 新息方差累积），变换尺度近似
        h = len(forecasts) + 1
        se_transformed = fit.sigma2 ** 0.5 * math.sqrt(
            sum(_ar_variance_weights(fit.phi, h)))
        forecasts.append({
            'month': m, 'forecast': True, 'horizon': h,
            'point_transformed': z_hat,
            'point_original': back['bias_corrected'],
            'point_original_plain': back['plain'],
            'forecast_se_transformed': se_transformed,
            'systematic_transformed': systematic,
            'ar_component': ar_part,
        })
        # 递归：以预测新息零均值推进，残差历史追加预测值的 AR 条件期望
        resid_hist.append(ar_part)
    return {'available': True, 'cutoff': cutoff, 'transform': transform.name,
            'forecasts': forecasts}


def _ar_variance_weights(phi, horizon):
    """多步预测方差权重 1 + psi1^2 + ... （用 AR 脉冲响应截断近似）。"""
    p = len(phi)
    if p == 0:
        return [1.0] * horizon
    psi = [1.0]
    for k in range(1, horizon):
        val = sum(phi[j] * psi[k - j - 1] for j in range(min(p, k)))
        psi.append(val)
    return [x * x for x in psi[:horizon]]


def _calendar(snapshot):
    calendar_rows = {}
    interventions = snapshot.metadata.get('interventions', [])
    for r in snapshot.rows('calendar'):
        calendar_rows[r['month']] = r
    return calendar_rows, interventions
