"""冻结结果的重放与校验：冻结后仍能追到实际月份、设计矩阵与模型参数。

freeze 负载包含输入摘要、月份与观测、设计矩阵、选模候选表、异常轨迹、
估计参数与诊断摘要。replay 不依赖原始快照即可由冻结的设计矩阵复算拟合，
并与冻结参数逐元素比对。
"""
import math

from .arima import fit_fgls
from .transform import Transform
from .calendar import DesignMatrix
from .stats import ljung_box, seasonal_dummy_test


class ReplayMismatch(AssertionError):
    pass


def replay(freeze, tolerance=1e-9):
    """从冻结负载重新估计并校验，返回复算结果与逐项核对结论。"""
    design = DesignMatrix(
        months=freeze['design']['months'],
        names=freeze['design']['names'],
        groups=freeze['design']['groups'],
        matrix=freeze['design']['matrix'],
        cutoff=freeze['cutoff'],
    )
    tf = Transform(freeze['transform']['transform'],
                   freeze['transform']['reason'], [])
    z = tf.apply(freeze['values'])
    fit = fit_fgls(design, z, freeze['chosen_spec']['ar_order'])
    est = freeze['estimated']
    checks = {
        'n_months_match': len(freeze['months']) == design.n_obs,
        'design_dimensions': (design.n_obs, design.n_cols) == (
            len(freeze['design']['matrix']), len(freeze['design']['names'])),
        'max_beta_abs_diff': max(abs(a - b) for a, b in zip(fit.beta, est['beta'])),
        'max_phi_abs_diff': max((abs(a - b) for a, b in zip(fit.phi, est['phi'])), default=0.0),
        'sigma2_abs_diff': abs(fit.sigma2 - est['sigma2']),
        'bic_abs_diff': abs(fit.bic - est['bic']),
        'spec': freeze['chosen_spec'],
        'tolerance': tolerance,
    }
    checks['reproduces'] = (
        checks['max_beta_abs_diff'] <= tolerance
        and checks['max_phi_abs_diff'] <= tolerance
        and checks['sigma2_abs_diff'] <= tolerance * max(1.0, abs(est['sigma2']))
    )
    innovations = [e for e in fit.innovations if e is not None]
    diagnostics = {
        'ljung_box_12': ljung_box(innovations, min(12, len(innovations) - 2),
                                  ar_terms=len(fit.phi)),
        'residual_seasonality': seasonal_dummy_test(fit.resid, design.months),
        'rank_report': design.rank_report(),
    }
    rows = _reassemble_rows(freeze, design, fit, tf)
    if not checks['reproduces']:
        raise ReplayMismatch(f"冻结重放未能复现：{checks}")
    return {
        'series': freeze['series'], 'cutoff': freeze['cutoff'],
        'input_digest': freeze['input_digest'],
        'checks': checks, 'rows': rows,
        'recomputed_diagnostics': diagnostics,
        'traceability': {
            'months': freeze['months'], 'record_ids': freeze['record_ids'],
            'revisions': freeze['revisions'],
            'design_names': design.names, 'design_groups': design.groups,
            'candidate_table_present': bool(freeze.get('candidate_table')),
            'outlier_trace_present': bool(freeze.get('outlier_trace')),
        },
    }


def _reassemble_rows(freeze, design, fit, transform):
    n = design.n_obs
    groups = {}
    for j, group in enumerate(design.groups):
        col = [design.matrix[i][j] * fit.beta[j] for i in range(n)]
        groups.setdefault(group, [0.0] * n)
        groups[group] = [groups[group][i] + col[i] for i in range(n)]
    rows = []
    for i in range(n):
        raw = freeze['values'][i]
        cal = groups.get('trading', [0.0] * n)[i] + groups.get('holiday', [0.0] * n)[i]
        seas = groups.get('seasonal', [0.0] * n)[i]
        if transform.name == 'log':
            cal_adj = raw * math.exp(-cal)
            sa = raw * math.exp(-(cal + seas))
        else:
            cal_adj = raw - cal
            sa = raw - cal - seas
        rows.append({'month': freeze['months'][i],
                     'record_id': freeze['record_ids'][i],
                     'revision': freeze['revisions'][i],
                     'unadjusted': raw, 'calendar_adjusted': cal_adj,
                     'seasonally_adjusted': sa})
    return rows
