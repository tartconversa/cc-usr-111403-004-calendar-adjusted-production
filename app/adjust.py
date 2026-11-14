"""季节与日历调整主管线：单信息截止时点（vintage）下的可复算调整。

每个序列独立估计参数；日历设计（DesignFrame）可跨序列复用。输出同时保存
未调整、日历调整与季调三条序列，并给出残留季节性、残差相关与不可识别诊断。
"""
import math
from dataclasses import dataclass

from .calendar import DesignFrame
from .transform import choose_transform
from .outlier import detect_outliers
from .selection import select_model
from .arima import fit_fgls
from .stats import ljung_box, seasonal_dummy_test


@dataclass
class VintageData:
    """某 cutoff 下一个序列的可见月度面板（月序连续）。"""
    series: str
    cutoff: int
    months: list
    values: list
    record_ids: list
    revisions: list


def extract_vintage(snapshot, series, cutoff, calendar_cache=None):
    """按信息截止时点取该序列可见月值，同 series/month 取最高 revision。"""
    active = {}
    versions = {}
    for r in snapshot.rows('months'):
        if r['series'] != series or r['published'] > cutoff:
            continue
        key = r['month']
        version = (key, r['revision'])
        if version in versions and versions[version] != r['value']:
            raise ValueError('月值冲突')
        versions[version] = r['value']
        if key not in active or active[key]['revision'] < r['revision']:
            active[key] = r
    months = sorted(active)
    if not months:
        raise ValueError(f'cutoff={cutoff} 下序列 {series} 无可见观测')
    _check_continuous(months)
    return VintageData(series, cutoff, months,
                       [float(active[m]['value']) for m in months],
                       [active[m]['id'] for m in months],
                       [active[m]['revision'] for m in months])


def _check_continuous(months):
    """小整数月编码要求连续；YYYYMM 要求月份逐月连续。"""
    for a, b in zip(months, months[1:]):
        if a >= 10000:
            y1, m1 = divmod(a, 100)
            expect = a + 1 if m1 < 12 else (y1 + 1) * 100 + 1
            if b != expect:
                raise ValueError(f'月份不连续：{a} 之后应为 {expect}，实际 {b}')
        elif b != a + 1:
            raise ValueError(f'月份不连续：{a} 之后应为 {a + 1}，实际 {b}')


def build_calendar_frame(snapshot):
    """构造与序列无关、可复用的日历设计仓库。"""
    calendar_rows = {}
    for r in snapshot.rows('calendar'):
        if r['month'] in calendar_rows:
            raise ValueError('日历月份重复')
        calendar_rows[r['month']] = r
    interventions = snapshot.metadata.get('interventions', [])
    _validate_interventions(interventions)
    return calendar_rows, interventions


def _validate_interventions(interventions):
    for iv in interventions:
        for field_name in ('name', 'kind', 'month'):
            if field_name not in iv:
                raise ValueError(f'干预缺少字段 {field_name}')
        if iv['kind'] not in ('pulse', 'level_shift'):
            raise ValueError("干预 kind 应为 pulse 或 level_shift")
        if not isinstance(iv['name'], str) or not iv['name']:
            raise ValueError('干预名称应为非空字符串')


def list_series(snapshot):
    return sorted({r['series'] for r in snapshot.rows('months')})


def list_vintages(snapshot):
    """所有不同发布时点，升序；不超过 metadata.as_of。"""
    as_of = snapshot.metadata['as_of']
    return sorted({r['published'] for r in snapshot.rows('months') if r['published'] <= as_of})


def make_frame(snapshot, months):
    """按给定观测月份构造日历设计仓库（设计行与响应行严格对齐）。

    同一 cutoff 下观测月份集合相同的多个序列可复用同一 frame；
    日历资料按月份键查找，与序列无关。
    """
    calendar_rows, interventions = build_calendar_frame(snapshot)
    for m in months:
        if m not in calendar_rows:
            raise ValueError(f'日历缺少月份 {m}')
    return DesignFrame(list(months), calendar_rows, interventions)


def adjust_series(snapshot, series, cutoff, frame=None, fixed_spec=None,
                  outlier_threshold=3.5, force_transform=None, forecast_horizon=None):
    """对单个序列执行完整调整管线。

    fixed_spec: {'seasonal': ..., 'ar_order': ...}，给定则跳过 BIC 选模（固定口径对照）。
    forecast_horizon: 非 None 时启用端点延拓，结果单列于 endpoint_extension，
                      不与观测月的三条序列混合。
    返回包含三条序列、成分、诊断、选模过程与冻结要素的字典。
    """
    vintage = extract_vintage(snapshot, series, cutoff)
    if frame is None:
        frame = make_frame(snapshot, vintage.months)
    elif frame.months != vintage.months:
        # 传入的 frame 月份窗口与该 vintage 观测不一致时，按观测月重建，杜绝错位
        frame = make_frame(snapshot, vintage.months)
    transform = choose_transform(vintage.values, force_transform)
    z = transform.apply(vintage.values)

    pulses, trace = detect_outliers(frame, z, cutoff, threshold=outlier_threshold)
    if fixed_spec is None:
        selection = select_model(frame, z, cutoff, outlier_dummies=pulses)
        if selection['chosen'] is None:
            raise ValueError(
                f"cutoff={cutoff} 序列 {series} 无可行模型：{selection['infeasible_reasons']}")
        chosen = selection['chosen']
    else:
        chosen = dict(fixed_spec)
        selection = {'chosen': chosen, 'criterion': 'fixed', 'cutoff': cutoff,
                     'candidates': [], 'n_candidates': 0, 'n_feasible': 0}
    design = frame.build(chosen['seasonal'], cutoff, pulses)
    if design.n_obs != len(z):
        raise ValueError('设计矩阵行数与观测数不一致')
    fit = fit_fgls(design, z, chosen['ar_order'])

    components = _decompose(design, fit, transform)
    rows = _assemble_rows(vintage, transform, fit, components)
    diagnostics = _diagnostics(vintage, design, fit, transform)
    endpoint_extension = None
    if forecast_horizon is not None:
        from .endpoint import extend_endpoint
        endpoint_extension = extend_endpoint(
            snapshot, series, cutoff, frame, vintage, transform, chosen,
            pulses, fit, horizon=forecast_horizon)
    freeze = _freeze_payload(snapshot, series, cutoff, vintage, transform, chosen,
                             selection, pulses, trace, design, fit, diagnostics)
    return {
        'series': series,
        'cutoff': cutoff,
        'n_obs': vintage.n_obs if hasattr(vintage, 'n_obs') else len(vintage.months),
        'months': vintage.months,
        'transform': transform.describe(),
        'selection': selection,
        'outliers': [{'name': n, 'month': next(iter(p))} for n, p in pulses],
        'outlier_trace': trace,
        'ar_phi': fit.phi,
        'sigma2': fit.sigma2,
        'converged': fit.converged,
        'iterations': fit.iterations,
        'rows': rows,
        'diagnostics': diagnostics,
        'endpoint_extension': endpoint_extension,
        'freeze': freeze,
    }


def _decompose(design, fit, transform):
    """按列分组累加变换尺度下的成分贡献。"""
    groups = {}
    for j, group in enumerate(design.groups):
        col = [design.matrix[i][j] * fit.beta[j] for i in range(design.n_obs)]
        groups.setdefault(group, [0.0] * design.n_obs)
        groups[group] = [groups[group][i] + col[i] for i in range(design.n_obs)]
    return groups


def _assemble_rows(vintage, transform, fit, components):
    n = len(vintage.months)
    rows = []
    intercept = components.get('intercept', [0.0] * n)
    trading = components.get('trading', [0.0] * n)
    holiday = components.get('holiday', [0.0] * n)
    seasonal = components.get('seasonal', [0.0] * n)
    intervention = components.get('intervention', [0.0] * n)
    outlier = components.get('outlier', [0.0] * n)
    for i in range(n):
        raw = vintage.values[i]
        calendar_effect = trading[i] + holiday[i]
        # 变换尺度（对数或恒等）上的精确成分，可严格加总复算
        comp_transformed = {
            'intercept': intercept[i], 'trading': trading[i],
            'holiday_spring': holiday[i], 'seasonal': seasonal[i],
            'intervention': intervention[i], 'outlier': outlier[i],
        }
        if transform.name == 'log':
            # 三条主序列均为精确表达式（对数效应整体指数化）
            cal_adj = raw * math.exp(-calendar_effect)
            seasonal_adj = raw * math.exp(-(calendar_effect + seasonal[i]))
            # 原始尺度逐项成分为一阶边际近似：乘法下各效应不严格可加，
            # 精确复算请使用 components_transformed 与三条主序列公式
            comp_orig = {
                'trading': raw * (math.exp(trading[i]) - 1.0),
                'holiday_spring': raw * (math.exp(holiday[i]) - 1.0),
                'seasonal': raw * (math.exp(seasonal[i]) - 1.0),
                'intervention': raw * (math.exp(intervention[i]) - 1.0),
                'outlier': raw * (math.exp(outlier[i]) - 1.0),
            }
            component_note = '对数（乘法）口径：原始尺度逐项成分为边际近似，精确分解见 components_transformed'
        else:
            cal_adj = raw - calendar_effect
            seasonal_adj = raw - calendar_effect - seasonal[i]
            comp_orig = {
                'trading': trading[i], 'holiday_spring': holiday[i],
                'seasonal': seasonal[i], 'intervention': intervention[i],
                'outlier': outlier[i],
            }
            component_note = '恒等（加法）口径：原始尺度成分精确可加'
        rows.append({
            'month': vintage.months[i],
            'record_id': vintage.record_ids[i],
            'revision': vintage.revisions[i],
            'unadjusted': raw,
            'calendar_adjusted': cal_adj,
            'seasonally_adjusted': seasonal_adj,
            'residual': fit.resid[i],
            'innovation': fit.innovations[i],
            'level_transformed': intercept[i],
            'components_original_scale': comp_orig,
            'components_transformed': comp_transformed,
            'component_note': component_note,
            'components_original_scale': comp_orig,
        })
    return rows


def _diagnostics(vintage, design, fit, transform):
    months_index = design.months
    innovations = [e for e in fit.innovations if e is not None]
    lb12 = ljung_box(innovations, min(12, len(innovations) - 2), ar_terms=len(fit.phi))
    lb24 = ljung_box(innovations, min(24, max(0, len(innovations) - 2)),
                     ar_terms=len(fit.phi)) if len(innovations) >= 16 else \
        {'statistic': None, 'p_value': None, 'df': None, 'evaluated': False}
    season = seasonal_dummy_test(fit.resid, months_index)
    rank = fit.design_rank_report
    cnum = rank.get('condition_number')
    cnum_ok = isinstance(cnum, (int, float)) and math.isfinite(cnum) and cnum < 1e10
    identifiable = bool(rank.get('full_rank')) and cnum_ok
    coefficient_signals = []
    for j, name in enumerate(design.names):
        coefficient_signals.append({'name': name, 'group': design.groups[j],
                                    'beta': fit.beta[j], 't_stat': fit.t_stats[j]})
    return {
        'residual_seasonality': season,
        'residual_autocorrelation': {'ljung_box_12': lb12, 'ljung_box_24': lb24,
                                     'ar_phi': fit.phi},
        'identifiability': {
            'full_rank': rank.get('full_rank'), 'rank': rank.get('rank'),
            'n_cols': rank.get('n_cols'),
            'condition_number': rank.get('condition_number'),
            'zero_columns': rank.get('zero_columns'),
            'dropped_interventions': design.dropped_interventions,
            'identifiable': identifiable,
        },
        'coefficients': coefficient_signals,
        'converged': fit.converged,
        'fgls_iterations': fit.iterations,
    }


def _freeze_payload(snapshot, series, cutoff, vintage, transform, chosen, selection,
                    pulses, trace, design, fit, diagnostics):
    """冻结一次结果所需的全部要素，可脱离原始快照重放。"""
    return {
        'schema_version': 1,
        'series': series,
        'cutoff': cutoff,
        'input_digest': snapshot.digest,
        'months': vintage.months,
        'values': vintage.values,
        'record_ids': vintage.record_ids,
        'revisions': vintage.revisions,
        'transform': transform.describe(),
        'chosen_spec': chosen,
        'criterion': selection.get('criterion'),
        'candidate_table': selection.get('candidates'),
        'outlier_pulses': [{'name': n, 'month': next(iter(p))} for n, p in pulses],
        'outlier_trace': trace,
        'design': {
            'months': design.months, 'names': design.names,
            'groups': design.groups, 'matrix': design.matrix,
        },
        'estimated': {
            'beta': fit.beta, 'phi': fit.phi, 'sigma2': fit.sigma2,
            'loglik': fit.loglik, 'bic': fit.bic, 'aicc': fit.aicc,
        },
        'diagnostics_summary': {
            'residual_seasonality_p': diagnostics['residual_seasonality'].get('p_value'),
            'ljung_box_12_p': diagnostics['residual_autocorrelation']['ljung_box_12'].get('p_value'),
            'identifiable': diagnostics['identifiability']['identifiable'],
        },
    }
