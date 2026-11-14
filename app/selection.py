"""季节表示与 AR 阶数的模型选择，只用信息截止时点可见的数据。

候选集：季节规格 {fixed, trig3, trig6} × AR 阶数 {0,1,2}，共 9 个候选；
按 BIC 选择（AICc 同时返回备查）。秩亏或样本不足的候选剔除并记录。
"""
from .arima import fit_fgls
from .calendar import SEASONAL_SPECS


def select_model(frame, y, cutoff, ar_orders=(0, 1, 2), outlier_dummies=None):
    candidates = []
    for spec in SEASONAL_SPECS:
        for ar in ar_orders:
            design = frame.build(spec, cutoff, outlier_dummies)
            entry = {'seasonal': spec, 'ar_order': ar, 'n_obs': design.n_obs,
                     'n_cols': design.n_cols}
            rank = design.rank_report()
            entry['rank_report'] = rank
            if not rank['full_rank']:
                entry.update({'rejected': True, 'reason': '设计矩阵秩亏或含零列'})
                candidates.append(entry)
                continue
            try:
                fit = fit_fgls(design, y, ar)
            except ValueError as exc:
                entry.update({'rejected': True, 'reason': str(exc)})
                candidates.append(entry)
                continue
            entry.update({'rejected': False, 'bic': fit.bic, 'aicc': fit.aicc,
                          'loglik': fit.loglik, 'converged': fit.converged,
                          'n_params': fit.n_params})
            candidates.append(entry)
    feasible = [c for c in candidates if not c['rejected']]
    if not feasible:
        reasons = sorted({c.get('reason', '未知') for c in candidates})
        return {
            'chosen': None,
            'candidates': candidates,
            'criterion': 'BIC',
            'cutoff': cutoff,
            'n_candidates': len(candidates),
            'n_feasible': 0,
            'bic_gap_to_next': None,
            'infeasible_reasons': reasons,
        }
    best = min(feasible, key=lambda c: c['bic'])
    return {
        'chosen': {'seasonal': best['seasonal'], 'ar_order': best['ar_order']},
        'candidates': candidates,
        'criterion': 'BIC',
        'cutoff': cutoff,
        'n_candidates': len(candidates),
        'n_feasible': len(feasible),
        'bic_gap_to_next': _bic_gap(feasible, best),
    }


def fit_selected(frame, y, cutoff, chosen, outlier_dummies=None):
    design = frame.build(chosen['seasonal'], cutoff, outlier_dummies)
    return fit_fgls(design, y, chosen['ar_order'])


def _bic_gap(feasible, best):
    others = [c['bic'] for c in feasible if c is not best]
    return min(others) - best['bic'] if others else None
