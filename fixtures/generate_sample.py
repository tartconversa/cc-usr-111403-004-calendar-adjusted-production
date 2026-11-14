"""确定性合成月度资料生成器（仅标准库，固定随机种子）。

DGP 明确写入生成文件的 metadata.dgp，便于检验季调是否还原已知成分：
- 乘法趋势 × 固定季节因子 × 交易日效应 × 春节窗口效应 × AR(1) 噪声；
- 事前已知干预（脉冲、水平迁移）与单点异常；
- 按批次发布形成多个 vintage，并对 3 个历史月发布 revision=2 的修订；
- 第三个序列包含零值，用于检验非正值时对数变换的明确退路。

用法：python fixtures/generate_sample.py
"""
import json
import math
import random
from pathlib import Path

SEED = 20261115
N_MONTHS = 72            # 6 年
rng = random.Random(SEED)

# 每 6 个月一个发布批次，形成 12 个 vintage
PUBLISH_BATCH = 6
# 被修订的历史月（相对批次发布后回溯 2 个月）
REVISION_MONTHS = {26: 1.06, 44: 0.95, 58: 1.03}
# 事前已知干预（写入 metadata，不从产量反推）；落在自由度充足、杠杆良好的月份
INTERVENTIONS = [
    {'name': 'policy_pulse_40', 'kind': 'pulse', 'month': 40, 'active_as_of': 40},
    {'name': 'plant_shift_55', 'kind': 'level_shift', 'month': 55, 'active_as_of': 55},
]
# 单点异常（DGP 中真实存在，检测应能识别；位于设计支撑充分、杠杆 <1 的后段）
# 44 月为强异常（约 4-5 个残差标准差，应越过阈值）；60 月为温和异常，展示阈值边界
OUTLIER_MONTHS = {44: 0.72, 60: 1.15}

SERIES_SPECS = {
    'industry_a': {'base': 100.0, 'trend': 0.004, 'season_amp': 0.12,
                   'td_beta': 0.004, 'holiday_beta': -0.22, 'ar_phi': 0.45,
                   'noise_sd': 0.012, 'allow_nonpositive': False},
    'industry_b': {'base': 80.0, 'trend': 0.002, 'season_amp': 0.08,
                   'td_beta': 0.0025, 'holiday_beta': -0.15, 'ar_phi': 0.30,
                   'noise_sd': 0.015, 'allow_nonpositive': False},
    'inventory_c': {'base': 30.0, 'trend': 0.001, 'season_amp': 0.05,
                    'td_beta': 0.0, 'holiday_beta': -0.08, 'ar_phi': 0.2,
                    'noise_sd': 0.02, 'allow_nonpositive': True},
}
SEASON_PHASE = [0.92, 0.82, 0.95, 1.02, 1.05, 1.08, 1.04, 1.03, 1.06, 1.09, 1.10, 0.84]


def build_calendar(month_id):
    """合成星期计数（确定性循环）与春节窗口权重（1-2 月分摊，逐年漂移）。"""
    m = month_id % 12 or 12
    base = [4] * 7
    extra = month_id % 7
    for k in range(extra):
        base[k % 7] += 1
    if month_id % 2 == 0:
        base[(month_id + 3) % 7] += 1
    # 春节窗口：1、2 月权重互补，模拟移动节日落点漂移
    if m in (1, 2):
        drift = (month_id // 12) % 4
        share = {1: [0.8, 0.6, 0.4, 0.2], 2: [0.2, 0.4, 0.6, 0.8]}[m][drift]
        holiday = round(share, 4)
    else:
        holiday = 0.0
    return base, holiday


def weekday_contrasts(counts):
    return [counts[j] - counts[6] for j in range(6)]


def simulate(spec, calendar_map):
    months = list(range(1, N_MONTHS + 1))
    eps = [rng.gauss(0.0, spec['noise_sd']) for _ in months]
    ar = []
    prev = 0.0
    for e in eps:
        prev = spec['ar_phi'] * prev + e
        ar.append(prev)
    values = {}
    for i, m in enumerate(months):
        moy = m % 12 or 12
        counts, holiday = calendar_map[m]
        td = sum(spec['td_beta'] * c for c in weekday_contrasts(counts))
        log_y = (math.log(spec['base']) + spec['trend'] * i
                 + math.log(SEASON_PHASE[moy - 1]) + td
                 + spec['holiday_beta'] * holiday + ar[i])
        value = math.exp(log_y)
        if m in OUTLIER_MONTHS:
            value *= OUTLIER_MONTHS[m]
        for iv in INTERVENTIONS:
            if iv['kind'] == 'pulse' and iv['month'] == m:
                value *= 0.90
            if iv['kind'] == 'level_shift' and m >= iv['month']:
                value *= 1.05
        if spec['allow_nonpositive'] and m == 17:
            value = 0.0  # 明确的零值，检验对数退路
        values[m] = round(value, 4)
    return values


def main():
    calendar_map = {m: build_calendar(m) for m in range(1, N_MONTHS + 1)}
    months_rows, calendar_rows = [], []
    for m, (counts, holiday) in calendar_map.items():
        calendar_rows.append({'id': f'c{m}', 'revision': 1, 'month': m,
                              'weekday_counts': counts, 'holiday_overlap': holiday})
    for series, spec in SERIES_SPECS.items():
        values = simulate(spec, calendar_map)
        for m, value in values.items():
            batch_end = min(((m - 1) // PUBLISH_BATCH + 1) * PUBLISH_BATCH, N_MONTHS)
            published = batch_end + 5
            months_rows.append({'id': f'{series}_m{m}_r1', 'series': series,
                                'month': m, 'value': value,
                                'published': published, 'revision': 1})
            if m in REVISION_MONTHS:
                revised = round(value * REVISION_MONTHS[m], 4)
                months_rows.append({'id': f'{series}_m{m}_r2', 'series': series,
                                    'month': m, 'value': revised,
                                    'published': min(m + PUBLISH_BATCH + 5, N_MONTHS + 5),
                                    'revision': 2})
    payload = {
        'schema_version': 1,
        'metadata': {
            'data_version': 'synthetic-long-1',
            'as_of': N_MONTHS + 5,
            'interventions': INTERVENTIONS,
            'dgp': {
                'seed': SEED, 'n_months': N_MONTHS,
                'publish_batch_size': PUBLISH_BATCH,
                'revision_months': {str(k): v for k, v in REVISION_MONTHS.items()},
                'outlier_months': {str(k): v for k, v in OUTLIER_MONTHS.items()},
                'season_phase': SEASON_PHASE,
                'series_specs': SERIES_SPECS,
                'note': '合成数据不对应真实企业或个人',
            },
        },
        'collections': {'months': months_rows, 'calendar': calendar_rows},
    }
    out = Path(__file__).resolve().parent / 'sample_long.json'
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'written {out} with {len(months_rows)} month rows, {len(calendar_rows)} calendar rows')


if __name__ == '__main__':
    main()
