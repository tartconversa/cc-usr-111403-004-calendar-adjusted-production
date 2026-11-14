"""日历与回归设计矩阵：交易日对比、春节移动窗口、季节项与事前已知干预。

设计矩阵可在多个序列间复用（同一日历），但各序列的参数与调整结果独立估计。
所有日历信息均为事前固定资料：weekday_counts 与 holiday_overlap 来自 calendar 集合，
干预清单来自 metadata.interventions，绝不从产量序列反向构造。
"""
import math

from .linalg import column_norms, condition_number, rank_pivot

TD_NAMES = [f'td_{name}' for name in ('mon', 'tue', 'wed', 'thu', 'fri', 'sat')]
SEASONAL_SPECS = ('fixed', 'trig3', 'trig6')


def month_of_year(month_id):
    """支持两种月编码：连续小整数（1,2,...）与 YYYYMM。1-12 月返回值 1..12。"""
    if month_id >= 10000:
        return month_id % 100
    m = month_id % 12
    return 12 if m == 0 else m


def _intervention_column(kind, anchor, months):
    if kind == 'pulse':
        return [1.0 if m == anchor else 0.0 for m in months]
    if kind == 'level_shift':
        return [1.0 if m >= anchor else 0.0 for m in months]
    raise ValueError(f"未知干预类型: {kind}（应为 pulse 或 level_shift）")


class DesignFrame:
    """按月份对齐的可复用设计列仓库，与具体序列无关。"""

    def __init__(self, months, calendar_rows, interventions_meta):
        self.months = months
        self.pos = {m: i for i, m in enumerate(months)}
        self._calendar = calendar_rows
        self._interventions_meta = interventions_meta
        self.base_columns = []  # (名称, 分组, 列)
        self._build_base()

    def _build_base(self):
        n = len(self.months)
        self.base_columns.append(('intercept', 'intercept', [1.0] * n))
        td = [[] for _ in range(6)]
        hol = []
        for m in self.months:
            row = self._calendar[m]
            counts = row['weekday_counts']
            if len(counts) != 7 or any(x < 0 for x in counts):
                raise ValueError(f'月份 {m} 的 weekday_counts 无效')
            sunday = counts[6]
            for j in range(6):
                td[j].append(float(counts[j] - sunday))
            hol.append(float(row['holiday_overlap']))
        for j, name in enumerate(TD_NAMES):
            self.base_columns.append((name, 'trading', td[j]))
        self.base_columns.append(('holiday_spring', 'holiday', hol))
        for spec in SEASONAL_SPECS:
            for name, col in self._seasonal_columns(spec):
                self.base_columns.append((name, f'seasonal:{spec}', col))
        for iv in self._interventions_meta:
            kind = iv['kind']
            col = _intervention_column(kind, iv['month'], self.months)
            self.base_columns.append((f"iv:{iv['name']}", 'intervention', col))

    def _seasonal_columns(self, spec):
        if spec == 'fixed':
            cols = []
            for m_ref in range(2, 13):  # 1 月为基准月
                cols.append((f'seas_m{m_ref:02d}',
                             [1.0 if month_of_year(m) == m_ref else 0.0 for m in self.months]))
            return cols
        harmonics = 3 if spec == 'trig3' else 6
        cols = []
        for k in range(1, harmonics + 1):
            sin_col, cos_col = [], []
            for m in self.months:
                angle = 2.0 * 3.141592653589793 * k * (month_of_year(m) - 1) / 12.0
                sin_col.append(math.sin(angle))
                cos_col.append(math.cos(angle))
            cols.append((f'seas_sin{k}', sin_col))
            cols.append((f'seas_cos{k}', cos_col))
        return cols

    def active_interventions(self, cutoff):
        """返回该信息截止时点已知的干预元数据。"""
        out = []
        for iv in self._interventions_meta:
            if iv.get('active_as_of', -10 ** 18) <= cutoff:
                out.append(iv)
        return out

    def window(self, end_month):
        """返回只含 <= end_month 月份的子 DesignFrame，基础列按子窗口重建。

        供无前瞻滚动检测使用：未来才发生的干预在子窗口内为零列并被剔除。
        """
        months = [m for m in self.months if m <= end_month]
        sub = DesignFrame.__new__(DesignFrame)
        sub.months = months
        sub.pos = {m: i for i, m in enumerate(months)}
        sub._calendar = self._calendar
        sub._interventions_meta = self._interventions_meta
        sub.base_columns = []
        sub._build_base()
        return sub

    def build(self, seasonal_spec, cutoff, outlier_dummies=None):
        """组装某序列、某 vintage 下的完整设计矩阵。

        - 只纳入 cutoff 时已知的干预（active_as_of）；
        - 已知但尚未在当前窗口发生的干预（脉冲月晚于窗口末端、或水平迁移
          起点晚于窗口末端）会形成零列，予以剔除并记录，避免秩亏；
        - outlier_dummies: [(name, {month_id: 1.0 脉冲})]，由无前瞻异常检测产生。
        """
        if seasonal_spec not in SEASONAL_SPECS:
            raise ValueError(f'季节规格应为 {SEASONAL_SPECS} 之一')
        window_end = self.months[-1] if self.months else cutoff
        active_iv = {f"iv:{iv['name']}" for iv in self.active_interventions(cutoff)}
        names, groups, cols, dropped = [], [], [], []
        for name, group, col in self.base_columns:
            if group.startswith('seasonal:') and group != f'seasonal:{seasonal_spec}':
                continue
            if group == 'intervention':
                if name not in active_iv:
                    continue
                if all(abs(v) < 1e-300 for v in col):
                    dropped.append({'name': name, 'reason': '干预在当前窗口尚未发生（零列剔除）'})
                    continue
            names.append(name)
            groups.append(group.split(':')[0])
            cols.append(col)
        for name, pulses in (outlier_dummies or []):
            names.append(name)
            groups.append('outlier')
            cols.append([pulses.get(m, 0.0) for m in self.months])
        matrix = [[cols[j][i] for j in range(len(cols))] for i in range(len(self.months))]
        dm = DesignMatrix(self.months, names, groups, matrix, cutoff)
        dm.dropped_interventions = dropped
        return dm


class DesignMatrix:
    """具体一次估计使用的设计矩阵：月份、列名、列分组与数值矩阵。"""

    def __init__(self, months, names, groups, matrix, cutoff):
        self.months = months
        self.names = names
        self.groups = groups
        self.matrix = matrix
        self.cutoff = cutoff
        self.n_obs = len(matrix)
        self.n_cols = len(names)
        self.dropped_interventions = []

    def column(self, name):
        j = self.names.index(name)
        return [self.matrix[i][j] for i in range(self.n_obs)]

    def group_indices(self, group):
        return [j for j, g in enumerate(self.groups) if g == group]

    def rank_report(self, tol=1e-10):
        norms = column_norms(self.matrix)
        zero_cols = [self.names[j] for j, nm in enumerate(norms) if nm <= 1e-300]
        try:
            rank, pivots, _ = rank_pivot(self.matrix, tol)
            cnum = condition_number([[self.matrix[i][j] / norms[j] for j in range(self.n_cols)]
                                     for i in range(self.n_obs)]) if not zero_cols else math.inf
        except ValueError:
            rank, cnum = None, math.inf
        return {
            'n_obs': self.n_obs, 'n_cols': self.n_cols, 'rank': rank,
            'full_rank': rank == self.n_cols if rank is not None else False,
            'zero_columns': zero_cols,
            'condition_number': (cnum if math.isfinite(cnum) else 'infinite'),
            'pivot_columns': [self.names[j] for j in pivots],
            'tolerance': tol,
        }
