"""无前瞻的异常观测检测：按时间顺序滚动，异常判定只用截至该月已见数据。

流程（对每个 vintage 同样适用；在实时分析中对最终 vintage 运行一次后，
旧 vintage 只取用其 cutoff 之前已确认的脉冲，天然保持因果性）：
1. 基设计含截距、交易日、春节窗口、固定季节虚拟与当时已发生的事前干预；
2. 逐月纳入观测，设计列集合随窗口变化（干预在发生月进入）；每步用列主元
   QR 最小二乘重拟合，对脉冲列早期仅单个非零元的近共线情形稳健；
3. 对当月计算外部学生化残差（externally studentized），超过阈值才标记脉冲；
4. 脉冲数设上限，避免虚拟饱和。

任何月份的标记决策不依赖其后任何观测，因此杜绝“用全样本平滑结果挑异常”。
"""
from .linalg import qr_solve, matvec


class WindowReg:
    """截至当前窗口的最小二乘状态，基于列主元 QR，支持设计列变化。"""

    def __init__(self, x, y):
        self.x = [row[:] for row in x]
        self.y = y[:]
        self._fit()

    def _fit(self):
        self.beta, self.pivots, self.rank = qr_solve(self.x, self.y)
        fitted = matvec(self.x, self.beta)
        self.resid = [self.y[i] - fitted[i] for i in range(len(self.y))]
        self.sse = sum(e * e for e in self.resid)
        self.k = len(self.x[0])
        self.dof = len(self.y) - self.rank

    def append(self, row, value):
        self.x.append(row[:])
        self.y.append(value)
        self._fit()

    def external_studentized(self, row, value):
        """新观测不参与拟合 -> 外部学生化残差（leave-one-out 口径）。

        杠杆 h = x0'(X'X)^- x0，用列主元 QR 的主元列三角因子求逆得到广义逆。
        """
        n = len(self.y)
        if self.dof <= 2:
            return None, None
        pred = sum(row[j] * self.beta[j] for j in range(self.k))
        resid = value - pred
        leverage = self._leverage(row)
        if leverage is None:
            return None, resid
        # h 可能 >1：新行落在历史列空间外推区域（设计外推），此时 (1+h) 仍正定；
        # 仅当完全共线（h 数值发散）才放弃。外推信息记录在轨迹中供研究者判断。
        self._last_leverage = leverage
        s2 = self.sse / self.dof
        return resid / (s2 * (1.0 + leverage)) ** 0.5, resid

    def _leverage(self, row):
        """新行杠杆 h=x0'(X'X)^-x0。

        直接对 X'X 求逆在脉冲列仅含单个非零元时病态，故先把各设计列标准化为
        单位长度，在标准化空间用 Cholesky（失败再列主元消元）求逆，最后还原。
        """
        import math
        from .linalg import matmul, transpose, inverse_spd, solve
        n, k = len(self.x), self.k
        scales = [math.sqrt(sum(self.x[i][j] ** 2 for i in range(n))) or 1.0
                  for j in range(k)]
        xs = [[self.x[i][j] / scales[j] for j in range(k)] for i in range(n)]
        try:
            inv_s = inverse_spd(matmul(transpose(xs), xs))
        except ValueError:
            xtx = matmul(transpose(xs), xs)
            try:
                inv_s = transpose([solve(xtx, [1.0 if i == c else 0.0 for i in range(k)])
                                   for c in range(k)])
            except ValueError:
                return None
        row_s = [row[j] / scales[j] for j in range(k)]
        mv = [sum(inv_s[a][b] * row_s[b] for b in range(k)) for a in range(k)]
        return sum(row_s[a] * mv[a] for a in range(k))


def detect_outliers(frame, y, cutoff, threshold=3.5, max_share=0.05,
                    seasonal_spec='fixed', warmup=None):
    """因果滚动异常检测，返回 (脉冲列表 [(name, {month: 1.0})], 检测轨迹)。

    评估第 i 个月时设计矩阵只含截至该月的行与已发生干预，无未来信息。
    """
    n = len(y)
    cap = max(1, int(max_share * n))
    pulses, trace, pulse_months = [], [], []

    def window_design(end_i):
        sub = frame.window(frame.months[end_i])
        return sub.build(seasonal_spec, cutoff, None)

    start_i = 0
    for i in range(n):
        dm = window_design(i)
        if dm.n_obs > dm.n_cols + 1:
            start_i = i
            break
    if warmup is not None:
        start_i = max(start_i, warmup - 1)

    col_names = list(window_design(start_i).names)
    history_x, history_y = [], []

    def aligned_row(end_i):
        dm = window_design(end_i)
        lookup = {dm.names[j]: dm.matrix[-1][j] for j in range(dm.n_cols)}
        pulse_set = {f'outlier_pulse_{frame.months[pos]}' for pos in pulse_months}
        names = [nm for nm in col_names]
        row = []
        for nm in names:
            if nm in pulse_set:
                pos_month = int(nm.split('_')[-1])
                row.append(1.0 if pos_month == frame.months[end_i] else 0.0)
            else:
                row.append(lookup.get(nm, 0.0))
        return row, dm.names

    # 初始化历史（截至 start_i）
    history_x, history_y = [], []
    for h in range(start_i + 1):
        history_x.append(aligned_row(h)[0])
        history_y.append(y[h])
    reg = WindowReg(history_x, history_y)

    def rebuild(through_i):
        rows = [aligned_row(h)[0] for h in range(through_i + 1)]
        return WindowReg(rows, y[:through_i + 1])

    for i in range(start_i + 1, n):
        current_base = window_design(i).names
        base_in_cols = [c for c in col_names if not c.startswith('outlier_')]
        if len(current_base) != len(base_in_cols):
            # 新干预进入窗口：重建列名后对对齐历史重拟合
            col_names = list(current_base) + [
                f'outlier_pulse_{frame.months[pos]}' for pos in pulse_months]
            reg = rebuild(i - 1)
        row, _ = aligned_row(i)
        tstat, raw = reg.external_studentized(row, y[i])
        leverage = getattr(reg, '_last_leverage', None)
        extrapolation = (leverage or 0.0) > 1.0
        decision = {'month': frame.months[i], 'position': i,
                    'studentized_residual': tstat, 'raw_residual': raw,
                    'leverage': leverage,
                    'design_extrapolation': extrapolation,
                    'threshold': threshold, 'flagged': False}
        # 设计外推点（h>1）的学生化值不可靠，不标记为异常，仅留痕
        flagged = (tstat is not None and not extrapolation
                   and abs(tstat) > threshold and len(pulses) < cap)
        if flagged:
            name = f'outlier_pulse_{frame.months[i]}'
            col_names.append(name)
            pulse_months.append(i)
            pulses.append((name, {frame.months[i]: 1.0}))
            decision['flagged'] = True
            decision['dummy'] = name
        # 无论是否标记都纳入当前观测；标记后列集合已变，需整体重拟合
        reg = rebuild(i) if flagged else WindowReg(reg.x + [row], reg.y + [y[i]])
        trace.append(decision)
    return pulses, trace
