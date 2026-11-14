"""纯标准库统计工具：分布函数、自相关与白噪声/季节诊断。

分布函数采用 Numerical Recipes 风格的连分式，仅供月度规模诊断使用。
"""
import math


# ---------------------------------------------------------------- 分布函数
def _gammap_ser(a, x):
    if x <= 0:
        return 0.0
    ap, total, term = a, 1.0 / a, 1.0 / a
    for _ in range(200):
        ap += 1.0
        term *= x / ap
        total += term
        if abs(term) < abs(total) * 1e-14:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gammaq_cf(a, x):
    b = x + 1.0 - a
    c = 1e300
    d = 1.0 / b
    h = d
    for i in range(1, 201):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-300:
            d = 1e-300
        c = b + an / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def gamma_p(a, x):
    """正则化下不完全伽马 P(a, x)。"""
    if x <= 0:
        return 0.0
    if x < a + 1.0:
        return _gammap_ser(a, x)
    return 1.0 - _gammaq_cf(a, x)


def chi2_sf(x, df):
    """卡方上尾概率。"""
    if x <= 0 or df <= 0:
        return 1.0
    return max(0.0, 1.0 - gamma_p(df / 2.0, x / 2.0))


def _betacf(a, b, x):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    h = d
    for m in range(1, 201):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h


def beta_i(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x)) / a
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x)
    return 1.0 - math.exp(lbeta + b * math.log(1.0 - x) + a * math.log(x)) / b * _betacf(b, a, 1.0 - x)


def f_sf(x, d1, d2):
    """F 分布上尾概率。"""
    if x <= 0 or d1 <= 0 or d2 <= 0:
        return 1.0
    return max(0.0, 1.0 - beta_i(d1 / 2.0, d2 / 2.0, (d1 * x) / (d1 * x + d2)))


def normal_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


# ---------------------------------------------------------------- 序列诊断
def acf(x, max_lag):
    n = len(x)
    mean = sum(x) / n
    dev = [v - mean for v in x]
    denom = sum(d * d for d in dev)
    if denom <= 1e-300:
        return [0.0] * (max_lag + 1)
    out = []
    for lag in range(max_lag + 1):
        out.append(1.0 if lag == 0 else sum(dev[i] * dev[i - lag] for i in range(lag, n)) / denom)
    return out


def ljung_box(x, max_lag, ar_terms=0):
    """Ljung-Box 多阶自相关检验，返回统计量与上尾 p 值（自由度扣减 AR 项）。"""
    n = len(x)
    if max_lag <= ar_terms or n < max_lag + 4:
        return {'statistic': None, 'p_value': None, 'df': max(0, max_lag - ar_terms), 'evaluated': False}
    rho = acf(x, max_lag)
    q = n * (n + 2) * sum(rho[k] ** 2 / (n - k) for k in range(1, max_lag + 1))
    df = max_lag - ar_terms
    return {'statistic': q, 'p_value': chi2_sf(q, df), 'df': df, 'evaluated': True}


def seasonal_dummy_test(resid, months_index):
    """残留季节性检验：按 12 个月份分组的单因素 ANOVA（F 检验）。

    直接由分组均值与组内方差构造，避免月份虚拟矩阵近奇异时的求逆问题。
    零假设为各月残差均值相等。自由度不足或组内方差近零时如实标记不可评估。
    """
    n = len(resid)
    groups = {m: [] for m in range(1, 13)}
    for i, e in enumerate(resid):
        groups[months_index[i] % 12 or 12].append(e)
    present = [m for m in range(1, 13) if groups[m]]
    means = {m: (sum(groups[m]) / len(groups[m])) for m in present}
    grand = sum(resid) / n
    k = len(present)
    d2 = n - k
    if k < 2 or d2 <= 0:
        return {'statistic': None, 'p_value': None, 'evaluated': False,
                'reason': f'可识别月份组数 {k} 不足或自由度 d2={d2} <= 0'}
    ssb = sum(len(groups[m]) * (means[m] - grand) ** 2 for m in present)
    sse = sum((e - means[m]) ** 2 for m in present for e in groups[m])
    if sse <= 1e-300:
        return {'statistic': None, 'p_value': None, 'evaluated': False,
                'reason': '组内残差方差近零（季节信号已被设计完全吸收）',
                'month_means': means}
    f = (ssb / (k - 1)) / (sse / d2)
    full_means = {m: means.get(m) for m in range(1, 13)}
    return {'statistic': f, 'p_value': f_sf(f, k - 1, d2),
            'df_num': k - 1, 'df_den': d2,
            'month_means': full_means, 'evaluated': True}
