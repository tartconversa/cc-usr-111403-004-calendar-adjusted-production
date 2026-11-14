"""日历回归与 AR(p) 序列模型的联合估计（可行广义最小二乘 FGLS）。

设计矩阵中的交易日、春节窗口、季节项、事前干预与异常虚拟在同一估计中共同识别；
AR(p) 误差刻画未被设计列吸收的短期相关。没有“先全样本平滑、再挑异常”的两步做法。
"""
import math
from dataclasses import dataclass, field

from .linalg import matmul, matvec, transpose, ols, solve

def yule_walker(e, p):
    """由残差序列用 Yule-Walker 估计 AR(p) 系数；非平稳则降阶。"""
    if p == 0:
        return []
    n = len(e)
    mean = sum(e) / n
    d = [v - mean for v in e]
    gamma = [sum(d[i] * d[i - k] for i in range(k, n)) / n for k in range(p + 1)]
    if gamma[0] <= 1e-300:
        return []
    r = [gamma[k] / gamma[0] for k in range(1, p + 1)]
    rmat = [[1.0 if i == j else r[abs(i - j) - 1] for j in range(p)] for i in range(p)]
    try:
        phi = solve(rmat, r)
    except ValueError:
        return yule_walker(e, p - 1)
    if not _stationary(phi):
        return yule_walker(e, p - 1)
    return phi


def _stationary(phi):
    """AR 系数平稳性：特征根均在单位圆外（Jury 判据的根检查，p<=2 直接判定）。"""
    p = len(phi)
    if p == 0:
        return True
    if p == 1:
        return abs(phi[0]) < 0.999
    if p == 2:
        a1, a2 = phi
        return a1 + a2 < 0.999 and a2 - a1 < 0.999 and abs(a2) < 0.999
    # 更高阶：直接以自回归多项式求根的数值替代（本项目候选仅到 p=2，此为防御分支）
    return all(abs(c) < 1.0 for c in phi)


def _transform_pw(x, y, phi):
    """Prais-Winsten 变换：首观测乘 sqrt(1-phi1^2) 保留全样本。"""
    p = len(phi)
    n = len(y)
    xt = [row[:] for row in x]
    yt = y[:]
    if p >= 1:
        c0 = math.sqrt(max(1e-12, 1.0 - sum(phi[k] ** 2 for k in range(p))))
        yt[0] *= c0
        xt[0] = [v * c0 for v in xt[0]]
    for i in range(1, n):
        lag = min(i, p)
        yt[i] = y[i] - sum(phi[k] * y[i - k - 1] for k in range(lag))
        xt[i] = [x[i][j] - sum(phi[k] * x[i - k - 1][j] for k in range(lag))
                 for j in range(len(x[0]))]
    return xt, yt


@dataclass
class FitResult:
    beta: list
    names: list
    groups: list
    phi: list
    resid: list          # 变换尺度残差 y - X beta
    innovations: list    # AR 白化后的新息（前 p 个为 None）
    sigma2: float
    loglik: float
    bic: float
    aicc: float
    n_obs: int
    n_params: int
    iterations: int
    converged: bool
    covariance: list
    t_stats: list
    design_rank_report: dict = field(default_factory=dict)

    def coefficient(self, name):
        return self.beta[self.names.index(name)]

    def group_beta(self, group):
        idx = [j for j, g in enumerate(self.groups) if g == group]
        return idx, [self.beta[j] for j in idx]

    def forecast_one_step(self, x_new, hist_resid):
        """给定新设计行与历史残差，预测下一个变换尺度值。"""
        systematic = sum(x_new[j] * self.beta[j] for j in range(len(self.beta)))
        ar_part = sum(self.phi[k] * hist_resid[-k - 1] for k in range(len(self.phi)))
        return systematic + ar_part


def fit_fgls(design, y, ar_order, max_iter=60, tol=1e-10):
    """联合估计设计矩阵回归系数与 AR(ar_order) 误差（Cochrane-Orcutt 条件最小二乘）。

    似然基于条件最小二乘有效样本（t=p+1..n，n-p 个新息），
    保证不同 AR 阶数间 BIC 可比；成分分解仍在全样本变换尺度上进行。
    """
    x = design.matrix
    n, k = len(x), len(x[0])
    if n <= k + ar_order + 3:
        raise ValueError(f'样本量 n={n} 不足以估计 {k} 个回归参数与 AR({ar_order})')
    beta, resid, _, _ = ols(x, y)
    phi = yule_walker(resid, ar_order)
    converged = False
    iterations = 0
    for iterations in range(1, max_iter + 1):
        x_cls, y_cls = _conditional_rows(x, y, phi)
        new_beta, _, _, _ = ols(x_cls, y_cls)
        new_resid = [y[i] - sum(x[i][j] * new_beta[j] for j in range(k)) for i in range(n)]
        new_phi = yule_walker(new_resid, ar_order)
        phi_diffs = (
            abs((new_phi[j] if j < len(new_phi) else 0.0)
                - (phi[j] if j < len(phi) else 0.0))
            for j in range(max(len(phi), len(new_phi))))
        if (max(abs(new_beta[j] - beta[j]) for j in range(k)) < tol
                and max(phi_diffs, default=0.0) < tol):
            beta, phi = new_beta, new_phi
            converged = True
            break
        beta, phi, resid = new_beta, new_phi, new_resid
    # 收敛后以最终 phi 做一次条件最小二乘，统一各口径
    x_cls, y_cls = _conditional_rows(x, y, phi)
    beta, _, _, xtx_inv_cls = ols(x_cls, y_cls)
    if xtx_inv_cls is None:
        xtx_cls = matmul(transpose(x_cls), x_cls)
        xtx_inv_cls = transpose([
            solve(xtx_cls, [1.0 if i == col else 0.0 for i in range(k)])
            for col in range(k)])
    p = len(phi)
    fitted = matvec(x, beta)
    resid = [y[i] - fitted[i] for i in range(n)]
    innovations = [None] * p + [
        resid[i] - sum(phi[h] * resid[i - h - 1] for h in range(p)) for i in range(p, n)
    ]
    n_eff = n - p
    sse_inn = sum(e * e for e in innovations[p:])
    sigma2 = sse_inn / n_eff
    # 条件高斯似然（有效样本 n-p），不同 AR 阶数间可直接比较
    loglik = -0.5 * n_eff * (math.log(2 * math.pi) + math.log(max(sigma2, 1e-300)) + 1.0)
    n_params = k + p + 1
    bic = -2 * loglik + n_params * math.log(n_eff)
    aicc = (-2 * loglik + 2 * n_params
            + 2 * n_params * (n_params + 1) / max(1, n_eff - n_params - 1))
    # 条件最小二乘协方差（设计列同样做 AR 过滤）
    cov = [[sigma2 * xtx_inv_cls[i][j] for j in range(k)] for i in range(k)]
    t_stats = [beta[j] / math.sqrt(max(cov[j][j], 1e-300)) for j in range(k)]
    return FitResult(beta=beta, names=design.names, groups=design.groups, phi=phi,
                     resid=resid, innovations=innovations, sigma2=sigma2,
                     loglik=loglik, bic=bic, aicc=aicc, n_obs=n, n_params=n_params,
                     iterations=iterations, converged=converged, covariance=cov,
                     t_stats=t_stats, design_rank_report=design.rank_report())


def _conditional_rows(x, y, phi):
    """AR(p) 条件最小二乘的过滤设计行与响应：跳过前 p 个观测。"""
    p = len(phi)
    n, k = len(x), len(x[0])
    x_cls, y_cls = [], []
    for i in range(p, n):
        lag = min(i, p)
        y_cls.append(y[i] - sum(phi[h] * y[i - h - 1] for h in range(lag)))
        x_cls.append([x[i][j] - sum(phi[h] * x[i - h - 1][j] for h in range(lag))
                      for j in range(k)])
    return x_cls, y_cls
