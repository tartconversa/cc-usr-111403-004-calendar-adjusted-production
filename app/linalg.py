"""纯标准库线性代数：OLS、Cholesky、秩诊断与对称矩阵特征值。

仅依赖 math，矩阵为 list[list[float]]、向量为 list[float]。
规模面向月度资料（n 数百、列数十），不追求大规模性能。
"""
import math


def matmul(a, b):
    n, k, m = len(a), len(b), len(b[0])
    bt = list(zip(*b))
    return [[sum(a[i][t] * bt[j][t] for t in range(k)) for j in range(m)] for i in range(n)]


def matvec(a, v):
    return [sum(row[j] * v[j] for j in range(len(v))) for row in a]


def transpose(a):
    return [list(col) for col in zip(*a)]


def identity(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def solve(a, b):
    """高斯消元（部分主元）解 A x = b，A 须非奇异。"""
    n = len(a)
    m = [a[i][:] + [b[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-14:
            raise ValueError('矩阵奇异或近乎奇异')
        if pivot != col:
            m[col], m[pivot] = m[pivot], m[col]
        piv = m[col][col]
        for r in range(col + 1, n):
            factor = m[r][col] / piv
            if factor:
                m[r][col] = 0.0
                for c in range(col + 1, n + 1):
                    m[r][c] -= factor * m[col][c]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (m[i][n] - sum(m[i][j] * x[j] for j in range(i + 1, n))) / m[i][i]
    return x


def qr_solve(x, y, tol=1e-11):
    """列主元修正 Gram-Schmidt 最小二乘，对近共线稳健，返回 (beta, 主元, 秩)。"""
    n, k = len(x), len(x[0])
    norms = column_norms(x)
    qs = []  # 正交单位向量
    r_rows = [[] for _ in range(k)]
    pivots, used = [], set()
    remaining = list(range(k))
    # 逐列选主元
    work = [[x[i][j] for i in range(n)] for j in range(k)]
    rank = 0
    scale = max(norms) if norms else 1.0
    for _ in range(k):
        if not remaining:
            break
        j = max(remaining, key=lambda c: math.sqrt(sum(v * v for v in work[c])))
        nm = math.sqrt(sum(v * v for v in work[j]))
        if nm <= tol * max(scale, 1e-300):
            break
        q = [v / nm for v in work[j]]
        qs.append(q)
        pivots.append(j)
        remaining.remove(j)
        rank += 1
        for c in remaining:
            proj = sum(q[i] * work[c][i] for i in range(n))
            work[c] = [work[c][i] - proj * q[i] for i in range(n)]
    # R 矩阵（主元列上三角）与 Q'y
    qty = [sum(qs[r][i] * y[i] for i in range(n)) for r in range(rank)]
    rmat = [[0.0] * rank for _ in range(rank)]
    for r in range(rank):
        for c in range(r, rank):
            col = pivots[c]
            q = qs[r]
            rmat[r][c] = sum(q[i] * x[i][col] for i in range(n))
    # 回代解 R beta_piv = Q'y
    beta_piv = [0.0] * rank
    for r in range(rank - 1, -1, -1):
        beta_piv[r] = (qty[r] - sum(rmat[r][c] * beta_piv[c] for c in range(r + 1, rank))) / rmat[r][r]
    beta = [0.0] * k
    for r, col in enumerate(pivots):
        beta[col] = beta_piv[r]
    return beta, pivots, rank


def hat_diag_qr(x, tol=1e-11):
    """利用列主元 QR 计算帽子矩阵对角（投影到列空间的杠杆值）。"""
    n, k = len(x), len(x[0])
    norms = column_norms(x)
    scale = max(norms) if norms else 1.0
    work = [[x[i][j] for i in range(n)] for j in range(k)]
    qs, remaining, rank = [], list(range(k)), 0
    for _ in range(k):
        if not remaining:
            break
        j = max(remaining, key=lambda c: math.sqrt(sum(v * v for v in work[c])))
        nm = math.sqrt(sum(v * v for v in work[j]))
        if nm <= tol * max(scale, 1e-300):
            break
        q = [v / nm for v in work[j]]
        qs.append(q)
        remaining.remove(j)
        rank += 1
        for c in remaining:
            proj = sum(q[i] * work[c][i] for i in range(n))
            work[c] = [work[c][i] - proj * q[i] for i in range(n)]
    return [sum(q[i] ** 2 for q in qs) for i in range(n)], rank


def cholesky(a):
    """对称正定矩阵的 Cholesky 下三角 L（A = L L'）。"""
    n = len(a)
    l = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(l[i][k] * l[j][k] for k in range(j))
            if i == j:
                d = a[i][i] - s
                if d <= 1e-18:  # 近奇异由上层 QR/消元回退处理
                    raise ValueError('矩阵非正定')
                l[i][j] = math.sqrt(d)
            else:
                l[i][j] = (a[i][j] - s) / l[j][j]
    return l


def inverse_spd(a):
    """正定矩阵求逆，基于 Cholesky 前代/回代。"""
    n = len(a)
    l = cholesky(a)
    cols = []
    for k in range(n):
        y = [0.0] * n
        for i in range(n):
            y[i] = (1.0 if i == k else 0.0) - sum(l[i][j] * y[j] for j in range(i))
            y[i] /= l[i][i]
        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            x[i] = (y[i] - sum(l[j][i] * x[j] for j in range(i + 1, n))) / l[i][i]
        cols.append(x)
    return transpose(cols)


def ols(x, y):
    """普通最小二乘，返回 beta、残差、SSE 与 (X'X)^-1。

    正定情形走 Cholesky（快速且稳定）；近奇异时回退列主元 QR 最小二乘，
    此时不返回协方差逆矩阵（调用方仅需残差/SSE 的诊断场景）。
    """
    n, k = len(x), len(x[0])
    if n <= k:
        raise ValueError('观测数不足以估计设计矩阵')
    xt = transpose(x)
    xtx = matmul(xt, x)
    xty = matvec(xt, y)
    try:
        xtx_inv = inverse_spd(xtx)
        beta = matvec(xtx_inv, xty)
    except ValueError:
        beta, pivots, rank = qr_solve(x, y)
        if len(beta) != k:
            raise ValueError('OLS 回退求解维度不一致')
        xtx_inv = None
    fitted = matvec(x, beta)
    resid = [y[i] - fitted[i] for i in range(n)]
    sse = sum(e * e for e in resid)
    return beta, resid, sse, xtx_inv


def column_norms(x):
    return [math.sqrt(sum(x[i][j] ** 2 for i in range(len(x)))) for j in range(len(x[0]))]


def rank_pivot(x, tol=1e-10):
    """列主元修正 Gram-Schmidt，返回 (秩, 主元列下标, 列归一化后的 R 对角)。"""
    n, k = len(x), len(x[0])
    norms = column_norms(x)
    cols = []
    for j in range(k):
        nm = norms[j]
        cols.append([x[i][j] / nm for i in range(n)] if nm > 0 else [0.0] * n)
    available = set(range(k))
    pivots, diag = [], []
    rank = 0
    scale = None
    while available:
        j = max(available, key=lambda c: math.sqrt(sum(v * v for v in cols[c])))
        nm = math.sqrt(sum(v * v for v in cols[j]))
        if scale is None:
            scale = nm
        if nm <= tol * max(scale, 1e-300):
            break
        available.remove(j)
        pivots.append(j)
        diag.append(nm)
        rank += 1
        q = cols[j]
        for c in available:
            proj = sum(q[i] * cols[c][i] for i in range(n))
            cols[c] = [cols[c][i] - proj * q[i] for i in range(n)]
    return rank, pivots, diag


def jacobi_eigen(a, tol=1e-13, max_sweeps=80):
    """对称矩阵特征值（Jacobi 旋转），返回升序特征值。"""
    n = len(a)
    if n == 1:
        return [a[0][0]]
    m = [row[:] for row in a]
    for _ in range(max_sweeps):
        off = max((abs(m[i][j]) for i in range(n) for j in range(i + 1, n)), default=0.0)
        diag_scale = math.sqrt(sum(m[i][i] ** 2 for i in range(n)) / n)
        if off <= tol * max(diag_scale, 1e-300):
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(m[p][q]) <= 1e-300:
                    continue
                tau = (m[q][q] - m[p][p]) / (2.0 * m[p][q])
                t = math.copysign(1.0 / (abs(tau) + math.sqrt(1.0 + tau * tau)), tau) if tau != 0.0 else 1.0
                c = 1.0 / math.sqrt(1.0 + t * t)
                s = t * c
                for r in range(n):
                    if r != p and r != q:
                        arp, arq = m[r][p], m[r][q]
                        m[r][p] = c * arp - s * arq
                        m[p][r] = m[r][p]
                        m[r][q] = s * arp + c * arq
                        m[q][r] = m[r][q]
                app, aqq, apq = m[p][p], m[q][q], m[p][q]
                m[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq
                m[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq
                m[p][q] = m[q][p] = 0.0
    return sorted(m[i][i] for i in range(n))


def singular_values(x):
    """奇异值 = X'X 特征值的平方根（X 已按列归一化时条件数稳定）。"""
    xtx = matmul(transpose(x), x)
    vals = jacobi_eigen(xtx)
    return [math.sqrt(max(v, 0.0)) for v in vals]


def condition_number(x):
    sv = singular_values(x)
    smin = sv[0]
    if smin <= 1e-14:
        return math.inf
    return sv[-1] / smin
