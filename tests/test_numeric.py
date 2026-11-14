"""线性代数与统计分布的数值正确性（纯标准库实现）。"""
import math
import unittest

from app.linalg import ols, qr_solve, jacobi_eigen, condition_number, rank_pivot
from app.stats import f_sf, chi2_sf, ljung_box, seasonal_dummy_test, acf


class LinalgTests(unittest.TestCase):
    def test_ols_recovers_known_coefficients(self):
        x = [[1, 1], [1, 2], [1, 3], [1, 4], [1, 5]]
        y = [3 + 2 * r[1] for r in x]
        beta, resid, sse, inv = ols(x, y)
        self.assertAlmostEqual(beta[0], 3.0, places=10)
        self.assertAlmostEqual(beta[1], 2.0, places=10)
        self.assertAlmostEqual(sse, 0.0, places=12)
        self.assertIsNotNone(inv)

    def test_ols_fallback_on_nearly_singular_design(self):
        # 两列近乎相同 -> Cholesky 失败，QR 回退仍给出有限 beta
        x = [[1, 1.0], [1, 1.0000001], [1, 2.0], [1, 3.0], [1, 4.0], [1, 5.0]]
        y = [2.0, 2.1, 4.0, 6.0, 8.0, 10.0]
        beta, resid, sse, inv = ols(x, y)
        self.assertEqual(len(beta), 2)
        self.assertTrue(all(math.isfinite(b) for b in beta))

    def test_qr_solve_dimensions_and_rank(self):
        x = [[1, 0, 1], [1, 1, 0], [1, 2, 0], [1, 3, 1], [1, 4, 0]]
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        beta, pivots, rank = qr_solve(x, y)
        self.assertEqual(rank, 3)
        self.assertEqual(len(beta), 3)

    def test_jacobi_eigenvalues_sorted(self):
        vals = jacobi_eigen([[2.0, 1.0], [1.0, 2.0]])
        self.assertAlmostEqual(vals[0], 1.0, places=10)
        self.assertAlmostEqual(vals[1], 3.0, places=10)

    def test_rank_pivot_detects_zero_column(self):
        x = [[1, 0, 1], [1, 0, 2], [1, 0, 3], [1, 0, 4]]
        rank, pivots, _ = rank_pivot(x)
        self.assertEqual(rank, 2)


class DistributionTests(unittest.TestCase):
    def test_chi2_known_critical_values(self):
        # df=1 在 3.841 处上尾约 0.05；df=2 在 5.991 处约 0.05
        self.assertAlmostEqual(chi2_sf(3.841, 1), 0.05, places=2)
        self.assertAlmostEqual(chi2_sf(5.991, 2), 0.05, places=2)

    def test_f_known_critical_value(self):
        # F(2,10) 在 4.103 处上尾约 0.05
        self.assertAlmostEqual(f_sf(4.103, 2, 10), 0.05, places=2)

    def test_ljung_box_white_noise_not_rejected(self):
        # 固定种子的独立高斯白噪声，Ljung-Box 不应拒绝
        import random
        rng = random.Random(11)
        e = [rng.gauss(0, 1.0) for _ in range(120)]
        res = ljung_box(e, 12)
        self.assertTrue(res['evaluated'])
        self.assertGreater(res['p_value'], 0.05)

    def test_ljung_box_detects_strong_autocorrelation(self):
        # AR(1) 强持续序列应被 Ljung-Box 拒绝
        e, prev = [], 0.0
        for _ in range(120):
            prev = 0.9 * prev + 0.01
            e.append(prev)
        res = ljung_box(e, 12)
        self.assertTrue(res['evaluated'])
        self.assertLess(res['p_value'], 0.05)

    def test_seasonal_test_detects_seasonal_residual(self):
        # 强 12 月周期季节信号 + 轻微噪声（保证组内方差非零、F 检验可评估）
        import random
        rng = random.Random(3)
        phase = [1.5, -1.5, 1.5, -1.5, 1.5, -1.5, 1.5, -1.5, 1.5, -1.5, 1.5, -1.5]
        resid = [phase[i % 12] + rng.gauss(0, 0.05) for i in range(72)]
        months = list(range(1, 73))
        out = seasonal_dummy_test(resid, months)
        self.assertTrue(out['evaluated'])
        self.assertLess(out['p_value'], 0.05)

    def test_seasonal_test_clean_residual_passes(self):
        # 固定种子的温和白噪声，季节检验不应拒绝
        import random
        rng = random.Random(7)
        resid = [rng.gauss(0, 0.01) for _ in range(72)]
        out = seasonal_dummy_test(resid, list(range(1, 73)))
        self.assertTrue(out['evaluated'])
        self.assertGreater(out['p_value'], 0.01)


if __name__ == '__main__':
    unittest.main()
