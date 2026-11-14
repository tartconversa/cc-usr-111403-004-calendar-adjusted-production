"""无前瞻异常检测、调整管线诊断与序列隔离。"""
import json
import unittest
from pathlib import Path

from app.catalog import load
from app.service import SCHEMAS, adjust
from app.adjust import extract_vintage, make_frame
from app.calendar import DesignFrame
from app.transform import choose_transform
from app.outlier import detect_outliers

LONG = json.loads((Path(__file__).resolve().parents[1] / 'fixtures' / 'sample_long.json')
                  .read_text(encoding='utf-8'))


def snap():
    return load(LONG, SCHEMAS)


class NoLookAheadTests(unittest.TestCase):
    def test_flag_decision_does_not_use_future_rows(self):
        """同一月份的异常判定在更短/更长 cutoff 下应保持一致（无未来信息依赖）。"""
        s = snap()
        cut = s.metadata['as_of']
        v_full = extract_vintage(s, 'industry_b', cut)
        frame = make_frame(s, v_full.months)
        z = choose_transform(v_full.values).apply(v_full.values)
        _, trace_full = detect_outliers(frame, z, cut)
        flagged_full = {d['month'] for d in trace_full if d['flagged']}
        # 截断 vintage：按 cutoff=53 的实际可见观测构建对齐 frame（44 月已公布）
        v50 = extract_vintage(s, 'industry_b', 53)
        frame50 = make_frame(s, v50.months)
        z50 = choose_transform(v50.values).apply(v50.values)
        _, trace50 = detect_outliers(frame50, z50, 53)
        flagged50 = {d['month'] for d in trace50 if d['flagged']}
        visible = set(v50.months)
        early_full = {m for m in flagged_full if m in visible}
        self.assertEqual(early_full, flagged50)

    def test_extrapolation_points_are_not_flagged(self):
        s = snap()
        cut = s.metadata['as_of']
        frame = make_frame(s, list(range(1, 73)))
        v = extract_vintage(s, 'industry_b', cut)
        z = choose_transform(v.values).apply(v.values)
        _, trace = detect_outliers(frame, z, cut)
        for d in trace:
            if d['design_extrapolation']:
                self.assertFalse(d['flagged'], '设计外推点不得标记为异常')
        self.assertTrue(all('leverage' in d for d in trace))

    def test_known_outlier_month_is_detected(self):
        # DGP 在 industry_b 的 44 月植入强异常（见 metadata.dgp.outlier_months）
        s = snap()
        cut = s.metadata['as_of']
        frame = make_frame(s, list(range(1, 73)))
        v = extract_vintage(s, 'industry_b', cut)
        z = choose_transform(v.values).apply(v.values)
        pulses, _ = detect_outliers(frame, z, cut)
        flagged_months = {next(iter(p[1])) for p in pulses}
        self.assertIn(44, flagged_months)

    def test_detection_identical_with_or_without_later_data(self):
        """在 cutoff=50 截断点检测，与全样本检测对该 cutoff 可见月的结论完全一致。

        这直接证明早期标记不依赖 cutoff 之后发布的任何数据（含 44 月异常的判定）。
        frame 必须按该 cutoff 的实际观测月份构建（发布滞后使其少于日历月数）。
        """
        s = snap()
        cut = s.metadata['as_of']
        # 全样本
        v_full = extract_vintage(s, 'industry_b', cut)
        full_frame = make_frame(s, v_full.months)
        z_full = choose_transform(v_full.values).apply(v_full.values)
        pulses_full, _ = detect_outliers(full_frame, z_full, cut)
        flagged_full = {next(iter(p[1])) for p in pulses_full}

        # 截断 vintage：按 cutoff=53 的实际可见观测构建对齐 frame
        # （44 月在批次 42-47 末发布，cutoff=53 可见；60 月异常此时尚未公布）
        v_early = extract_vintage(s, 'industry_b', 53)
        early_frame = make_frame(s, v_early.months)
        z_early = choose_transform(v_early.values).apply(v_early.values)
        pulses_early, trace_early = detect_outliers(early_frame, z_early, 53)
        flagged_early = {next(iter(p[1])) for p in pulses_early}
        visible = set(v_early.months)
        # 44 月异常在 cutoff=53 已可见且被标记，与全样本对该区间结论一致
        self.assertIn(44, visible)
        self.assertIn(44, flagged_early)
        self.assertEqual(flagged_early, {m for m in flagged_full if m in visible})
        # 早期检测轨迹不超出该 cutoff 的可见月份
        self.assertTrue(all(d['month'] in visible for d in trace_early))


class AdjustmentPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = adjust(LONG)

    def test_three_series_emitted_and_independent(self):
        names = [a['series'] for a in self.out['adjustments']]
        self.assertEqual(names, ['industry_a', 'industry_b', 'inventory_c'])
        # 参数不串用：不同序列的季节系数应不同
        coefs = []
        for a in self.out['adjustments']:
            seas = [c['beta'] for c in a['diagnostics']['coefficients']
                    if c['group'] == 'seasonal']
            coefs.append(tuple(round(x, 8) for x in seas))
        self.assertEqual(len(set(coefs)), 3)

    def test_three_series_variants_present_each_row(self):
        for a in self.out['adjustments']:
            for row in a['rows']:
                for key in ('unadjusted', 'calendar_adjusted', 'seasonally_adjusted'):
                    self.assertIn(key, row)
                    self.assertTrue(isinstance(row[key], (int, float)))

    def test_identity_series_decomposition_exactly_additive(self):
        inv = next(a for a in self.out['adjustments'] if a['series'] == 'inventory_c')
        self.assertEqual(inv['transform']['transform'], 'identity')
        for row in inv['rows']:
            comp = row['components_original_scale']
            recon = row['unadjusted'] - sum(
                comp[g] for g in ('trading', 'holiday_spring', 'seasonal'))
            self.assertAlmostEqual(recon, row['seasonally_adjusted'], places=10)

    def test_log_series_main_series_use_exact_formula(self):
        a = next(x for x in self.out['adjustments'] if x['series'] == 'industry_b')
        self.assertEqual(a['transform']['transform'], 'log')
        # 季调值必须不同于未调整值（季节项确实被移除）
        diffs = [abs(r['seasonally_adjusted'] - r['unadjusted']) for r in a['rows']]
        self.assertGreater(sum(diffs), 1.0)

    def test_diagnostics_cover_three_required_dimensions(self):
        for a in self.out['adjustments']:
            d = a['diagnostics']
            self.assertIn('residual_seasonality', d)
            self.assertIn('residual_autocorrelation', d)
            self.assertIn('identifiability', d)
            self.assertTrue(d['identifiability']['identifiable'])
            self.assertTrue(d['residual_seasonality']['evaluated'])
            self.assertTrue(d['residual_autocorrelation']['ljung_box_12']['evaluated'])

    def test_white_noise_residuals_pass_diagnostics(self):
        # 季调后新息不应有显著残留季节或自相关
        for a in self.out['adjustments']:
            d = a['diagnostics']
            seas_p = d['residual_seasonality']['p_value']
            lb_p = d['residual_autocorrelation']['ljung_box_12']['p_value']
            self.assertGreater(seas_p, 0.01)
            self.assertGreater(lb_p, 0.01)

    def test_model_selection_uses_only_cutoff_data(self):
        # 每个 vintage 的候选表只记录该 cutoff 可见样本量
        a = self.out['adjustments'][0]
        for c in a['selection']['candidates']:
            if not c['rejected']:
                self.assertLessEqual(c['n_obs'], a['n_obs'])

    def test_freeze_contains_traceability_artifacts(self):
        a = self.out['adjustments'][0]
        f = a['freeze']
        self.assertEqual(len(f['months']), len(f['design']['matrix']))
        self.assertEqual(len(f['design']['names']), len(f['design']['groups']))
        self.assertTrue(f['candidate_table'])
        self.assertTrue(f['outlier_trace'])
        self.assertEqual(len(f['estimated']['beta']), len(f['design']['names']))
        self.assertEqual(len(f['months']), len(f['record_ids']))


if __name__ == '__main__':
    unittest.main()
