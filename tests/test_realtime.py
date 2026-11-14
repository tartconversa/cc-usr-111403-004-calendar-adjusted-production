"""实时修订三效应分解、固定口径对照与冻结重放。"""
import json
import unittest
from pathlib import Path

from app.service import adjust, realtime, replay_frozen
from app.freeze import replay

LONG = json.loads((Path(__file__).resolve().parents[1] / 'fixtures' / 'sample_long.json')
                  .read_text(encoding='utf-8'))


class RealtimeRevisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rt = realtime(LONG, series='industry_b', min_obs=24)
        cls.r = cls.rt['realtime']['series']['industry_b']

    def test_vintages_expand_sample_over_time(self):
        obs = [v['n_obs'] for v in self.r['evolving_vintages']]
        self.assertEqual(obs, sorted(obs))
        self.assertGreater(len(obs), 3)
        # 早期样本不足的 vintage 被跳过并记录原因
        self.assertTrue(self.r['skipped_vintages'])

    def test_three_effects_sum_exactly_to_total(self):
        for d in self.r['endpoint_revision_decomposition']:
            for e in d['effects_by_month']:
                s = (e['new_month_effect'] + e['revision_effect']
                     + e['reselection_effect'])
                self.assertAlmostEqual(s, e['total'], places=10)
                self.assertAlmostEqual(e['additivity_residual'], 0.0, places=10)

    def test_effects_are_separable_not_merged(self):
        summary = self.r['revision_summary']['rmse']
        # 三个效应分别有独立量级，而不是只给一个笼统的“修订小”
        for key in ('new_month_effect', 'revision_effect', 'reselection_effect'):
            self.assertIn(key, summary)
        # 新增月效应应显著（样本窗口扩展是主要修订来源）
        self.assertGreater(summary['new_month_effect'], 0.0)

    def test_fixed_track_has_zero_reselection_effect(self):
        for d in self.r['fixed_track_decomposition']:
            for e in d['effects_by_month']:
                self.assertEqual(e['reselection_effect'], 0.0)
        # 固定轨道在首个合格 vintage 锁定模型口径，之后不变
        specs = {tuple(sorted(v['frozen_spec'].items())) for v in
                 self.r['fixed_vintages']}
        self.assertEqual(len(specs), 1)

    def test_frozen_spec_matches_first_feasible_vintage(self):
        first = self.r['evolving_vintages'][0]['chosen_spec']
        self.assertEqual(self.r['frozen_spec'], first)

    def test_historical_revision_months_show_revision_effect(self):
        # DGP 修订月含 44、58；这些月在对应分解中应出现非零 revision_effect
        rev_months = {int(k) for k in LONG['metadata']['dgp']['revision_months']}
        seen = set()
        for d in self.r['endpoint_revision_decomposition']:
            for e in d['effects_by_month']:
                if e['month'] in rev_months and abs(e['revision_effect']) > 1e-9:
                    seen.add(e['month'])
        self.assertTrue(seen & rev_months)

    def test_vintage_table_records_spec_and_transform(self):
        for v in self.r['evolving_vintages']:
            self.assertIn('seasonal', v['chosen_spec'])
            self.assertIn('ar_order', v['chosen_spec'])
            self.assertIn(v['transform'], ('log', 'identity'))
            self.assertEqual(len(v['sa_by_month']), v['n_obs'])


class FreezeReplayTests(unittest.TestCase):
    def test_replay_reproduces_estimates_exactly(self):
        out = adjust(LONG, series='industry_a')
        freeze = out['adjustments'][0]['freeze']
        result = replay(freeze)
        self.assertTrue(result['checks']['reproduces'])
        self.assertEqual(result['checks']['max_beta_abs_diff'], 0.0)
        self.assertEqual(result['checks']['max_phi_abs_diff'], 0.0)

    def test_replay_traceable_to_months_design_and_model(self):
        out = adjust(LONG, series='industry_b')
        freeze = out['adjustments'][0]['freeze']
        result = replay(freeze)
        tr = result['traceability']
        self.assertEqual(tr['months'], freeze['months'])
        self.assertEqual(len(tr['design_names']), len(freeze['design']['names']))
        self.assertTrue(tr['candidate_table_present'])
        self.assertTrue(tr['outlier_trace_present'])
        # 重放行与冻结的实际月份、记录 id 对齐
        self.assertEqual([r['month'] for r in result['rows']], freeze['months'])
        self.assertEqual([r['record_id'] for r in result['rows']], freeze['record_ids'])

    def test_service_replay_entry(self):
        out = adjust(LONG, series='inventory_c')
        freeze = out['adjustments'][0]['freeze']
        wrapped = replay_frozen(freeze)
        self.assertEqual(wrapped['schema_version'], 1)
        self.assertTrue(wrapped['replay']['checks']['reproduces'])

    def test_freeze_carries_input_digest(self):
        out = adjust(LONG, series='industry_a')
        self.assertEqual(len(out['input_digest']), 64)
        self.assertEqual(out['adjustments'][0]['freeze']['input_digest'],
                         out['input_digest'])


class CrossSeriesIsolationTests(unittest.TestCase):
    def test_shared_design_but_separate_parameters(self):
        out = adjust(LONG)
        phis = {a['series']: tuple(a['ar_phi']) for a in out['adjustments']}
        # 三序列 AR 参数互不相同，证明参数未串用
        self.assertEqual(len(set(phis.values())), 3)
        # 日历/季节/干预等基础设计列在序列间一致（同一日历设计复用），
        # 而序列特异的异常脉冲列各自独立
        base_name_sets = [
            {c['name'] for c in a['diagnostics']['coefficients']
             if not c['name'].startswith('outlier_')}
            for a in out['adjustments']]
        self.assertEqual(base_name_sets[0], base_name_sets[1])
        outlier_cols = [
            {c['name'] for c in a['diagnostics']['coefficients']
             if c['name'].startswith('outlier_')}
            for a in out['adjustments']]
        # 含异常的序列其脉冲列不出现在无异常序列中（结果不串用）
        self.assertIn('outlier_pulse_44', outlier_cols[1])
        self.assertNotIn('outlier_pulse_44', outlier_cols[0])


if __name__ == '__main__':
    unittest.main()
