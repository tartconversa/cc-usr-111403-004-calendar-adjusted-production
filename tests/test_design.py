"""日历设计、变换退路与事前干预语义。"""
import json
import math
import unittest
from pathlib import Path

from app.calendar import DesignFrame, month_of_year, TD_NAMES
from app.catalog import load
from app.transform import choose_transform
from app.service import SCHEMAS
from app.adjust import make_frame, extract_vintage

LONG = json.loads((Path(__file__).resolve().parents[1] / 'fixtures' / 'sample_long.json')
                  .read_text(encoding='utf-8'))


def snapshot():
    return load(LONG, SCHEMAS)


def calendar_frame(snap, months):
    from app.adjust import make_frame
    return make_frame(snap, months)


class MonthCodingTests(unittest.TestCase):
    def test_month_of_year_small_integer_and_yyyymm(self):
        self.assertEqual(month_of_year(1), 1)
        self.assertEqual(month_of_year(12), 12)
        self.assertEqual(month_of_year(13), 1)
        self.assertEqual(month_of_year(202403), 3)
        self.assertEqual(month_of_year(202412), 12)


class CalendarDesignTests(unittest.TestCase):
    def test_trading_contrasts_use_sunday_baseline(self):
        snap = snapshot()
        frame = calendar_frame(snap, [1, 2])
        design = frame.build('fixed', snap.metadata['as_of'])
        td_idx = [design.names.index(n) for n in TD_NAMES]
        # 与契约 baseline 一致：前六个 weekday 计数减星期日计数
        cal = {r['month']: r for r in snap.rows('calendar')}
        for i, m in enumerate([1, 2]):
            w = cal[m]['weekday_counts']
            for j, idx in enumerate(td_idx):
                self.assertAlmostEqual(design.matrix[i][idx], w[j] - w[6])

    def test_holiday_window_is_single_preadjustment_regressor(self):
        snap = snapshot()
        frame = calendar_frame(snap, [1, 2])
        design = frame.build('fixed', snap.metadata['as_of'])
        j = design.names.index('holiday_spring')
        cal = {r['month']: r for r in snap.rows('calendar')}
        self.assertEqual(design.matrix[0][j], cal[1]['holiday_overlap'])
        self.assertEqual(design.matrix[1][j], cal[2]['holiday_overlap'])

    def test_future_intervention_dropped_before_its_month(self):
        snap = snapshot()
        # policy_pulse_40：在只到第 20 月的窗口内不应出现（零列剔除）
        frame = calendar_frame(snap, list(range(1, 21)))
        design = frame.build('fixed', snap.metadata['as_of'])
        self.assertNotIn('iv:policy_pulse_40', design.names)
        self.assertTrue(any('policy_pulse_40' in d['name']
                            for d in design.dropped_interventions))
        # 到第 40 月后应进入设计
        frame40 = calendar_frame(snap, list(range(1, 41)))
        design40 = frame40.build('fixed', snap.metadata['as_of'])
        self.assertIn('iv:policy_pulse_40', design40.names)

    def test_design_full_rank_on_full_window(self):
        snap = snapshot()
        frame = calendar_frame(snap, list(range(1, 73)))
        for spec in ('fixed', 'trig3', 'trig6'):
            design = frame.build(spec, snap.metadata['as_of'])
            report = design.rank_report()
            self.assertTrue(report['full_rank'], f'{spec} 应满秩')

    def test_seasonal_specs_have_expected_column_counts(self):
        snap = snapshot()
        frame = calendar_frame(snap, list(range(1, 73)))
        counts = {}
        for spec in ('fixed', 'trig3', 'trig6'):
            design = frame.build(spec, snap.metadata['as_of'])
            counts[spec] = len([g for g in design.groups if g == 'seasonal'])
        self.assertEqual(counts['fixed'], 11)   # 1 月基准
        self.assertEqual(counts['trig3'], 6)    # 3 谐波 × 正余弦
        self.assertEqual(counts['trig6'], 12)


class TransformFallbackTests(unittest.TestCase):
    def test_positive_series_uses_log_multiplicative(self):
        tf = choose_transform([1.0, 2.0, 3.0])
        self.assertEqual(tf.name, 'log')
        self.assertEqual(tf.family, 'multiplicative')
        self.assertAlmostEqual(tf.apply([math.e])[0], 1.0)

    def test_nonpositive_series_falls_back_to_identity(self):
        tf = choose_transform([1.0, 0.0, -2.0])
        self.assertEqual(tf.name, 'identity')
        self.assertEqual(tf.family, 'additive')
        self.assertIn('非正', tf.reason)
        self.assertEqual(len(tf.nonpositive), 2)

    def test_inventory_series_with_zero_uses_identity(self):
        snap = snapshot()
        from app.adjust import extract_vintage
        v = extract_vintage(snap, 'inventory_c', snap.metadata['as_of'])
        tf = choose_transform(v.values)
        self.assertEqual(tf.name, 'identity')

    def test_force_log_rejects_nonpositive(self):
        with self.assertRaises(ValueError):
            choose_transform([1.0, 0.0], force='log')


if __name__ == '__main__':
    unittest.main()
