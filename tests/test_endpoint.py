"""端点延拓：预测标注、与观测序列分离及无未来日历时的退路。"""
import json
import unittest
from pathlib import Path

from app.service import adjust

LONG = json.loads((Path(__file__).resolve().parents[1] / 'fixtures' / 'sample_long.json')
                  .read_text(encoding='utf-8'))


class EndpointExtensionTests(unittest.TestCase):
    def test_forecasts_are_separate_and_labeled(self):
        out = adjust(LONG, series='industry_b', cutoff=60, forecast_horizon=6)
        a = out['adjustments'][0]
        ext = a['endpoint_extension']
        self.assertTrue(ext['available'])
        self.assertEqual(len(ext['forecasts']), 6)
        obs_months = {r['month'] for r in a['rows']}
        horizons = []
        for f in ext['forecasts']:
            self.assertTrue(f['forecast'])
            self.assertNotIn(f['month'], obs_months)  # 绝不混入观测序列
            horizons.append(f['horizon'])
        self.assertEqual(horizons, [1, 2, 3, 4, 5, 6])

    def test_forecast_standard_error_grows_with_horizon(self):
        out = adjust(LONG, series='industry_a', cutoff=60, forecast_horizon=6)
        fs = out['adjustments'][0]['endpoint_extension']['forecasts']
        ses = [f['forecast_se_transformed'] for f in fs]
        self.assertEqual(ses, sorted(ses))  # 多步不确定性单调不减
        self.assertGreater(ses[-1], ses[0])

    def test_extension_disabled_by_default(self):
        out = adjust(LONG, series='industry_b', cutoff=60)
        self.assertIsNone(out['adjustments'][0]['endpoint_extension'])

    def test_no_extension_without_future_calendar(self):
        # 以最终 as_of 为 cutoff 时没有未来日历月
        out = adjust(LONG, series='industry_b', forecast_horizon=6)
        ext = out['adjustments'][0]['endpoint_extension']
        self.assertFalse(ext['available'])
        self.assertIn('无日历', ext['reason'])
        self.assertEqual(ext['forecasts'], [])


if __name__ == '__main__':
    unittest.main()
