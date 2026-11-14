"""发布带资料身份的业务摘要与季节/日历调整结果。

inspect 保持既有语义不变；新增 adjust / realtime / replay 三个独立入口。
"""
import json
import sys
from pathlib import Path

from .catalog import load
from .baseline import analyze
from .adjust import adjust_series, list_series, extract_vintage, make_frame
from .realtime import realtime as _realtime
from .freeze import replay as _replay

SCHEMAS = {'months': ['series', 'month', 'value', 'published', 'revision'],
           'calendar': ['month', 'weekday_counts', 'holiday_overlap']}


def inspect(payload):
    snapshot = load(payload, SCHEMAS)
    return {'schema_version': 1, 'input_digest': snapshot.digest,
            'data_version': snapshot.metadata.get('data_version'), 'summary': analyze(snapshot)}


def adjust(payload, series=None, cutoff=None, fixed_spec=None,
           outlier_threshold=3.5, force_transform=None, forecast_horizon=None):
    """季节与日历调整入口；日历设计跨序列复用、参数与结果不串用。"""
    snapshot = load(payload, SCHEMAS)
    cutoff = cutoff if cutoff is not None else snapshot.metadata['as_of']
    names = [series] if series else list_series(snapshot)
    outputs = []
    frame_cache = {}
    for name in names:
        vintage = extract_vintage(snapshot, name, cutoff)
        key = tuple(vintage.months)
        if key not in frame_cache:  # 同观测月份集合复用同一日历设计
            frame_cache[key] = make_frame(snapshot, vintage.months)
        outputs.append(adjust_series(snapshot, name, cutoff, frame=frame_cache[key],
                                     fixed_spec=fixed_spec,
                                     outlier_threshold=outlier_threshold,
                                     force_transform=force_transform,
                                     forecast_horizon=forecast_horizon))
    return {'schema_version': 1, 'input_digest': snapshot.digest,
            'data_version': snapshot.metadata.get('data_version'),
            'as_of': snapshot.metadata['as_of'], 'cutoff': cutoff,
            'adjustments': outputs}


def realtime(payload, series=None, min_obs=24, outlier_threshold=3.5):
    """逐历史发布的实时修订分析（含固定规格对照与三效应分解）。"""
    snapshot = load(payload, SCHEMAS)
    report = _realtime(snapshot, series=series, min_obs=min_obs,
                       outlier_threshold=outlier_threshold)
    return {'schema_version': 1, 'input_digest': snapshot.digest,
            'data_version': snapshot.metadata.get('data_version'),
            'realtime': report}


def replay_frozen(freeze):
    """从冻结负载重放并校验。"""
    return {'schema_version': 1, 'replay': _replay(freeze)}


def main():
    if len(sys.argv) < 2:
        raise SystemExit('用法: python -m app.service <资料JSON> [inspect|adjust|realtime]')
    path = Path(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else 'inspect'
    payload = json.loads(path.read_text(encoding='utf-8'))
    if mode == 'inspect':
        out = inspect(payload)
    elif mode == 'adjust':
        out = adjust(payload)
    elif mode == 'realtime':
        out = realtime(payload)
    else:
        raise SystemExit(f'未知模式 {mode}（inspect|adjust|realtime）')
    print(json.dumps(out, ensure_ascii=False, allow_nan=False, indent=2))


if __name__ == '__main__':
    main()
