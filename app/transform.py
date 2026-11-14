"""序列变换：对数（乘法分解）与恒等（加法分解），含非正值明确退路。

选择规则在每个 vintage 内基于该 cutoff 可见观测独立判定，不使用任何未来信息：
- 全部为正且对数尺度下离散度可接受：对数变换（乘法分解）；
- 出现零或负值：恒等变换（加法分解），记录触发原因；
调用方可通过 force='identity' 强制加法口径，选择理由写入冻结结果。
"""
import math


def choose_transform(values, force=None):
    if force not in (None, 'log', 'identity'):
        raise ValueError("force 取值应为 'log' 或 'identity'")
    nonpositive = [(i, v) for i, v in enumerate(values) if v <= 0]
    if force == 'identity':
        spec = 'identity'
        reason = '调用方强制恒等变换'
    elif force == 'log':
        if nonpositive:
            raise ValueError(f'强制对数变换但存在 {len(nonpositive)} 个非正观测')
        spec, reason = 'log', '调用方强制对数变换'
    elif nonpositive:
        spec = 'identity'
        worst = min(v for _, v in nonpositive)
        reason = f'存在 {len(nonpositive)} 个非正观测（最小值 {worst:g}），对数无定义，退回恒等（加法）分解'
    else:
        spec, reason = 'log', '全部观测为正，采用对数（乘法）分解'
    return Transform(spec, reason, [v for _, v in nonpositive])


class Transform:
    def __init__(self, name, reason, nonpositive):
        self.name = name
        self.reason = reason
        self.nonpositive = nonpositive

    def apply(self, values):
        if self.name == 'log':
            return [math.log(v) for v in values]
        return [float(v) for v in values]

    def invert(self, z, residual_variance=0.0):
        """变换尺度 -> 原始尺度。对数情形提供无偏校正与朴素指数两种口径。"""
        if self.name == 'log':
            bias = 0.5 * residual_variance
            return {
                'plain': math.exp(z),
                'bias_corrected': math.exp(z + bias),
            }
        return {'plain': z, 'bias_corrected': z}

    @property
    def family(self):
        return 'multiplicative' if self.name == 'log' else 'additive'

    def describe(self):
        return {'transform': self.name, 'family': self.family,
                'reason': self.reason, 'nonpositive_count': len(self.nonpositive)}
