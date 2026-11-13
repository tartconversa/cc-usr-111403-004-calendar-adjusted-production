# 资料与现有计算约定

## 修订快照

schema_version为整数1，metadata为业务方案字典，collections为业务集合字典。每条记录有非空字符串id和正整数revision。
同集合相同id取最高revision；同id同revision内容不同即冲突，即便存在更高修订也拒绝。deleted=true为撤回，之后更高修订可恢复。
撤回记录不要求携带业务字段。输入顺序与相同记录重试不改变语义；非有限数字拒绝。
Snapshot.rows返回独立副本；input_digest绑定有效修订和全部metadata。输出data_version沿用metadata，不由程序自造资料版本。
既有摘要入口为app.service.inspect(payload)，输出schema_version、input_digest、data_version和summary。

## 业务资料

months含series、month、value、published、revision，calendar含month、weekday_counts(7项)、holiday_overlap。holiday_overlap为事前固定窗口分摊权重，不从产量拟合。交易日对比以星期日为基准，weekday_counts前六项分别减最后一项。as_of筛发布时间后同series/month取最高revision，冲突拒绝。每次公布使用不同id，保留旧公布事件，published为版本首次可见时间。baseline.analyze返回当时可见月值及原日历对比列。

各集合必要字段：{"months": ["series", "month", "value", "published", "revision"], "calendar": ["month", "weekday_counts", "holiday_overlap"]}。

## 当前能力

app.baseline.analyze(snapshot)产生已有摘要，app.service.inspect负责绑定输入身份；新增分析可以增设独立入口，已有接口及上述数据语义保持兼容。
