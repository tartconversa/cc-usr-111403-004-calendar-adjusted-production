"""现有业务摘要，不推断未观察到的业务关系。"""
def analyze(snapshot):
    calendar={}
    for r in snapshot.rows('calendar'):
        if r['month'] in calendar:raise ValueError('日历月份重复')
        calendar[r['month']]=r
    active={};versions={}
    for r in snapshot.rows('months'):
        if r['published']>snapshot.metadata['as_of']:continue
        key=r['series'],r['month']; old=active.get(key)
        version=(*key,r['revision'])
        if version in versions and versions[version]!=r['value']:raise ValueError('月值冲突')
        versions[version]=r['value']
        if old and old['revision']==r['revision'] and old['value']!=r['value']:raise ValueError('月值冲突')
        if not old or old['revision']<r['revision']:active[key]=r
    result=[]
    for key,r in sorted(active.items()):
        c=calendar[r['month']];w=c['weekday_counts']
        if len(w)!=7 or any(x<0 for x in w):raise ValueError('日历计数无效')
        result.append({'series':r['series'],'month':r['month'],'value':r['value'],'trading_contrasts':[x-w[-1] for x in w[:6]],'holiday':c['holiday_overlap']})
    return result
