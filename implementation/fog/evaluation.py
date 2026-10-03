"""Offline inter-event association. Never called by the online state machine."""
import numpy as np
import pandas as pd

EPS=1e-9
MAIN=['Persistence','AC1','Variance','Recovery','Reddening','State-history']
COUNTS=['n_reference','covered','missed','alarms','successful_warnings','repeated_successes',
        'during_event_or_boundary','unresolved_tail','observed_seconds','active_seconds']

def evaluate_record(r,method,episodes,trace,q):
    iv=np.asarray(r['gt_all'],float).reshape(-1,2)
    start=r['test_start_time']
    refs=np.asarray([g for g,e in r['gt_test']])
    common=dict(tag=r['tag'],participant=r['participant'],method=method)
    alarm_rows=[]
    for a in episodes.itertuples():
        inside=bool(np.any((iv[:,0]-EPS<=a.issued_at)&(a.issued_at<=iv[:,1]+EPS)))
        nxt=np.flatnonzero(iv[:,0]>a.issued_at+EPS)
        k=int(nxt[0]) if len(nxt) else len(iv)
        g=float(iv[k,0]) if len(nxt) else np.nan
        prev=float(iv[k-1,1]) if k else np.nan
        status=('during_reference_event_or_boundary' if inside else
                'successful_between_events' if len(nxt) else 'no_later_onset_observed')
        if status=='successful_between_events':
            assert np.any(np.isclose(refs,g,atol=1e-8,rtol=0))
        alarm_rows.append(dict(**common,alarm_id=a.alarm_id,issued_at=a.issued_at,
            observed_end=a.observed_end,right_censored=a.right_censored,next_onset=g,
            previous_reference_end=prev,status=status,
            lead_sec=g-a.issued_at if status=='successful_between_events' else np.nan))
    event_rows=[]
    for g in refs:
        k=int(np.flatnonzero(np.isclose(iv[:,0],g,atol=1e-8,rtol=0))[0])
        prev=float(iv[k-1,1]) if k else np.nan
        eligible=(episodes.issued_at<g-EPS)&(episodes.issued_at>=start-EPS)
        if np.isfinite(prev):eligible &= episodes.issued_at>prev+EPS
        hits=episodes[eligible]
        first=float(hits.issued_at.min()) if len(hits) else np.nan
        event_rows.append(dict(**common,reference_onset=g,previous_reference_end=prev,
            scored_test_start=start,left_truncated=int(not np.isfinite(prev) or prev<start),
            covered=int(len(hits)>0),n_successful_warnings=len(hits),first_warning=first,lead_sec=g-first))
    success=sum(a['status']=='successful_between_events' for a in alarm_rows)
    covered=sum(e['covered'] for e in event_rows)
    assert success==sum(e['n_successful_warnings'] for e in event_rows)
    times=trace.loc[trace.evaluable,'anchor_time'].to_numpy()
    observed=float(np.maximum(0,np.minimum(times+.05,r['t'][-1])-times).sum())
    row=dict(**common,q=q,n_reference=len(refs),covered=covered,missed=len(refs)-covered,
        alarms=len(episodes),successful_warnings=success,repeated_successes=success-covered,
        during_event_or_boundary=sum(a['status']=='during_reference_event_or_boundary' for a in alarm_rows),
        unresolved_tail=sum(a['status']=='no_later_onset_observed' for a in alarm_rows),
        observed_seconds=observed,active_seconds=float(episodes.observed_duration.sum()))
    return row,alarm_rows,event_rows

def summarize(records,events):
    summary=records.groupby('method',sort=False)[COUNTS].sum()
    summary['coverage']=summary.covered/summary.n_reference
    summary['success_fraction']=summary.successful_warnings/summary.alarms.replace(0,np.nan)
    summary['activations_per_minute']=summary.alarms*60/summary.observed_seconds
    summary['active_fraction']=summary.active_seconds/summary.observed_seconds
    summary['q']=records.groupby('method').q.first()
    for m in summary.index:
        lead=events.loc[events.method.eq(m)&events.covered.eq(1),'lead_sec']
        for name,value in dict(median_lead=lead.median(),mean_lead=lead.mean(),
            lead_q25=lead.quantile(.25),lead_q75=lead.quantile(.75),lead_max=lead.max(),lead_min=lead.min()).items():
            summary.loc[m,name]=value
    assert (summary.successful_warnings+summary.during_event_or_boundary+summary.unresolved_tail==summary.alarms).all()
    return summary

def bootstrap(records,events,summary,seed=20261002,n=2000):
    people=sorted(records.participant.unique());rng=np.random.default_rng(seed)
    draws=rng.integers(0,len(people),size=(n,len(people)))
    weights=np.stack([(draws==i).sum(axis=1) for i in range(len(people))],axis=1)
    rep={};ci=[];differences=[]
    metrics=[('coverage','covered','n_reference',1),('success_fraction','successful_warnings','alarms',1),
             ('activations_per_minute','alarms','observed_seconds',60),('active_fraction','active_seconds','observed_seconds',1)]
    for method in MAIN:
        grp=records[records.method.eq(method)].groupby('participant')[COUNTS].sum().reindex(people)
        sums=weights@grp.to_numpy();idx={v:i for i,v in enumerate(COUNTS)}
        for metric,num,den,factor in metrics:
            v=factor*sums[:,idx[num]]/sums[:,idx[den]];rep[method,metric]=v
            lo,hi=np.quantile(v,[.025,.975])
            ci.append(dict(method=method,metric=metric,estimate=summary.loc[method,metric],lower95=lo,upper95=hi))
    for comparator in MAIN[:-1]:
        for metric,_,_,_ in metrics:
            v=rep['State-history',metric]-rep[comparator,metric];lo,hi=np.quantile(v,[.025,.975])
            differences.append(dict(comparator=comparator,metric=metric,
                difference=summary.loc['State-history',metric]-summary.loc[comparator,metric],lower95=lo,upper95=hi))
    paired=events[events.method.isin(['State-history','Persistence'])].pivot(
        index=['tag','participant','reference_onset'],columns='method',values='lead_sec').dropna()
    paired['difference']=paired['State-history']-paired['Persistence']
    ps=dict(n=len(paired),median_difference=float(paired.difference.median()),mean_difference=float(paired.difference.mean()),
        operator_earlier=int((paired.difference>EPS).sum()),persistence_earlier=int((paired.difference< -EPS).sum()),ties=int((abs(paired.difference)<=EPS).sum()))
    return pd.DataFrame(ci),pd.DataFrame(differences),paired.reset_index(),ps
