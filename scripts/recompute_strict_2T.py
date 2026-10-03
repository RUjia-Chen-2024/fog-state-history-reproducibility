"""Recompute strict inter-event + 2T sensitivity, without changing any warnings."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
OUT=Path(__file__).resolve().parents[1]
def put(name,text): (OUT/name).write_text(text.strip()+'\n',encoding='utf-8')
DATA=OUT/'analysis/inter_event_revision'
B=OUT/'analysis/strict_2T_sensitivity'
B.mkdir(exist_ok=True)
alarms=pd.read_csv(DATA/'alarms.csv')
events=pd.read_csv(DATA/'events.csv')
scales=pd.read_csv(B/'movement_scales.csv')
T=scales.set_index('tag').T_sec.to_dict()
alarms['window_sec']=2*alarms.tag.map(T)
alarms['eligible_2T']=(alarms.status.eq('successful_between_events') & (alarms.lead_sec<=alarms.window_sec+1e-9))
rows=[]
for e in events.itertuples():
    a=alarms[(alarms.method==e.method)&(alarms.tag==e.tag)&alarms.eligible_2T & np.isclose(alarms.next_onset,e.reference_onset,atol=1e-9,rtol=0)]
    first=a.issued_at.min() if len(a) else np.nan
    rows.append(dict(method=e.method,tag=e.tag,participant=e.participant,reference_onset=e.reference_onset,previous_reference_end=e.previous_reference_end,window_sec=2*T[e.tag],covered=int(len(a)>0),n_eligible=len(a),first_activation=first,lead_sec=e.reference_onset-first))
bounded=pd.DataFrame(rows)
summary=[]
for m,ee in bounded.groupby('method',sort=False):
    aa=alarms[alarms.method==m]
    ls=ee.lead_sec.dropna()
    summary.append(dict(method=m,events=len(ee),covered=int(ee.covered.sum()),activations=len(aa),associated=int(aa.eligible_2T.sum()),coverage=ee.covered.mean(),association_fraction=aa.eligible_2T.mean(),median_lead=ls.median(),mean_lead=ls.mean(),q25=ls.quantile(.25),q75=ls.quantile(.75)))
bs=pd.DataFrame(summary)
bs.to_csv(B/'summary.csv',index=False)
bounded.to_csv(B/'events.csv',index=False)
alarms.to_csv(B/'activations.csv',index=False)
# Same paired participant resampling as the primary comparison, fixed forecasts and rules.
participants=sorted(events.participant.unique())
rng=np.random.default_rng(20261002)
draws=rng.integers(0,len(participants),(2000,len(participants)))
counts=np.stack([(draws==i).sum(axis=1) for i in range(len(participants))],axis=1)
boot={}
for m in bs.method:
    vv=[]
    for p in participants:
        ee=bounded[(bounded.method==m)&(bounded.participant==p)]
        aa=alarms[(alarms.method==m)&(alarms.participant==p)]
        vv.append([ee.covered.sum(),len(ee),aa.eligible_2T.sum(),len(aa)])
    c=counts@np.array(vv)
    boot[m]=np.column_stack([c[:,0]/c[:,1],c[:,2]/c[:,3]])
ci=[]
for m,v in boot.items():
    for k,label in enumerate(['coverage','association_fraction']):
        lo,hi=np.quantile(v[:,k],[.025,.975]);ci.append(dict(method=m,metric=label,lower95=lo,upper95=hi))
pd.DataFrame(ci).to_csv(B/'participant_cluster_intervals.csv',index=False)
delta=boot['State-history']-boot['Persistence']
put('analysis/strict_2T_sensitivity/protocol.json',json.dumps(dict(rule='previous_reference_end < actual_activation < next_reference_onset and lead <= 2T',denominator='All actual activations, unchanged from unlimited analysis; not complete-follow-up precision',events=88,participants=participants,bootstrap=2000,seed=20261002,paired_state_minus_persistence_ci=np.quantile(delta,[.025,.975],axis=0).tolist()),indent=2))
print(bs.to_string(index=False))
