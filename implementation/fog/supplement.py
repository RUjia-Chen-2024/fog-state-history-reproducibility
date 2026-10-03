"""Separate forecast-level and bounded-2T appendix analyses."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .coordinates import prepare_record
from .pipeline import ROOT,dump

def main():
    out=ROOT/'outputs/paper';dest=out/'supplement';dest.mkdir(exist_ok=True)
    cfg=json.loads((ROOT/'config/records.json').read_text());scales=pd.read_csv(out/'movement_scales.csv').set_index('tag')
    rows=[];bounded=[];bev=[]
    for item in cfg:
        tag=item['tag'];safe=tag.replace(' ','_');r=prepare_record(tag)
        f=pd.read_csv(out/'forecasts'/f'{safe}.csv.gz')
        ai=f.anchor_index.to_numpy(int);ti=f.target_index.to_numpy(int);hs=ti-ai
        raw=np.c_[r['se2'],r['me2']];pr=f[['SE2_pred','ME2_pred']].to_numpy()
        cross=np.zeros(len(f),bool)
        for g,e in r['gt_test']:cross |= (r['t'][ai]<g)&(r['t'][ti]>=g)
        drop=np.maximum(np.log(np.maximum(raw[ai],1e-4))-np.log(np.maximum(raw[ti],1e-4)),0).max(axis=1)>=.15
        for regime,mask in [('all',np.ones(len(f),bool)),('onset_crossing',cross),('coordinate_decline',drop)]:
            for h in range(5,151,5):
                use=mask&(hs==h)
                for j,c in enumerate(['RA','RE']):
                    rows.append(dict(tag=tag,participant=r['participant'],horizon_samples=h,horizon_sec=h/100,
                        regime=regime,coordinate=c,n=int(use.sum()),model_absolute_error_sum=float(np.abs(pr[use,j]-raw[ti[use],j]).sum()),
                        persistence_absolute_error_sum=float(np.abs(raw[ai[use],j]-raw[ti[use],j]).sum())))
        tr=pd.read_csv(out/'activations'/safe/'State-history_trace.csv')
        times=tr.loc[tr.evaluable,'anchor_time'].to_numpy();end=r['t'][-1];W=2*scales.loc[tag,'T_sec']
        refs=np.array([g for g,e in r['gt_test']])
        for method in ['State-history','Persistence','AC1','Variance','Recovery','Reddening']:
            a=pd.read_csv(out/'activations'/safe/f'{method}.csv');issued=a.issued_at.to_numpy()
            ii=np.searchsorted(refs,issued,side='right');future=ii<len(refs)
            g=np.full(len(a),np.nan);g[future]=refs[ii[future]]
            hit=future&(g-issued<=W+1e-9);full=issued+W<=end+1e-9
            lead=[float(np.max(ref-issued[hit&(ii==i)])) if np.any(hit&(ii==i)) else np.nan for i,ref in enumerate(refs)]
            for ref,l in zip(refs,lead):bev.append(dict(tag=tag,method=method,onset=ref,lead_sec=l))
            bounded.append(dict(tag=tag,participant=r['participant'],method=method,n_reference=len(refs),covered=int(np.isfinite(lead).sum()),
                alarms=len(a),full_followup_alarms=int(full.sum()),full_followup_hits=int((hit&full).sum()),unmatched=int((~hit&full).sum()),
                unmatched_followup_incomplete=int((~hit&~full).sum()),
                exposure_seconds=float(np.maximum(0,np.minimum(times+.05,end-W)-times).sum())))
    record=pd.DataFrame(rows);keys=['horizon_samples','horizon_sec','regime','coordinate']
    pooled=record.groupby(keys,as_index=False)[['n','model_absolute_error_sum','persistence_absolute_error_sum']].sum()
    for frame in [record,pooled]:
        frame['model_MAE']=frame.model_absolute_error_sum/frame.n.replace(0,np.nan)
        frame['persistence_MAE']=frame.persistence_absolute_error_sum/frame.n.replace(0,np.nan)
        frame['MAE_skill']=1-frame.model_MAE/frame.persistence_MAE
    reference=pd.read_csv(ROOT/'reference/forecast_pooled.csv').set_index(keys).sort_index()
    actual=pooled.set_index(keys).sort_index()
    assert np.array_equal(actual.n,reference.n),'Retrospective stratum pair counts differ'
    assert np.allclose(actual[['model_MAE','persistence_MAE']],reference[['model_MAE','persistence_MAE']],atol=1e-6)
    record.to_csv(dest/'forecast_per_record.csv',index=False);pooled.to_csv(dest/'forecast_pooled.csv',index=False)
    people=sorted(record.participant.unique());rng=np.random.default_rng(20261002)
    draw=rng.integers(0,len(people),(20000,len(people)))
    w=np.stack([(draw==i).sum(axis=1) for i in range(len(people))],axis=1);cis=[]
    for reg in ['all','onset_crossing','coordinate_decline']:
        for h in [25,150]:
            for c in ['RA','RE']:
                x=record[record.regime.eq(reg)&record.horizon_samples.eq(h)&record.coordinate.eq(c)]
                ag=x.groupby('participant')[['model_absolute_error_sum','persistence_absolute_error_sum']].sum().reindex(people,fill_value=0).to_numpy()
                agg=w@ag
                with np.errstate(divide='ignore',invalid='ignore'):skill=100*(1-agg[:,0]/agg[:,1])
                lo,hi=np.quantile(skill[np.isfinite(skill)],[.025,.975])
                cis.append(dict(regime=reg,horizon_sec=h/100,coordinate=c,point=100*(1-ag[:,0].sum()/ag[:,1].sum()),lower=lo,upper=hi))
    cis=pd.DataFrame(cis);cis.to_csv(dest/'forecast_skill_cluster_intervals.csv',index=False)
    old=pd.read_csv(ROOT/'reference/forecast_cluster_intervals.csv').query("category == 'forecast_skill'")
    keys=['regime','horizon_sec','coordinate']
    assert np.allclose(cis.set_index(keys).sort_index()[['point','lower','upper']],old.set_index(keys).sort_index()[['point','lower','upper']],atol=.001)
    b=pd.DataFrame(bounded);s=b.groupby('method').sum(numeric_only=True)
    s['coverage']=s.covered/s.n_reference;s['full_followup_precision']=s.full_followup_hits/s.full_followup_alarms
    s['unmatched_per_minute']=60*s.unmatched/s.exposure_seconds
    s['median_lead']=pd.DataFrame(bev).groupby('method').lead_sec.median()
    b.to_csv(dest/'bounded_2T_per_record.csv',index=False);s.to_csv(dest/'bounded_2T_summary.csv')
    assert s.loc['State-history','covered']==69 and s.loc['State-history','full_followup_hits']==83 and s.loc['State-history','full_followup_alarms']==183
    dump(dest/'verification.json',dict(forecast_pair_counts_match=True,forecast_MAE_matches=True,forecast_cluster_CI_matches=True,
        forecast_bootstrap_replicates=20000,forecast_bootstrap_seed=20261002,bounded_2T_model_counts_match=True,
        role='Supplementary analyses only; the main association is inter-event.'))
    print(s.to_string());print(cis.to_string(index=False))
