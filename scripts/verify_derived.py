"""Validate archived original forecasts and recompute forecast errors from derived coordinates."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parents[1]
D=R/'derived'
hashes=json.loads((D/'sha256.json').read_text())
for name,h in hashes.items():assert hashlib.sha256((D/name).read_bytes()).hexdigest()==h,name
records=json.loads((D/'recordings.json').read_text())
refs=json.loads((D/'reference_intervals.json').read_text())
rows=[];total=0;onsets=0
for rec in records:
    c=pd.read_csv(D/rec['coordinates']);p=pd.read_csv(D/rec['predictions'])
    ai=p.anchor_index.to_numpy(int);ti=p.target_index.to_numpy(int)
    assert len(c)==rec['samples'] and len(p)==rec['forecast_pairs']
    assert np.array_equal(c.sample_index.to_numpy(),np.arange(len(c)))
    assert (ai>=rec['test_start_index']).all() and (ti>ai).all() and (ti<len(c)).all()
    assert np.allclose(p.anchor_time,c.time_sec.to_numpy()[ai])
    assert np.allclose(p.target_time,c.time_sec.to_numpy()[ti])
    z=c[['RA','RE']].to_numpy();pr=p[['RA_pred','RE_pred']].to_numpy()
    assert np.isfinite(z).all() and np.isfinite(pr).all()
    assert (pr>=0).all() and (pr<=2).all()
    cross=np.zeros(len(p),bool)
    testrefs=[(g,e) for g,e in refs[rec['tag']] if g>=rec['test_start_time']]
    for g,e in testrefs:cross|=(p.anchor_time.to_numpy()<g)&(p.target_time.to_numpy()>=g)
    decline=np.maximum(np.log(np.maximum(z[ai],1e-4))-np.log(np.maximum(z[ti],1e-4)),0).max(axis=1)>=.15
    for regime,mask in [('all',np.ones(len(p),bool)),('onset_crossing',cross),('coordinate_decline',decline)]:
        for h in [5,25,50,75,100,125,150]:
            use=mask&((ti-ai)==h)
            for j,coord in enumerate(['RA','RE']):
                for model,pred in [('State-history',pr),('Persistence',z[ai])]:
                    rows.append(dict(tag=rec['tag'],participant=rec['participant'],regime=regime,horizon_sec=h/100,coordinate=coord,method=model,n=int(use.sum()),absolute_error_sum=float(np.abs(pred[use,j]-z[ti[use],j]).sum())))
    total+=len(p);onsets+=len(testrefs)
assert len(records)==21 and total==385035 and onsets==88
errors=pd.DataFrame(rows)
key=['tag','participant','regime','horizon_sec','coordinate','method']
actual=errors.set_index(key).sort_index()
expected=pd.read_csv(R/'provenance/replayed_forecast_errors_per_record.csv').set_index(key).sort_index()
assert actual.index.equals(expected.index)
assert np.array_equal(actual.n,expected.n)
mean_difference=np.abs(actual.absolute_error_sum-expected.absolute_error_sum)/np.maximum(actual.n,1)
assert mean_difference.max()<2e-6
errors.to_csv(R/'provenance/original_saved_forecast_errors_per_record.csv',index=False)
agg=errors.groupby(['regime','horizon_sec','coordinate','method'])[['n','absolute_error_sum']].sum()
agg['MAE']=agg.absolute_error_sum/agg.n.replace(0,np.nan)
agg.to_csv(R/'provenance/original_saved_forecast_errors.csv')
report=dict(records=21,participants=len(set(r['participant'] for r in records)),forecast_pairs=total,test_onsets=onsets,all_derived_sha256_match=True,max_per_record_MAE_difference_from_independent_replay=float(mean_difference.max()),models_retrained=False)
(R/'provenance/derived_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
