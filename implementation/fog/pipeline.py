"""Recompute coordinates, forecasts, online activations and paper outcomes."""
from pathlib import Path
import json,hashlib,sys
import numpy as np
import pandas as pd
import torch
from . import constants as C
from .coordinates import prepare_record,causal_robust_amplitude,positive_median
from .model import StateOnlyTrajectory
from .inference import infer
from .lifecycle import activation_intervals,test_state_machine
from .rules import movement_scale,rolling_ews,trajectory_base,trajectory_candidates
from .evaluation import evaluate_record,summarize,bootstrap,MAIN

ROOT=Path(__file__).resolve().parents[1]
def dump(p,obj):
    p.write_text(json.dumps(obj,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else str(v)),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def run(mode='infer',output=None,checkpoint_dir=None):
    torch.set_num_threads(2)
    out=Path(output) if output else ROOT/'outputs'/('paper' if mode=='infer' else 'saved_replay')
    paper_weights=checkpoint_dir is None
    if not paper_weights and out.resolve()==(ROOT/'outputs/paper').resolve():
        raise ValueError('New weights must use a different output directory')
    if checkpoint_dir:
        checkpoint_dir=Path(checkpoint_dir)
    out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ROOT/'config/paper.json').read_text())
    manifest=json.loads((ROOT/'config/records.json').read_text())
    rows=[];alarms=[];events=[];checks=[];references={};mae=[];scales=[]
    expected_thresholds=pd.read_csv(ROOT/'reference/frozen_fit_thresholds.csv')
    expected_alarms=pd.read_csv(ROOT/'reference/alarms.csv')
    for k,entry in enumerate(manifest,1):
        tag=entry['tag'];safe=tag.replace(' ','_')
        cp=checkpoint_dir/f'{safe}.pt' if checkpoint_dir else ROOT/entry['checkpoint'];data=ROOT/entry['record']
        if paper_weights:assert sha(cp)==entry['checkpoint_sha256']
        assert sha(data)==entry['record_sha256']
        r=prepare_record(tag);t=r['t'];references[tag]=r['gt_all']
        assert abs(r['dt']-.01)<1e-6
        with np.load(data) as z:saved=pd.DataFrame(z['forecast'],columns=z['forecast_columns'])
        if mode=='infer':
            checkpoint=torch.load(cp,map_location='cpu',weights_only=False)
            model=StateOnlyTrajectory(state_dim=4,phys_dim=0,history_window=100,max_time=r['max_time'])
            model.load_state_dict(checkpoint['model_state_dict'])
            df=infer(r,model,r['test_start_idx'],len(t))
            assert np.array_equal(df[['anchor_index','target_index']],saved[['anchor_index','target_index']])
            error=float(np.max(np.abs(df[['SE2_pred','ME2_pred']].to_numpy()-saved[['SE2_pred','ME2_pred']].to_numpy())))
            if paper_weights:assert error<1e-4,(tag,error)
        else:df=saved;error=0.
        (out/'forecasts').mkdir(exist_ok=True)
        df.to_csv(out/'forecasts'/f'{safe}.csv.gz',index=False)
        scale,_=movement_scale(r);T=scale['T_sec'];scales.append(dict(tag=tag,**scale))
        amp=causal_robust_amplitude(r['x']-r['y'],r['fs'])
        signal={'A':amp[::5]/positive_median(amp[:r['sel_idx']],fallback=1),'RA':r['se2'][::5]}
        tt=t[::5];fits=tt<t[r['sel_idx']]
        series=pd.DataFrame(dict(time=tt,A=signal['A'],RA=r['se2'][::5],RE=r['me2'][::5]))
        scores={};fit_stats={};thresholds={}
        for inp,w in [('A',5),('A',3),('A',7),('RA',5)]:
            v=rolling_ews(signal[inp],T,w)
            for j,m in enumerate(['AC1','Variance','Recovery','Reddening']):
                key=m if (inp,w)==('A',5) else f'{m}_{inp}{w}T'
                raw=v[:,j];series[key]=raw;scores[key]=-raw if m=='Recovery' else raw
                finite=raw[fits&np.isfinite(raw)]
                fit_stats[key]=dict(median=float(np.median(finite)) if len(finite) else np.nan,
                    iqr=float(np.subtract(*np.quantile(finite,[.75,.25]))) if len(finite) else np.nan)
                sf=scores[key][fits&np.isfinite(scores[key])];q=cfg['selected_quantiles'][key]
                thresholds[key]=float(np.quantile(sf,q)) if len(sf) else np.nan
                old=expected_thresholds[expected_thresholds.tag.eq(tag)&expected_thresholds.method.eq(key)&np.isclose(expected_thresholds.q,q)]
                assert len(old)==1 and np.isclose(thresholds[key],old.threshold.iloc[0],equal_nan=True,rtol=1e-8,atol=1e-9),(tag,key)
            series[f'phi_{inp}{w}T']=v[:,4]
        folder=out/'series'/safe;folder.mkdir(parents=True,exist_ok=True)
        series.to_csv(folder/'causal_indicators.csv',index=False)
        dump(folder/'fit_statistics.json',fit_stats)
        dump(folder/'reference.json',dict(tag=tag,participant=r['participant'],T=T,
            test_onsets=[g for g,e in r['gt_test']],test_start=r['test_start_time'],record_end=float(t[-1])))
        bases={m:trajectory_base(vars(C),r,df,m=='Persistence') for m in ['State-history','Persistence']}
        ii=bases['State-history'][0];available=bases['State-history'][-1].sum(axis=1)>=3
        for method,q in cfg['selected_quantiles'].items():
            if method in bases:candidate=trajectory_candidates(vars(C),r,bases[method],q)
            else:
                ss=scores[method][ii//5];candidate=np.isfinite(ss)&(ss>thresholds[method])
            trace=pd.DataFrame(dict(anchor_time=t[ii],anchor_index=ii,candidate=candidate,evaluable=available))
            ints,state=activation_intervals(trace,cfg['exit_duration'],cfg['enter_confirmations'],record_end=t[-1])
            old=expected_alarms[expected_alarms.tag.eq(tag)&expected_alarms.method.eq(method)]
            if paper_weights or method!='State-history':
                assert len(ints)==len(old),(tag,method,len(ints),len(old))
                assert np.allclose(ints[['issued_at','observed_end']].to_numpy(float),old[['issued_at','observed_end']].to_numpy(float),atol=1e-8,rtol=0),(tag,method)
            dest=out/'activations'/safe;dest.mkdir(parents=True,exist_ok=True)
            ints.to_csv(dest/f'{method}.csv',index=False)
            if method in bases:state.merge(trace,on='anchor_time').to_csv(dest/f'{method}_trace.csv',index=False)
            row,ar,er=evaluate_record(r,method,ints,trace,q)
            rows.append(row);alarms.extend(ar);events.extend(er)
        # Forecast regimes overlap by design; predictions use no reference labels.
        ai=df.anchor_index.to_numpy(int);ti=df.target_index.to_numpy(int)
        raw=np.c_[r['se2'],r['me2']];pred=df[['SE2_pred','ME2_pred']].to_numpy()
        cross=np.zeros(len(df),bool)
        for onset,end in r['gt_test']:cross |= (t[ai]<onset)&(t[ti]>=onset)
        drop=np.maximum(np.log(np.maximum(raw[ai],1e-4))-np.log(np.maximum(raw[ti],1e-4)),0).max(axis=1)>=.15
        for regime,mask in [('all',np.ones(len(df),bool)),('onset_crossing',cross),('coordinate_decline',drop)]:
            for h in [5,25,50,75,100,125,150]:
                use=mask&((ti-ai)==h)
                for j,coord in enumerate(['RA','RE']):
                    for modelname,pr in [('State-history',pred),('Persistence',raw[ai])]:
                        mae.append(dict(tag=tag,participant=r['participant'],regime=regime,horizon_sec=h*r['dt'],
                            coordinate=coord,method=modelname,n=int(use.sum()),absolute_error_sum=float(np.abs(pr[use,j]-raw[ti[use],j]).sum())))
        checks.append(dict(tag=tag,forecast_pairs=len(df),max_forecast_difference=error,all_18_activation_streams_match=True if paper_weights else None))
        print(f'[{k}/21] {tag}: difference from saved forecast={error:.3g}; paper weights={paper_weights}',flush=True)
    records=pd.DataFrame(rows);ev=pd.DataFrame(events);al=pd.DataFrame(alarms)
    summary=summarize(records,ev)
    expected=pd.read_csv(ROOT/'reference/summary.csv').set_index('method')
    assert set(summary.index)==set(expected.index)
    for col in expected.columns:
        if paper_weights:assert np.allclose(summary.loc[expected.index,col],expected[col],atol=1e-8,rtol=1e-8,equal_nan=True),col
    ci,dif,paired,ps=bootstrap(records,ev,summary,cfg['bootstrap_seed'],cfg['bootstrap_replicates'])
    for name,frame in [('summary',summary.reset_index()),('per_record',records),('events',ev),('alarms',al),
                       ('participant_cluster_intervals',ci),('participant_cluster_differences',dif),('paired_leads',paired)]:
        frame.to_csv(out/f'{name}.csv',index=False)
    for name,keys in [('participant_cluster_intervals',['method','metric']),('participant_cluster_differences',['comparator','metric'])]:
        a=pd.read_csv(out/f'{name}.csv').set_index(keys).sort_index()
        b=pd.read_csv(ROOT/'reference'/f'{name}.csv').set_index(keys).sort_index()
        if paper_weights:assert np.allclose(a,b,atol=1e-8,rtol=1e-8),name
    mf=pd.DataFrame(mae);mf['horizon_sec']=mf.horizon_sec.round(2)
    mf.to_csv(out/'forecast_errors_per_record.csv',index=False)
    ms=mf.groupby(['regime','horizon_sec','coordinate','method'])[['n','absolute_error_sum']].sum()
    ms['MAE']=ms.absolute_error_sum/ms.n.replace(0,np.nan);ms.to_csv(out/'forecast_errors.csv')
    pd.DataFrame(scales).to_csv(out/'movement_scales.csv',index=False)
    dump(out/'reference_intervals.json',references);dump(out/'paired_lead_summary.json',ps)
    dump(out/'verification.json',dict(mode=mode,model_training=False,records=21,participants=7,reference_onsets=88,
        summary_all_18_matches_paper=True if paper_weights else None,bootstrap_matches_paper=True if paper_weights else None,checks=checks,state_machine=test_state_machine(),
        checkpoint_directory=str(checkpoint_dir) if checkpoint_dir else 'bundled paper weights',
        input_integrity_verified=True))
    dump(out/'protocol.json',cfg)
    from .report import make_report
    make_report(out,summary,records,ev)
    (out/'COMPLETE').write_text('Verified against current manuscript; all 18 configurations match.\n' if paper_weights else 'New checkpoint evaluation complete; not the published paper numbers.\n')
    print(summary.loc[MAIN,['covered','alarms','successful_warnings','coverage','success_fraction','median_lead']].to_string(),flush=True)
    return out
