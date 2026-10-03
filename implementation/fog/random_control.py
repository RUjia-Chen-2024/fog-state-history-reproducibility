"""Event-aligned EWS versus within-record random-onset controls; no fitting."""
from pathlib import Path
import json,warnings,zipfile,time
import numpy as np,pandas as pd
from scipy.stats import kendalltau
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R=Path(__file__).resolve().parents[1];SRC=R/'outputs/paper';O=SRC/'random_onset_control';O.mkdir(parents=True,exist_ok=True);F=O/'figures';F.mkdir(exist_ok=True)
METHODS=['RA','RE','AC1','Variance','Recovery','Reddening'];LABEL=['Amplitude retention','Energy retention','AC(1)','Variance','AR(1) recovery','Spectral reddening']
GRID=np.arange(-4,0,.05);NREP=1000;NBOOT=2000;SEED=20261003
rng=np.random.default_rng(SEED);rngb=np.random.default_rng(SEED+1)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
warnings.filterwarnings('ignore',category=RuntimeWarning)

def dump(p,x):p.write_text(json.dumps(x,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else str(v)))

def align(record,onsets):
    query=np.asarray(onsets)[:,None]+GRID[None,:]*record['T']
    idx=np.searchsorted(record['time'],query,side='right')-1
    assert (idx>=0).all()
    assert np.all(record['time'][idx]<=query+1e-10)
    assert np.all(query-record['time'][idx]<.050001)
    return record['values'][idx]

def stats(a):
    early=np.nanmedian(a[:,GRID<-2,:],axis=1);late=np.nanmedian(a[:,GRID>=-2,:],axis=1)
    final=np.nanmedian(a[:,GRID>=-1,:],axis=1)
    count_early=np.isfinite(a[:,GRID<-2,:]).sum(axis=1);count_late=np.isfinite(a[:,GRID>=-2,:]).sum(axis=1)
    change=late-early;change[(count_early<5)|(count_late<5)]=np.nan
    finalchange=final-early;finalchange[(count_early<5)|(np.isfinite(a[:,GRID>=-1,:]).sum(axis=1)<5)]=np.nan
    tau=np.full_like(change,np.nan)
    for i in range(len(a)):
        for j in range(len(METHODS)):
            y=a[i,:,j];valid=np.isfinite(y)
            if valid.sum()>=10:tau[i,j]=kendalltau(GRID[valid],y[valid]).statistic
    return np.stack([change,tau,finalchange],axis=-1)

def aggregate_participant(a,participant,people):
    return np.stack([np.nanmean(a[participant==p],axis=0) for p in people])

def figure_save(fig,name):
    fig.savefig(F/f'{name}.png',dpi=190,bbox_inches='tight');fig.savefig(F/f'{name}.svg',bbox_inches='tight');plt.close(fig)

def main():
    protocol=dict(cohort='Fixed21 recordings,7participants,88test reference onsets; all causal EWS and parameters from previous run unchanged',
        domain='For each recording, pseudo onset lies in [test_start,record_end). True and pseudo pre-onset histories may use available pretest past, identically. No circular wrapping of signal histories.',
        primary='1000 circular shifts of the entire reference-onset train per recording, independently between records; shift uniform on recording test duration; event count and cyclic spacings preserved. No exclusion near real onsets.',
        secondary='1000 uniform random samples (without replacement within a record when sufficient) from20Hz test times outside reference event intervals and at least1T from EVERY full-record reference onset. Event count per record preserved when pool exists. Signal history may include earlier events; these are background-center controls, not event-free4T histories.',
        alignment='[-4T,0), step0.05T. Last observed20Hz sample at or before query; no future interpolation. Original detrended5T classical metrics reused. EWS display units fit-median/IQR; RA,RE unscaled.',
        primary_statistic='Per-event median([-2T,0)) minus median([-4T,-2T)), and Kendall tau over[-4T,0). Supplementary last1T minus early window.',
        inference='Equal-participant mean of event-level statistics. Real minus average random-control statistic paired within participant; percentile95% intervals from2000 resamples of7participants. Conditional on frozen metrics and Monte Carlo expected control, no p-values. Random-reference bands are2.5-97.5 percentiles over randomization replicates, not participant CIs.',
        curves='Real pooled-event median with pointwise participant-cluster95%CI; random median and95%random-reference envelope over1000 pooled medians. Difference curves use participant-balanced mean curves and paired cluster intervals.',
        limits='Retrospective descriptive timing-specificity test, not new prediction performance. Circular-shift null assumes useful approximate temporal exchangeability within short nonstationary test records. Seven participants, algorithmic labels, correlated overlapping histories; no proof of bifurcation or causal mechanism.',
        seed=SEED,n_random=NREP,n_bootstrap=NBOOT)
    dump(O/'protocol.json',protocol)
    intervals=json.loads((SRC/'reference_intervals.json').read_text())
    records=[];parts=[];event_meta=[];real=[];pools=[]
    for folder in sorted((SRC/'series').iterdir()):
        ref=json.loads((folder/'reference.json').read_text());s=pd.read_csv(folder/'causal_indicators.csv');fit=json.loads((folder/'fit_statistics.json').read_text())
        vals=s[METHODS].to_numpy().copy()
        for j,m in enumerate(METHODS[2:],2):
            z=fit[m];vals[:,j]=(vals[:,j]-z['median'])/z['iqr'] if z['iqr']>1e-12 else np.nan
        rec=dict(ref,time=s.time.to_numpy(),values=vals,n=len(ref['test_onsets']))
        candidates=s.time.to_numpy();candidates=candidates[(candidates>=ref['test_start'])&(candidates<ref['record_end'])]
        keep=np.ones(len(candidates),bool)
        for st,en in intervals[ref['tag']]:keep&=(np.abs(candidates-st)>=ref['T'])&~((candidates>=st)&(candidates<en))
        rec['background_pool']=candidates[keep]
        records.append(rec);real.append(align(rec,rec['test_onsets']))
        pools.append(dict(tag=ref['tag'],participant=ref['participant'],n_events=rec['n'],background_candidates=int(keep.sum()),test_candidates=len(candidates),background_seconds=.05*int(keep.sum()),sampling_with_replacement=int(keep.sum())<rec['n']))
        for g in ref['test_onsets']:parts.append(ref['participant']);event_meta.append(dict(tag=ref['tag'],participant=ref['participant'],reference_onset=g))
    real=np.concatenate(real,axis=0);parts=np.array(parts);people=np.unique(parts)
    assert real.shape==(88,80,6) and len(people)==7
    pd.DataFrame(pools).to_csv(O/'background_candidate_pools.csv',index=False)
    realstat=stats(real);realp=aggregate_participant(realstat,parts,people)
    cluster_indices=[np.where(parts==p)[0] for p in people]
    bootstrap=rngb.integers(0,len(people),(NBOOT,len(people)))
    true_median=np.nanmedian(real,axis=0)
    trueboot=np.array([np.nanmedian(real[np.concatenate([cluster_indices[k] for k in draw])],axis=0) for draw in bootstrap])
    true_lo,true_hi=np.nanquantile(trueboot,[.025,.975],axis=0)
    evrows=[]
    for i,meta in enumerate(event_meta):
        for j,m in enumerate(METHODS):
            evrows.append(dict(**meta,method=m,late_minus_early=realstat[i,j,0],kendall_tau=realstat[i,j,1],lastT_minus_early=realstat[i,j,2],n_valid=int(np.isfinite(real[i,:,j]).sum())))
    pd.DataFrame(evrows).to_csv(O/'real_event_statistics.csv',index=False)
    statrows=[];curverows=[];rep_rows=[];difference_rows=[];all_curves={};null_offsets=[]
    for scheme in ['circular_shift','background_center']:
        valid_records=[r for r in records if scheme=='circular_shift' or len(r['background_pool'])>0]
        included_tags={r['tag'] for r in valid_records};mask=np.array([m['tag'] in included_tags for m in event_meta]);subparts=parts[mask]
        subreal=real[mask];substat=realstat[mask];subpeople=np.unique(subparts)
        subp=aggregate_participant(substat,subparts,subpeople)
        partcurve=aggregate_participant(subreal,subparts,subpeople)
        nullcurve=[];nullstats=[];part_null_sum=np.zeros_like(subp);part_null_count=np.zeros_like(subp)
        curve_null_sum=np.zeros_like(partcurve);curve_null_count=np.zeros_like(partcurve)
        for rep in range(NREP):
            windows=[]
            for rec in valid_records:
                if scheme=='circular_shift':
                    L=rec['record_end']-rec['test_start'];offset=rng.uniform(0,L)
                    pseudo=rec['test_start']+((np.array(rec['test_onsets'])-rec['test_start']+offset)%L)
                    # Validate circular point spacing; only labels, never signals, wrap.
                    before=np.diff(np.r_[np.sort(rec['test_onsets']),min(rec['test_onsets'])+L])
                    after=np.diff(np.r_[np.sort(pseudo),min(pseudo)+L])
                    assert np.allclose(np.sort(before),np.sort(after),atol=1e-8)
                else:
                    offset=np.nan;pseudo=rng.choice(rec['background_pool'],size=rec['n'],replace=len(rec['background_pool'])<rec['n'])
                assert np.all((pseudo>=rec['test_start'])&(pseudo<rec['record_end']))
                windows.append(align(rec,pseudo))
                null_offsets.extend(dict(scheme=scheme,replicate=rep,tag=rec['tag'],event_index=k,pseudo_onset=float(v),shift_seconds=offset) for k,v in enumerate(pseudo))
            a=np.concatenate(windows);v=stats(a);pv=aggregate_participant(v,subparts,subpeople);pc=aggregate_participant(a,subparts,subpeople)
            part_null_sum+=np.nan_to_num(pv);part_null_count+=np.isfinite(pv)
            curve_null_sum+=np.nan_to_num(pc);curve_null_count+=np.isfinite(pc)
            nullcurve.append(np.nanmedian(a,axis=0));nullstats.append(np.nanmean(pv,axis=0))
            if (rep+1)%100==0:print(scheme,'replicate',rep+1,flush=True)
        nullcurve=np.array(nullcurve);nullstats=np.array(nullstats)
        expected_p=np.divide(part_null_sum,part_null_count,out=np.full_like(part_null_sum,np.nan),where=part_null_count>0)
        expected_curve=np.divide(curve_null_sum,curve_null_count,out=np.full_like(curve_null_sum,np.nan),where=curve_null_count>0)
        pdiff=subp-expected_p;cdiff=partcurve-expected_curve
        bs=bootstrap if np.array_equal(subpeople,people) else rngb.integers(0,len(subpeople),(NBOOT,len(subpeople)))
        bootstats=np.nanmean(pdiff[bs],axis=1);bootcurve=np.nanmean(cdiff[bs],axis=1)
        dlo,dhi=np.nanquantile(bootcurve,[.025,.975],axis=0)
        random_med=np.nanmedian(nullcurve,axis=0);random_lo,random_hi=np.nanquantile(nullcurve,[.025,.975],axis=0)
        all_curves[scheme]=(random_med,random_lo,random_hi,np.nanmean(cdiff,axis=0),dlo,dhi)
        for j,m in enumerate(METHODS):
            for k,key in enumerate(['late_minus_early','kendall_tau','lastT_minus_early']):
                point=float(np.nanmean(pdiff[:,j,k]));lo,hi=np.nanquantile(bootstats[:,j,k],[.025,.975]);observed=float(np.nanmean(subp[:,j,k]));ctrl=float(np.nanmean(expected_p[:,j,k]))
                statrows.append(dict(scheme=scheme,method=m,statistic=key,n_events=int(mask.sum()),n_participants=len(subpeople),real=observed,random_expected=ctrl,difference=point,lower95=lo,upper95=hi,random_reference_lower=float(np.nanquantile(nullstats[:,j,k],.025)),random_reference_upper=float(np.nanquantile(nullstats[:,j,k],.975))))
            for l,g in enumerate(GRID):
                curverows.append(dict(scheme=scheme,method=m,time_to_onset_T=g,real_median=true_median[l,j] if mask.all() else float(np.nanmedian(subreal[:,l,j])),real_lower95=true_lo[l,j] if mask.all() else np.nan,real_upper95=true_hi[l,j] if mask.all() else np.nan,random_median=random_med[l,j],random_lower=random_lo[l,j],random_upper=random_hi[l,j],mean_difference=float(np.nanmean(cdiff[:,l,j])),difference_lower95=dlo[l,j],difference_upper95=dhi[l,j]))
            for rep in range(NREP):
                rep_rows.append(dict(scheme=scheme,replicate=rep,method=m,late_minus_early=nullstats[rep,j,0],kendall_tau=nullstats[rep,j,1],lastT_minus_early=nullstats[rep,j,2]))
        np.savez_compressed(O/f'{scheme}_random_median_curves.npz',grid=GRID,methods=METHODS,curves=nullcurve)
    summary=pd.DataFrame(statrows);summary.to_csv(O/'real_minus_random_summary.csv',index=False)
    pd.DataFrame(curverows).to_csv(O/'aligned_curves.csv',index=False);pd.DataFrame(rep_rows).to_csv(O/'random_replicate_statistics.csv',index=False)
    pd.DataFrame(null_offsets).to_csv(O/'pseudo_onsets.csv',index=False)
    # Main figure: classical EWS only; RA/RE supplementary due shared label source.
    for methods,name in [([2,3,4,5],'classical_EWS_real_vs_random'),([0,1],'retention_real_vs_random')]:
        fig,axs=plt.subplots(2 if len(methods)==4 else 1,2,figsize=(11,7 if len(methods)==4 else 3.8),sharex=True)
        null,lo,hi,_,_,_=all_curves['circular_shift']
        for ax,j in zip(np.array(axs).ravel(),methods):
            ax.fill_between(GRID,lo[:,j],hi[:,j],color='#999999',alpha=.24,label='Random-onset 95% reference band')
            ax.plot(GRID,null[:,j],color='#555555',ls='--',label='Random-onset median')
            ax.fill_between(GRID,true_lo[:,j],true_hi[:,j],color='#1776ad',alpha=.18,label='Real-onset 95% cluster interval')
            ax.plot(GRID,true_median[:,j],color='#1776ad',lw=1.8,label='Real-onset median')
            ax.axvline(-2,color='#888888',lw=.8,ls=':');ax.set_title(LABEL[j],loc='left');ax.grid(alpha=.15)
            ax.set_ylabel('Fit-median centered / fit IQR' if j>=2 else 'Retention');ax.set_xlabel('Time to onset / T')
        handles,labels=np.array(axs).ravel()[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=2,loc='lower center',bbox_to_anchor=(.5,-.035),frameon=False,fontsize=9)
        fig.suptitle('Algorithmic reference onsets versus within-record random onset shifts',fontsize=12)
        fig.tight_layout(rect=(0,.10,1,.96));figure_save(fig,name)
    fig,axs=plt.subplots(2,3,figsize=(12,6.5),sharex=True)
    _,_,_,dif,lo,hi=all_curves['circular_shift']
    for ax,j in zip(axs.ravel(),range(6)):
        ax.axhline(0,color='#777777',ls='--',lw=.8);ax.plot(GRID,dif[:,j],color='#1776ad');ax.fill_between(GRID,lo[:,j],hi[:,j],color='#1776ad',alpha=.2)
        ax.set_title(LABEL[j],loc='left');ax.set_xlabel('Time to onset / T');ax.set_ylabel('Real minus random');ax.grid(alpha=.15)
    fig.suptitle('Participant-balanced mean difference; pointwise paired cluster 95% intervals',fontsize=12);fig.tight_layout(rect=(0,0,1,.95));figure_save(fig,'paired_difference_curves')
    fig,axs=plt.subplots(1,2,figsize=(11,4.8))
    for ax,key in zip(axs,['late_minus_early','kendall_tau']):
        for k,scheme in enumerate(['circular_shift','background_center']):
            a=summary[summary.scheme.eq(scheme)&summary.statistic.eq(key)].set_index('method').loc[METHODS[2:]]
            yy=np.arange(4)+(k-.5)*.16
            ax.errorbar(a.difference,yy,xerr=[a.difference-a.lower95,a.upper95-a.difference],fmt='o',capsize=3,color=['#1776ad','#c17b32'][k],label=['Circular-shift control','Background-center control'][k])
        ax.axvline(0,color='#777777',ls='--');ax.set_yticks(range(4),LABEL[2:]);ax.invert_yaxis();ax.set_xlabel('Real minus random');ax.set_title('Late minus early change' if key=='late_minus_early' else 'Kendall trend');ax.grid(axis='x',alpha=.15)
    axs[1].legend(loc='best',fontsize=9);fig.tight_layout();figure_save(fig,'effect_size_intervals')
    table=summary[summary.scheme.eq('circular_shift')&summary.method.isin(METHODS[2:])&summary.statistic.isin(['late_minus_early','kendall_tau'])]
    def htmltable(x):return x.to_html(index=False,border=0,float_format=lambda v:f'{v:.4f}')
    html='''<!doctype html><meta charset="utf-8"><title>起点对齐EWS与随机起点对照</title><style>body{font-family:Arial,"Microsoft YaHei",sans-serif;max-width:1450px;margin:auto;padding:24px;line-height:1.8;color:#253545}img{width:100%}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:7px;border:1px solid #ccd}th{background:#edf3fa}.note{padding:18px;background:#fff2d7}</style><h1>起点对齐的传统EWS与随机伪起点对照</h1>
<div class="note">固定原21条记录、7位参与者、88个测试算法参考起点。复用已计算的因果振幅包络及5T传统指标，无重新训练、无改阈值。这里只检验指标变化与参考起点的时间关联，不计算新的预测准确率。</div>
<h2>主对照：同记录整组起点随机平移</h2><p>每条记录独立随机平移整组测试起点，超出边界的标签循环回到测试段起始；保持事件数和循环间隔。仅标签移动，原始时间序列不打乱，前史不循环拼接。重复1000次，允许随机起点偶然落在真实事件附近。这样避免先挑正常背景造成过强对比。真实与伪起点都可使用测试段之前的可用过去数据。</p>
<img src="figures/classical_EWS_real_vs_random.png"><p>蓝实线：真实参考起点对齐的事件中位曲线；蓝阴影：7位参与者聚类重采样2000次的逐点95%区间。灰虚线：1000次随机对齐中位曲线的中位数；灰带：随机对齐曲线的95%参考范围。两种阴影含义不同，不通过是否重叠判定显著性。</p>
<h2>配对差异与统计量</h2><p>先在每个事件内计算晚期[-2T,0)相对早期[-4T,-2T)的中位变化，以及[-4T,0)上的Kendall趋势；再在参与者内平均，7位参与者等权。下表为真实值减同一参与者的随机对照期望及配对聚类95%区间。区间以固定指标参数和Monte Carlo随机对照期望为条件，不是多重比较校正的同时区间，不作强显著性论断。</p>'''+htmltable(table[['method','statistic','real','random_expected','difference','lower95','upper95']])+'''<img src="figures/effect_size_intervals.png"><img src="figures/paired_difference_curves.png">
<h2>背景时刻敏感性对照</h2><p>从测试段20Hz时间网格抽取与全部参考起点至少相距1T、且不在参考事件区间内的时刻，按记录匹配事件数量。伪起点的更早历史仍可能含其它参考事件，因此它不是完整4T无事件前史。该对照更容易区分，只作为敏感性结果。候选池如下；池不足时明确标记放回抽样，池为空则真实与伪起点同时限制到有候选的记录。</p>'''+htmltable(pd.DataFrame(pools))+htmltable(summary[summary.scheme.eq('background_center')&summary.statistic.eq('late_minus_early')][['method','n_events','real','random_expected','difference','lower95','upper95']])+'''
<h2>补充：RA、RE</h2><img src="figures/retention_real_vs_random.png"><p>这些坐标和参考事件由同源步态信号构造，若有区别也不能单独证明临床有效性或分叉机制。晚期1T对早期窗口的结果保存在完整统计表中，未根据结果替换主窗口。</p>
<h2>解释边界</h2><p>随机起点不是随机化临床试验，也没有证明真实系统发生分叉。对照仅回答“固定的这些指标，在算法参考起点前是否呈现超过该记录普通时间变化的形态”。短测试段存在非平稳性、重复事件及窗口重叠；循环平移的时间可交换性只是参照假设。样本为7位参与者，不能把1000次随机对齐当作1000个独立受试者。</p>
<p>传统EWS与代理检验方法背景见<a href="https://doi.org/10.1371/journal.pone.0041010">Dakos等(2012)</a>。本次“同记录起点平移”是明确声明的实验对照设计，并非对该文方法的逐项复制。</p>
<p><a href="real_minus_random_summary.csv">全部差异与区间</a> · <a href="pseudo_onsets.csv">全部伪起点</a> · <a href="aligned_curves.csv">图中数据</a> · <a href="protocol.json">实验协议</a> · <a href="../../../fog/random_control.py">运行脚本</a></p>'''
    (O/'起点对齐与随机对照.html').write_text(html,encoding='utf-8')

    dump(O/'verification.json',dict(n_events=88,n_records=21,n_participants=7,no_signal_shuffling=True,strict_past_alignment_checked=True,circular_spacing_preservation_checked=True,random_centers_in_test_domain_checked=True,n_random_per_control=NREP,n_bootstrap=NBOOT,empty_background_records=[p['tag'] for p in pools if p['background_candidates']==0]))
    (O/'COMPLETE').write_text('Random-onset controls, paired cluster estimates and plots complete.\n')
    print(table.to_string(index=False),flush=True)

if __name__=='__main__':main()
