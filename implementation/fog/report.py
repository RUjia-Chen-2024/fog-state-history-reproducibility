from pathlib import Path
import html,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .evaluation import MAIN

def make_report(out,summary,records,events):
    figdir=out/'figures';figdir.mkdir(exist_ok=True)
    refs=json.loads((out/'reference_intervals.json').read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none'})
    panels=[]
    for tag in records.tag.drop_duplicates():
        safe=tag.replace(' ','_');folder=out/'series'/safe
        s=pd.read_csv(folder/'causal_indicators.csv');meta=json.loads((folder/'reference.json').read_text())
        sel=s.time>=meta['test_start'];fig,ax=plt.subplots(figsize=(12,2.6))
        ax.plot(s.time[sel],s.RA[sel],lw=1.1,color='#c93632',label='$R_A$')
        ax.plot(s.time[sel],s.RE[sel],lw=1.1,color='#237bb4',label='$R_E$')
        for j,(g,e) in enumerate(refs[tag]):
            if e<meta['test_start']:continue
            ax.axvspan(max(g,meta['test_start']),e,color='#c8c8c8',alpha=.4,label='Cycle-derived reference' if not any(x[1]>=meta['test_start'] for x in refs[tag][:j]) else None)
        for m,y,color in [('State-history',-.20,'#23915e'),('Persistence',-.42,'#ae8737')]:
            ep=pd.read_csv(out/'activations'/safe/f'{m}.csv')
            ax.broken_barh([(a.issued_at,a.observed_end-a.issued_at) for a in ep.itertuples()],(y,.14),facecolors=color,label=m+' active')
            ax.scatter(ep.issued_at,np.full(len(ep),y+.07),s=10,marker='|',color='black',zorder=4)
        ax.set(xlim=(meta['test_start'],meta['record_end']),ylim=(-.52,2.05),xlabel='Recording time (s)',ylabel='Retention / warning bands')
        ax.set_title(tag,loc='left');ax.legend(loc='upper center',ncol=5,fontsize=8)
        ax.spines[['top','right']].set_visible(False);fig.tight_layout()
        fig.savefig(figdir/f'{safe}.png',dpi=150);fig.savefig(figdir/f'{safe}.svg');plt.close(fig)
        panels.append(f'<h3>{html.escape(tag)}</h3><img src="figures/{safe}.png">')
    columns=['covered','n_reference','alarms','successful_warnings','coverage','success_fraction','median_lead','mean_lead','active_fraction']
    table=summary.loc[MAIN,columns].to_html(float_format=lambda x:f'{x:.4f}')
    main=records[records.method.eq('State-history')].copy();main['coverage']=main.covered/main.n_reference
    text='''<!doctype html><meta charset="utf-8"><title>独立论文代码：事件间预警评价</title>
<style>body{font:16px/1.7 Arial,"Microsoft YaHei",sans-serif;max-width:1250px;margin:auto;padding:25px;color:#243343}table{border-collapse:collapse;font-size:13px}th,td{border:1px solid #ccd;padding:7px}img{width:100%}.note{background:#edf4f7;padding:18px}</style>
<h1>论文独立代码：状态历史预测与事件间预警</h1>
<div class="note">固定21条记录、7位参与者、88个测试参考起点。参考事件为步态周期算法标签，并非独立临床标注。主规则：上一次参考事件结束 &lt; 实际预警激活 &lt; 下一次参考起点。没有最大提前时间限制，也不要求预警保持到起点。参考事件边界只参与事后评价。</div>
<h2>主比较</h2><p>coverage：事件覆盖率；success_fraction：成功预警次数/全部激活次数；median_lead：每个成功事件的首次合格预警提前时间中位数（秒）。success_fraction 不是固定时间窗口下的预警精确率。一个事件间隔内多次激活分别计数，事件只覆盖一次。</p>'''+table+'''
<h2>逐记录结果</h2>'''+main[['tag','n_reference','covered','alarms','successful_warnings','during_event_or_boundary','unresolved_tail','coverage']].to_html(index=False)+'''
<h2>真实坐标、参考事件与实际激活区间</h2><p>灰色：周期算法参考事件。绿色：State-history实际激活区间。棕色：Persistence实际激活区间。黑色短线：实际发出预警的时刻。所有曲线和色带均由本次运行生成。</p>'''+''.join(panels)
    (out/'report.html').write_text(text,encoding='utf-8')
