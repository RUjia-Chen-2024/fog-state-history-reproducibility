"""Render vector figures from unchanged recorded coordinates and saved activations."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'analysis/inter_event_revision'
FIG = ROOT / 'figures'
summary = pd.read_csv(DATA / 'summary.csv').set_index('method')
intervals = pd.read_csv(DATA / 'participant_cluster_intervals.csv')
events = pd.read_csv(DATA / 'events.csv')
alarms = pd.read_csv(DATA / 'alarms.csv')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
    'svg.fonttype': 'none', 'pdf.fonttype': 42, 'axes.spines.top': False,
    'axes.spines.right': False})
BLUE, ORANGE, GRAY = '#206d9e', '#bb7049', '#858d96'

def save(fig, name):
    for ext in ['pdf', 'svg', 'png']:
        fig.savefig(FIG / f'{name}.{ext}', dpi=240, bbox_inches='tight')
    plt.close(fig)

methods = ['Persistence', 'AC1', 'Variance', 'Recovery', 'Reddening', 'State-history']
labels = ['Persistence', 'Autocorrelation', 'Variance', 'AR(1) proxy', 'Spectral reddening', 'State-history']
colors = [GRAY, '#70929b', '#82a191', '#b9a779', '#9c8b9e', BLUE]
fig, axes = plt.subplots(1, 3, figsize=(11.8, 4.1), layout='constrained',
    gridspec_kw={'width_ratios':[1,1,1.65]})
y = np.arange(6)
for ax, metric, title in zip(axes[:2], ['coverage', 'success_fraction'],
        ['a  Event advance coverage', 'b  Activation association']):
    values = summary.loc[methods, metric].to_numpy()*100
    ax.barh(y, values, color=colors, height=.55)
    ci = intervals[intervals.metric.eq(metric)].set_index('method').loc[methods]
    ax.errorbar(values,y,xerr=np.vstack((values-ci.lower95.to_numpy()*100,
        ci.upper95.to_numpy()*100-values)),fmt='none',ecolor='#26323b',capsize=2.5,elinewidth=.9)
    ax.set(xlim=(0,105),xticks=[0,25,50,75,100],xlabel='Percent',yticks=y,ylim=(5.65,-.65))
    for yy,v in zip(y,values):
        ax.text(.02,yy+.29,f'{v:.1f}%',transform=ax.get_yaxis_transform(),fontsize=8.5,va='top',color='#313a43')
    ax.set_title(title,loc='left',fontsize=10.3,weight='bold',pad=12)
    ax.set_axisbelow(True);ax.grid(axis='x',color='#e3e7ea',lw=.55)
axes[0].set_yticklabels(labels,fontsize=10)
axes[1].set_yticklabels([])
ax=axes[2]
other=summary.loc[[m for m in summary.index if m not in methods+['AC1_A3T']]]
ax.scatter(other.activations_per_minute,other.coverage*100,s=25,marker='D',
    facecolors='none',edgecolors='#b7bfc6',linewidths=.9,label='Other EWS settings',zorder=2)
short=['Persistence','AC1, 5T','Variance, 5T','AR(1), 5T','Reddening, 5T','State-history']
offsets=[(5,-13),(5,-12),(-2,-15),(5,-13),(5,9),(5,9)]
for m,label,col,off in zip(methods,short,colors,offsets):
    s=summary.loc[m]
    ax.scatter(s.activations_per_minute,s.coverage*100,s=62,color=col,
        edgecolors='white',linewidths=.8,zorder=4)
    ax.annotate(label,(s.activations_per_minute,s.coverage*100),xytext=off,
        textcoords='offset points',fontsize=8.7,color=col,weight='bold' if m=='State-history' else 'normal')
s=summary.loc['AC1_A3T']
ax.scatter(s.activations_per_minute,s.coverage*100,s=66,marker='D',color=ORANGE,edgecolors='white',linewidths=.7,zorder=4)
ax.annotate('AC1, 3T',(s.activations_per_minute,s.coverage*100),xytext=(-3,-16),
    ha='right',textcoords='offset points',fontsize=8.7,color=ORANGE)
ax.set(xlim=(0,47),ylim=(15,106),xticks=[0,10,20,30,40],yticks=[25,50,75,100],
    xlabel='Actual activations / min',ylabel='Event advance coverage (%)')
ax.set_title('c  Coverage and warning burden',loc='left',fontsize=10.3,weight='bold',pad=12)
ax.set_axisbelow(True);ax.grid(color='#e3e7ea',lw=.55)
ax.legend(loc='lower right',frameon=False,fontsize=8.5,handletextpad=.3)
save(fig, 'classical_three_metrics')

# Retain the same recording selected for the preceding figure, without reselection.
tag = 'Pt 20_FD20_3'
meta = json.loads((ROOT / 'analysis/recording_example/metadata.json').read_text(encoding='utf-8'))
assert meta['tag'] == tag
raw_path = ROOT / 'analysis/recording_example' / meta['file']
trace_path = ROOT / 'analysis/online_states/Pt_20_FD20_3/state_trace.csv'
raw = pd.read_csv(raw_path)
trace = pd.read_csv(trace_path)
raw = raw[raw.time_s >= trace.anchor_time.min()]
start, end = raw.time_s.min(), raw.time_s.max()
ee = events[events.tag.eq(tag) & events.method.eq('State-history')].sort_values('reference_onset')
aa = alarms[alarms.tag.eq(tag) & alarms.method.eq('State-history')].sort_values('issued_at')
assert len(ee) == 5 and ee.covered.sum() == 5
assert len(aa) == 9 and aa.status.eq('successful_between_events').sum() == 6
fig, (ax, b) = plt.subplots(2, 1, figsize=(10, 4.6), sharex=True,
    gridspec_kw={'height_ratios': [2, 1]}, layout='constrained')
for st, en in meta['intervals']:
    if en > start and st < end:
        ax.axvspan(max(st, start), min(en, end), color='#d5d7da', alpha=.4)
        b.axvspan(max(st, start), min(en, end), color='#d5d7da', alpha=.25, zorder=-1)
ax.plot(raw.time_s, raw.RA, color='#c94847', lw=1.3, label=r'Observed $R_A$')
ax.plot(raw.time_s, raw.RE, color='#574689', ls='--', lw=1.3, label=r'Observed $R_E$')
for e in ee.itertuples():
    ax.axvline(e.reference_onset, color='#777', lw=.6, alpha=.65)
    ax.plot(e.reference_onset, 2.05, marker='v', color=BLUE, ms=5)
    ax.text(e.reference_onset, 2.22, f'{e.lead_sec:.3f} s', ha='center',
        va='center', color=BLUE, fontsize=8.5)
    ax.plot([e.first_warning, e.reference_onset], [1.95, 1.95], color=BLUE,
        lw=.8, marker='|', markersize=5, markeredgewidth=.8)
ax.set(ylim=(-.05, 2.4), ylabel='Retention coordinate')
ax.set_title(f'{tag} | 5/5 onsets covered between reference events',
    loc='left', fontsize=11, weight='bold', pad=17)
ax.legend(frameon=False, ncol=2, loc='lower right', bbox_to_anchor=(1, 1.025), fontsize=9)
for a in aa.itertuples():
    color = BLUE if a.status == 'successful_between_events' else ORANGE
    b.broken_barh([(a.issued_at, a.observed_end-a.issued_at)], (.16, .38),
        facecolors=color, alpha=.5)
    b.plot(a.issued_at, .64, marker='^', color=color, ms=5)
    if a.right_censored:
        b.plot(a.observed_end, .35, marker='>', color=color, ms=5)
for e in ee.itertuples():
    b.axvline(e.reference_onset, color='#555', lw=.8)
b.set(ylim=(0, .95), yticks=[], xlabel='Time from recording start (s)',
    ylabel='Warning', xlim=(start, end))
b.legend(handles=[Patch(color=BLUE, alpha=.5, label='Activation between events'),
    Patch(color=ORANGE, alpha=.5, label='Activation during event')],
    loc='lower left', bbox_to_anchor=(0, 1), ncol=2, frameon=False, fontsize=8.5)
unknown = trace[trace.status.eq('not_evaluable')]
if len(unknown):
    for axis in (ax, b):
        axis.axvspan(unknown.anchor_time.min(), end, color='#ddd', alpha=.4, hatch='///')
save(fig, 'inter_event_example')

# Compact vector workflow in the appendix; no artificial signal curves.
fig, ax = plt.subplots(figsize=(10, 4.9))
ax.set(xlim=(0, 10), ylim=(0, 5))
ax.axis('off')
def box(x, y, w, h, title, body, fill):
    ax.add_patch(FancyBboxPatch((x,y), w,h, boxstyle='round,pad=0.08,rounding_size=0.09',
        facecolor=fill, edgecolor='#8294a5', linewidth=.8))
    ax.text(x+w/2, y+h-.27, title, ha='center', va='center', weight='bold', fontsize=10.5)
    ax.text(x+w/2, y+h/2-.17, body, ha='center', va='center', fontsize=9.6, linespacing=1.65)
def arrow(p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=14,
        linewidth=1.1, color='#4d6d85'))
box(.2, 3.1, 2.7, 1.45, '1  Available movement',
    'Paired force signals: x, y\nCausal retention: $R_A$, $R_E$\nOne-second state history', '#edf3f8')
box(3.65, 3.1, 2.7, 1.45, '2  Future-state forecast',
    'Frozen state-history operator\nPersistence-residual correction\nQueries: 0.05–1.50 s', '#e6f0f8')
box(7.1, 3.1, 2.7, 1.45, '3  Actual activation',
    '3 future points + current support\n3 rolling anchors → activation\n0.10-s absence → clearance', '#e9f2eb')
arrow((3.0,3.82),(3.5,3.82)); arrow((6.45,3.82),(6.95,3.82))
box(.2, .7, 4.2, 1.62, '4  Offline inter-event verification',
    '$e_{i-1}<a_j<g_i$\nPrevious reference end < activation < next onset\nNo maximum lead; no overlap requirement', '#fbf1e5')
box(5.2, .7, 4.6, 1.62, '5  Primary evaluation',
    'Event advance coverage\nInter-event activation association fraction\nFirst qualifying activation → lead per event', '#edf3f8')
arrow((8.45,3.0),(8.45,2.67)); arrow((8.45,2.67),(2.3,2.67))
arrow((2.3,2.67),(2.3,2.42)); arrow((4.52,1.5),(5.05,1.5))
ax.text(5, .16, 'Reference intervals enter offline verification only; warning duration does not predict event duration.',
    ha='center', fontsize=9, color='#4d5965')
fig.subplots_adjust(left=0,right=1,top=1,bottom=0)
save(fig, 'inter_event_workflow')

# Confirm that the retained main framework example meets the revised eligibility.
fw = alarms[alarms.tag.eq('Pt 10_FD10_1') & alarms.method.eq('State-history') &
            np.isclose(alarms.issued_at,67.5)]
assert len(fw)==1 and fw.iloc[0].status == 'successful_between_events'
assert np.isclose(fw.iloc[0].previous_reference_end,65.06)
manifest = {
    'framework': {'unchanged_graphic':True, 'actual_activation':67.5,
        'previous_reference_end':65.06, 'next_reference_onset':float(fw.iloc[0].next_onset)},
    'recording_example': {'tag':tag,'selection':'Retained from the preceding illustration; not reselected for the new evaluation.',
        'covered':5,'events':5,'successful_activations':6,'activations':9,
        'recorded_curves_unchanged':True},
    'model_training':False,
    'input_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [raw_path,trace_path,DATA/'summary.csv',DATA/'events.csv',DATA/'alarms.csv',
                  DATA/'participant_cluster_intervals.csv']}}
(DATA / 'figure_provenance.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print('Rendered comparison, retained recorded example and appendix workflow in PDF/SVG/PNG.')
