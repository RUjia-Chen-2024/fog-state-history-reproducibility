"""Reassociate saved actual activations with strictly between-event reference gaps.

No forecasting, training, threshold selection, or activation replay is performed.
"""
from pathlib import Path
import hashlib
import json
import shutil

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'analysis/inter_event_revision'
SRC = OUT / 'source'
SRC.mkdir(parents=True, exist_ok=True)
EPS = 1e-9
MAIN = ['Persistence', 'AC1', 'Variance', 'Recovery', 'Reddening', 'State-history']
if not (SRC / 'reference_intervals.json').exists():
    raise FileNotFoundError('Required bundled input analysis/inter_event_revision/source/reference_intervals.json is missing')
intervals = json.loads((SRC / 'reference_intervals.json').read_text(encoding='utf-8'))
metadata = {m['tag']: m for m in json.loads((SRC / 'recording_metadata.json').read_text(encoding='utf-8'))}
old = pd.read_csv(ROOT / 'analysis/classical_ews_revision/source/results/selected_test_results.csv').set_index('method')
old_records = pd.read_csv(ROOT / 'analysis/classical_sensitivity/selected_record_rows.csv')
old_events = pd.read_csv(ROOT / 'analysis/classical_sensitivity/selected_event_rows.csv')
tags = old_records.tag.drop_duplicates().tolist()
methods = MAIN + [m for m in old.index if m not in MAIN]
assert len(tags) == 21 and len(methods) == 18
alarm_rows, event_rows, record_rows = [], [], []
for tag in tags:
    iv = np.array(intervals[tag], dtype=float)
    assert np.allclose(iv, metadata[tag]['intervals'])
    assert np.all(np.diff(iv[:, 0]) > 0) and np.all(iv[:-1, 1] < iv[1:, 0])
    ref = old_events[(old_events.tag == tag) & (old_events.method == 'State-history')].reference_onset.to_numpy()
    start = float(metadata[tag]['evaluation_start'])
    participant = old_records[old_records.tag.eq(tag)].participant.iloc[0]
    for method in methods:
        episodes = pd.read_csv(SRC / 'selected_alarms' / tag.replace(' ', '_') / f'{method}.csv')
        prior = old_records[(old_records.tag == tag) & (old_records.method == method)].iloc[0]
        assert len(episodes) == prior.alarms
        assert np.isclose((episodes.observed_end - episodes.issued_at).sum(), prior.active_seconds)
        assert (episodes.issued_at >= start - EPS).all()
        ar = []
        for aid, a in enumerate(episodes.itertuples(), 1):
            inside = bool(np.any((iv[:, 0] - EPS <= a.issued_at) & (a.issued_at <= iv[:, 1] + EPS)))
            nxt = np.flatnonzero(iv[:, 0] > a.issued_at + EPS)
            if inside:
                status = 'during_reference_event_or_boundary'
            elif len(nxt):
                status = 'successful_between_events'
            else:
                status = 'no_later_onset_observed'
            k = int(nxt[0]) if len(nxt) else len(iv)
            g = float(iv[k, 0]) if len(nxt) else np.nan
            prev = float(iv[k-1, 1]) if k else np.nan
            if status == 'successful_between_events':
                assert np.isnan(prev) or a.issued_at > prev + EPS
                assert np.any(np.isclose(ref, g, rtol=0, atol=1e-8))
            ar.append(dict(tag=tag, participant=participant, method=method, alarm_id=aid,
                issued_at=a.issued_at, observed_end=a.observed_end, right_censored=int(a.right_censored),
                next_onset=g, previous_reference_end=prev, status=status,
                lead_sec=g-a.issued_at if status == 'successful_between_events' else np.nan))
        ar = pd.DataFrame(ar, columns=['tag','participant','method','alarm_id','issued_at','observed_end',
            'right_censored','next_onset','previous_reference_end','status','lead_sec'])
        er = []
        for g in ref:
            k = np.flatnonzero(np.isclose(iv[:, 0], g, rtol=0, atol=1e-8))
            assert len(k) == 1
            k = int(k[0]); prev = float(iv[k-1, 1]) if k else np.nan
            eligible = (episodes.issued_at < g-EPS)
            if np.isfinite(prev): eligible &= episodes.issued_at > prev+EPS
            eligible &= episodes.issued_at >= start-EPS
            hits = episodes[eligible]
            first = float(hits.issued_at.min()) if len(hits) else np.nan
            er.append(dict(tag=tag, participant=participant, method=method, reference_onset=float(g),
                previous_reference_end=prev, scored_test_start=start,
                left_truncated=int(not np.isfinite(prev) or prev < start),
                covered=int(len(hits)>0), n_successful_warnings=len(hits),
                first_warning=first, lead_sec=float(g-first)))
        er = pd.DataFrame(er)
        success = int(ar.status.eq('successful_between_events').sum())
        assert success == er.n_successful_warnings.sum()
        assert np.isclose(ar[ar.status.eq('successful_between_events')].groupby('next_onset').issued_at.min().sort_index(),
                          er[er.covered.eq(1)].sort_values('reference_onset').first_warning).all()
        record_rows.append(dict(tag=tag, participant=participant, method=method, q=float(prior.q),
            n_reference=len(ref), covered=int(er.covered.sum()), missed=int((1-er.covered).sum()),
            alarms=len(ar), successful_warnings=success, repeated_successes=success-int(er.covered.sum()),
            during_event_or_boundary=int(ar.status.eq('during_reference_event_or_boundary').sum()),
            unresolved_tail=int(ar.status.eq('no_later_onset_observed').sum()),
            observed_seconds=float(prior.observed_seconds), active_seconds=float(prior.active_seconds)))
        alarm_rows.extend(ar.to_dict('records')); event_rows.extend(er.to_dict('records'))
alarms = pd.DataFrame(alarm_rows); events = pd.DataFrame(event_rows); records = pd.DataFrame(record_rows)
cols = ['n_reference','covered','missed','alarms','successful_warnings','repeated_successes',
    'during_event_or_boundary','unresolved_tail','observed_seconds','active_seconds']
summary = records.groupby('method', sort=False)[cols].sum()
summary['coverage'] = summary.covered/summary.n_reference
summary['success_fraction'] = summary.successful_warnings/summary.alarms
summary['activations_per_minute'] = summary.alarms*60/summary.observed_seconds
summary['active_fraction'] = summary.active_seconds/summary.observed_seconds
summary['q'] = old.q
for m in methods:
    lead = events.loc[events.method.eq(m)&events.covered.eq(1), 'lead_sec']
    for name, value in dict(median_lead=lead.median(),mean_lead=lead.mean(),lead_q25=lead.quantile(.25),
        lead_q75=lead.quantile(.75),lead_max=lead.max(),lead_min=lead.min()).items():
        summary.loc[m,name] = value
    assert summary.loc[m,'alarms'] == old.loc[m,'alarms']
    assert np.isclose(summary.loc[m,'active_fraction'], old.loc[m,'active_fraction'])
assert (summary.n_reference == 88).all()
assert (summary.successful_warnings + summary.during_event_or_boundary + summary.unresolved_tail == summary.alarms).all()
m = summary.loc['State-history']
for col, val in dict(covered=71, missed=17, alarms=194, successful_warnings=117,
                    repeated_successes=46, during_event_or_boundary=76, unresolved_tail=1).items():
    assert m[col] == val, (col,m[col])
supplied = json.loads((SRC / 'supplied_model_summary.json').read_text())
for key, skey in [('median_lead','median_lead'),('mean_lead','mean_lead'),('lead_q25','q25_lead'),('lead_q75','q75_lead')]:
    assert np.isclose(m[key], supplied[skey])

# Keep the participant-cluster uncertainty calculation aligned with the new outcomes.
participants = sorted(records.participant.unique()); rng = np.random.default_rng(20261002)
draws = rng.integers(0,len(participants),size=(2000,len(participants)))
weights = np.stack([(draws == i).sum(axis=1) for i in range(len(participants))], axis=1)
replicates = {}; ci = []
for method in MAIN:
    grp = records[records.method.eq(method)].groupby('participant')[cols].sum().reindex(participants)
    sums = weights @ grp.to_numpy()
    idx = {v:i for i,v in enumerate(cols)}
    for metric,num,den,factor in [('coverage','covered','n_reference',1),
        ('success_fraction','successful_warnings','alarms',1),
        ('activations_per_minute','alarms','observed_seconds',60),
        ('active_fraction','active_seconds','observed_seconds',1)]:
        v = factor*sums[:,idx[num]]/sums[:,idx[den]]
        replicates[(method,metric)] = v
        lo,hi = np.quantile(v,[.025,.975])
        ci.append(dict(method=method,metric=metric,estimate=summary.loc[method,metric],lower95=lo,upper95=hi))
differences=[]
for comparator in MAIN[:-1]:
    for metric in ['coverage','success_fraction','activations_per_minute','active_fraction']:
        v = replicates[('State-history',metric)]-replicates[(comparator,metric)]
        lo,hi = np.quantile(v,[.025,.975])
        differences.append(dict(comparator=comparator,metric=metric,
            difference=summary.loc['State-history',metric]-summary.loc[comparator,metric],lower95=lo,upper95=hi))
paired = events[events.method.isin(['State-history','Persistence'])].pivot(
    index=['tag','participant','reference_onset'],columns='method',values='lead_sec').dropna()
paired['difference'] = paired['State-history']-paired['Persistence']
pstats = dict(n=len(paired),median_difference=float(paired.difference.median()),
    mean_difference=float(paired.difference.mean()),operator_earlier=int((paired.difference>EPS).sum()),
    persistence_earlier=int((paired.difference< -EPS).sum()),ties=int((abs(paired.difference)<=EPS).sum()))
for name, df in [('alarms',alarms),('events',events),('per_record',records),
    ('summary',summary.reset_index()),('participant_cluster_intervals',pd.DataFrame(ci)),
    ('participant_cluster_differences',pd.DataFrame(differences)),('paired_leads',paired.reset_index())]:
    df.to_csv(OUT/f'{name}.csv', index=False)
(OUT/'paired_lead_summary.json').write_text(json.dumps(pstats,indent=2),encoding='utf-8')
hashes = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
          for p in sorted(SRC.rglob('*')) if p.is_file()}
protocol = dict(definition='Strict previous reference end < actual activation < next reference onset. No maximum lead and no persistence-to-onset requirement.',
    boundary_tolerance_sec=EPS, cohort_records=21,participants=7,reference_onsets=88,
    activation_streams_unchanged=True, models_retrained=False, thresholds_retuned=False,
    original_selection='EWS quantiles and exit setting were originally chosen on validation using the earlier bounded 2T criterion; this is a post hoc reassociation of the same held-out activations.',
    first_test_event='Use preceding event end from full-record labels, including pretest; only actual scored-test activations are eligible. No invented earlier warning.',
    left_truncated_reference_gaps=int(events[events.method.eq('State-history')].left_truncated.sum()),
    eligibility_uses_reference_end_offline=True, online_reference_end_gating=False,
    denominator_for_success_fraction='All actual activations, including in-event/boundary prompts and unresolved non-event tail prompts; not complete-follow-up precision.',
    bootstrap_replicates=2000, bootstrap_seed=20261002, bootstrap_clusters=participants,
    old_bounded_results_retained_as_sensitivity=True, model_user_numbers_reproduced=True,
    source_sha256=hashes, paired_leads=pstats)
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
print(summary.loc[MAIN+['AC1_A3T'],['covered','alarms','successful_warnings','during_event_or_boundary',
    'unresolved_tail','coverage','success_fraction','median_lead','mean_lead','lead_q25','lead_q75','lead_max']].to_string())
print('PAIRED',json.dumps(pstats))
print(pd.DataFrame(differences).query("comparator in ['Persistence','Recovery'] and metric in ['coverage','success_fraction']").to_string(index=False))
