"""Online warning lifecycle. No future labels or retrospective event merging."""
import numpy as np
import pandas as pd

def activation_intervals(trace, exit_duration=.1, enter_confirmations=3, record_end=None):
    """Input: anchor_time, candidate, evaluable. Stops are never backdated.
    Unknown input terminates the observable interval by censoring, not clearance.
    """
    active=False;run=0;exit_start=None;current=None;intervals=[];states=[];last=None
    for row in trace.sort_values('anchor_time').itertuples():
        now=float(row.anchor_time)
        if not bool(row.evaluable):
            if active:
                current.update(cleared_at=np.nan,observed_end=now,clear_reason='insufficient_future_points',right_censored=1)
                intervals.append(current);current=None
            active=False;run=0;exit_start=None
            states.append(dict(anchor_time=now,active=np.nan,enter_run=0,exit_elapsed=np.nan,status='not_evaluable'))
            last=now;continue
        if last is not None and now-last>.050001:
            if active:
                current.update(cleared_at=np.nan,observed_end=last+.05,clear_reason='data_gap',right_censored=1)
                intervals.append(current);current=None
            active=False;run=0;exit_start=None
        elapsed=0.
        if not active:
            run=run+1 if bool(row.candidate) else 0
            if run>=enter_confirmations:
                active=True;exit_start=None
                current=dict(alarm_id=len(intervals)+1,issued_at=now)
        else:
            if bool(row.candidate):exit_start=None
            else:
                if exit_start is None:exit_start=now
                elapsed=now-exit_start
                if elapsed+1e-9>=exit_duration:
                    current.update(cleared_at=now,observed_end=now,clear_reason='sustained_candidate_absence',right_censored=0)
                    intervals.append(current);current=None;active=False;run=0;exit_start=None
        states.append(dict(anchor_time=now,active=int(active),enter_run=run,exit_elapsed=elapsed,status='active' if active else 'inactive'))
        last=now
    if active:
        end=min(last+.05,record_end) if record_end is not None else last+.05
        current.update(cleared_at=np.nan,observed_end=end,clear_reason='record_end',right_censored=1)
        intervals.append(current)
    cols=['alarm_id','issued_at','cleared_at','observed_end','clear_reason','right_censored']
    alarms=pd.DataFrame(intervals,columns=cols)
    if len(alarms):alarms['observed_duration']=alarms.observed_end-alarms.issued_at
    else:alarms['observed_duration']=pd.Series(dtype=float)
    return alarms,pd.DataFrame(states)

def test_state_machine():
    def run(c,e=None,delay=.1):
        return activation_intervals(pd.DataFrame({'anchor_time':np.arange(len(c))*.05,'candidate':c,'evaluable':e if e is not None else [1]*len(c)}),delay,record_end=len(c)*.05)
    a,s=run([1,1,1,0,1,0,0,0]);assert len(a)==1 and abs(a.issued_at.iloc[0]-.1)<1e-9 and abs(a.cleared_at.iloc[0]-.35)<1e-9
    a,s=run([1,1,1,0,0],[1,1,1,0,0]);assert len(a)==1 and a.right_censored.iloc[0]==1 and pd.isna(a.cleared_at.iloc[0]) and s.active.iloc[-1]!=s.active.iloc[-1]
    a,s=run([1,1,1,0,1,1,1],delay=0);assert len(a)==2 and np.allclose(a.issued_at,[.1,.3])
    full=pd.DataFrame({'anchor_time':np.arange(10)*.05,'candidate':[1,1,1,0,1,0,0,0,1,1],'evaluable':[1]*10})
    _,x=activation_intervals(full,.1);_,y=activation_intervals(full.iloc[:6],.1);assert x.iloc[:6].reset_index(drop=True).equals(y.reset_index(drop=True))
    return dict(brief_dropout_held=True,clearance_not_backdated=True,unknown_not_recovery=True,three_confirmations_preserved=True,prefix_invariant=True)

if __name__=='__main__':print(test_state_machine())
