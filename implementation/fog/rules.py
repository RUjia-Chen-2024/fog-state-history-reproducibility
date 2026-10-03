import numpy as np
import pandas as pd
from scipy.signal import detrend,periodogram,find_peaks
def movement_scale(r):
 # Exactly the original detector's peak extraction, confined to fit prefix.
 t=r['t'][:r['sel_idx']];v=np.abs(r['x'][:r['sel_idx']]-r['y'][:r['sel_idx']]);fs=r['fs'];T0=1.13
 distance=max(round(.45*T0*fs),2)
 coarse,_=find_peaks(v,distance=distance,prominence=max(.15*np.std(v),1e-8))
 scale=max(float(np.mean(v[coarse])),1e-8) if len(coarse) else 1.
 peaks,_=find_peaks(v/scale,distance=distance,height=.4,prominence=.08)
 gaps=np.diff(t[peaks]);keep=(gaps<=1.5*T0+1e-9)&(gaps>0)
 normal=gaps[keep];fallback=len(normal)<5
 T=T0 if fallback else float(np.median(normal))
 return dict(T_sec=T,frequency_hz=1/T,normal_gaps=len(normal),all_gaps=len(gaps),excluded_long_gaps=int((~keep).sum()),q25=float(np.quantile(normal,.25)) if len(normal) else np.nan,q75=float(np.quantile(normal,.75)) if len(normal) else np.nan,fallback=fallback,calibration_end=float(t[-1])),pd.DataFrame(dict(peak_time=t[peaks][:-1],next_peak_time=t[peaks][1:],gap_sec=gaps,used_for_scale=keep.astype(int)))
def rolling_ews(u,T,multiple=5):
    """Trailing windows, linear detrend inside each window. Input is at20Hz."""
    N=max(12,int(round(multiple*T/.05))+1)
    out=np.full((len(u),5),np.nan)
    if len(u)<N:return out
    x=np.lib.stride_tricks.sliding_window_view(np.asarray(u,float),N)
    z=detrend(x,axis=1,type='linear')
    ss=np.sum(z*z,axis=1);lagss=np.sum(z[:,:-1]**2,axis=1)
    num=np.sum(z[:,:-1]*z[:,1:],axis=1)
    valid=ss>1e-16
    ac=np.divide(num,ss,out=np.full(len(ss),np.nan),where=valid)
    var=np.var(z,axis=1,ddof=1)
    phi=np.divide(num,lagss,out=np.full(len(ss),np.nan),where=lagss>1e-16)
    stable=(phi>0)&(phi<1)&valid
    kap=np.full(len(ss),np.nan);kap[stable]=-np.log(phi[stable])/.05
    freq,power=periodogram(z,fs=20,window='hann',detrend=False,axis=1)
    low=(freq>0)&(freq<=.5/T);high=(freq>.5/T)&(freq<=min(2/T,10))
    lo=power[:,low].sum(axis=1);hi=power[:,high].sum(axis=1)
    sr=np.divide(lo,hi,out=np.full(len(ss),np.nan),where=(hi>1e-16)&valid)
    out[N-1:]=np.c_[ac,var,kap,sr,phi]
    return out

def trajectory_base(ns,r,df,persist):
    ii=np.sort(df.anchor_index.unique().astype(int));n=len(ii)
    ridx=np.searchsorted(ii,df.anchor_index);cidx=((df.target_index.to_numpy()-df.anchor_index.to_numpy())//5-1).astype(int)
    s=np.full((n,30),np.nan);e=s.copy();dt=s.copy()
    s[ridx,cidx]=df.SE2_pred;e[ridx,cidx]=df.ME2_pred;dt[ridx,cidx]=df.horizon_sec
    valid=np.isfinite(dt)
    if persist:s=np.where(valid,r['se2'][ii,None],np.nan);e=np.where(valid,r['me2'][ii,None],np.nan)
    ps=np.c_[r['se2'][ii],s[:,:-1]];pe=np.c_[r['me2'][ii],e[:,:-1]]
    step=np.diff(np.c_[np.zeros(n),dt],axis=1)
    with np.errstate(invalid='ignore',divide='ignore'):
        ss=(np.log(np.maximum(s,ns['EPS']))-np.log(np.maximum(ps,ns['EPS'])))/np.maximum(step,1e-8)
        es=(np.log(np.maximum(e,ns['EPS']))-np.log(np.maximum(pe,ns['EPS'])))/np.maximum(step,1e-8)
    return ii,s,e,(ss<r['se_slope_th'])&valid,(es<r['me_slope_th'])&valid,valid

def trajectory_candidates(ns,r,b,q):
    ii,s,e,st,et,valid=b
    sth=np.quantile(r['se2'][:r['split_idx']],q);eth=np.quantile(r['me2'][:r['split_idx']],q)
    ls=(s<sth)&valid;le=(e<eth)&valid;soft=(ls&(le|et))|(le&(ls|st))
    f3=(soft[:,:-2]&soft[:,1:-1]&soft[:,2:]).any(axis=1)
    support=(r['se2'][ii]<=ns['CURRENT_LOW_FACTOR']*sth)|(r['me2'][ii]<=ns['CURRENT_LOW_FACTOR']*eth)|(r['dlog_se2'][ii]<r['se_slope_th'])|(r['dlog_me2'][ii]<r['me_slope_th'])
    return f3&support
