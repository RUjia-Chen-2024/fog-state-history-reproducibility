"""CPU batched checkpoint inference, independent of saved predictions."""
import numpy as np
import pandas as pd
import torch
def infer(r,model,start,stop):
    feat=torch.as_tensor(r['features_norm'][:,:4],dtype=torch.float32);off=torch.arange(-100,0);hs=np.arange(5,151,5);frames=[]
    model.eval()
    with torch.no_grad():
        for arr in np.array_split(np.arange(start,stop-5,5),max(1,int(np.ceil((stop-start)/5/64)))):
            if not len(arr):continue
            shared=model.encode_shared(feat[torch.tensor(arr)[:,None]+off])
            horizon=torch.tensor(hs*r['dt'],dtype=torch.float32).reshape(1,30,1).expand(len(arr),-1,-1)
            corr=model.trajectory_from_shared(shared[:,None,:].expand(-1,30,-1).reshape(-1,shared.shape[-1]),horizon.reshape(-1,1)).reshape(len(arr),30,2).numpy()
            pred=r['target_scaler'].inverse_transform((r['targets_norm'][arr,None,:]+corr).reshape(-1,2)).reshape(len(arr),30,2).clip(0,2)
            for k,i in enumerate(arr):
                valid=i+hs<stop;j=i+hs[valid]
                frames.append(pd.DataFrame(dict(anchor_index=i,anchor_time=r['t'][i],target_index=j,target_time=r['t'][j],horizon_sec=r['t'][j]-r['t'][i],SE2_pred=pred[k,valid,0],ME2_pred=pred[k,valid,1])))
    return pd.concat(frames,ignore_index=True)
