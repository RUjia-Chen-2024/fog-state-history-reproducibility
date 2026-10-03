"""Focused tests for lifecycle, leakage boundary, loss, and model scaling."""
import numpy as np
import pandas as pd
import torch
from .model import StateOnlyTrajectory,TransitionWeightedResidualLoss
from .lifecycle import activation_intervals,test_state_machine
from .evaluation import evaluate_record
from .rules import rolling_ews

def run_tests():
    result=test_state_machine()
    m=StateOnlyTrajectory(phys_dim=0)
    assert sum(p.numel() for p in m.parameters())==68282
    # Random head checks zero horizon even after learning; not just initialization.
    torch.nn.init.normal_(m.residual_out.weight);torch.nn.init.normal_(m.residual_out.bias)
    m.eval()
    with torch.no_grad():
        h=torch.randn(4,100,4);s=m.encode_shared(h)
        assert torch.equal(m.trajectory_from_shared(s,torch.zeros(4,1)),torch.zeros(4,2))
    pred=torch.tensor([[.2,-.1],[-.2,.4]],requires_grad=True)
    target=torch.tensor([[-.5,.1],[.1,-.3]])
    strength=torch.tensor([[.3],[0.]])
    tm=torch.tensor([[1.],[0.]]);sm=1-tm
    got=TransitionWeightedResidualLoss()(pred,target,strength,tm,sm)
    # Forecast: (.1325*5 + .145)/2; stable .25*.1; direction .5*.35.
    expected=.40375+.025+.175
    assert abs(got.item()-expected)<1e-6,(got.item(),expected)
    got.backward();assert torch.isfinite(pred.grad).all()
    # A warning ending before onset still succeeds; boundaries/inside do not.
    ep=pd.DataFrame(dict(alarm_id=range(1,7),issued_at=[2.,3.,4.,5.,8.,10.],
        observed_end=[2.1,3.1,4.1,5.1,8.1,10.1],right_censored=[0]*6,observed_duration=[.1]*6))
    r=dict(tag='test',participant='P',t=np.arange(0,11,.05),test_start_time=2.,
        gt_all=[(1.,2.),(4.,5.),(8.,9.)],gt_test=[(4.,5.),(8.,9.)])
    tr=pd.DataFrame(dict(anchor_time=np.arange(2,11,.05),evaluable=True))
    row,a,e=evaluate_record(r,'test',ep,tr,.175)
    assert row['successful_warnings']==1 and row['covered']==1
    assert row['during_event_or_boundary']==4 and row['unresolved_tail']==1
    assert e[0]['first_warning']==3. and e[1]['covered']==0
    # EWS is prefix causal once the fit-prefix scale is frozen.
    x=np.random.default_rng(2026).normal(size=350)
    assert np.allclose(rolling_ews(x,.8)[:250],rolling_ews(x[:250],.8),equal_nan=True)
    result.update(parameter_count=68282,zero_horizon_after_nonzero_head=True,
        exact_loss_and_gradient=True,strict_boundaries_and_nonpersistent_warning=True,ews_prefix_causal=True)
    return result
