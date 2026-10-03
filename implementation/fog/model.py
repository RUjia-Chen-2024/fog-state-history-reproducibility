import numpy as np
import torch
from torch import nn
from .constants import *

class FeatureGate(nn.Module):
    def __init__(self, init_weights):
        super().__init__()
        w = torch.as_tensor(
            init_weights,
            dtype=torch.float32,
        )
        if torch.any(w <= 0):
            raise ValueError(
                "FeatureGate weights must be > 0"
            )
        w = w / w.sum()
        self.logits = nn.Parameter(torch.log(w))

    def weights(self):
        return torch.softmax(self.logits, dim=0)

    def forward(self, x):
        w = self.weights()
        return (
            x
            * w.view(1, 1, -1)
            * x.shape[-1]
        )

class StateOnlyTrajectory(nn.Module):
    """State-history operator with a horizon-scaled standardized residual.
    Inputs are state history and horizon. The optional phys_hist compatibility
    argument is ignored. Raw forecasts add sigma * residual to the current state.
    """

    def __init__(
        self,
        state_dim=4,
        phys_dim=0,
        history_window=HISTORY_WINDOW,
        max_time=1.5,
        p_dim=MODEL_P_DIM,
        branch_hidden=MODEL_BRANCH_HIDDEN,
        init_state_weights=None,
    ):
        super().__init__()

        self.max_time = float(max_time)
        self.state_dim = int(state_dim)
        self.phys_dim = int(phys_dim)
        self.history_window = int(history_window)

        if init_state_weights is None:
            init_state_weights = [1.0 / state_dim] * state_dim

        self.feature_gate = FeatureGate(init_state_weights)

        decay_rates = torch.exp(
            torch.linspace(-3, 0, history_window)
        )
        self.register_buffer(
            "state_attention",
            decay_rates.view(1, -1, 1),
        )

        self.state_branch = nn.Sequential(
            nn.Linear(
                state_dim * history_window,
                branch_hidden,
            ),
            nn.LayerNorm(branch_hidden),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(branch_hidden, p_dim),
            nn.LayerNorm(p_dim),
            nn.SiLU(),
        )

        self.trunk = nn.Sequential(
            nn.Linear(3, 64),
            nn.SiLU(),
            nn.Linear(64, p_dim),
            nn.LayerNorm(p_dim),
        )

        self.residual_hidden = nn.Sequential(
            nn.Linear(p_dim, 64),
            nn.SiLU(),
        )
        self.residual_out = nn.Linear(64, 2)

        # Initialize exactly at persistence.
        nn.init.zeros_(self.residual_out.weight)
        nn.init.zeros_(self.residual_out.bias)

    def get_state_feature_weights(self):
        return (
            self.feature_gate
            .weights()
            .detach()
            .cpu()
            .numpy()
        )

    def encode_shared(
        self,
        state_hist,
        phys_hist=None,
    ):
        gated_state = self.feature_gate(state_hist)
        attended_state = (
            gated_state * self.state_attention
        )

        v_state = self.state_branch(
            attended_state.reshape(
                attended_state.size(0), -1
            )
        )

        # Only the state-history branch is used.
        return v_state

    def trajectory_from_shared(
        self,
        shared,
        t_in,
    ):
        t_norm = (
            t_in
            / max(self.max_time, 1e-8)
        )
        t_poly = torch.cat(
            [
                t_norm,
                t_norm**2,
                t_norm**3,
            ],
            dim=-1,
        )

        t_feat = self.trunk(t_poly)
        fused = shared * t_feat

        residual = self.residual_out(
            self.residual_hidden(fused)
        )

        return residual * t_norm

    def forward(
        self,
        state_hist,
        phys_hist,
        t_in,
    ):
        shared = self.encode_shared(
            state_hist,
            phys_hist,
        )
        return self.trajectory_from_shared(
            shared,
            t_in,
        )

class TransitionWeightedResidualLoss(nn.Module):
    def __init__(
        self,
        transition_alpha=TRANSITION_WEIGHT_ALPHA,
        transition_cap=TRANSITION_WEIGHT_CAP,
        stable_lambda=LAMBDA_STABLE_RESIDUAL,
        direction_lambda=LAMBDA_TRANSITION_DIRECTION,
        transition_scale=TRANSITION_LOG_DROP_MIN,
    ):
        super().__init__()

        self.transition_alpha = float(
            transition_alpha
        )
        self.transition_cap = float(
            transition_cap
        )
        self.stable_lambda = float(
            stable_lambda
        )
        self.direction_lambda = float(
            direction_lambda
        )
        self.transition_scale = max(
            float(transition_scale),
            1e-8,
        )

    @staticmethod
    def _smooth_l1_elementwise(diff):
        ad = torch.abs(diff)
        return torch.where(
            ad < 1.0,
            0.5 * diff**2,
            ad - 0.5,
        )

    def forward(
        self,
        pred_residual,
        true_residual,
        transition_strength,
        transition_mask,
        stable_mask,
    ):
        diff = (
            pred_residual
            - true_residual
        )

        base_per_dim = (
            self._smooth_l1_elementwise(diff)
        )
        base_per_sample = (
            base_per_dim
            .mean(dim=1, keepdim=True)
        )

        normalized_strength = torch.clamp(
            transition_strength
            / self.transition_scale,
            min=0.0,
            max=self.transition_cap,
        )

        sample_weight = (
            1.0
            + self.transition_alpha
            * normalized_strength
        )

        weighted_forecast = (
            sample_weight
            * base_per_sample
        ).mean()

        pred_resid_energy = (
            pred_residual**2
        ).mean(
            dim=1,
            keepdim=True,
        )
        stable_denom = torch.clamp(
            stable_mask.sum(),
            min=1.0,
        )
        stable_loss = (
            stable_mask
            * pred_resid_energy
        ).sum() / stable_denom

        true_drop_mask = (
            true_residual < 0
        ).float()

        missed_drop = torch.relu(
            pred_residual
            - true_residual
        )

        direction_per_sample = (
            true_drop_mask
            * missed_drop
        ).mean(
            dim=1,
            keepdim=True,
        )

        trans_denom = torch.clamp(
            transition_mask.sum(),
            min=1.0,
        )
        direction_loss = (
            transition_mask
            * direction_per_sample
        ).sum() / trans_denom

        total = (
            weighted_forecast
            + self.stable_lambda
            * stable_loss
            + self.direction_lambda
            * direction_loss
        )

        return total
