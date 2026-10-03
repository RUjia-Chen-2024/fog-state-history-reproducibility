import random
import numpy as np
import torch
from .constants import *
from .model import StateOnlyTrajectory,TransitionWeightedResidualLoss

class TrajectoryDataset(torch.utils.data.Dataset):
    """
    Each anchor i uses only history [i-H, i).

    Supervised target:
        normalized future coordinate residual
        z_norm(i+h) - z_norm(i)

    Reference-event labels are NOT used to train this dataset.
    """

    def __init__(
        self,
        feats,
        target_coords_norm,
        target_coords_raw,
        t_seq,
        history_window=HISTORY_WINDOW,
        max_dt=MAX_DT,
        step=DATASET_STEP,
        state_dim=4,
        anchor_start=None,
        anchor_stop=None,
    ):
        feats = np.asarray(feats, float)
        target_coords_norm = np.asarray(target_coords_norm, float)
        target_coords_raw = np.asarray(target_coords_raw, float)
        t_seq = np.asarray(t_seq, float)

        if target_coords_norm.ndim != 2 or target_coords_norm.shape[1] != 2:
            raise ValueError("target_coords_norm must have shape [N,2]")
        if target_coords_raw.ndim != 2 or target_coords_raw.shape[1] != 2:
            raise ValueError("target_coords_raw must have shape [N,2]")
        if not (
            len(feats) == len(target_coords_norm)
            == len(target_coords_raw) == len(t_seq)
        ):
            raise ValueError("feats / targets / time length mismatch")

        self.state_hists = []
        self.phys_hists = []
        self.trunks = []
        self.targets = []
        self.transition_strengths = []
        self.transition_masks = []
        self.stable_masks = []

        n = len(feats)
        default_start = history_window
        default_stop = n - max_dt - 1

        if anchor_start is None:
            anchor_start = default_start
        if anchor_stop is None:
            anchor_stop = default_stop

        anchor_start = max(int(anchor_start), history_window)
        anchor_stop = min(int(anchor_stop), default_stop)

        if anchor_stop <= anchor_start:
            raise ValueError(
                f"empty dataset anchor range: {anchor_start=} "
                f"{anchor_stop=} N={n}"
            )

        for i in range(anchor_start, anchor_stop, step):
            state_hist = feats[i-history_window:i, :state_dim]
            phys_hist = feats[i-history_window:i, state_dim:]

            s0 = max(float(target_coords_raw[i, 0]), LOSS_LOG_FLOOR)
            m0 = max(float(target_coords_raw[i, 1]), LOSS_LOG_FLOOR)

            for dt_step in range(5, max_dt + 1, 5):
                j = i + dt_step
                if j >= n:
                    continue

                dt = float(t_seq[j] - t_seq[i])
                delta_norm = (
                    target_coords_norm[j]
                    - target_coords_norm[i]
                )

                s1 = max(
                    float(target_coords_raw[j, 0]),
                    LOSS_LOG_FLOOR,
                )
                m1 = max(
                    float(target_coords_raw[j, 1]),
                    LOSS_LOG_FLOOR,
                )

                log_change_s = float(np.log(s1) - np.log(s0))
                log_change_m = float(np.log(m1) - np.log(m0))

                drop_s = max(-log_change_s, 0.0)
                drop_m = max(-log_change_m, 0.0)

                transition_strength = max(drop_s, drop_m)
                transition_mask = float(
                    transition_strength >= TRANSITION_LOG_DROP_MIN
                )

                stable_log_change = max(
                    abs(log_change_s),
                    abs(log_change_m),
                )
                stable_mask = float(
                    stable_log_change <= STABLE_LOG_CHANGE_MAX
                )

                self.state_hists.append(state_hist)
                self.phys_hists.append(phys_hist)
                self.trunks.append([dt])
                self.targets.append(delta_norm)
                self.transition_strengths.append(
                    [transition_strength]
                )
                self.transition_masks.append([transition_mask])
                self.stable_masks.append([stable_mask])

        if len(self.targets) == 0:
            raise ValueError("constructed trajectory dataset is empty")

        self.state_hists = torch.tensor(
            np.stack(self.state_hists),
            dtype=torch.float32,
        )
        self.phys_hists = torch.tensor(
            np.stack(self.phys_hists),
            dtype=torch.float32,
        )
        self.trunks = torch.tensor(
            np.stack(self.trunks),
            dtype=torch.float32,
        )
        self.targets = torch.tensor(
            np.stack(self.targets),
            dtype=torch.float32,
        )
        self.transition_strengths = torch.tensor(
            np.stack(self.transition_strengths),
            dtype=torch.float32,
        )
        self.transition_masks = torch.tensor(
            np.stack(self.transition_masks),
            dtype=torch.float32,
        )
        self.stable_masks = torch.tensor(
            np.stack(self.stable_masks),
            dtype=torch.float32,
        )

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (
            self.state_hists[idx],
            self.phys_hists[idx],
            self.trunks[idx],
            self.targets[idx],
            self.transition_strengths[idx],
            self.transition_masks[idx],
            self.stable_masks[idx],
        )

def set_seed(seed=2026):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_state_only_model(train_ds, val_ds, max_time, seed, device="cpu", epochs=EPOCHS):
    set_seed(seed)

    generator = torch.Generator()
    generator.manual_seed(int(seed))

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        generator=generator,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )

    if len(train_loader) == 0 or len(val_loader) == 0:
        raise ValueError("empty train/validation loader")

    model = StateOnlyTrajectory(
        state_dim=4,
        phys_dim=0,
        history_window=HISTORY_WINDOW,
        max_time=max_time,
        init_state_weights=STATE_INIT_WEIGHTS,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
    criterion = TransitionWeightedResidualLoss()

    best_val = float("inf")
    best_epoch = 1
    best_state = None
    no_improve = 0
    train_losses = []
    val_losses = []

    for ep in range(epochs):
        model.train()
        total = 0.0

        for batch in train_loader:
            state_hist, phys_hist, h, y, trans_strength, trans_mask, stable_mask = batch

            state_hist = state_hist.to(device)
            phys_hist = phys_hist.to(device)
            h = h.to(device)
            y = y.to(device)
            trans_strength = trans_strength.to(device)
            trans_mask = trans_mask.to(device)
            stable_mask = stable_mask.to(device)

            opt.zero_grad()
            pred = model(state_hist, phys_hist, h)
            loss = criterion(
                pred,
                y,
                trans_strength,
                trans_mask,
                stable_mask,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item())

        train_loss = total / len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        total = 0.0
        with torch.no_grad():
            for batch in val_loader:
                state_hist, phys_hist, h, y, trans_strength, trans_mask, stable_mask = batch
                pred = model(
                    state_hist.to(device),
                    phys_hist.to(device),
                    h.to(device),
                )
                loss = criterion(
                    pred,
                    y.to(device),
                    trans_strength.to(device),
                    trans_mask.to(device),
                    stable_mask.to(device),
                )
                total += float(loss.item())

        val_loss = total / len(val_loader)
        val_losses.append(val_loss)
        scheduler.step()

        if val_loss < best_val - 1e-12:
            best_val = val_loss
            best_epoch = ep + 1
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            no_improve = 0
        else:
            no_improve += 1

        if ep == 0 or (ep + 1) % 10 == 0:
            print(
                f"    seed={seed} epoch={ep+1:3d} "
                f"train={train_loss:.6f} val={val_loss:.6f} "
                f"best={best_epoch:3d}"
            )

        if no_improve >= VAL_PATIENCE:
            break

    if best_state is None:
        raise RuntimeError("best checkpoint not captured")

    model.load_state_dict(best_state)
    model = model.to(device)
    model.eval()

    return model, best_epoch, best_val, train_losses, val_losses

def build_state_datasets(record):
    return (
        TrajectoryDataset(
            record["features_norm"][:record["sel_idx"]],
            record["targets_norm"][:record["sel_idx"]],
            record["built"]["targets_raw"][:record["sel_idx"]],
            record["t"][:record["sel_idx"]],
            anchor_start=HISTORY_WINDOW,
            anchor_stop=record["sel_idx"] - MAX_DT - 1,
            state_dim=4,
        ),
        TrajectoryDataset(
            record["features_norm"][:record["split_idx"]],
            record["targets_norm"][:record["split_idx"]],
            record["built"]["targets_raw"][:record["split_idx"]],
            record["t"][:record["split_idx"]],
            anchor_start=record["sel_idx"],
            anchor_stop=record["split_idx"] - MAX_DT - 1,
            state_dim=4,
        ),
    )
