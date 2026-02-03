import os
import random
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.multitask_model import MultiTaskTransformerModel, TransformerMTLConfig
from mtl.ntkmtl import NTKMTL
from mtl.fairgrad import FairGrad

from metrics.metrics import (
    evaluate_dataset,
    aggregate_multidataset,
    process_emotions,
    mf1, uar,
    acc_func, ccc,
    mf1_ah, uar_ah
)


def _fmt(v, ndigits=4):
    if v is None:
        return "N/A"
    try:
        return f"{float(v):.{ndigits}f}"
    except Exception:
        return str(v)


def pretty_print_train(epoch, num_epochs, loss, metrics: dict):
    print()
    print(f"Train loss: {_fmt(loss)}")

    mF1 = metrics.get("mF1")
    mUAR = metrics.get("mUAR")
    mF1_resd = metrics.get("mF1_RESD")
    mUAR_resd = metrics.get("mUAR_RESD")
    acc = metrics.get("ACC")
    ccc_val = metrics.get("CCC")
    mf1_ah_val = metrics.get("MF1_AH")
    uar_ah_val = metrics.get("UAR_AH")

    print(f"    • CMU-MOSEI: mF1={_fmt(mF1)}, mUAR={_fmt(mUAR)}")
    print(f"    • RESD     : mF1={_fmt(mF1_resd)}, mUAR={_fmt(mUAR_resd)}")
    print(f"    • FIV2     : ACC={_fmt(acc)}, CCC={_fmt(ccc_val)}")
    print(f"    • BAH      : MF1_AH={_fmt(mf1_ah_val)}, UAR_AH={_fmt(uar_ah_val)}")
    print()


def pretty_print_test(metrics: dict):
    print("Test metrics:")

    mean_all = metrics.get("mean_all")

    print("  ─ Aggregated:")
    print(f"    • mean_all = {_fmt(mean_all)}")

    if "by_dataset" in metrics:
        print("  ─ Per-dataset raw:")
        for ds in metrics["by_dataset"]:
            name = ds.get("name", "unknown")
            kv = ", ".join(f"{k}={_fmt(v)}" for k, v in ds.items() if k != "name")
            print(f"    • {name}: {kv}")
    print()


class LogCoshLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        diff = pred - tgt
        return torch.log(torch.cosh(diff)).mean()



class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        label_smoothing: float = 0.0,
        regression_loss_type: str = "mse",
    ):
        super().__init__()

        rtype = regression_loss_type.lower()
        if rtype == "mse":
            self.regression_loss = nn.MSELoss(reduction="mean")
        elif rtype == "smoothl1":
            self.regression_loss = nn.SmoothL1Loss(beta=1.0, reduction="mean")
        elif rtype == "logcosh":
            self.regression_loss = LogCoshLoss()
        else:
            raise ValueError(f"Unknown regression_loss_type: {regression_loss_type}")

        self.ce_resd = nn.CrossEntropyLoss(reduction="mean", label_smoothing=label_smoothing)
        self.ce_ah   = nn.CrossEntropyLoss(reduction="mean", label_smoothing=label_smoothing)

    def forward(self, outputs, batch, return_per_task: bool = False):
        device = outputs["emotion_mosei_pred"].device

        loss_emo_m = torch.tensor(0.0, device=device)
        loss_emo_r = torch.tensor(0.0, device=device)
        loss_pers  = torch.tensor(0.0, device=device)
        loss_ah    = torch.tensor(0.0, device=device)

        # CMU-MOSEI:
        if batch["has_emo_mosei"].any():
            pred = outputs["emotion_mosei_pred"][batch["has_emo_mosei"]]
            tgt  = batch["emotion_mosei"][batch["has_emo_mosei"]]
            loss_emo_m = self.regression_loss(pred, tgt)

        # RESD:
        if batch["has_emo_resd"].any():
            pred = outputs["emotion_resd_logits"][batch["has_emo_resd"]]
            tgt  = batch["emotion_resd"][batch["has_emo_resd"]]
            loss_emo_r = self.ce_resd(pred, tgt)

        # FIV2:
        if batch["has_pers"].any():
            pred = outputs["personality_preds"][batch["has_pers"]]
            tgt  = batch["personality"][batch["has_pers"]]
            loss_pers = self.regression_loss(pred, tgt)

        # BAH:
        if batch["has_ah"].any():
            pred = outputs["ah_logits"][batch["has_ah"]]
            tgt  = batch["ah"][batch["has_ah"]]
            loss_ah = self.ce_ah(pred, tgt)

        losses_vec = torch.stack([loss_emo_m, loss_emo_r, loss_pers, loss_ah])

        if return_per_task:
            return losses_vec

        return losses_vec.sum()


class SelectiveTaskGroupUpdater:
    """
    Порядок задач:
    0: CMU-MOSEI
    1: RESD
    2: FIV2
    3: BAH
    """
    def __init__(
        self,
        n_tasks: int,
        beta: float = 1e-3,
        regroup_interval: int = 50,
        affinity_threshold: float = 0.0,
        device: torch.device = torch.device("cpu"),
    ):
        self.n_tasks = n_tasks
        self.beta = beta
        self.regroup_interval = regroup_interval
        self.affinity_threshold = affinity_threshold
        self.device = device

        self.B = torch.zeros(n_tasks, n_tasks, dtype=torch.float32, device=device)
        self.groups: List[List[int]] = [[i] for i in range(n_tasks)]
        self.step_idx = 0

    def _update_affinity(self, group: List[int], base_losses: torch.Tensor, new_losses: torch.Tensor):
        eps = 1e-8
        K = self.n_tasks
        with torch.no_grad():
            for j in range(K):
                if base_losses[j].item() <= 0.0:
                    continue
                ratio = new_losses[j] / (base_losses[j] + eps)
                b_val = 1.0 - ratio
                for i in group:
                    old = self.B[i, j]
                    self.B[i, j] = (1.0 - self.beta) * old + self.beta * b_val

    def _recompute_groups(self):
        K = self.n_tasks
        visited = [False] * K
        new_groups: List[List[int]] = []

        with torch.no_grad():
            for i in range(K):
                if visited[i]:
                    continue
                group = [i]
                visited[i] = True
                stack = [i]
                while stack:
                    u = stack.pop()
                    for v in range(K):
                        if visited[v]:
                            continue
                        if self.B[u, v] > self.affinity_threshold and self.B[v, u] > self.affinity_threshold:
                            visited[v] = True
                            stack.append(v)
                            group.append(v)
                new_groups.append(group)

        if len(new_groups) == 0:
            new_groups = [[i] for i in range(K)]

        self.groups = new_groups

    def step(self, model, optimizer, criterion, batch, device):
        self.step_idx += 1
        model.train()

        has_any = (
            batch["has_emo_mosei"].any()
            or batch["has_emo_resd"].any()
            or batch["has_pers"].any()
            or batch["has_ah"].any()
        )
        if not has_any:
            with torch.no_grad():
                out = model(batch)
                losses_vec = criterion(out, batch, return_per_task=True)
                total_loss_scalar = float(losses_vec.sum().item())
            return total_loss_scalar, out

        group_order = list(range(len(self.groups)))
        random.shuffle(group_order)

        last_out = None
        last_total_loss = 0.0

        for g_idx in group_order:
            group = self.groups[g_idx]

            optimizer.zero_grad()
            out = model(batch)
            losses_vec = criterion(out, batch, return_per_task=True)  # (K,)
            base_losses = losses_vec.detach().clone()

            group_indices = torch.tensor(group, dtype=torch.long, device=losses_vec.device)
            group_loss = losses_vec.index_select(0, group_indices).sum()

            if group_loss.item() == 0.0:
                continue

            group_loss.backward()
            optimizer.step()

            with torch.no_grad():
                out_new = model(batch)
                new_losses_vec = criterion(out_new, batch, return_per_task=True)

            self._update_affinity(group, base_losses, new_losses_vec.detach().clone())

            last_out = out_new
            last_total_loss = float(new_losses_vec.sum().item())

        if last_out is None:
            with torch.no_grad():
                out = model(batch)
                losses_vec = criterion(out, batch, return_per_task=True)
                last_total_loss = float(losses_vec.sum().item())
            last_out = out

        if self.regroup_interval > 0 and (self.step_idx % self.regroup_interval == 0):
            self._recompute_groups()

        return last_total_loss, last_out


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    weight_method=None,
    group_method: Optional[SelectiveTaskGroupUpdater] = None,
):
    model.train()

    total_loss = 0.0
    n_samples = 0

    emo_preds, emo_tgts = [], []
    resd_preds, resd_tgts = [], []
    pkl_preds, pkl_tgts = [], []
    ah_preds, ah_tgts = [], []

    for batch in tqdm(loader, desc="Train", leave=False):
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

        # 1) update
        if group_method is not None:
            loss_scalar, out = group_method.step(model, optimizer, criterion, batch, device)
        else:
            optimizer.zero_grad()
            out = model(batch)

            if weight_method is None:
                loss = criterion(out, batch)
                loss.backward()
                optimizer.step()
                loss_scalar = float(loss.item())
            else:
                losses_vec = criterion(out, batch, return_per_task=True)

                shared_params = list(model.encoder.parameters())
                task_params = (
                    list(model.emo_mosei_head.parameters())
                    + list(model.emo_resd_head.parameters())
                    + list(model.personality_head.parameters())
                    + list(model.ah_head.parameters())
                )

                ret = weight_method.backward(
                    losses=losses_vec,
                    shared_parameters=shared_params,
                    task_specific_parameters=task_params,
                    last_shared_parameters=None,
                    representation=None,
                )

                with torch.no_grad():
                    loss = losses_vec.sum()

                    if isinstance(ret, (np.ndarray, list)) and not isinstance(ret, tuple):
                        w = torch.as_tensor(ret, device=losses_vec.device, dtype=losses_vec.dtype)
                        if w.numel() == losses_vec.numel():
                            loss = (losses_vec * w).sum()

                    elif isinstance(ret, tuple) and len(ret) == 2 and isinstance(ret[1], dict) and "weights" in ret[1]:
                        w = torch.as_tensor(ret[1]["weights"], device=losses_vec.device, dtype=losses_vec.dtype)
                        if w.numel() == losses_vec.numel():
                            loss = (losses_vec * w).sum()

                optimizer.step()
                loss_scalar = float(loss.item())

        bs = batch["x"].shape[0]
        total_loss += loss_scalar * bs
        n_samples += bs

        if batch["has_emo_mosei"].any():
            pred = out["emotion_mosei_pred"][batch["has_emo_mosei"]]
            tgt  = batch["emotion_mosei"][batch["has_emo_mosei"]]
            p, t = process_emotions(pred, tgt)
            emo_preds.extend(p)
            emo_tgts.extend(t)

        if batch["has_emo_resd"].any():
            pred = out["emotion_resd_logits"][batch["has_emo_resd"]].argmax(dim=1)
            tgt  = batch["emotion_resd"][batch["has_emo_resd"]]
            resd_preds.extend(pred.cpu().numpy())
            resd_tgts.extend(tgt.cpu().numpy())

        if batch["has_pers"].any():
            pred = out["personality_preds"][batch["has_pers"]].detach().cpu().numpy()
            tgt  = batch["personality"][batch["has_pers"]].cpu().numpy()
            pkl_preds.append(pred)
            pkl_tgts.append(tgt)

        if batch["has_ah"].any():
            pred = out["ah_logits"][batch["has_ah"]].argmax(dim=1)
            tgt  = batch["ah"][batch["has_ah"]]
            ah_preds.extend(pred.cpu().numpy())
            ah_tgts.extend(tgt.cpu().numpy())

    results = {}

    if emo_tgts:
        emo_tgt = np.asarray(emo_tgts)
        emo_prd = np.asarray(emo_preds)
        results["mF1"] = mf1(emo_tgt, emo_prd)
        results["mUAR"] = uar(emo_tgt, emo_prd)

    if resd_tgts:
        results["mF1_RESD"] = mf1_ah(np.array(resd_tgts), np.array(resd_preds))
        results["mUAR_RESD"] = uar_ah(np.array(resd_tgts), np.array(resd_preds))

    if pkl_tgts:
        tgt = np.vstack(pkl_tgts)
        prd = np.vstack(pkl_preds)
        results["ACC"] = acc_func(tgt, prd)
        ccs = []
        for i in range(tgt.shape[1]):
            mask = ~np.isnan(tgt[:, i])
            if mask.sum() > 0:
                ccs.append(ccc(tgt[mask, i], prd[mask, i]))
        if ccs:
            results["CCC"] = float(np.mean(ccs))

    if ah_tgts:
        results["MF1_AH"] = mf1_ah(np.array(ah_tgts), np.array(ah_preds))
        results["UAR_AH"] = uar_ah(np.array(ah_tgts), np.array(ah_preds))

    train_loss = total_loss / max(1, n_samples)
    return train_loss, results



def train_model(cfg, train_loader: DataLoader, test_loaders: Dict[str, DataLoader]):
    device = torch.device(cfg.device)

    model_cfg = TransformerMTLConfig(
        in_dim=1024,
        encoder_type=getattr(cfg, "encoder_type", "transformer"),
        zeros_use_norm=getattr(cfg, "zeros_use_norm", True),

        d_model=getattr(cfg, "d_model", 256),
        n_heads=getattr(cfg, "n_heads", 4),
        num_layers=getattr(cfg, "num_layers", 4),
        dim_feedforward=getattr(cfg, "dim_feedforward", 1024),
        dropout=getattr(cfg, "dropout", 0.1),
        max_len=getattr(cfg, "max_len", 5000),

        mamba_d_state=getattr(cfg, "mamba_d_state", 16),
        mamba_d_conv=getattr(cfg, "mamba_d_conv", 3),
        mamba_expand=getattr(cfg, "mamba_expand", 2),

        emo_mosei_out_dim=7,
        emo_resd_out_dim=7,
        pers_out_dim=5,
        ah_out_dim=2,
    )

    model = MultiTaskTransformerModel(model_cfg).to(device)

    criterion = MultiTaskLoss(
        label_smoothing=getattr(cfg, "label_smoothing", 0.0),
        regression_loss_type=getattr(cfg, "regression_loss_type", "mse"),
    )

    base_lr = float(getattr(cfg, "lr", 1e-4))
    enc_mult = float(getattr(cfg, "encoder_lr_mult", 1.0))
    head_mult = float(getattr(cfg, "heads_lr_mult", 3.0))

    encoder_lr = base_lr * enc_mult
    heads_lr = base_lr * head_mult

    optimizer = torch.optim.Adam(
        [
            {"params": model.encoder.parameters(), "lr": encoder_lr},
            {"params": model.emo_mosei_head.parameters(), "lr": heads_lr},
            {"params": model.emo_resd_head.parameters(), "lr": heads_lr},
            {"params": model.personality_head.parameters(), "lr": heads_lr},
            {"params": model.ah_head.parameters(), "lr": heads_lr},
        ]
    )

    weight_method = None
    group_method = None

    mt_method = getattr(cfg, "mt_weight_method", "none")
    if mt_method == "ntkmtl":
        weight_method = NTKMTL(
            n_tasks=4,
            device=device,
            max_norm=getattr(cfg, "mt_max_norm", 1.0),
            ntk_exp=getattr(cfg, "ntk_exp", 0.5),
        )
    elif mt_method == "taskgroup":
        group_method = SelectiveTaskGroupUpdater(
            n_tasks=4,
            beta=getattr(cfg, "group_decay", 1e-3),
            regroup_interval=getattr(cfg, "group_regroup_interval", 50),
            affinity_threshold=getattr(cfg, "group_affinity_threshold", 0.0),
            device=device,
        )
    elif mt_method == "fairgrad":
        weight_method = FairGrad(
            n_tasks=4,
            device=device,
            alpha=getattr(cfg, "fairgrad_alpha", 1.0),
            max_norm=getattr(cfg, "mt_max_norm", 1.0),
        )

    best_score = -float("inf")
    best_test = {}

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    early_stop_cnt = 0

    for epoch in range(cfg.num_epochs):
        print(f"\n====== EPOCH {epoch+1}/{cfg.num_epochs} ======")

        train_loss, train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            weight_method=weight_method,
            group_method=group_method,
        )

        pretty_print_train(epoch, cfg.num_epochs, train_loss, train_metrics)

        test_results = {}
        for ds, loader in test_loaders.items():
            test_results[ds] = evaluate_dataset(model, loader, device)

        test_final = aggregate_multidataset(test_results)
        pretty_print_test(test_final)

        score = test_final.get(cfg.selection_metric, -999)
        early_stop_cnt += 1

        if score > best_score:
            early_stop_cnt = 0
            best_score = score
            best_test = test_final

            ckpt_path = Path(cfg.checkpoint_dir) / "best_model.pt"
            torch.save(model.state_dict(), ckpt_path)
            print("✔ Saved best model:", ckpt_path)

        if early_stop_cnt >= 5:
            break

    return best_test
