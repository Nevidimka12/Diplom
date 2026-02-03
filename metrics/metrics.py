from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, mean_absolute_error
from sklearn.metrics import recall_score, f1_score, accuracy_score


def mf1(targets, predicts):
    targets = np.array(targets)
    predicts = np.array(predicts)

    f1s = []
    for i in range(predicts.shape[1]):
        cr = classification_report(
            targets[:, i],
            predicts[:, i],
            output_dict=True,
            zero_division=0,
        )
        f1s.append(cr['macro avg']['f1-score'])
    return float(np.mean(f1s))


def uar(targets, predicts):
    targets = np.array(targets)
    predicts = np.array(predicts)

    uars = []
    for i in range(predicts.shape[1]):
        cr = classification_report(
            targets[:, i],
            predicts[:, i],
            output_dict=True,
            zero_division=0,
        )
        uars.append(cr['macro avg']['recall'])
    return float(np.mean(uars))


def mf1_ah(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def uar_ah(y_true, y_pred):
    return recall_score(y_true, y_pred, average="macro", zero_division=0)


def acc_func(trues, preds):
    trues = np.array(trues)
    preds = np.array(preds)
    mae = np.mean(np.abs(trues - preds), axis=0)
    return float(np.mean(1 - mae))


def ccc(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mean_true = y_true.mean()
    mean_pred = y_pred.mean()
    var_true = ((y_true - mean_true) ** 2).mean()
    var_pred = ((y_pred - mean_pred) ** 2).mean()
    cov = ((y_true - mean_true) * (y_pred - mean_pred)).mean()
    return float((2 * cov) / (var_true + var_pred + (mean_true - mean_pred) ** 2))



def transform_matrix(matrix: np.ndarray) -> np.ndarray:
    threshold1 = 1 - 1 / 7
    threshold2 = 1 / 7

    mask1 = matrix[:, 0] >= threshold1
    result = np.zeros_like(matrix[:, 1:])
    transformed = (matrix[:, 1:] >= threshold2).astype(int)
    result[~mask1] = transformed[~mask1]
    return result


def process_emotions(logits: torch.Tensor, targets: torch.Tensor):

    pred = F.softmax(logits, dim=1).detach().cpu().numpy()
    pred = transform_matrix(pred)

    true = targets.detach().cpu().numpy()
    true = np.where(true > 0, 1, 0)[:, 1:]

    return pred.tolist(), true.tolist()


@torch.no_grad()
def evaluate_dataset(model, loader, device):

    model.eval()

    # CMU-MOSEI
    emo_preds, emo_tgts = [], []

    # FIV2
    pkl_preds, pkl_tgts = [], []

    # BAH
    ah_preds, ah_tgts = [], []

    # RESD
    resd_preds, resd_tgts = [], []

    for batch in loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }

        out = model(batch)

        if "emotion_mosei_pred" in out and out["emotion_mosei_pred"] is not None:
            logits = out["emotion_mosei_pred"]          # (B, 7)
            targets = batch["emotion_mosei"]            # (B, 7)
            has_mask = batch.get("has_emo_mosei", None) # (B,)

            if has_mask is not None:
                logits = logits[has_mask]
                targets = targets[has_mask]

            if logits.size(0) > 0:
                p, t = process_emotions(logits, targets)
                emo_preds.extend(p)
                emo_tgts.extend(t)

        if "emotion_resd_logits" in out and out["emotion_resd_logits"] is not None:
            logits_resd = out["emotion_resd_logits"]    # (B, 7)
            targets_resd = batch["emotion_resd"]        # (B,)
            has_mask_resd = batch.get("has_emo_resd", None)

            if has_mask_resd is not None:
                logits_resd = logits_resd[has_mask_resd]
                targets_resd = targets_resd[has_mask_resd]

            if logits_resd.size(0) > 0:
                preds = logits_resd.argmax(dim=1).detach().cpu().numpy()
                tgts = targets_resd.long().detach().cpu().numpy()
                resd_preds.append(preds)
                resd_tgts.append(tgts)

        if "personality_preds" in out and out["personality_preds"] is not None:
            preds_p = out["personality_preds"]     # (B, 5)
            targets_p = batch["personality"]       # (B, 5)
            has_mask_p = batch.get("has_pers", None)

            if has_mask_p is not None:
                preds_p = preds_p[has_mask_p]
                targets_p = targets_p[has_mask_p]

            if preds_p.size(0) > 0:
                pkl_preds.append(preds_p.detach().cpu().numpy())
                pkl_tgts.append(targets_p.detach().cpu().numpy())

        if "ah_logits" in out and out["ah_logits"] is not None:
            logits_ah = out["ah_logits"]       # (B, 2)
            targets_ah = batch["ah"]           # (B,)
            has_mask_ah = batch.get("has_ah", None)

            if has_mask_ah is not None:
                logits_ah = logits_ah[has_mask_ah]
                targets_ah = targets_ah[has_mask_ah]

            if logits_ah.size(0) > 0:
                preds = logits_ah.argmax(dim=1).detach().cpu().numpy()
                tgts = targets_ah.long().detach().cpu().numpy()
                ah_preds.append(preds)
                ah_tgts.append(tgts)

    metrics = {}

    # CMU-MOSEI
    if emo_tgts:
        tgt = np.asarray(emo_tgts)
        prd = np.asarray(emo_preds)
        metrics["mF1"] = mf1(tgt, prd)
        metrics["mUAR"] = uar(tgt, prd)

    # RESD
    if resd_tgts:
        tgt = np.concatenate(resd_tgts, axis=0)
        prd = np.concatenate(resd_preds, axis=0)
        metrics["mF1_RESD"] = mf1_ah(tgt, prd)
        metrics["mUAR_RESD"] = uar_ah(tgt, prd)

    # FIV2
    if pkl_tgts:
        tgt = np.vstack(pkl_tgts)
        prd = np.vstack(pkl_preds)
        metrics["ACC"] = acc_func(tgt, prd)

        ccs = []
        for i in range(tgt.shape[1]):
            mask = ~np.isnan(tgt[:, i])
            if mask.sum() > 0:
                ccs.append(ccc(tgt[mask, i], prd[mask, i]))
        if ccs:
            metrics["CCC"] = float(np.mean(ccs))

    # BAH
    if ah_tgts:
        tgt = np.concatenate(ah_tgts, axis=0)
        prd = np.concatenate(ah_preds, axis=0)
        metrics["MF1_AH"] = mf1_ah(tgt, prd)
        metrics["UAR_AH"] = uar_ah(tgt, prd)

    return metrics


def aggregate_multidataset(results: dict[str, dict]):

    final = {}
    by_ds = []

    # Целевые метрики
    TASK_METRIC_KEYS = (
        "mF1", "mUAR",          # MOSEI
        "mF1_RESD", "mUAR_RESD",# RESD
        "ACC", "CCC",           # FIV2
        "MF1_AH", "UAR_AH",     # BAH
    )

    all_metric_values = []

    for ds_name, m in results.items():
        entry = {"name": ds_name}
        entry.update(m)
        by_ds.append(entry)

        for k, v in m.items():
            final[f"{k}_{ds_name}"] = v

        for k in TASK_METRIC_KEYS:
            if k in m:
                all_metric_values.append(float(m[k]))

    if all_metric_values:
        final["mean_all"] = float(np.mean(all_metric_values))

    final["by_dataset"] = by_ds
    return final


__all__ = [
    "mf1", "uar", "mf1_ah", "uar_ah",
    "acc_func", "ccc",
    "transform_matrix", "process_emotions",
    "evaluate_dataset", "aggregate_multidataset",
]
