from __future__ import annotations

import copy
import os
from dataclasses import asdict
from itertools import product
from typing import Any, Dict, List, Tuple, Callable

METRIC_ORDER = [
    "mean_all",

    # CMU-MOSEI
    "mF1_mosei",
    "mUAR_mosei",

    # RESD
    "mF1_RESD_resd",
    "mUAR_RESD_resd",

    # FIV2
    "ACC_fiv2",
    "CCC_fiv2",

    # BAH
    "MF1_AH_bah",
    "UAR_AH_bah",
]


def _fmt(x: Any, ndigits: int = 4) -> str:
    try:
        return f"{float(x):.{ndigits}f}"
    except Exception:
        return str(x)


def _pick_score(metrics: Dict[str, Any], primary: str = "mean_all") -> float:
    if primary in metrics:
        return float(metrics[primary])

    if "mean_all" in metrics:
        return float(metrics["mean_all"])

    return float("-inf")


def _format_box(
    combo_id: int,
    cfg_dict: Dict[str, Any],
    metrics: Dict[str, Any],
    is_best: bool,
    selection_metric: str = "mean_all",
) -> str:

    title = f"Combo #{combo_id}"
    cfg_lines = [f"{k} = {v}" for k, v in cfg_dict.items()]

    # метрики
    metric_lines: List[str] = []
    ordered = METRIC_ORDER + sorted(set(metrics.keys()) - set(METRIC_ORDER))
    for k in ordered:
        if k not in metrics:
            continue
        if k == "by_dataset":
            continue

        val = metrics[k]
        line = f"{k.upper():12} = {_fmt(val)}"
        if is_best and k == selection_metric:
            line += "  ✅"
        metric_lines.append(line)

    content_lines: List[str] = [title, "  Params:"] + [
        f"    {ln}" for ln in cfg_lines
    ]
    content_lines.append("  Metrics:")
    content_lines += [f"    {ln}" for ln in metric_lines]

    max_width = max(len(l) for l in content_lines)
    top = "┌" + "─" * (max_width + 2) + "┐"
    bot = "└" + "─" * (max_width + 2) + "┘"

    box_lines = [top]
    for l in content_lines:
        box_lines.append(f"│ {l.ljust(max_width)} │")
    box_lines.append(bot)
    return "\n".join(box_lines)


def grid_search(
    base_cfg,
    train_loader,
    test_loaders: Dict[str, Any],
    train_fn: Callable[..., Dict[str, Any]],
    param_grid: Dict[str, List[Any]],
    log_file: str = "grid_search.log",
    selection_metric: str = "mean_all",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:

    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)

    all_param_names = list(param_grid.keys())
    combos = list(product(*(param_grid[p] for p in all_param_names)))

    best_score = float("-inf")
    best_cfg_dict: Dict[str, Any] = {}
    best_metrics: Dict[str, Any] = {}

    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"=== Grid search ===\n")
        f.write(f"Params: {all_param_names}\n\n")

    for combo_id, values in enumerate(combos, start=1):

        cfg = copy.deepcopy(base_cfg)
        param_values = dict(zip(all_param_names, values))
        for k, v in param_values.items():
            setattr(cfg, k, v)

        if hasattr(cfg, "checkpoint_dir"):
            combo_dir = os.path.join(cfg.checkpoint_dir, f"combo_{combo_id}")
            os.makedirs(combo_dir, exist_ok=True)
            cfg.checkpoint_dir = combo_dir

        print(f"\n[GRID] Combo {combo_id}/{len(combos)}: {param_values}")

        metrics = train_fn(cfg, train_loader, test_loaders)

        score = _pick_score(metrics, primary=selection_metric)
        is_best = score > best_score

        box = _format_box(combo_id, param_values, metrics, is_best, selection_metric)

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(box + "\n\n")

        print(box)

        if is_best:
            best_score = score
            best_cfg_dict = param_values
            best_metrics = metrics

    with open(log_file, "a", encoding="utf-8") as f:
        f.write("=== BEST COMBINATION ===\n")
        for k, v in best_cfg_dict.items():
            f.write(f"{k} = {v}\n")
        f.write(f"\nBest {selection_metric} = {_fmt(best_score)}\n")

    print("\n=== GRID SEARCH FINISHED ===")
    print("Best params:", best_cfg_dict)
    print("Best", selection_metric, "=", _fmt(best_score))

    return best_cfg_dict, best_metrics


def greedy_search(
    base_cfg,
    train_loader,
    test_loaders: Dict[str, Any],
    train_fn: Callable[..., Dict[str, Any]],
    param_grid: Dict[str, List[Any]],
    log_file: str = "greedy_search.log",
    selection_metric: str = "mean_all",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:

    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)

    param_names = list(param_grid.keys())

    current_best_params: Dict[str, Any] = {
        name: getattr(base_cfg, name) if hasattr(base_cfg, name) else values[0]
        for name, values in param_grid.items()
    }
    current_best_metrics: Dict[str, Any] = {}
    current_best_score = float("-inf")

    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"=== Greedy search ===\n")
        f.write(f"Params (order): {param_names}\n\n")

    combo_id = 0

    for step_idx, param_name in enumerate(param_names, start=1):
        values = param_grid[param_name]

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n=== STEP {step_idx}: tune '{param_name}' ===\n")

        best_param_value = current_best_params[param_name]
        best_param_score = float("-inf")
        best_param_metrics: Dict[str, Any] = {}

        for val in values:
            combo_id += 1

            cfg = copy.deepcopy(base_cfg)
            for k, v in current_best_params.items():
                setattr(cfg, k, v)
            setattr(cfg, param_name, val)

            if hasattr(cfg, "checkpoint_dir"):
                combo_dir = os.path.join(
                    cfg.checkpoint_dir,
                    f"step{step_idx}_{param_name}_{val}",
                )
                os.makedirs(combo_dir, exist_ok=True)
                cfg.checkpoint_dir = combo_dir

            param_values = copy.deepcopy(current_best_params)
            param_values[param_name] = val

            print(
                f"\n[GREEDY] Step {step_idx}/{len(param_names)} | "
                f"{param_name}={val} | fixed="
                f"{ {k: current_best_params[k] for k in param_names if k != param_name} }"
            )

            metrics = train_fn(cfg, train_loader, test_loaders)
            score = _pick_score(metrics, primary=selection_metric)

            is_best_for_param = score > best_param_score
            if is_best_for_param:
                best_param_score = score
                best_param_value = val
                best_param_metrics = metrics

            box = _format_box(
                combo_id,
                param_values,
                metrics,
                is_best=is_best_for_param,
                selection_metric=selection_metric,
            )

            with open(log_file, "a", encoding="utf-8") as f:
                f.write(box + "\n\n")

            print(box)

        current_best_params[param_name] = best_param_value
        current_best_metrics = best_param_metrics
        current_best_score = best_param_score

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(
                f"--> STEP {step_idx} best {param_name} = {best_param_value}, "
                f"{selection_metric} = {_fmt(current_best_score)}\n"
            )

        print(
            f"\n[GREEDY] After step {step_idx}: best {param_name}={best_param_value}, "
            f"{selection_metric}={_fmt(current_best_score)}"
        )

    with open(log_file, "a", encoding="utf-8") as f:
        f.write("\n=== FINAL GREEDY PARAMS ===\n")
        for k, v in current_best_params.items():
            f.write(f"{k} = {v}\n")
        f.write(f"\nBest {selection_metric} = {_fmt(current_best_score)}\n")

    print("\n=== GREEDY SEARCH FINISHED ===")
    print("Best params:", current_best_params)
    print("Best", selection_metric, "=", _fmt(current_best_score))

    return current_best_params, current_best_metrics
