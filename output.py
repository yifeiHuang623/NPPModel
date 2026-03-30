import argparse
from collections import Counter
import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from model.ROTAN.ROTAN import ROTAN
from utils.exargs import ConfigResolver
from utils.logger import get_logger, set_dataset_name, set_log_file_name, set_model_name
from utils.register import DATALOADER_REGISTRY, VIEW_REGISTRY
import utils.dataloader
import utils.eval
import utils.views


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ROTAN")
    parser.add_argument("--dataset", default="NYC_time_interval")
    parser.add_argument("--task", default="NPP")
    parser.add_argument("--device", default=None)
    parser.add_argument("--saved_model", default=None)
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args()


def to_python_scalar(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def normalize_json_value(value):
    value = to_python_scalar(value)
    if isinstance(value, dict):
        return {k: normalize_json_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize_json_value(v) for v in value]
    if isinstance(value, tuple):
        return [normalize_json_value(v) for v in value]
    return value


def batch_to_device(batch_data, device):
    batch_data["quad_key"] = torch.as_tensor(batch_data["quad_key"], dtype=torch.long)
    batch_data["y_POI_id"]["quad_key"] = torch.as_tensor(batch_data["y_POI_id"]["quad_key"], dtype=torch.long)

    batch_data_device = {
        k: v.to(device, non_blocking=True) for k, v in batch_data.items() if k != "y_POI_id"
    }
    batch_data_device["y_POI_id"] = {
        k: v.to(device, non_blocking=True) for k, v in batch_data["y_POI_id"].items()
    }
    return batch_data_device


def sample_history_from_batch(batch_data, idx):
    history_len = int(batch_data["mask"][idx].item())
    history = batch_data["POI_id"][idx][:history_len]
    return [int(v) for v in history.detach().cpu().tolist()]


def sample_truth_from_batch(batch_data, idx):
    return int(batch_data["y_POI_id"]["POI_id"][idx].item())


def load_raw_df(dataset_name, task, pre_views):
    dataset_prefix = dataset_name.split("_")[0]
    module = importlib.import_module(f"utils.dataloader.{task}.dataloader_{dataset_prefix}")
    raw_df = module.pre_process_func()
    raw_df = raw_df.sort_values("timestamps").reset_index(drop=True)
    raw_df["row_id"] = np.arange(len(raw_df), dtype=np.int64)

    view_value = {}
    for view in pre_views:
        raw_df, view_value = VIEW_REGISTRY[view](raw_df, view_value)
    return raw_df


def build_full_history_lookup(raw_df):
    lookup = {}
    group_key = ["user_id", "trajectory_id"] if "trajectory_id" in raw_df.columns else ["user_id"]

    for _, group in raw_df.groupby(group_key, sort=False):
        group = group.sort_values("timestamps")
        poi_ids = group["POI_id"].astype(np.int64).tolist()
        row_ids = group["row_id"].astype(np.int64).tolist()

        for idx, row_id in enumerate(row_ids):
            lookup[int(row_id)] = poi_ids[:idx]

    return lookup


def build_next_poi_lookup(raw_df):
    lookup = {}
    group_key = ["user_id", "trajectory_id"] if "trajectory_id" in raw_df.columns else ["user_id"]

    for _, group in raw_df.groupby(group_key, sort=False):
        group = group.sort_values("timestamps")
        poi_ids = group["POI_id"].astype(np.int64).tolist()

        for cur_poi, next_poi in zip(poi_ids[:-1], poi_ids[1:]):
            lookup.setdefault(int(cur_poi), Counter())
            lookup[int(cur_poi)][int(next_poi)] += 1

    return lookup


def build_user_timeslot_poi_lookup(raw_df):
    lookup = {}
    for _, group in raw_df.groupby(["user_id", "time_id"], sort=False):
        group = group.sort_values("timestamps")
        user_id = int(group["user_id"].iloc[0])
        time_id = int(group["time_id"].iloc[0])
        counts = Counter(int(v) for v in group["POI_id"].astype(np.int64).tolist())
        lookup[(user_id, time_id)] = counts
    return lookup


def build_recent_pattern_lookup(raw_df, max_pattern_len=3):
    lookup = {k: {} for k in range(1, max_pattern_len + 1)}
    group_key = ["user_id", "trajectory_id"] if "trajectory_id" in raw_df.columns else ["user_id"]

    for _, group in raw_df.groupby(group_key, sort=False):
        group = group.sort_values("timestamps")
        poi_ids = group["POI_id"].astype(np.int64).tolist()

        for next_idx in range(1, len(poi_ids)):
            next_poi = int(poi_ids[next_idx])
            for k in range(1, max_pattern_len + 1):
                if next_idx - k < 0:
                    continue
                pattern = tuple(int(v) for v in poi_ids[next_idx - k:next_idx])
                lookup[k].setdefault(pattern, Counter())
                lookup[k][pattern][next_poi] += 1

    return lookup


def top3_poi_counts(history):
    counts = Counter(int(poi_id) for poi_id in history)
    top3 = counts.most_common(3)
    return [{"POI_id": poi_id, "count": count} for poi_id, count in top3]


def top3_next_pois(last_poi_id, next_poi_lookup):
    if last_poi_id is None:
        return []
    counts = next_poi_lookup.get(int(last_poi_id), Counter())
    return [{"POI_id": poi_id, "count": count} for poi_id, count in counts.most_common(3)]


def top3_same_timeslot_pois(user_id, time_id, user_timeslot_poi_lookup):
    counts = user_timeslot_poi_lookup.get((int(user_id), int(time_id)), Counter())
    return [{"POI_id": poi_id, "count": count} for poi_id, count in counts.most_common(3)]


def top3_recent_pattern_next_pois(history, recent_pattern_lookup, max_pattern_len=3):
    if not history:
        return []

    for k in range(min(max_pattern_len, len(history)), 0, -1):
        pattern = tuple(int(v) for v in history[-k:])
        counts = recent_pattern_lookup.get(k, {}).get(pattern)
        if counts:
            return [{"POI_id": poi_id, "count": count} for poi_id, count in counts.most_common(3)]
    return []


def evaluate_split(
    model,
    dataloader,
    split_name,
    device,
    full_history_lookup,
    next_poi_lookup,
    user_timeslot_poi_lookup,
    recent_pattern_lookup,
):
    rows = []

    model.eval()
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc=f"export_{split_name}"):
            batch_size = len(batch_data["user_id"])
            batch_data_device = batch_to_device(batch_data, device)
            logits = model.predict(batch_data_device)
            pred_ids = torch.argmax(logits, dim=-1).detach().cpu().numpy()
            truth_ids = batch_data["y_POI_id"]["POI_id"].detach().cpu().numpy()

            for i in range(batch_size):
                pred_id = int(pred_ids[i])
                truth_id = int(truth_ids[i])
                history = sample_history_from_batch(batch_data, i)
                last_poi_id = history[-1] if history else None
                row = {
                    "split": split_name,
                    "user_id": int(batch_data["user_id"][i].item()),
                    "trajectory_id": int(batch_data["trajectory_id"][i].item()),
                    "label_row_id": int(batch_data["y_POI_id"]["row_id"][i].item()),
                    "history_length": int(batch_data["mask"][i].item()),
                    "end_ts": int(batch_data["end_ts"][i].item()),
                    "is_correct": int(pred_id == truth_id),
                    "historical": json.dumps(history, ensure_ascii=False),
                    "predict": pred_id,
                    "truth": sample_truth_from_batch(batch_data, i),
                    "same_timeslot_top_pois": json.dumps(
                        top3_same_timeslot_pois(
                            int(batch_data["user_id"][i].item()),
                            int(batch_data["y_POI_id"]["time_id"][i].item()),
                            user_timeslot_poi_lookup,
                        ),
                        ensure_ascii=False,
                    ),
                    "recent_pattern_top_next_pois": json.dumps(
                        top3_recent_pattern_next_pois(history, recent_pattern_lookup),
                        ensure_ascii=False,
                    ),
                    "last_poi_top_next_pois": json.dumps(
                        top3_next_pois(last_poi_id, next_poi_lookup),
                        ensure_ascii=False,
                    ),
                    "user_top3_history_pois": json.dumps(
                        top3_poi_counts(
                            full_history_lookup.get(int(batch_data["y_POI_id"]["row_id"][i].item()), [])
                        ),
                        ensure_ascii=False,
                    ),
                }
                rows.append(row)

    return pd.DataFrame(rows)


def save_split_csv(df, output_dir, split_name):
    correct_dir = output_dir / "correct"
    wrong_dir = output_dir / "wrong"
    correct_dir.mkdir(parents=True, exist_ok=True)
    wrong_dir.mkdir(parents=True, exist_ok=True)

    correct_path = correct_dir / f"{split_name}.csv"
    wrong_path = wrong_dir / f"{split_name}.csv"

    df[df["predict"] == df["truth"]].to_csv(correct_path, index=False)
    df[df["predict"] != df["truth"]].to_csv(wrong_path, index=False)
    return correct_path, wrong_path


def main():
    args = parse_args()

    set_model_name(args.model)
    set_dataset_name(args.dataset)
    set_log_file_name()
    log = get_logger("output")

    utils.dataloader.register_all(task=args.task)

    if args.model != "ROTAN":
        raise ValueError("output.py currently supports ROTAN only.")

    model_args = ConfigResolver(f"model/{args.model}/{args.model}.yaml").parse()
    pre_views = ["ROTAN_preview"]
    post_views = ["ROTAN_post_view"]
    dataloader = DATALOADER_REGISTRY[args.dataset](
        model_args=model_args,
        pre_views=pre_views,
        post_views=post_views,
    )
    raw_df = load_raw_df(args.dataset, args.task, pre_views)
    full_history_lookup = build_full_history_lookup(raw_df)
    next_poi_lookup = build_next_poi_lookup(raw_df)
    user_timeslot_poi_lookup = build_user_timeslot_poi_lookup(raw_df)
    recent_pattern_lookup = build_recent_pattern_lookup(raw_df, max_pattern_len=3)

    with open(f"./model/{args.model}/{args.model}.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.update(dataloader.view_value)

    if args.device:
        device = args.device
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg["device"] = device

    model = ROTAN(SimpleNamespace(**cfg)).to(device)

    saved_model = args.saved_model or f"./saved_models/{args.dataset}/{args.model}/{args.model}.pkl"
    state_dict = torch.load(saved_model, map_location=device)
    model.load_state_dict(state_dict)

    output_dir = Path(args.output_dir or f"./outputs/{args.dataset}/{args.model}")
    output_dir.mkdir(parents=True, exist_ok=True)

    valid_df = evaluate_split(
        model=model,
        dataloader=dataloader.val_dataloader,
        split_name="valid",
        device=device,
        full_history_lookup=full_history_lookup,
        next_poi_lookup=next_poi_lookup,
        user_timeslot_poi_lookup=user_timeslot_poi_lookup,
        recent_pattern_lookup=recent_pattern_lookup,
    )
    test_df = evaluate_split(
        model=model,
        dataloader=dataloader.test_dataloader,
        split_name="test",
        device=device,
        full_history_lookup=full_history_lookup,
        next_poi_lookup=next_poi_lookup,
        user_timeslot_poi_lookup=user_timeslot_poi_lookup,
        recent_pattern_lookup=recent_pattern_lookup,
    )

    valid_correct, valid_wrong = save_split_csv(valid_df, output_dir, "valid")
    test_correct, test_wrong = save_split_csv(test_df, output_dir, "test")

    summary = {
        "saved_model": str(Path(saved_model).resolve()),
        "output_dir": str(output_dir.resolve()),
        "valid_total": int(len(valid_df)),
        "valid_correct": int((valid_df["is_correct"] == 1).sum()),
        "valid_wrong": int((valid_df["is_correct"] == 0).sum()),
        "test_total": int(len(test_df)),
        "test_correct": int((test_df["is_correct"] == 1).sum()),
        "test_wrong": int((test_df["is_correct"] == 0).sum()),
        "files": {
            "valid_correct": str(valid_correct.resolve()),
            "valid_wrong": str(valid_wrong.resolve()),
            "test_correct": str(test_correct.resolve()),
            "test_wrong": str(test_wrong.resolve()),
        },
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    log.info("Export finished: %s", summary_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
