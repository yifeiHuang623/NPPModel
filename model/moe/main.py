import os
import shutil
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.EarlyStopping import EarlyStopping
from utils.logger import get_logger
from utils.register import EVAL_REGISTRY

from .moe import moe

log = get_logger(__name__)

pre_views = ["moe_preview"]
post_views = ["moe_post_view"]

model = None


def _set_modules_grad(modules, requires_grad: bool):
    for module in modules:
        for param in module.parameters():
            param.requires_grad = requires_grad


def _load_causal_backbone(model: moe, dataset_name: str):
    causal_ckpt = f"./saved_models/{dataset_name}/causal/causal.pkl"
    if not os.path.exists(causal_ckpt):
        log.info("No causal checkpoint found at %s, training moe from scratch", causal_ckpt)
        return False

    state_dict = torch.load(causal_ckpt, map_location="cpu")
    mapped_state = {}
    for key, value in state_dict.items():
        if key.startswith("encoder."):
            mapped_state[key.replace("encoder.", "shared_encoder.", 1)] = value
        elif key.startswith(("emb_loc.", "emb_reg.", "emb_time.", "emb_user.", "pos_encoder.", "decoder.")):
            mapped_state[key] = value

    missing, unexpected = model.load_state_dict(mapped_state, strict=False)
    log.info(
        "Loaded causal backbone from %s (missing=%d unexpected=%d)",
        causal_ckpt,
        len(missing),
        len(unexpected),
    )
    return True


def train_model(model: moe, train_dataloader: DataLoader, valid_dataloader: DataLoader, metric_keys, args):
    model = model.to(args.device)

    backbone_modules = [
        model.emb_loc,
        model.emb_reg,
        model.emb_time,
        model.emb_user,
        model.pos_encoder,
        model.shared_encoder,
        model.decoder,
    ]
    moe_modules = [
        model.short_expert,
        model.medium_expert,
        model.memory_expert,
        model.time_skip_expert,
        model.router,
        model.segment_context_proj,
    ]
    backbone_params = []
    for module in backbone_modules:
        backbone_params.extend(list(module.parameters()))
    backbone_param_ids = {id(p) for p in backbone_params}
    moe_params = [p for p in model.parameters() if id(p) not in backbone_param_ids]

    opt = torch.optim.AdamW(
        [
            {
                "params": backbone_params,
                "lr": args.lr * getattr(args, "backbone_lr_scale", 0.2),
                "weight_decay": args.weight_decay,
            },
            {
                "params": moe_params,
                "lr": args.lr * getattr(args, "moe_lr_scale", 1.0),
                "weight_decay": args.weight_decay,
            },
        ]
    )
    early_stopping = EarlyStopping(
        patience=args.patience,
        save_model_folder=args.save_model_folder,
        save_model_name=f"{args.save_model_name}",
        logger=log,
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        "max",
        patience=args.lr_step,
        factor=args.lr_decay,
        threshold=args.schedule_threshold,
    )

    for epoch in range(args.num_epochs):
        model.set_epoch(epoch, args.num_epochs)
        freeze_backbone = bool(getattr(args, "init_from_causal", False)) and epoch < getattr(args, "freeze_backbone_epochs", 0)
        _set_modules_grad(backbone_modules, not freeze_backbone)
        if freeze_backbone:
            log.info("Epoch: %d, backbone frozen for warm start adaptation", epoch + 1)
        log.info(
            "Epoch: %d, expert scale: %.4f, memory scale: %.4f, router tau: %.4f",
            epoch + 1,
            model.current_expert_scale,
            model.current_memory_scale,
            model.router.current_tau,
        )

        model.train()
        total_loss = 0.0
        total_router_loss = 0.0
        route_usage_sum = None
        route_usage_steps = 0
        route_max_sum = 0.0
        route_entropy_sum = 0.0
        train_data_loader_tqdm = tqdm(train_dataloader, ncols=120)
        for batch_idx, batch_data in enumerate(train_data_loader_tqdm):
            loss = model.calculate_loss(batch_data)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            if model.last_router_loss is not None:
                total_router_loss += float(model.last_router_loss.detach().cpu().item())
            if model.last_route_usage is not None:
                current_usage = model.last_route_usage.detach().cpu()
                route_usage_sum = current_usage if route_usage_sum is None else route_usage_sum + current_usage
                route_usage_steps += 1
            if model.last_route_max is not None:
                route_max_sum += float(model.last_route_max.detach().cpu().item())
            if model.last_route_entropy is not None:
                route_entropy_sum += float(model.last_route_entropy.detach().cpu().item())

            train_data_loader_tqdm.set_description(
                f"Epoch: {epoch + 1}, train for the {batch_idx + 1}-th batch, train loss: {loss.item()}"
            )
        log.info(f"Epoch: {epoch + 1}, train loss: {total_loss / len(train_dataloader)}")
        if route_usage_steps > 0:
            mean_route_usage = (route_usage_sum / route_usage_steps).tolist()
            log.info(
                "Epoch: %d, router aux loss: %.6f, route usage [short, segment, memory, restart]: %s, route max: %.4f, route entropy: %.4f",
                epoch + 1,
                total_router_loss / route_usage_steps,
                [round(float(v), 4) for v in mean_route_usage],
                route_max_sum / route_usage_steps,
                route_entropy_sum / route_usage_steps,
            )

        y_predict, y_truth = inference(valid_dataloader)
        scores = {}
        for metric_name in metric_keys:
            score = EVAL_REGISTRY[metric_name](y_predict, y_truth)
            scores[metric_name] = score
            log.info("%-12s : %.6f", metric_name, score)

        selection_score = (
            0.2 * scores.get("NDCG1", 0.0) +
            0.3 * scores.get("NDCG5", 0.0) +
            0.5 * scores.get("NDCG10", 0.0)
        )
        log.info("SelectionScore : %.6f", selection_score)

        val_metric_indicator = [("SelectionScore", selection_score, True)]
        lr_scheduler.step(selection_score)
        early_stop = early_stopping.step(val_metric_indicator, model)
        if early_stop:
            break

    early_stopping.load_checkpoint(model)


def train(dataloader: dict[DataLoader], dataset_name, **kv):
    global model

    with open("./model/moe/moe.yaml", "r", encoding="utf-8") as f:
        args = yaml.safe_load(f)
    args.update(dataloader.view_value)
    args = SimpleNamespace(**args)

    train_dataloader, val_dataloader = dataloader.train_dataloader, dataloader.val_dataloader
    model = moe(args)
    if getattr(args, "init_from_causal", False):
        _load_causal_backbone(model, dataset_name)

    metric_keys = list(EVAL_REGISTRY)

    args.save_model_name = f"{args.model_name}"
    save_model_folder = f"./saved_models/{dataset_name}/{args.save_model_name}/"
    shutil.rmtree(save_model_folder, ignore_errors=True)
    os.makedirs(save_model_folder, exist_ok=True)
    args.save_model_folder = save_model_folder

    train_model(model, train_dataloader, val_dataloader, metric_keys, args)


def inference(dataloader: DataLoader, **kv):
    global model
    log.info("=============begin valid/test==============")

    if kv.get("model_para", False):
        with open("./model/moe/moe.yaml", "r", encoding="utf-8") as f:
            args = yaml.safe_load(f)
        args.update(kv.get("view_value", {}))
        args = SimpleNamespace(**args)
        log.info("=============args==============")
        log.info(args)
        model = moe(args).to(args.device)
        model.load_state_dict(kv["model_para"])

    model.eval()
    log.info("the number of batches: %d", len(dataloader))

    with torch.no_grad():
        y_predict_list, y_truth_list = [], []
        for batch_data in tqdm(dataloader):
            y_predict = model.predict(batch_data)
            y_predict_list.extend(y_predict.detach().cpu().numpy().tolist())
            y_truth_list.extend(batch_data["y_POI_id"]["POI_id"].detach().cpu().numpy().tolist())

    y_predict = np.array(y_predict_list)
    y_truth = np.array(y_truth_list)
    return y_predict, y_truth
