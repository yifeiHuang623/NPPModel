
import torch
from torch.utils.data import DataLoader
import yaml
import numpy as np
import shutil
import os

from utils.logger import get_logger
from types import SimpleNamespace

from .causal import causal
from tqdm import tqdm
from utils.EarlyStopping import EarlyStopping
from utils.register import EVAL_REGISTRY

log = get_logger(__name__)

pre_views = ["causal_preview"]
post_views = []

model = None


def train_model(
    model: causal,
    train_dataloader: DataLoader,
    valid_dataloader: DataLoader,
    metric_keys,
    args,
):
    model = model.to(args.device)

    opt = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
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
        model.train()
        total_loss = 0.0
        train_data_loader_tqdm = tqdm(train_dataloader, ncols=120)
        for batch_idx, batch_data in enumerate(train_data_loader_tqdm):
            loss = model.calculate_loss(batch_data)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() / len(batch_data)

        train_data_loader_tqdm.set_description(
            f"Epoch: {epoch + 1}, train for the {batch_idx + 1}-th batch, train loss: {loss.item()}"
        )

        log.info(
            f"Epoch: {epoch + 1}, train loss: {total_loss / len(train_dataloader)}"
        )
        y_predict, y_truth = inference(valid_dataloader)
        scores = {}
        for m in metric_keys:
            score = EVAL_REGISTRY[m](y_predict, y_truth)
            scores[m] = score
            log.info("%-12s : %.6f", m, score)

        val_metric_indicator = []
        for metric_name, value in scores.items():
            val_metric_indicator.append((metric_name, value, True))

        lr_scheduler.step(scores["NDCG1"])
        early_stop = early_stopping.step(val_metric_indicator, model)

        # if early_stop or opt.param_groups[0]["lr"] < args.early_stop_lr:
        if early_stop:
            break

    early_stopping.load_checkpoint(model)


def train(dataloader: dict[DataLoader], dataset_name, **kv):
    global model
    with open("./model/causal/causal.yaml", "r", encoding="utf-8") as f:
        args = yaml.safe_load(f)
    args.update(dataloader.view_value)
    args = SimpleNamespace(**args)

    train_dataloader, val_dataloader = (
        dataloader.train_dataloader,
        dataloader.val_dataloader,
    )

    model = causal(args)
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
    
    if kv.get('model_para', False):   
        with open('./model/causal/causal.yaml', 'r', encoding='utf-8') as f:
            args = yaml.safe_load(f)
        args.update(kv.get('view_value', {}))
        args = SimpleNamespace(**args)
        log.info('=============args==============')
        log.info(args)
        model = causal(args).to(args.device)
        model.load_state_dict(kv['model_para'])
    
    model.eval()
    y_predict_list = []
    log.info('the number of batches: %d', len(dataloader))

    with torch.no_grad():
        y_predict_list, y_truth_list = [], []
        for batch_data in tqdm(dataloader):
            y_predict = model.predict(batch_data)
            y_predict_list.extend(y_predict.detach().cpu().numpy().tolist())
            y_truth_list.extend(
                batch_data["y_POI_id"]["POI_id"].detach().cpu().numpy().tolist()
            )

    y_predict = np.array(y_predict_list)
    y_truth = np.array(y_truth_list)

    return y_predict, y_truth
