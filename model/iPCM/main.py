import random
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import yaml
import pandas as pd
import numpy as np

from .iPCM import iPCM
from .iPCM_utils import construct_graph
from utils.EarlyStopping import EarlyStopping
from tqdm import tqdm
from utils.register import EVAL_REGISTRY
from types import SimpleNamespace
import shutil, os

from utils.logger import get_logger
from tqdm import tqdm
log = get_logger(__name__)

pre_views = ['iPCM_preview']
post_views = []

model = None

def train_model(model: iPCM, train_dataloader: DataLoader, valid_dataloader: DataLoader, metric_keys, args):
    model = model.to(args.device)
    early_stopping = EarlyStopping(patience=args.patience, save_model_folder=args.save_model_folder, save_model_name=f"{args.save_model_name}", logger=log)
    
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'max', factor=args.lr_scheduler_factor, patience=args.patience)
    
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0
    
        train_data_loader_tqdm = tqdm(train_dataloader, ncols=120)
        for batch_idx, batch_data in enumerate(train_data_loader_tqdm):
            batch_data_device = {k: v.to(args.device, non_blocking=True) for k, v in batch_data.items() if 'y_POI_id' not in k}
            batch_data_device['y_POI_id'] = {k: v.to(args.device, non_blocking=True) for k, v in batch_data['y_POI_id'].items()}
            loss = model(batch_data_device)

            opt.zero_grad()
            loss.backward()
            opt.step()
        
            total_loss += loss.item() / len(batch_data)
            train_data_loader_tqdm.set_description(f'Epoch: {epoch + 1}, train for the {batch_idx + 1}-th batch, train loss: {loss.item()}')
            
        log.info(f'Epoch: {epoch + 1}, train loss: {total_loss / len(train_dataloader)}')
        
        y_predict, y_truth = inference(valid_dataloader)
        scores = {}
        for m in metric_keys:
            if type(y_predict) == dict:
                poi = y_predict["poi"]
                poi_y_truth = y_truth["poi"]
                
            score = EVAL_REGISTRY[m](poi, poi_y_truth)
            scores[m] = score
            log.info("%-12s : %.6f", m, score)
            
        val_metric_indicator = []
        for metric_name, value in scores.items():
            if metric_name not in ["MSE", "MAE"]:
                val_metric_indicator.append((metric_name, value, True))
            
        lr_scheduler.step(scores['NDCG1'])
        early_stop = early_stopping.step(val_metric_indicator, model)

        if early_stop:
            break
    
    early_stopping.load_checkpoint(model)

def train(dataloader: dict[DataLoader], dataset_name, **kw):
    global model
    with open('./model/iPCM/iPCM.yaml', 'r', encoding='utf-8') as f:
        args = yaml.safe_load(f)
    args.update(dataloader.view_value)
    args = SimpleNamespace(**args)
    
    train_dataloader, val_dataloader = dataloader.train_dataloader, dataloader.val_dataloader
    
    def dict_dataloader_to_df(dataloader):
        trajectory_id = 0
        trajectory_list, user_id_list, POI_id_list, region_id_list, time_period_list = [], [], [], [], []
        
        for batch in dataloader:
            bs = len(next(iter(batch.values())))
            for i in range(bs):
                data_len = batch['mask'][i]
                trajectory_list.extend([trajectory_id] * data_len)
                user_id_list.extend([batch['user_id'][i].item()] * data_len)
                POI_id_list.extend(batch['POI_id'][i][:data_len].tolist())
                region_id_list.extend(batch['region_id'][i][:data_len].tolist())
                time_period_list.extend(batch['time_period'][i][:data_len].tolist())
                trajectory_id += 1
                
        records = pd.DataFrame({
            'trajectory_id': trajectory_list,
            'user_id': user_id_list,
            'POI_id': POI_id_list,
            'region_id': region_id_list,
            'time_period': time_period_list
        })    
        return records
    
    train_df = dict_dataloader_to_df(train_dataloader)
    graph_dict = construct_graph(args, train_df)
    
    model = iPCM(args, graph_dict)
    
    metric_keys = (list(EVAL_REGISTRY))
    
    args.save_model_name = f'{args.model_name}'
    save_model_folder = f"./saved_models/{dataset_name}/{args.save_model_name}/"
    shutil.rmtree(save_model_folder, ignore_errors=True)
    os.makedirs(save_model_folder, exist_ok=True)
    args.save_model_folder = save_model_folder
    
    train_model(model, train_dataloader, val_dataloader, metric_keys, args)

def inference(dataloader: DataLoader, **kv):
    global model
    
    model.eval()
    y_predict_list = []
    
    with torch.no_grad():
        y_predict_list, y_truth_list = [], []
        y_predict_time_list, y_truth_time_list = [], []
        for batch_data in tqdm(dataloader):
            batch_data_device = {k: v.to(model.device, non_blocking=True) for k, v in batch_data.items() if 'y_POI_id' not in k}
            batch_data_device['y_POI_id'] = {k: v.to(model.device, non_blocking=True) for k, v in batch_data['y_POI_id'].items()}

            y_predict, y_predict_time, y_truth_time = model.predict(batch_data_device)
            
            y_predict_list.extend(y_predict.tolist())
            y_truth_list.extend(batch_data['y_POI_id']['POI_id'].detach().cpu().numpy().tolist())
            
            y_predict_time_list.extend(y_predict_time)
            y_truth_time_list.extend(y_truth_time)
            
    y_predict = np.array(y_predict_list)
    y_truth = np.array(y_truth_list)
    y_predict_time = np.array(y_predict_time_list)
    y_truth_time = np.array(y_truth_time_list)

    return {'poi': y_predict, 'time': y_predict_time}, {'poi': y_truth, 'time': y_truth_time}