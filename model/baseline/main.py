import random
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import yaml
import pandas as pd
import numpy as np
import shutil, os

from utils.logger import get_logger
from types import SimpleNamespace

from .baseline import baseline
from utils.EarlyStopping import EarlyStopping
from tqdm import tqdm
from utils.register import EVAL_REGISTRY
from timm.scheduler.cosine_lr import CosineLRScheduler
from .baseline_utils import drift_report, PandasEncoder
import json
from collections import defaultdict

log = get_logger(__name__)

pre_views = ['baseline_preview']
post_views = []  

model = None

torch.set_num_threads(1)

def dict_dataloader_to_df(dataloader):
    trajectory_id = 0
    trajectory_list, user_id_list, POI_id_list, latitude_list, longitude_list, timestamp_list = [], [], [], [], [], []
    
    for batch in dataloader:
        bs = len(next(iter(batch.values())))
        for i in range(bs):
            data_len = batch['mask'][i]
            trajectory_list.extend([trajectory_id] * data_len)
            user_id_list.extend([batch['user_id'][i].item()] * data_len)
            POI_id_list.extend(batch['POI_id'][i][:data_len].tolist())
            latitude_list.extend(batch['latitude'][i][:data_len].tolist())
            longitude_list.extend(batch['longitude'][i][:data_len].tolist())
            timestamp_list.extend(batch['timestamps'][i][:data_len].tolist())
            trajectory_id += 1
            
    records = pd.DataFrame({
        'trajectory_id': trajectory_list,
        'user_id': user_id_list,
        'POI_id': POI_id_list,
        'latitude': latitude_list,
        'longitude': longitude_list,
        'timestamps': timestamp_list
    })    
    return records

def train(dataloader: dict[DataLoader], dataset_name, **kv):
    global model
    
    with open('./model/baseline/baseline.yaml', 'r', encoding='utf-8') as f:
        args = yaml.safe_load(f)
    args.update(dataloader.view_value)
    args = SimpleNamespace(**args)
    
    train_dataloader,  val_dataloader, test_dataloader = dataloader.train_dataloader, dataloader.val_dataloader, dataloader.test_dataloader
    
    model = baseline(args)
    
    metric_keys = (list(EVAL_REGISTRY))
    
    args.save_model_name = f'{args.model_name}'
    save_model_folder = f"./saved_models/{dataset_name}/{args.save_model_name}/"
    shutil.rmtree(save_model_folder, ignore_errors=True)
    os.makedirs(save_model_folder, exist_ok=True)
    args.save_model_folder = save_model_folder
    
    # user_id_dict = defaultdict(list)
    # for item in val_dataloader:
    #     for i in range(len(item["POI_id"])):
    #         user_id_dict[item["user_id"][i].item()].append((item["POI_id"][i], item["timestamps"][i], item["y_POI_id"]["POI_id"][i])) 
    # print(user_id_dict[1])
    # exit()
    
    train_df = dict_dataloader_to_df(train_dataloader)
    model.fit_from_train_df(train_df)
    # test_df = dict_dataloader_to_df(test_dataloader)
    
    # statistics = drift_report(train_df, test_df)
    # with open(f'./statistics/{dataset_name}/statistics.json', 'w', encoding='utf-8') as f:
    #     json.dump(statistics, f, cls=PandasEncoder, ensure_ascii=False, indent=2)
    # exit()

def inference(dataloader: DataLoader, **kv):
    global model
    log.info('=============begin valid/test==============')
    
    if kv.get('model_para', False):   
        with open('./model/baseline/baseline.yaml', 'r', encoding='utf-8') as f:
            args = yaml.safe_load(f)
        args.update(kv.get('view_value', {}))
        args = SimpleNamespace(**args)
        log.info('=============args==============')
        log.info(args)
        model = baseline(args).to(args.device)
        model.load_state_dict(kv['model_para'])
    
    model.eval()
    y_predict_list = []
    
    with torch.no_grad():
        y_predict_list, y_truth_list = [], []
        for batch_data in tqdm(dataloader):
            batch_data_device = {k: v.to(model.device, non_blocking=True) for k, v in batch_data.items() if 'y_POI_id' not in k}
            batch_data_device['y_POI_id'] = {k: v.to(model.device, non_blocking=True) for k, v in batch_data['y_POI_id'].items()}
            
            y_predict = model.predict(batch_data_device, 'user_frequent')
            y_predict_list.extend(y_predict.detach().cpu().numpy().tolist())
            y_truth_list.extend(batch_data['y_POI_id']['POI_id'].detach().cpu().numpy().tolist())
            
    y_predict = np.array(y_predict_list)
    y_truth = np.array(y_truth_list)
    
    return y_predict, y_truth