from operator import ne
import pandas as pd
import random
from torch.utils.data import Dataset
import numpy as np

from utils.register import register_view
from utils.logger import get_logger
from datetime import datetime

logger = get_logger(__name__)

@register_view("causal_preview")
def causal_preview(raw_df: pd.DataFrame, view_value: dict = None) -> tuple[Dataset, dict]:
    from model.causal.causal_utils import get_norm_time96, get_day_norm7, get_time_slot_id, get_all_permutations_dict, get_quad_keys
    """
    A preprocessing view for causal that prepares the dataset for training.
    """
    logger.info("Applying causal_preview to dataset")
    
    num_users = raw_df['user_id'].nunique()
    num_pois = raw_df['POI_id'].nunique()
    view_value['num_users'] = num_users + 1
    view_value['num_pois'] = num_pois + 1
    
    raw_df['norm_time'] = raw_df['timestamps'].apply(get_norm_time96)
    raw_df['day_time'] = raw_df['timestamps'].apply(get_day_norm7)
    raw_df['time_id'] = raw_df['timestamps'].apply(lambda x: 
        (datetime.fromtimestamp(x).weekday() * 24 + datetime.fromtimestamp(x).hour) / (24 * 7))
    view_value['num_times'] = raw_df['time_id'].nunique() + 1
    
    permutations_dict = get_all_permutations_dict(6)
    raw_df['quad_key'] = raw_df.apply(lambda row: get_quad_keys(row['latitude'], row['longitude'], permutations_dict), axis=1)
    
    return raw_df, view_value

@register_view("causal_post_view")
def causal_post_view(raw_df: pd.DataFrame, view_value: dict = None) -> tuple[Dataset, dict]:
    """
    A preprocessing view for causal.
    """
    logger.info("Applying causal_post_view to dataset")

    for seq_data in raw_df:        
        quad_key = seq_data['quad_key']
        new_quad_key = []
        for quad in quad_key:
            if isinstance(quad, list):
                new_quad_key.append(quad)
            else:
                new_quad_key.append([0] * len(quad_key[0]))
        new_quad_key = np.array(new_quad_key, dtype=np.int64)
        seq_data['quad_key'] = new_quad_key
    
    return raw_df, view_value