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
    from model.causal.causal_utils import build_region_id
    """
    A preprocessing view for causal that prepares the dataset for training.
    """
    logger.info("Applying causal_preview to dataset")
    
    num_users = raw_df['user_id'].nunique()
    num_pois = raw_df['POI_id'].nunique()
    num_poi_cats = raw_df['POI_catid'].nunique()
    view_value['num_users'] = num_users + 1
    view_value['num_pois'] = num_pois + 1
    view_value['num_poi_cats'] = num_poi_cats + 1
    
    raw_df['time_id'] = raw_df['timestamps'].apply(lambda x: 
        datetime.fromtimestamp(x).weekday() * 24 + datetime.fromtimestamp(x).hour + 1)
    view_value['num_times'] = raw_df['time_id'].nunique() + 1
    
    region_id_dict = build_region_id(raw_df, num_clusters=300)
    view_value['poi2region'] = region_id_dict
    raw_df['region_id'] = raw_df['POI_id'].map(region_id_dict)
    view_value['num_regions'] = raw_df['region_id'].nunique() + 1

    return raw_df, view_value