from operator import ne
import pandas as pd
import random
from torch.utils.data import Dataset
import numpy as np

from utils.register import register_view
from utils.logger import get_logger

logger = get_logger(__name__)

@register_view("LoTNext_preview")
def LoTNext_preview(raw_df: pd.DataFrame, view_value: dict = None) -> tuple[Dataset, dict]:
    """
    A preprocessing view for LoTNext that prepares the dataset for training.
    """
    logger.info("Applying LoTNext_preview to dataset")
    
    num_users = raw_df['user_id'].nunique()
    num_pois = raw_df['POI_id'].nunique()
    num_poi_types = raw_df['POI_catid'].nunique()
    view_value['num_users'] = num_users + 1
    view_value['num_pois'] = num_pois + 1
    view_value['num_poi_types'] = num_poi_types + 1
    
    raw_df['time_slot'] = (pd.to_datetime(raw_df['timestamps'], unit='s').dt.weekday * 24 +
                       pd.to_datetime(raw_df['timestamps'], unit='s').dt.hour)
    
    return raw_df, view_value
