import numpy as np
import pandas as pd
from datetime import datetime
from torch.utils.data import Dataset

from utils.logger import get_logger
from utils.register import register_view

logger = get_logger(__name__)


@register_view("moe_preview")
def moe_preview(raw_df: pd.DataFrame, view_value: dict = None) -> tuple[Dataset, dict]:
    from model.causal.causal_utils import build_region_id

    logger.info("Applying moe_preview to dataset")

    num_users = raw_df["user_id"].nunique()
    num_pois = raw_df["POI_id"].nunique()
    num_poi_cats = raw_df["POI_catid"].nunique()
    view_value["num_users"] = num_users + 1
    view_value["num_pois"] = num_pois + 1
    view_value["num_poi_cats"] = num_poi_cats + 1

    raw_df["time_id"] = raw_df["timestamps"].apply(
        lambda x: datetime.fromtimestamp(x).weekday() * 24 + datetime.fromtimestamp(x).hour + 1
    )
    view_value["num_times"] = raw_df["time_id"].nunique() + 1

    region_id_dict = build_region_id(raw_df, num_clusters=300)
    view_value["poi2region"] = region_id_dict
    raw_df["region_id"] = raw_df["POI_id"].map(region_id_dict)
    view_value["num_regions"] = raw_df["region_id"].nunique() + 1

    return raw_df, view_value


@register_view("moe_post_view")
def moe_post_view(raw_df: pd.DataFrame, view_value: dict = None) -> tuple[Dataset, dict]:
    logger.info("Applying moe_post_view to dataset")

    for seq_data in raw_df:
        timestamps = np.asarray(seq_data["timestamps"])
        valid_len = int(seq_data["mask"])
        time_delta = np.zeros_like(timestamps, dtype=np.float32)

        if valid_len > 0:
            valid_ts = timestamps[:valid_len].astype(np.float64)
            delta_hours = np.diff(valid_ts, prepend=valid_ts[0]) / 3600.0
            time_delta[:valid_len] = np.clip(delta_hours, a_min=0.0, a_max=None).astype(np.float32)
            target_ts = float(seq_data["y_POI_id"]["timestamps"])
            target_delta = max((target_ts - valid_ts[valid_len - 1]) / 3600.0, 0.0)
        else:
            target_delta = 0.0

        seq_data["time_delta"] = time_delta
        seq_data["y_POI_id"]["time_delta"] = np.float32(target_delta)

    return raw_df, view_value
