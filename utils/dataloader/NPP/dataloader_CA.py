# dataloader_CA.py
import torch
from pathlib import Path
import pandas as pd

from utils.register import register_dataloader
from utils.logger import get_logger
from utils.dataloader.NPP.dataloader_base import BaseDataset, BaseDataLoader
from utils.exargs import ConfigResolver

logger = get_logger(__name__)
dataset_name = "CA"
root_path = Path(__file__).resolve().parent.parent.parent.parent
args = ConfigResolver(f"{root_path}/data/{dataset_name}/{dataset_name}.yaml").parse()


def pre_process_func() -> pd.DataFrame:
    df = pd.read_csv(f"{root_path}/data/{dataset_name}/{dataset_name}.csv")
    df = df[["UTCTimeOffsetEpoch", "POI_id", "user_id", "latitude", "longitude", "POI_catid_code"]]

    df["POI_id"] = pd.factorize(df["POI_id"])[0] + 1
    df["user_id"] = pd.factorize(df["user_id"])[0] + 1
    df["POI_catid"] = pd.factorize(df["POI_catid_code"])[0] + 1

    df["timestamps"] = pd.to_numeric(df["UTCTimeOffsetEpoch"], errors="coerce").astype("int64")
    df = df.drop(columns=["UTCTimeOffsetEpoch"])

    df["user_id"] = df["user_id"].astype("int64")
    df["POI_id"] = df["POI_id"].astype("int64")
    df["POI_catid"] = df["POI_catid"].astype("int64")
    df = df.sort_values("timestamps")
    return df


class MyDatasetTimeInterval(BaseDataset):
    preprocess_func = staticmethod(pre_process_func)  # 新增这一行
    split_mode = "time_interval"

    def __init__(self, df=None, pre_views=None, view_value={}, post_views=None, is_train=True, allowed_label_rowids=None) -> None:
        super().__init__(
            preprocess_func=pre_process_func,
            df=df,
            view_value=view_value,
            split_mode="time_interval",
            sequence_length=args.get("sequence_length", 30),
            max_gap_seconds=86400,
            n_jobs=args.get("n_jobs", 1),
            pre_views=pre_views,
            post_views=post_views,
            is_train=is_train,
            allowed_label_rowids=allowed_label_rowids
        )

class MyDatasetFixedLength(BaseDataset):
    preprocess_func = staticmethod(pre_process_func)
    split_mode = "fixed_length"

    def __init__(self, df=None, pre_views=None, view_value={}, post_views=None, is_train=True, allowed_label_rowids=None) -> None:
        super().__init__(
            preprocess_func=pre_process_func,
            df=df,
            view_value=view_value,
            split_mode="fixed_length",
            fixed_len=args.get("fixed_len", 20),
            n_jobs=args.get("n_jobs", 1),
            pre_views=pre_views,
            post_views=post_views,
            is_train=is_train,
            allowed_label_rowids=allowed_label_rowids
        )
        
class MyDatasetRecentK(BaseDataset):
    preprocess_func = staticmethod(pre_process_func)
    split_mode = "recent_k"

    def __init__(self, df=None, pre_views=None, view_value={}, post_views=None, is_train=True, context_df=None, label_df=None, k=100, allowed_label_rowids=None) -> None:
        super().__init__(
            preprocess_func=pre_process_func,
            df=df,
            view_value=view_value,
            split_mode="recent_k",
            n_jobs=args.get("n_jobs", 1),
            pre_views=pre_views,
            post_views=post_views,
            is_train=is_train,
            context_df=context_df,
            label_df=label_df,
            max_recent=k,
            allowed_label_rowids=allowed_label_rowids
        )


@register_dataloader(name="CA_time_interval")
class CA_TimeInterval_DataLoader(BaseDataLoader):
    def __init__(self, model_args=None, pre_views=None, post_views=None) -> None:
        super().__init__(
            MyDataset=MyDatasetTimeInterval,
            dataset_name="CA_time_interval",
            logger=logger,
            args=args,
            model_args=model_args or {},
            pre_views=pre_views,
            post_views=post_views,
            enforce_unique_user_in_batch=True,
            drop_last=False,
        )

@register_dataloader(name="CA_fixed_length")
class CA_FixedLength_DataLoader(BaseDataLoader):
    def __init__(self, model_args=None, pre_views=None, post_views=None) -> None:
        super().__init__(
            MyDataset=MyDatasetFixedLength,
            dataset_name="CA_fixed_length",
            logger=logger,
            args=args,
            model_args=model_args or {},
            pre_views=pre_views,
            post_views=post_views,
            enforce_unique_user_in_batch=True,
            drop_last=False,
        )

@register_dataloader(name="CA_recent_k")
class CA_Recent_DataLoader(BaseDataLoader):
    def __init__(self, model_args=None, pre_views=None, post_views=None) -> None:
        super().__init__(
            MyDataset=MyDatasetRecentK,
            dataset_name="CA_recent_k",
            logger=logger,
            args=args,
            model_args=model_args or {},
            pre_views=pre_views,
            post_views=post_views,
            enforce_unique_user_in_batch=True,
            drop_last=False,
            with_context=True
        )

if __name__ == "__main__":
    d = CA_TimeInterval_DataLoader(model_args={"batch_size": 32})
    for batch in d.train_dataloader:
        # 验证 batch 内 user_id 唯一
        u = batch["user_id"]
        if isinstance(u, torch.Tensor):
            assert len(set(u.tolist())) == len(u.tolist())
        print("ok batch")
        break