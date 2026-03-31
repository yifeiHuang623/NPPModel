from torch.utils.data import Dataset, DataLoader, Sampler
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd

from utils.register import VIEW_REGISTRY

from typing import Any, Dict, List, Any, Iterator, Tuple, Callable, Optional
from collections import OrderedDict, deque, defaultdict
import math

import os, json
from typing import Set

from torch.utils.data import Sampler
from typing import List, Iterator, Dict
from collections import deque, defaultdict

class UniqueUserBatchSampler(Sampler[List[int]]):
    """
    目标：
    - batch 内 user_id 唯一
    - 不丢样本（drop_last=False）
    - 保证同一用户的 idx 按 dataset 顺序 FIFO 出现（per-user time order）
    
    关键点：
    - 不在 __init__ 缓存 user_ids，__iter__ 时从 dataset.samples 读，避免错配
    - 遇到重复 user：放入 per-user 队列 deferred[uid]（FIFO）
    - 每个 batch 先从主流 buffer 取；取不到再从 deferred 中取，但每个 uid 也只能取一次
    - deferred 的取出顺序按“最早出现的 deferred idx”优先，尽量接近全局顺序
    """
    def __init__(self, dataset, batch_size: int, drop_last: bool = False):
        assert batch_size > 0
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[List[int]]:
        n = len(self.dataset)

        # 动态取 user_ids（避免 dataset 顺序变化导致错配）
        user_ids = [int(s["user_id"]) for s in self.dataset.samples]

        buffer = deque(range(n))                 # 主序列：严格按 dataset 顺序
        deferred: Dict[int, deque] = defaultdict(deque)  # uid -> deque[idx]，FIFO 保证该用户顺序

        while buffer or any(deferred.values()):
            batch: List[int] = []
            seen = set()

            # 1) 先从 buffer 取，保持整体尽量按原顺序
            while buffer and len(batch) < self.batch_size:
                idx = buffer.popleft()
                uid = user_ids[idx]
                if uid in seen:
                    deferred[uid].append(idx)   # 该 uid 的后续样本排队（FIFO）
                else:
                    batch.append(idx)
                    seen.add(uid)

            # 2) 如果 batch 还没满，尝试从 deferred 填充
            #    选择策略：每次拿“所有 deferred 队列头部 idx 中最小的那个”
            #    这样更接近全局顺序；且每个 uid FIFO，不会乱该用户时间顺序
            while len(batch) < self.batch_size:
                # 找到当前可用（uid 不在 seen 且该 uid 队列非空）的最早 idx
                best_uid = None
                best_idx = None
                for uid, q in deferred.items():
                    if not q or uid in seen:
                        continue
                    head = q[0]
                    if best_idx is None or head < best_idx:
                        best_idx = head
                        best_uid = uid

                if best_uid is None:
                    break

                batch.append(deferred[best_uid].popleft())
                seen.add(best_uid)

            # 清理空队列，避免 any(deferred.values()) 反复扫到空
            empty_uids = [uid for uid, q in deferred.items() if not q]
            for uid in empty_uids:
                del deferred[uid]

            if len(batch) == self.batch_size:
                yield batch
            else:
                if not self.drop_last and len(batch) > 0:
                    yield batch

    def __len__(self) -> int:
        return sum(1 for _ in iter(self))

# class UniqueUserBatchSampler(Sampler[List[int]]):
#     """
#     每个 batch 内 user_id 严格唯一。
#     若无法凑满 batch_size（剩余可用的新 user 不足），则输出更小的 batch（batch_size 灵活调整）。
#     不丢样本：drop_last=False 时，所有 index 都会被 yield 一次且仅一次。

#     近似保持原顺序：按 buffer 顺序取样；重复 user 暂存到 stash，等新 batch 再回放。
#     """
#     def __init__(self, dataset: Dataset, batch_size: int, drop_last: bool = False):
#         assert batch_size > 0
#         self.dataset = dataset
#         self.batch_size = batch_size
#         self.drop_last = drop_last

#         self.user_ids = []
#         for i in range(len(dataset)):
#             uid = dataset[i]["user_id"]
#             if isinstance(uid, torch.Tensor):
#                 uid = int(uid.item())
#             self.user_ids.append(int(uid))

#     def __iter__(self) -> Iterator[List[int]]:
#         n = len(self.user_ids)
#         buffer = deque(range(n))
#         stash = defaultdict(deque)  # uid -> indices

#         while buffer or any(stash.values()):
#             batch: List[int] = []
#             seen_users = set()

#             # 为新 batch：先把 stash 回填到 buffer 头部（FIFO），尽量不偏离顺序
#             if any(stash.values()):
#                 for u in list(stash.keys()):
#                     while stash[u]:
#                         buffer.appendleft(stash[u].popleft())
#                     del stash[u]

#             # 填充 batch：能填多少填多少（最多 batch_size），如果填不满就缩小
#             while buffer and len(batch) < self.batch_size:
#                 idx = buffer.popleft()
#                 uid = self.user_ids[idx]

#                 if uid in seen_users:
#                     stash[uid].append(idx)
#                     continue

#                 batch.append(idx)
#                 seen_users.add(uid)

#             # 如果 batch 空，但 buffer 里都是同一个 uid 的重复（极端情况），
#             # 上面逻辑会把它们都进 stash 导致 batch 为空。
#             # 这时为了不死循环，我们允许取 1 个样本组成 batch（仍然唯一）。
#             if not batch and buffer:
#                 idx = buffer.popleft()
#                 batch = [idx]

#             # 输出 batch：若 drop_last=True 且 batch 不满，则丢弃该 batch（但你一般不想）
#             if len(batch) == self.batch_size:
#                 yield batch
#             else:
#                 if not self.drop_last:
#                     yield batch
#                 # drop_last=True 时，这里会丢掉不足 batch_size 的 batch（会变少）

#     def __len__(self) -> int:
#         # 变长 batch 无法用公式给出；提供一个安全的下界或精确值。
#         # 精确值会遍历一次（可能较慢）
#         return sum(1 for _ in iter(self))


def flex_collate(batch):
    all_keys = set().union(*batch)
    collated = {}
    for k in all_keys:
        values = [b.get(k, None) for b in batch]

        if all(isinstance(v, torch.Tensor) for v in values):
            collated[k] = torch.stack(values)

        elif all(isinstance(v, (int, float, np.integer, np.floating)) for v in values):
            collated[k] = torch.tensor(values)

        elif all(isinstance(v, dict) for v in values if v is not None):

            sub_batch = [{**(v or {})} for v in values]
            collated[k] = flex_collate(sub_batch)
        else:
            
            collated[k] = values

    return collated

def _maybe_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (int, float, np.integer, np.floating)):
        return torch.tensor(x, dtype=torch.long)
    if isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.number):
        return torch.as_tensor(x)
    return x   

def _maybe_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (int, float, np.integer, np.floating)):
        return torch.tensor(x, dtype=torch.long)
    if isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.number):
        return torch.as_tensor(x)
    return x


class BaseDataset(Dataset):
    def __init__(
        self,
        preprocess_func: Callable[..., pd.DataFrame] = None,
        df: pd.DataFrame = None,
        view_value: dict = None,
        split_mode: str = "time_interval",
        sequence_length: int = 30,
        fixed_len: int = 20,
        max_gap_seconds: int = 86400,
        n_jobs: int = 1,
        pre_views: List[str] = None,
        post_views: List[str] = None,
        is_train: bool = True,
        context_df: pd.DataFrame = None,
        label_df: pd.DataFrame = None,
        max_recent: int = 100,
        allowed_label_rowids: Set = None
    ) -> None:
        super().__init__()

        # if df is None:
        #     if preprocess_func is None:
        #         raise ValueError("Either preprocess_func or df must be provided.")
        #     raw_df = preprocess_func()
        # else:
        #     raw_df = df

        # view_value = {}
        # if pre_views:
        #     for view in pre_views:
        #         if view not in VIEW_REGISTRY:
        #             raise ValueError(f"View '{view}' not registered")
        #         raw_df, view_value = VIEW_REGISTRY[view](raw_df, view_value)
        raw_df = df

        # 只在这里切轨迹段
        if split_mode == "time_interval":
            samples = split_trajectories_time_interval(
                raw_df, sequence_length=sequence_length, max_gap_seconds=max_gap_seconds, is_train=is_train, allowed_label_rowids=allowed_label_rowids
            )
        elif split_mode == "fixed_length":
            samples = split_trajectories_fixed_length(
                raw_df, fixed_len=fixed_len, drop_last=False, is_train=is_train, allowed_label_rowids=allowed_label_rowids
            )
        elif split_mode == "recent_k":
            samples = split_recent_k_prefix(
                context_df=context_df, label_df=label_df, k=max_recent,  is_train=is_train,allowed_label_rowids=allowed_label_rowids
        )
        else:
            raise ValueError(f"Unknown split_mode: {split_mode}")

        if post_views:
            for view in post_views:
                if view not in VIEW_REGISTRY:
                    raise ValueError(f"View '{view}' not registered")
                samples, view_value = VIEW_REGISTRY[view](samples, view_value)

        self.samples = samples
        self.view_value = view_value

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        ordered = OrderedDict()
        for k, v in s.items():
            ordered[k] = _maybe_tensor(v)
        return ordered
    
class BaseDataLoader:
    def __init__(
        self,
        MyDataset,
        dataset_name,
        logger,
        args=None,
        model_args=None,
        pre_views=None,
        post_views=None,
        enforce_unique_user_in_batch: bool = True,
        drop_last: bool = False,
        split_by_user: bool = True,
        with_context: bool = False
    ) -> None:

        model_args = model_args or {}
        args = args or {}

        # 1) preprocess 出 raw_df（只做一次）
        raw_df = MyDataset.preprocess_func()
        raw_df = raw_df.sort_values("timestamps").reset_index(drop=True)
        raw_df["row_id"] = np.arange(len(raw_df), dtype=np.int64)
        
        view_value = {}
        if pre_views:
            for view in pre_views:
                if view not in VIEW_REGISTRY:
                    raise ValueError(f"View '{view}' not registered")
                raw_df, view_value = VIEW_REGISTRY[view](raw_df, view_value)

        # 2) 先对 df 做 user-wise 8:1:1 划分（在“点”层面）
        if split_by_user:
            train_df, val_df, test_df = userwise_time_split_df_8_1_1(
                raw_df, ratios=(0.8, 0.1, 0.1), min_per_user=3
            )
        else:
            # 全局按时间切 df（不建议但给你留口）
            raw_df = raw_df.sort_values("timestamps").reset_index(drop=True)
            n = len(raw_df)
            n_train = int(0.8 * n)
            n_val = int(0.1 * n)
            train_df = raw_df.iloc[:n_train]
            val_df = raw_df.iloc[n_train:n_train + n_val]
            test_df = raw_df.iloc[n_train + n_val:]

        split_path = f"./data/{dataset_name.split('_')[0]}/{dataset_name.split('_')[0]}_valtest_rowids.json"

        intersection_params = {
            "max_gap_seconds": args.get("max_gap_seconds", model_args.get("max_gap_seconds", 86400)),
            "fixed_len": args.get("fixed_len", model_args.get("fixed_len", 20)),
            "sequence_length": args.get("sequence_length", model_args.get("sequence_length", 30)),
            "recent_k": model_args.get("recent_k", args.get("recent_k", 100)),
        }

        need_rebuild_split = True
        if os.path.exists(split_path):
            loaded = load_valtest_rowids(split_path)
            loaded_params = (loaded.get("meta", {}) or {}).get("params", {})
            if loaded_params == intersection_params:
                allowed_val = loaded["val_rowids"]
                allowed_test = loaded["test_rowids"]
                need_rebuild_split = False
                logger.info("Loaded val/test row_id file: %s (val=%d test=%d)", split_path, len(allowed_val), len(allowed_test))
            else:
                logger.info(
                    "Existing val/test row_id file params mismatch, rebuilding: %s old=%s new=%s",
                    split_path,
                    loaded_params,
                    intersection_params,
                )

        if need_rebuild_split:
            allowed_val, allowed_test, meta = build_valtest_rowids_intersection(
                raw_df=raw_df,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                max_gap_seconds=intersection_params["max_gap_seconds"],
                fixed_len=intersection_params["fixed_len"],
                sequence_length=intersection_params["sequence_length"],
                recent_k=intersection_params["recent_k"],
            )
            save_valtest_rowids(split_path, allowed_val, allowed_test, meta=meta)
            logger.info("Saved val/test row_id file: %s meta=%s", split_path, meta)


        if with_context:
             # 训练集：只用 train_df（每用户一个样本）
            train_dataset = MyDataset(
                context_df=train_df,
                label_df=train_df,
                view_value=view_value,
                k=model_args.get("recent_k", 100),
                pre_views=pre_views, post_views=post_views,
                is_train=True,
            )

            # 验证集：context= train+val，label=val（只评估 val 的点）
            val_context = pd.concat([train_df, val_df], ignore_index=True)
            val_dataset = MyDataset(
                context_df=val_context,
                label_df=val_df,
                view_value=view_value,
                k=model_args.get("recent_k", 100),
                pre_views=pre_views, post_views=post_views,
                is_train=False,
                allowed_label_rowids=allowed_val
            )

            # 测试集：context= train+val+test，label=test
            test_context = pd.concat([train_df, val_df, test_df], ignore_index=True)
            test_dataset = MyDataset(
                context_df=test_context,
                label_df=test_df,
                view_value=view_value,
                k=model_args.get("recent_k", 100),
                pre_views=pre_views, post_views=post_views,
                is_train=False,
                allowed_label_rowids=allowed_test
            )
        else:
            # 3) 再分别在 train/val/test 内切轨迹段生成 samples（满足你的“先划分再切段”）
            train_dataset = MyDataset(df=train_df, view_value=view_value, pre_views=pre_views, post_views=post_views)
            val_dataset   = MyDataset(df=val_df,   view_value=view_value, pre_views=pre_views, post_views=post_views, is_train=False, allowed_label_rowids=allowed_val)
            test_dataset  = MyDataset(df=test_df,  view_value=view_value, pre_views=pre_views, post_views=post_views, is_train=False, allowed_label_rowids=allowed_test)

        train_batch_size = model_args.get("train_batch_size", model_args.get("batch_size", 32))
        val_batch_size   = model_args.get("val_batch_size",   model_args.get("batch_size", 32))
        test_batch_size  = model_args.get("test_batch_size",  model_args.get("batch_size", 32))

        if enforce_unique_user_in_batch:
            train_sampler = UniqueUserBatchSampler(train_dataset, train_batch_size, drop_last=drop_last)
            val_sampler   = UniqueUserBatchSampler(val_dataset,   val_batch_size,   drop_last=drop_last)
            test_sampler  = UniqueUserBatchSampler(test_dataset,  test_batch_size,  drop_last=drop_last)

            self.train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler, shuffle=False, collate_fn=flex_collate)
            self.val_dataloader   = DataLoader(val_dataset,   batch_sampler=val_sampler, shuffle=False,  collate_fn=flex_collate)
            self.test_dataloader  = DataLoader(test_dataset,  batch_sampler=test_sampler, shuffle=False, collate_fn=flex_collate)
        else:
            self.train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=False, collate_fn=flex_collate, drop_last=drop_last)
            self.val_dataloader   = DataLoader(val_dataset,   batch_size=val_batch_size,   shuffle=False, collate_fn=flex_collate, drop_last=drop_last)
            self.test_dataloader  = DataLoader(test_dataset,  batch_size=test_batch_size,  shuffle=False, collate_fn=flex_collate, drop_last=drop_last)

        # view_value：现在每个 split 会各自算一份
        # 通常用 train 的统计（避免泄漏）。
        self.view_value = getattr(train_dataset, "view_value", {})

        logger.info("DataLoader initialized with dataset: %s", dataset_name)
        logger.info("Split (points): train=%d val=%d test=%d", len(train_df), len(val_df), len(test_df))
        logger.info("After trajectory split (samples): train=%d val=%d test=%d",
                    len(train_dataset), len(val_dataset), len(test_dataset))
        

def split_trajectories_time_interval(
    df: pd.DataFrame,
    sequence_length: int,
    max_gap_seconds: int = 86400,
    is_train: bool = True,
    mode: str = "time interval",
    allowed_label_rowids: Optional[Set[int]] = None
) -> List[Dict[str, Any]]:
    """
    模式A：
    1) 按相邻点时间差 > max_gap_seconds 断开成子轨迹
    2) 若某个子轨迹长度为 1，则合并到前一个子轨迹（如果存在）
    3) 训练：每个子轨迹只取 1 个 label（默认：min(end_idx, block_start+sequence_length)）
       验证/测试：前缀预测，label 从子轨迹的第 2 个点一路到最后
    4) allowed_label_rowids：如果提供，则仅当 label_row["row_id"] 在集合中才产出样本
    """
    df = df.sort_values(by="timestamps")
    group_key = ["user_id", "trajectory_id"] if "trajectory_id" in df.columns else ["user_id"]
    groupby_user = df.groupby(group_key, sort=False)

    data_list: List[Dict[str, Any]] = []
    for index, data in tqdm(groupby_user, desc="split_time_interval"):
        user_id = index[0] if isinstance(index, tuple) else np.int64(index)
        data = data.sort_values(by="timestamps")
        cols = data.columns

        if len(data) < 2:
            continue

        # 找断点：相邻时间差 > max_gap_seconds
        ts = data["timestamps"].values
        time_diff = np.diff(ts, prepend=ts[0])
        time_diff[0] = 0
        breakpoints = time_diff > max_gap_seconds
        break_pos = np.where(breakpoints)[0]

        # 初始切段边界： [0, ..., len)
        cuts = np.concatenate(([0], break_pos, [len(data)])).astype(int)

        # 生成初始 segments (start, end_exclusive)
        segments: List[Tuple[int, int]] = []
        for i in range(len(cuts) - 1):
            s, e = int(cuts[i]), int(cuts[i + 1])
            if e > s:
                segments.append((s, e))

        # 合并长度为 1 的 segment 到前一个（若存在）
        merged: List[Tuple[int, int]] = []
        for s, e in segments:
            if (e - s) == 1 and merged:
                ps, pe = merged[-1]
                merged[-1] = (ps, e)  # 扩展前一段到当前段末尾
            else:
                merged.append((s, e))

        # 对每个连续块内部再按 sequence_length 生成样本
        for block_start, block_end_exclusive in merged:
            end_idx = block_end_exclusive - 1

            # block 至少要有 2 个点才能形成 (input -> label)
            if end_idx - block_start < 1:
                continue

            if is_train:
                # 每段只取 1 个 label（保持你原本策略）
                end_candidates = [min(end_idx, block_start + sequence_length)]
            else:
                # 前缀预测：丢弃该子轨迹第一个点作为 label，从第二个点开始
                end_candidates = range(block_start + 1, end_idx + 1)

            for slice_end in end_candidates:
                label_row = data.iloc[slice_end]
                if allowed_label_rowids is not None:
                    rid = int(label_row["row_id"])
                    if rid not in allowed_label_rowids:
                        continue

                # 输入起点：最多 sequence_length，超长截断最近的 L 个
                start_idx = max(block_start, slice_end - sequence_length)

                real_len = slice_end - start_idx  # 输入长度
                if real_len <= 0:
                    continue

                pad_num = sequence_length - real_len

                sample = {
                    "end_ts": int(label_row["timestamps"]),
                    "user_id": int(user_id),
                    "mask": int(real_len),
                    "trajectory_id": len(data_list),
                    "y_POI_id": label_row.to_dict(),
                    "end": 0
                }
                sample["y_POI_id"]["trajectory_id"] = sample["trajectory_id"]

                for col in cols:
                    if col in sample:
                        continue
                    seq = data[col].iloc[start_idx:slice_end].to_numpy()
                    pad_val = 0 if np.issubdtype(seq.dtype, np.number) else None
                    if pad_num > 0:
                        seq = np.pad(seq, (0, pad_num), constant_values=pad_val)
                    sample[col] = seq

                data_list.append(sample)
                
            data_list[-1]["end"] = 1
        # 保持近似时间顺序（你原代码在每个 user 内 sort；这里仍然按 end_ts 排）
        data_list.sort(key=lambda s: s["end_ts"])

    return data_list


def split_trajectories_fixed_length(
    df: pd.DataFrame,
    fixed_len: int = 20,
    drop_last: bool = False,
    is_train: bool = True,
    allowed_label_rowids: Optional[Set[int]] = None
) -> List[Dict[str, Any]]:
    """
    模式B：按固定长度 fixed_len 切分轨迹段（每段输入长度 fixed_len-1，label 是段内最后一个点）
    - 为了和你现有 sample 格式兼容：输入序列长度 = fixed_len-1，pad 到 fixed_len-1
      （如果你希望输入长度就是 fixed_len，也可以改）
    """
    df = df.sort_values(by="timestamps")
    group_key = ["user_id", "trajectory_id"] if "trajectory_id" in df.columns else ["user_id"]
    groupby_user = df.groupby(group_key, sort=False)

    data_list: List[Dict[str, Any]] = []
    for index, data in tqdm(groupby_user, desc="split_fixed_length"):
        user_id = index[0] if isinstance(index, tuple) else np.int64(index)
        data = data.sort_values(by="timestamps")
        cols = data.columns

        n = len(data)

        # chunk 起点为 0, fixed_len, 2*fixed_len, ...
        max_k = math.ceil(n / fixed_len)
        for k in range(max_k):
            start = k * fixed_len
            end_exclusive = min((k + 1) * fixed_len, n)
            chunk = data.iloc[start:end_exclusive]
            if len(chunk) < 2:
                continue
            seq_len = fixed_len - 1  # 模型输入长度（不含 label）

            if is_train:
                # 每个 chunk 只取 1 个样本：label=最后一个点，input=前面所有点(<=seq_len)，不足 pad
                label_pos_list = [len(chunk) - 1]
            else:
                # 前缀预测：label 从第1个点到最后一个点
                label_pos_list = range(1, len(chunk))

            for label_pos in label_pos_list:
                label_row = chunk.iloc[label_pos]
                if allowed_label_rowids is not None:
                    rid = int(label_row["row_id"])
                    if rid not in allowed_label_rowids:
                        continue
                    
                # input 是 label 之前的所有点；若超过 seq_len 则取最近 seq_len
                input_end = label_pos
                input_start = max(0, input_end - seq_len)
                input_part = chunk.iloc[input_start:input_end]

                real_len = len(input_part)
                if real_len <= 0:
                    continue

                pad_num = seq_len - real_len

                sample = {
                    "end_ts": int(label_row["timestamps"]),
                    "user_id": int(user_id),
                    "mask": int(real_len),
                    "trajectory_id": len(data_list),
                    "y_POI_id": label_row.to_dict(),
                    "end": 0
                }
                sample["y_POI_id"]["trajectory_id"] = sample["trajectory_id"]

                for col in cols:
                    if col in sample:
                        continue
                    seq = input_part[col].to_numpy()
                    pad_val = 0 if np.issubdtype(seq.dtype, np.number) else None
                    if pad_num > 0:
                        seq = np.pad(seq, (0, pad_num), constant_values=pad_val)
                    sample[col] = seq

                data_list.append(sample)
            data_list[-1]["end"] = 1

        data_list.sort(key=lambda s: s["end_ts"])
    return data_list

def split_recent_k_prefix(
    context_df: pd.DataFrame,
    label_df: pd.DataFrame,
    k: int = 100,
    is_train: bool = True,
    pad_numeric: float = 0.0,
    pad_object: Any = None,
    allowed_label_rowids: Optional[Set[int]] = None
) -> List[Dict[str, Any]]:
    """
    使用滑动窗口：每个样本输入长度固定为 (k-1)，label 是某个点。
    - Train: 每 group 只做 1 个样本（最后一个点当 label）
    - Val/Test: label 只来自 label_df，但输入上下文来自 context_df（含 train 历史）
    """
    assert "row_id" in context_df.columns and "row_id" in label_df.columns

    context_df = context_df.sort_values("timestamps")
    label_df = label_df.sort_values("timestamps")

    group_key = ["user_id", "trajectory_id"] if "trajectory_id" in context_df.columns else ["user_id"]

    # label row_id 集合：用来过滤“哪些点可以当 label”
    label_row_ids = set(label_df["row_id"].astype(np.int64).tolist())
    if allowed_label_rowids is not None:
        label_row_ids &= set(map(int, allowed_label_rowids))

    data_list: List[Dict[str, Any]] = []
    for index, g in tqdm(context_df.groupby(group_key, sort=False), desc=f"split_recent_{k}_prefix"):
        user_id = index[0] if isinstance(index, tuple) else np.int64(index)
        g = g.sort_values("timestamps")
        cols = g.columns

        if len(g) < 2:
            continue

        if is_train:
            # 只取最后一个点当 label
            label_positions = [len(g) - 1]
        else:
            # 只对 label_df 中的点构造样本；并且 label 至少要有 1 个历史点
            label_positions = []
            for pos in range(1, len(g)):
                rid = int(g.iloc[pos]["row_id"])
                if rid in label_row_ids:
                    label_positions.append(pos)

        seq_len = k - 1

        for pos in label_positions:
            label_row = g.iloc[pos]
            input_end = pos
            input_start = max(0, input_end - seq_len)
            input_part = g.iloc[input_start:input_end]

            real_len = len(input_part)
            if real_len <= 0:
                continue

            pad_num = seq_len - real_len

            sample = {
                "end_ts": int(label_row["timestamps"]),
                "user_id": int(user_id),
                "mask": int(real_len),
                "trajectory_id": len(data_list),
                "y_POI_id": label_row.to_dict(),
                "end": 0
            }
            sample["y_POI_id"]["trajectory_id"] = sample["trajectory_id"]

            for col in cols:
                if col in sample:
                    continue

                seq = input_part[col].to_numpy()

                # pad 到固定长度 seq_len
                if pad_num > 0:
                    if np.issubdtype(seq.dtype, np.number):
                        pad_val = pad_numeric
                    else:
                        pad_val = pad_object
                    seq = np.pad(seq, (0, pad_num), constant_values=pad_val)

                sample[col] = seq

            data_list.append(sample)
        data_list[-1]["end"] = 1

    data_list.sort(key=lambda s: s["end_ts"])
    return data_list

def userwise_time_split_df_8_1_1(
    df: pd.DataFrame,
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    min_per_user: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    先对原始 check-in df 做 user-wise temporal split（每个用户内部按 timestamps 排序后切 8/1/1）。
    返回 train_df, val_df, test_df（保持列不变）。
    """
    r0, r1, r2 = ratios
    assert abs((r0 + r1 + r2) - 1.0) < 1e-6
    assert "user_id" in df.columns and "timestamps" in df.columns

    # 保证时间有序（切分前必须）
    df = df.sort_values("timestamps")

    train_parts = []
    val_parts = []
    test_parts = []

    group_key = ["user_id", "trajectory_id"] if "trajectory_id" in df.columns else ["user_id"]
    for _, g in df.groupby(group_key, sort=False):
        g = g.sort_values("timestamps")
        n = len(g)

        if n < min_per_user:
            # 策略：太短的用户全部放 train（你也可以选择丢弃或全放 train）
            train_parts.append(g)
            continue

        n_val = int(n * r1)
        n_test = int(n * r2)
        n_train = n - n_val - n_test

        train_parts.append(g.iloc[:n_train])
        val_parts.append(g.iloc[n_train:n_train + n_val])
        test_parts.append(g.iloc[n_train + n_val:])

    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else df.iloc[:0].copy()
    val_df   = pd.concat(val_parts, ignore_index=True)   if val_parts else df.iloc[:0].copy()
    test_df  = pd.concat(test_parts, ignore_index=True)  if test_parts else df.iloc[:0].copy()

    # 维持整体时间排序（可选）
    train_df = train_df.sort_values("timestamps").reset_index(drop=True)
    val_df = val_df.sort_values("timestamps").reset_index(drop=True)
    test_df = test_df.sort_values("timestamps").reset_index(drop=True)

    return train_df, val_df, test_df


def save_valtest_rowids(path: str, val_rowids: Set[int], test_rowids: Set[int], meta: Dict[str, Any] = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "val_rowids": sorted(map(int, val_rowids)),
        "test_rowids": sorted(map(int, test_rowids)),
        "meta": meta or {},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def load_valtest_rowids(path: str) -> Dict[str, Set[int]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return {
        "val_rowids": set(map(int, payload.get("val_rowids", []))),
        "test_rowids": set(map(int, payload.get("test_rowids", []))),
        "meta": payload.get("meta", {}),
    }

from typing import Set, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd

def build_valtest_rowids_intersection(
    raw_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    max_gap_seconds: int = 86400,
    fixed_len: int = 20,
    sequence_length: int = 30,
    recent_k: int = 100,
    # 对齐策略：让 val/test 数量一致
    align: str = "min",   # "min" | "val" | "test" | "none"
    # 抽样对齐时的排序规则：尽量保持时间顺序
    sort_by: str = "timestamps",  # "timestamps" | "row_id"
) -> Tuple[Set[int], Set[int], Dict[str, Any]]:
    """
    目标：找“三种样本构造方式”都能产生的 label row_id 的交集，
    并在需要时让 val/test 数量一致（同口径可比）。

    三种方式（你这里的语境）：
    1) time_interval
    2) fixed_length
    3) recent_k_prefix

    返回：
      val_set, test_set: Set[row_id]
      meta: 统计信息
    """
    assert "row_id" in raw_df.columns and "timestamps" in raw_df.columns
    assert "row_id" in train_df.columns and "row_id" in val_df.columns and "row_id" in test_df.columns

    # ---------- 0) 先确定“候选 label 域” ----------
    # 评估时 label 必须来自各自 split 的点集合（你现在也是这么做的）
    val_candidates = set(val_df["row_id"].astype(np.int64).tolist())
    test_candidates = set(test_df["row_id"].astype(np.int64).tolist())

    # ---------- 1) 构造三种方式各自允许的 label row_id 集合 ----------
    # 注意：这里不需要真的生成 samples（很慢），只需要知道哪些 row_id 能作为 label
    # 最稳妥：直接复用你 split 函数的 allowed_label_rowids 机制，跑一遍“只收集label”的轻量版
    # 为了少改你的代码，我这里直接调用 split 但只取 y 的 row_id（如果数据很大建议再优化）

    def _collect_labels_from_samples(samples) -> Set[int]:
        out = set()
        for s in samples:
            y = s.get("y_POI_id", None)
            if isinstance(y, dict) and "row_id" in y:
                out.add(int(y["row_id"]))
        return out

    # --- time_interval: 在 val_df/test_df 内切（与你 DataLoader 的逻辑一致） ---
    time_val = _collect_labels_from_samples(
        split_trajectories_time_interval(
            val_df,
            sequence_length=sequence_length,
            max_gap_seconds=max_gap_seconds,
            is_train=False,
            allowed_label_rowids=val_candidates,  # 只让 val 点当 label
        )
    )
    time_test = _collect_labels_from_samples(
        split_trajectories_time_interval(
            test_df,
            sequence_length=sequence_length,
            max_gap_seconds=max_gap_seconds,
            is_train=False,
            allowed_label_rowids=test_candidates,  # 只让 test 点当 label
        )
    )

    # --- fixed_length ---
    fixed_val = _collect_labels_from_samples(
        split_trajectories_fixed_length(
            val_df,
            fixed_len=fixed_len,
            drop_last=False,
            is_train=False,
            allowed_label_rowids=val_candidates,
        )
    )
    fixed_test = _collect_labels_from_samples(
        split_trajectories_fixed_length(
            test_df,
            fixed_len=fixed_len,
            drop_last=False,
            is_train=False,
            allowed_label_rowids=test_candidates,
        )
    )

    # --- recent_k_prefix: 评估时 context 用 train+val / train+val+test，但 label 必须来自 val/test ---
    val_context = pd.concat([train_df, val_df], ignore_index=True)
    test_context = pd.concat([train_df, val_df, test_df], ignore_index=True)

    recent_val = _collect_labels_from_samples(
        split_recent_k_prefix(
            context_df=val_context,
            label_df=val_df,
            k=recent_k,
            is_train=False,
            allowed_label_rowids=val_candidates,
        )
    )
    recent_test = _collect_labels_from_samples(
        split_recent_k_prefix(
            context_df=test_context,
            label_df=test_df,
            k=recent_k,
            is_train=False,
            allowed_label_rowids=test_candidates,
        )
    )

    # ---------- 2) 三种方式交集 ----------
    val_set = time_val & fixed_val & recent_val
    test_set = time_test & fixed_test & recent_test

    meta = {
        "candidates": {"val": len(val_candidates), "test": len(test_candidates)},
        "time": {"val": len(time_val), "test": len(time_test)},
        "fixed": {"val": len(fixed_val), "test": len(fixed_test)},
        "recent": {"val": len(recent_val), "test": len(recent_test)},
        "intersection_after_align": {"val": len(val_set), "test": len(test_set)},
        "align": align,
        "sort_by": sort_by,
        "params": {
            "max_gap_seconds": max_gap_seconds,
            "fixed_len": fixed_len,
            "sequence_length": sequence_length,
            "recent_k": recent_k,
        },
    }

    return val_set, test_set, meta
