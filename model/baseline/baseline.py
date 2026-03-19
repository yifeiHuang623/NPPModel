import torch
import torch.nn as nn
import pandas as pd

class baseline(nn.Module):

    def __init__(self, args):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.device = args.device
        self.num_users = args.num_users
        self.num_pois = args.num_pois

        # 统计表（用 register_buffer，能跟着 .to(device)，也会进 state_dict 但不参与梯度）
        self.register_buffer("global_poi_cnt", torch.zeros(self.num_pois, dtype=torch.long))
        self.register_buffer("user_poi_cnt", torch.zeros(self.num_users, self.num_pois, dtype=torch.long))

        self.global_poi_cnt = self.global_poi_cnt.to(self.device)
        self.user_poi_cnt = self.user_poi_cnt.to(self.device)
        # 可选：避免推荐 padding=0
        self.pad_poi_id = 0
        
    @torch.no_grad()
    def fit_from_train_df(self, train_df: pd.DataFrame):
        """
        train_df 至少包含: user_id, POI_id
        注意：请用训练集统计，别用全量数据避免泄漏。
        """
        # 全局热门
        g = train_df["POI_id"].value_counts()
        for poi, cnt in g.items():
            if 0 <= int(poi) < self.num_pois:
                self.global_poi_cnt[int(poi)] = int(cnt)

        # 用户常去
        # groupby 后逐个累加（省内存写法；user_poi_cnt 很大时你可能要稀疏化）
        for (u, p), cnt in train_df.groupby(["user_id", "POI_id"]).size().items():
            u = int(u); p = int(p)
            if 0 <= u < self.num_users and 0 <= p < self.num_pois:
                self.user_poi_cnt[u, p] = int(cnt)
    
    @torch.no_grad()
    def predict(self, batch_data, mode: str = "popularity"):
        """
        mode:
          - "popularity": 全局热门 POI
          - "user_frequent": 用户历史最常去 POI
        return:
          y_pred: [B, num_pois]，分数越大越靠前
        """
        # batch_data['user_id'] 需要是 shape [B] 的 LongTensor
        user_ids = batch_data["user_id"].to(self.device).long()
        B = user_ids.size(0)

        if mode == "popularity":
            # 每个用户都用同一套全局分数
            y_pred = self.global_poi_cnt.float().unsqueeze(0).expand(B, -1).clone()

        elif mode == "user_frequent":
            # 每个用户用自己的访问频次
            y_pred = self.user_poi_cnt[user_ids].float().clone() + self.global_poi_cnt.float().unsqueeze(0).expand(B, -1).clone() * 0.001

            # 可选：如果用户从未出现（冷启动），退化成全局热门
            cold = (self.user_poi_cnt[user_ids].sum(dim=1) == 0)  # [B]
            if cold.any():
                y_pred[cold] = self.global_poi_cnt.float().unsqueeze(0).expand(cold.sum(), -1)

        else:
            raise ValueError(f"Unknown mode: {mode}")

        # 不推荐 padding=0（以及可选：不推荐已mask的 POI）
        y_pred[:, self.pad_poi_id] = -1e9

        return y_pred