import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import statistics
from datetime import datetime
from .causal_utils import rotate, rotate_batch, damped_rotate, damped_rotate_batch
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.utils.rnn import pad_sequence


class GPSEncoder(nn.Module):
    def __init__(self, embed_size, nhead, nhid, nlayers, dropout) -> None:
        super(GPSEncoder, self).__init__()
        encoder_layers = TransformerEncoderLayer(embed_size, nhead, nhid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embed_size = embed_size
        self.norm = nn.LayerNorm(embed_size)

    # s*l*d
    def forward(self, src, mask=None, src_key_mask=None):
        src = src * math.sqrt(self.embed_size)
        x = self.transformer_encoder(src, mask=mask, src_key_padding_mask=src_key_mask)
        x = torch.mean(x, -2)
        return self.norm(x)


def t2v(tau, f, out_features, w, b, w0, b0, arg=None):
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    return torch.cat([v1, v2], 1)


class SineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.sin

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)


class RightPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=600):
        super(RightPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return self.dropout(x)


class CosineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.cos

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)


class OriginTime2Vec(nn.Module):
    def __init__(self, activation, out_dim):
        super(OriginTime2Vec, self).__init__()
        if activation == "sin":
            self.l1 = SineActivation(1, out_dim)
        elif activation == "cos":
            self.l1 = CosineActivation(1, out_dim)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x):
        fea = x.view(-1, 1)
        return self.l1(fea)


class causal(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.device = args.device

        self.user_embed_dim = args.user_embed_dim
        self.poi_embed_dim = args.poi_embed_dim
        self.time_embed_dim = args.time_embed_dim
        self.gps_embed_dim = args.gps_embed_dim

        self.gps_embed_model = nn.Embedding(num_embeddings=4096, embedding_dim=self.gps_embed_dim)
        self.user_embed_model = nn.Embedding(num_embeddings=args.num_users, embedding_dim=self.user_embed_dim)
        self.poi_embed_model = nn.Embedding(num_embeddings=args.num_pois, embedding_dim=self.poi_embed_dim)

        self.total_embed_dim = self.user_embed_dim + self.poi_embed_dim
        self.rot_dim = int(0.5 * self.total_embed_dim)

        # rot_dim 必须为偶数
        if self.rot_dim % 2 != 0:
            self.rot_dim -= 1

        self.time_embed_model_user = OriginTime2Vec('sin', self.rot_dim)
        self.time_embed_model_user_tgt = OriginTime2Vec('sin', self.rot_dim)

        self.time_embed_model_user_day = OriginTime2Vec('sin', self.rot_dim)
        self.time_embed_model_user_day_tgt = OriginTime2Vec('sin', self.rot_dim)

        self.n_head = args.transformer_nhead
        self.dropout = args.transformer_dropout
        self.n_layers = args.transformer_nlayers
        self.hidden_size = args.transformer_nhid

        self.pos_encoder1 = RightPositionalEncoding(self.total_embed_dim, self.dropout)

        encoder_layers1 = TransformerEncoderLayer(
            self.total_embed_dim,
            self.n_head,
            self.hidden_size,
            self.dropout,
            batch_first=True
        )
        self.transformer_encoder1 = TransformerEncoder(encoder_layers1, self.n_layers)

        self.decoder_poi1 = nn.Linear(self.user_embed_dim + 2 * self.poi_embed_dim, args.num_pois)

        self.criterion_poi = nn.CrossEntropyLoss(ignore_index=0)

        # -----------------------------
        # 新增：阻尼参数
        # -----------------------------
        self.time_decay = nn.Parameter(torch.tensor(0.1, dtype=torch.float))
        self.day_decay = nn.Parameter(torch.tensor(0.05, dtype=torch.float))

        # hour / day 自适应融合
        self.hour_day_fusion = nn.Parameter(torch.tensor([0.7, 0.3], dtype=torch.float))

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder_poi1.bias.data.zero_()
        self.decoder_poi1.weight.data.uniform_(-initrange, initrange)

    def compute_poi_prob(
        self,
        src1,
        src_mask,
        target_hour,
        target_day,
        history_time,
        history_day,
        target_time,
        target_day_value,
        poi_embeds,
        src_key_mask
    ):
        src1 = src1 * math.sqrt(self.total_embed_dim)
        src1 = self.pos_encoder1(src1)
        src1 = self.transformer_encoder1(src=src1, mask=src_mask, src_key_padding_mask=src_key_mask)

        # 相对时间差：目标时刻 vs 历史时刻
        delta_time = torch.abs(target_time - history_time).unsqueeze(-1)   # [B, L, 1]
        delta_day = torch.abs(target_day_value - history_day).unsqueeze(-1)

        src1_hour = damped_rotate_batch(
            src1,
            target_hour[:, :, :self.rot_dim],
            self.rot_dim,
            self.time_decay,
            delta_t=delta_time,
            device=self.device
        )

        src1_day = damped_rotate_batch(
            src1,
            target_day[:, :, :self.rot_dim],
            self.rot_dim,
            self.day_decay,
            delta_t=delta_day,
            device=self.device
        )

        fusion_weight = torch.softmax(self.hour_day_fusion, dim=0)
        src1 = fusion_weight[0] * src1_hour + fusion_weight[1] * src1_day

        src1 = torch.cat((src1, poi_embeds), dim=-1)

        out_poi_prob1 = self.decoder_poi1(src1)
        out_poi_prob = out_poi_prob1

        return out_poi_prob

    def handle_sequence(self, batch_data):
        poi_id, norm_time, day_time = batch_data['POI_id'], batch_data['norm_time'], batch_data['day_time']
        user_id = batch_data['user_id'].unsqueeze(dim=1).expand(poi_id.shape[0], poi_id.shape[1])

        seq_len = batch_data['mask']

        y_poi_id = batch_data['y_POI_id']['POI_id']
        y_norm_time = batch_data['y_POI_id']['norm_time']
        y_day_time = batch_data['y_POI_id']['day_time']

        B, L = batch_data['POI_id'].size()
        y_poi_seq = torch.full((B, L), 0, dtype=torch.long, device=self.device)
        y_norm_time_seq = torch.full((B, L), 0, dtype=torch.float, device=self.device)
        y_day_time_seq = torch.full((B, L), 0, dtype=torch.float, device=self.device)

        for i in range(B):
            end = seq_len[i].item()
            if end > 0:
                y_poi_seq[i, :end] = torch.cat((poi_id[i, 1:end], y_poi_id[i].unsqueeze(dim=-1)), dim=-1)
                y_norm_time_seq[i, :end] = torch.cat((norm_time[i, 1:end], y_norm_time[i].unsqueeze(dim=-1)), dim=-1)
                y_day_time_seq[i, :end] = torch.cat((day_time[i, 1:end], y_day_time[i].unsqueeze(dim=-1)), dim=-1)

        batch_data['user_id'] = user_id
        batch_data['y_POI_id']['POI_id'] = y_poi_seq
        batch_data['y_POI_id']['norm_time'] = y_norm_time_seq
        batch_data['y_POI_id']['day_time'] = y_day_time_seq

        mask = batch_data['mask']
        indices = torch.arange(user_id.shape[1]).unsqueeze(0).expand(user_id.shape[0], user_id.shape[1]).to(self.device)
        mask = (indices >= mask.unsqueeze(1)).to(torch.bool)
        batch_data['mask'] = mask

        return batch_data

    def get_predict(self, batch_data):
        batch_data = self.handle_sequence(batch_data)

        seq_len = batch_data['POI_id'].size(1)

        # causal mask: 上三角可见
        mask = torch.triu(torch.ones(seq_len, seq_len, device=self.device)).transpose(0, 1)
        src_mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

        x1, batch_target_time_embed, batch_target_day_embed, poi_embeds_padded = self.get_rotation_and_loss(
            batch_data, src_mask, batch_data['mask']
        )

        y_poi = batch_data['y_POI_id']['POI_id']

        y_pred_poi = self.compute_poi_prob(
            x1,
            src_mask,
            batch_target_time_embed,
            batch_target_day_embed,
            history_time=batch_data['norm_time'].to(torch.float),
            history_day=batch_data['day_time'].to(torch.float),
            target_time=batch_data['y_POI_id']['norm_time'].to(torch.float),
            target_day_value=batch_data['y_POI_id']['day_time'].to(torch.float),
            poi_embeds=poi_embeds_padded,
            src_key_mask=batch_data['mask']
        )

        return y_pred_poi, y_poi

    def forward(self, batch_data):
        y_pred_poi, y_poi = self.get_predict(batch_data)
        loss_poi = self.criterion_poi(y_pred_poi.transpose(1, 2), y_poi)
        return loss_poi

    def predict(self, batch_data):
        y_pred_poi, _ = self.get_predict(batch_data)

        # original mask: False False False True True ...
        valid_len = y_pred_poi.shape[1] - torch.sum(batch_data['mask'], dim=-1)
        batch_indices = torch.arange(valid_len.shape[0], device=self.device)
        y_pred = y_pred_poi[batch_indices, valid_len - 1]
        return y_pred

    def get_rotation_and_loss(self, batch_data, mask, src_key_mask):
        # Parse sample
        u_id = batch_data['user_id']
        poi_id = batch_data['POI_id']
        time = batch_data['norm_time'].to(torch.float)
        day_time = batch_data['day_time'].to(torch.float)
        gps_id = batch_data['quad_key']

        target_time = batch_data['y_POI_id']['norm_time'].to(torch.float)
        target_day_time = batch_data['y_POI_id']['day_time'].to(torch.float)

        # embeddings
        user_embeddings = self.user_embed_model(u_id)
        seq_poi_embeddings = self.poi_embed_model(poi_id)
        poi_embeds = seq_poi_embeddings

        user_times = self.time_embed_model_user(time).reshape(poi_id.shape[0], poi_id.shape[1], -1)
        user_day_times = self.time_embed_model_user_day(day_time).reshape(poi_id.shape[0], poi_id.shape[1], -1)

        user_next_times = self.time_embed_model_user_tgt(target_time).reshape(poi_id.shape[0], poi_id.shape[1], -1)
        user_next_day_times = self.time_embed_model_user_day_tgt(target_day_time).reshape(poi_id.shape[0], poi_id.shape[1], -1)

        # base input
        user_embeddings = torch.cat((user_embeddings, seq_poi_embeddings), dim=-1)

        # 相对时间差：目标时间 vs 历史时间
        delta_time = torch.abs(target_time - time).unsqueeze(-1)      # [B, L, 1]
        delta_day = torch.abs(target_day_time - day_time).unsqueeze(-1)

        user_rotate_hour = damped_rotate(
            user_embeddings,
            user_times,
            self.rot_dim,
            self.time_decay,
            delta_t=delta_time,
            device=self.device
        )

        user_rotate_day = damped_rotate(
            user_embeddings,
            user_day_times,
            self.rot_dim,
            self.day_decay,
            delta_t=delta_day,
            device=self.device
        )

        fusion_weight = torch.softmax(self.hour_day_fusion, dim=0)
        user_rotate = fusion_weight[0] * user_rotate_hour + fusion_weight[1] * user_rotate_day

        seq_embedding1 = user_rotate
        seq_embedding3 = user_next_times
        seq_embedding4 = user_next_day_times

        return seq_embedding1, seq_embedding3, seq_embedding4, poi_embeds