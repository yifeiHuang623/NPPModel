import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from .ROTAN_utils import rotate, rotate_batch


def t2v(tau, f, out_features, w, b, w0, b0, arg=None):
    if tau.dim() == 1:
        tau = tau.unsqueeze(-1)
    if arg is not None:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    return torch.cat([v1, v2], dim=-1)


class SineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.Parameter(torch.randn(1))
        self.w = nn.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.Parameter(torch.randn(out_features - 1))
        self.f = torch.sin

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)


class CosineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.Parameter(torch.randn(1))
        self.w = nn.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.Parameter(torch.randn(out_features - 1))
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
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        elif x.dim() == 2:
            x = x.unsqueeze(-1)
        return self.l1(x)


class RightPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=600):
        super(RightPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].detach()
        return self.dropout(x)


class DeltaBucketizer(nn.Module):
    def __init__(self, boundaries=None):
        super().__init__()
        if boundaries is None:
            boundaries = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0]
        self.register_buffer("boundaries", torch.tensor(boundaries, dtype=torch.float))

    def forward(self, delta_t):
        if delta_t is None:
            return None
        return torch.bucketize(delta_t, self.boundaries)


class IntervalAdaptiveGate(nn.Module):
    def __init__(self, embed_dim, bucket_num=9, bucket_emb_dim=16, hidden_dim=None, dropout=0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim

        self.bucket_emb = nn.Embedding(bucket_num, bucket_emb_dim)
        self.cont_proj = nn.Linear(1, bucket_emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(bucket_emb_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, base_x, rotated_x, delta_t, delta_bucket=None):
        if delta_t is None:
            gate = 0.5
            return gate * rotated_x + (1.0 - gate) * base_x

        if delta_t.dim() == 1:
            delta_t = delta_t.unsqueeze(-1)

        delta_log = torch.log1p(torch.clamp(delta_t, min=0.0))
        cont_feat = self.cont_proj(delta_log.unsqueeze(-1) if delta_log.dim() <= 2 else delta_log)
        if cont_feat.dim() == 4 and cont_feat.size(-2) == 1:
            cont_feat = cont_feat.squeeze(-2)

        if delta_bucket is None:
            delta_bucket = torch.zeros_like(delta_t, dtype=torch.long, device=delta_t.device)
        else:
            delta_bucket = delta_bucket.long()

        bucket_feat = self.bucket_emb(delta_bucket)
        if bucket_feat.dim() == 4 and bucket_feat.size(-2) == 1:
            bucket_feat = bucket_feat.squeeze(-2)

        feat = torch.cat([cont_feat, bucket_feat], dim=-1)
        gate = torch.sigmoid(self.mlp(feat))
        return gate * rotated_x + (1.0 - gate) * base_x


class RotaryAdaptiveFusion(nn.Module):
    def __init__(self, embed_dim, rot_dim, hidden_dim=None, gate_hidden_dim=None, bucket_emb_dim=16, dropout=0.1):
        super().__init__()
        self.rot_dim = rot_dim
        self.hour_logit = nn.Parameter(torch.tensor(0.7))
        self.day_logit = nn.Parameter(torch.tensor(0.3))
        self.bucketizer = DeltaBucketizer()
        gate_hidden_dim = gate_hidden_dim if gate_hidden_dim is not None else embed_dim
        self.hour_gate = IntervalAdaptiveGate(embed_dim, 9, bucket_emb_dim, gate_hidden_dim, dropout)
        self.day_gate = IntervalAdaptiveGate(embed_dim, 9, bucket_emb_dim, gate_hidden_dim, dropout)

    def forward(self, x, hour_t, day_t, rotate_fn, device, hour_delta=None, day_delta=None):
        hour_rot = rotate_fn(x, hour_t, self.rot_dim, device)
        day_rot = rotate_fn(x, day_t, self.rot_dim, device)
        hour_bucket = self.bucketizer(hour_delta) if hour_delta is not None else None
        day_bucket = self.bucketizer(day_delta) if day_delta is not None else None
        hour_out = self.hour_gate(x, hour_rot, hour_delta, hour_bucket)
        day_out = self.day_gate(x, day_rot, day_delta, day_bucket)
        weights = torch.softmax(torch.stack([self.hour_logit, self.day_logit]), dim=0)
        return weights[0] * hour_out + weights[1] * day_out


class RotaryAdaptiveFusionBatch(nn.Module):
    def __init__(self, embed_dim, rot_dim, hidden_dim=None, gate_hidden_dim=None, bucket_emb_dim=16, dropout=0.1):
        super().__init__()
        self.rot_dim = rot_dim
        self.hour_logit = nn.Parameter(torch.tensor(0.7))
        self.day_logit = nn.Parameter(torch.tensor(0.3))
        self.bucketizer = DeltaBucketizer()
        gate_hidden_dim = gate_hidden_dim if gate_hidden_dim is not None else embed_dim
        self.hour_gate = IntervalAdaptiveGate(embed_dim, 9, bucket_emb_dim, gate_hidden_dim, dropout)
        self.day_gate = IntervalAdaptiveGate(embed_dim, 9, bucket_emb_dim, gate_hidden_dim, dropout)

    def forward(self, x, hour_t, day_t, rotate_batch_fn, device, hour_delta=None, day_delta=None):
        hour_rot = rotate_batch_fn(x, hour_t, self.rot_dim, device)
        day_rot = rotate_batch_fn(x, day_t, self.rot_dim, device)
        hour_bucket = self.bucketizer(hour_delta) if hour_delta is not None else None
        day_bucket = self.bucketizer(day_delta) if day_delta is not None else None
        hour_out = self.hour_gate(x, hour_rot, hour_delta, hour_bucket)
        day_out = self.day_gate(x, day_rot, day_delta, day_bucket)
        weights = torch.softmax(torch.stack([self.hour_logit, self.day_logit]), dim=0)
        return weights[0] * hour_out + weights[1] * day_out


class DeformableHistoryAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_points, max_distance, offset_scale, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_points = num_points
        self.max_distance = max_distance
        self.offset_scale = offset_scale

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.offset_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_heads * num_points),
        )
        self.dropout = nn.Dropout(dropout)

        anchors = torch.linspace(0.1, 0.9, num_points)
        self.register_buffer("anchors", anchors)

    def _build_sample_positions(self, x, padding_mask):
        bsz, seq_len, _ = x.shape
        device = x.device
        valid_len = seq_len - padding_mask.sum(dim=-1)
        valid_len = valid_len.clamp_min(1)

        positions = torch.arange(seq_len, device=device).view(1, seq_len, 1).expand(bsz, seq_len, self.num_points)
        base = self.anchors.view(1, 1, self.num_points) * positions.to(torch.float)
        offsets = torch.tanh(self.offset_mlp(x)).view(bsz, seq_len, self.num_heads, self.num_points)
        offsets = offsets.mean(dim=2) * self.offset_scale
        sampled = base + offsets

        min_allowed = torch.clamp(positions - self.max_distance, min=0).to(torch.float)
        max_allowed = positions.to(torch.float)
        sampled = torch.maximum(sampled, min_allowed)
        sampled = torch.minimum(sampled, max_allowed)

        per_batch_max = (valid_len - 1).view(bsz, 1, 1).to(torch.float)
        sampled = torch.minimum(sampled, per_batch_max)
        return sampled.round().long()

    def forward(self, x, padding_mask, extra_invalid_mask=None):
        bsz, seq_len, _ = x.shape
        sampled_pos = self._build_sample_positions(x, padding_mask)

        k = self.k_proj(x)
        v = self.v_proj(x)
        q = self.q_proj(x)

        gather_index = sampled_pos.unsqueeze(-1).expand(-1, -1, -1, self.embed_dim)
        k_sparse = torch.gather(k.unsqueeze(1).expand(-1, seq_len, -1, -1), 2, gather_index)
        v_sparse = torch.gather(v.unsqueeze(1).expand(-1, seq_len, -1, -1), 2, gather_index)

        scores = (q.unsqueeze(2) * k_sparse).sum(dim=-1) / math.sqrt(self.embed_dim)

        sampled_invalid = torch.gather(
            padding_mask.unsqueeze(1).expand(-1, seq_len, -1),
            2,
            sampled_pos,
        )
        query_pos = torch.arange(seq_len, device=x.device).view(1, seq_len, 1)
        causal_invalid = sampled_pos > query_pos
        invalid_mask = sampled_invalid | causal_invalid
        if extra_invalid_mask is not None:
            extra_invalid = torch.gather(
                extra_invalid_mask.unsqueeze(0).expand(bsz, -1, -1),
                2,
                sampled_pos,
            )
            invalid_mask = invalid_mask | extra_invalid
        scores = scores.masked_fill(invalid_mask, float("-inf"))

        all_invalid = torch.isinf(scores).all(dim=-1, keepdim=True)
        scores = scores.masked_fill(all_invalid, 0.0)
        attn = torch.softmax(scores, dim=-1)
        attn = attn.masked_fill(invalid_mask, 0.0)
        attn = self.dropout(attn)
        out = (attn.unsqueeze(-1) * v_sparse).sum(dim=2)
        return self.out_proj(out)


class ShortLongDeformableBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        hidden_dim,
        dropout,
        num_points,
        max_distance,
        offset_scale,
        local_window=4,
        short_term_len=4,
    ):
        super().__init__()
        self.local_window = local_window
        self.short_term_len = short_term_len
        self.local_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.long_attn = DeformableHistoryAttention(embed_dim, num_heads, num_points, max_distance, offset_scale, dropout=dropout)
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def _local_causal_mask(self, seq_len, device):
        pos = torch.arange(seq_len, device=device)
        future = pos.view(1, -1) > pos.view(-1, 1)
        short_window = min(self.local_window, self.short_term_len)
        too_far = (pos.view(-1, 1) - pos.view(1, -1)) >= short_window
        return future | too_far

    def _long_memory_mask(self, seq_len, device):
        pos = torch.arange(seq_len, device=device)
        recent_boundary = pos.view(-1, 1) - (self.short_term_len - 1)
        return pos.view(1, -1) >= torch.clamp(recent_boundary, min=0)

    def forward(self, x, padding_mask):
        local_mask = self._local_causal_mask(x.size(1), x.device)
        local_out, _ = self.local_attn(x, x, x, attn_mask=local_mask, key_padding_mask=padding_mask, need_weights=False)
        long_memory_mask = self._long_memory_mask(x.size(1), x.device)
        long_out = self.long_attn(x, padding_mask, extra_invalid_mask=long_memory_mask)
        gate = torch.sigmoid(self.gate(torch.cat([x, local_out, long_out], dim=-1)))
        fused = gate * local_out + (1.0 - gate) * long_out
        x = self.norm1(x + fused)
        x = self.norm2(x + self.ffn(x))
        return x


class ShortLongDeformableEncoder(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        hidden_dim,
        num_layers,
        dropout,
        num_points,
        max_distance,
        offset_scale,
        local_window=4,
        short_term_len=4,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            ShortLongDeformableBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                dropout=dropout,
                num_points=num_points,
                max_distance=max_distance,
                offset_scale=offset_scale,
                local_window=local_window,
                short_term_len=short_term_len,
            )
            for _ in range(num_layers)
        ])

    def forward(self, x, padding_mask):
        for layer in self.layers:
            x = layer(x, padding_mask)
        return x


class ROTAN(nn.Module):
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

        self.user_fused_dim = self.user_embed_dim + self.poi_embed_dim
        assert self.user_fused_dim % 2 == 0, "user_embed_dim + poi_embed_dim must be even for complex rotation."
        self.user_rot_dim = self.user_fused_dim // 2

        self.time_embed_model_user = OriginTime2Vec("sin", self.user_rot_dim)
        self.time_embed_model_user_tgt = OriginTime2Vec("sin", self.user_rot_dim)
        self.time_embed_model_user_day = OriginTime2Vec("sin", self.user_rot_dim)
        self.time_embed_model_user_day_tgt = OriginTime2Vec("sin", self.user_rot_dim)

        self.n_head = args.transformer_nhead
        self.dropout = args.transformer_dropout
        self.n_layers = args.transformer_nlayers
        self.hidden_size = args.transformer_nhid

        self.pos_encoder1 = RightPositionalEncoding(self.user_fused_dim, self.dropout)
        self.transformer_encoder1 = ShortLongDeformableEncoder(
            embed_dim=self.user_fused_dim,
            num_heads=self.n_head,
            hidden_dim=self.hidden_size,
            num_layers=self.n_layers,
            dropout=self.dropout,
            num_points=getattr(args, "deformable_num_points", 8),
            max_distance=getattr(args, "deformable_max_distance", 128),
            offset_scale=getattr(args, "deformable_offset_scale", 8.0),
            local_window=getattr(args, "deformable_local_window", 4),
            short_term_len=getattr(args, "short_term_len", 4),
        )

        self.history_time_fusion = RotaryAdaptiveFusion(
            embed_dim=self.user_fused_dim,
            rot_dim=self.user_rot_dim,
            hidden_dim=self.user_fused_dim,
            gate_hidden_dim=self.user_fused_dim,
            bucket_emb_dim=16,
            dropout=self.dropout,
        )
        self.target_time_fusion = RotaryAdaptiveFusionBatch(
            embed_dim=self.user_fused_dim,
            rot_dim=self.user_rot_dim,
            hidden_dim=self.user_fused_dim,
            gate_hidden_dim=self.user_fused_dim,
            bucket_emb_dim=16,
            dropout=self.dropout,
        )

        self.decoder_poi1 = nn.Linear(self.user_fused_dim + self.poi_embed_dim, args.num_pois)
        self.criterion_poi = nn.CrossEntropyLoss(ignore_index=0)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder_poi1.bias.data.zero_()
        self.decoder_poi1.weight.data.uniform_(-initrange, initrange)

    def _build_causal_mask(self, seq_len):
        return torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=self.device), diagonal=1)

    def _get_time_delta(self, batch_data, key="time_delta", target=False):
        if not target:
            if key in batch_data:
                return batch_data[key].to(torch.float)
            return None
        if "y_POI_id" in batch_data and key in batch_data["y_POI_id"]:
            return batch_data["y_POI_id"][key].to(torch.float)
        return None

    def compute_poi_prob(self, src1, src_mask, target_hour, target_day, poi_embeds, src_key_mask, target_delta=None):
        src1 = src1 * math.sqrt(self.user_fused_dim)
        src1 = self.pos_encoder1(src1)
        src1 = self.transformer_encoder1(src1, src_key_mask)
        src1 = self.target_time_fusion(
            src1,
            target_hour,
            target_day,
            rotate_batch_fn=rotate_batch,
            device=self.device,
            hour_delta=target_delta,
            day_delta=target_delta,
        )
        src1 = torch.cat((src1, poi_embeds), dim=-1)
        return self.decoder_poi1(src1)

    def handle_sequence(self, batch_data):
        poi_id = batch_data["POI_id"]
        norm_time = batch_data["norm_time"]
        day_time = batch_data["day_time"]
        seq_len = batch_data["mask"]

        user_id = batch_data["user_id"].unsqueeze(dim=1).expand(poi_id.shape[0], poi_id.shape[1])
        y_poi_id = batch_data["y_POI_id"]["POI_id"]
        y_norm_time = batch_data["y_POI_id"]["norm_time"]
        y_day_time = batch_data["y_POI_id"]["day_time"]

        bsz, seq_total_len = poi_id.size()
        y_poi_seq = torch.zeros((bsz, seq_total_len), dtype=torch.long, device=self.device)
        y_norm_time_seq = torch.zeros((bsz, seq_total_len), dtype=torch.float, device=self.device)
        y_day_time_seq = torch.zeros((bsz, seq_total_len), dtype=torch.float, device=self.device)

        has_delta = "time_delta" in batch_data and "time_delta" in batch_data["y_POI_id"]
        if has_delta:
            time_delta = batch_data["time_delta"]
            y_time_delta = batch_data["y_POI_id"]["time_delta"]
            time_delta_seq = torch.zeros((bsz, seq_total_len), dtype=torch.float, device=self.device)

        for i in range(bsz):
            end = seq_len[i].item()
            if end <= 0:
                continue
            y_poi_seq[i, :end] = torch.cat((poi_id[i, 1:end], y_poi_id[i].unsqueeze(dim=-1)), dim=-1)
            y_norm_time_seq[i, :end] = torch.cat((norm_time[i, 1:end], y_norm_time[i].unsqueeze(dim=-1)), dim=-1)
            y_day_time_seq[i, :end] = torch.cat((day_time[i, 1:end], y_day_time[i].unsqueeze(dim=-1)), dim=-1)
            if has_delta:
                time_delta_seq[i, :end] = torch.cat((time_delta[i, 1:end], y_time_delta[i].unsqueeze(dim=-1)), dim=-1)

        batch_data["user_id"] = user_id
        batch_data["y_POI_id"]["POI_id"] = y_poi_seq
        batch_data["y_POI_id"]["norm_time"] = y_norm_time_seq
        batch_data["y_POI_id"]["day_time"] = y_day_time_seq
        if has_delta:
            batch_data["y_POI_id"]["time_delta"] = time_delta_seq

        lengths = batch_data["mask"]
        indices = torch.arange(user_id.shape[1], device=self.device).unsqueeze(0).expand(user_id.shape[0], user_id.shape[1])
        padding_mask = (indices >= lengths.unsqueeze(1)).to(torch.bool)
        batch_data["mask"] = padding_mask
        batch_data["seq_len"] = lengths
        return batch_data

    def get_predict(self, batch_data):
        batch_data = self.handle_sequence(batch_data)
        seq_len = batch_data["POI_id"].size(1)
        src_mask = self._build_causal_mask(seq_len)
        x1, batch_target_time, batch_target_day, poi_embeds_padded, target_delta = self.get_rotation_and_loss(
            batch_data, src_mask, batch_data["mask"]
        )
        y_poi = batch_data["y_POI_id"]["POI_id"]
        y_pred_poi = self.compute_poi_prob(
            x1,
            src_mask,
            batch_target_time,
            batch_target_day,
            poi_embeds_padded,
            batch_data["mask"],
            target_delta=target_delta,
        )
        return y_pred_poi, y_poi

    def forward(self, batch_data):
        y_pred_poi, y_poi = self.get_predict(batch_data)
        return self.criterion_poi(y_pred_poi.transpose(1, 2), y_poi)

    def predict(self, batch_data):
        y_pred_poi, _ = self.get_predict(batch_data)
        padding_mask = batch_data["mask"]
        valid_len = y_pred_poi.shape[1] - torch.sum(padding_mask, dim=-1)
        batch_indices = torch.arange(valid_len.shape[0], device=self.device)
        return y_pred_poi[batch_indices, valid_len - 1]

    def get_rotation_and_loss(self, batch_data, mask, src_key_mask):
        u_id = batch_data["user_id"]
        poi_id = batch_data["POI_id"]
        time = batch_data["norm_time"].to(torch.float)
        day_time = batch_data["day_time"].to(torch.float)
        target_time = batch_data["y_POI_id"]["norm_time"].to(torch.float)
        target_day_time = batch_data["y_POI_id"]["day_time"].to(torch.float)
        time_delta = self._get_time_delta(batch_data, key="time_delta", target=False)
        target_time_delta = self._get_time_delta(batch_data, key="time_delta", target=True)

        user_embeddings = self.user_embed_model(u_id)
        seq_poi_embeddings = self.poi_embed_model(poi_id)
        poi_embeds = seq_poi_embeddings

        user_times = self.time_embed_model_user(time)
        user_day_times = self.time_embed_model_user_day(day_time)
        user_next_times = self.time_embed_model_user_tgt(target_time)
        user_next_day_times = self.time_embed_model_user_day_tgt(target_day_time)

        user_embeddings = torch.cat((user_embeddings, seq_poi_embeddings), dim=-1)
        seq_embedding1 = self.history_time_fusion(
            user_embeddings,
            user_times,
            user_day_times,
            rotate_fn=rotate,
            device=self.device,
            hour_delta=time_delta,
            day_delta=time_delta,
        )

        return seq_embedding1, user_next_times, user_next_day_times, poi_embeds, target_time_delta
