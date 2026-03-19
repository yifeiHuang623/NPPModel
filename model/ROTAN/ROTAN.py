import math
import torch
import torch.nn as nn
import torch.nn.functional as F
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


class DeformableTemporalSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        num_points=8,
        max_distance=128,
        offset_scale=8.0,
        dropout=0.1
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_points = num_points
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.max_distance = max(1, int(max_distance))
        self.offset_scale = float(offset_scale)
        self.delta_feature_dim = max(1, embed_dim // 4)

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.delta_bucketizer = DeltaBucketizer()
        self.delta_bucket_emb = nn.Embedding(9, self.delta_feature_dim)
        self.delta_cont_proj = nn.Linear(1, self.delta_feature_dim)
        self.temporal_context_proj = nn.Linear(embed_dim + 2 * self.delta_feature_dim, embed_dim)
        self.offset_proj = nn.Linear(embed_dim, num_heads * num_points)
        self.point_score_proj = nn.Linear(embed_dim, num_heads * num_points)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)

        anchor_positions = self._build_anchor_positions(num_points, self.max_distance)
        self.register_buffer("anchor_positions", anchor_positions, persistent=False)
        self.point_bias = nn.Parameter(torch.zeros(num_heads, num_points))
        self.time_distance_weight = nn.Parameter(torch.ones(num_heads, num_points))
        self.time_distance_bias = nn.Parameter(torch.zeros(num_heads, num_points))

    @staticmethod
    def _build_anchor_positions(num_points, max_distance):
        if num_points <= 1:
            anchors = torch.zeros(1, dtype=torch.float)
        else:
            anchors = torch.logspace(
                0,
                math.log2(max_distance + 1),
                steps=num_points,
                base=2.0,
                dtype=torch.float
            ) - 1.0
            anchors[0] = 0.0
        return anchors

    def _reshape_heads(self, x):
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2).contiguous()

    def _build_temporal_context(self, x, time_delta):
        if time_delta is None:
            return x

        delta_t = torch.clamp(time_delta.to(x.dtype), min=0.0)
        delta_log = torch.log1p(delta_t).unsqueeze(-1)
        delta_bucket = self.delta_bucketizer(delta_t).long()

        cont_feat = self.delta_cont_proj(delta_log)
        bucket_feat = self.delta_bucket_emb(delta_bucket)
        temporal_feat = torch.cat([cont_feat, bucket_feat], dim=-1)
        return self.temporal_context_proj(torch.cat([x, temporal_feat], dim=-1))

    def _gather_with_linear_interp(self, sequence, positions):
        batch_size, num_heads, seq_len, head_dim = sequence.shape
        num_points = positions.size(-1)

        left_idx = positions.floor().long().clamp_(0, seq_len - 1)
        right_idx = (left_idx + 1).clamp_(0, seq_len - 1)
        alpha = (positions - left_idx.to(positions.dtype)).unsqueeze(-1)

        expanded = sequence.unsqueeze(3).expand(-1, -1, -1, num_points, -1)
        left = torch.gather(
            expanded,
            2,
            left_idx.unsqueeze(-1).expand(-1, -1, -1, -1, head_dim)
        )
        right = torch.gather(
            expanded,
            2,
            right_idx.unsqueeze(-1).expand(-1, -1, -1, -1, head_dim)
        )
        return left * (1.0 - alpha) + right * alpha

    def _compute_temporal_score(self, sample_positions, time_delta, x_dtype):
        if time_delta is None:
            return 0.0

        elapsed = torch.cumsum(torch.clamp(time_delta.to(x_dtype), min=0.0), dim=-1)
        elapsed = elapsed.unsqueeze(1).unsqueeze(-1).expand(-1, self.num_heads, -1, 1)

        sampled_elapsed = self._gather_with_linear_interp(elapsed, sample_positions).squeeze(-1)
        query_elapsed = elapsed.squeeze(-1).unsqueeze(-1)
        relative_elapsed = torch.clamp(query_elapsed - sampled_elapsed, min=0.0)
        relative_elapsed = torch.log1p(relative_elapsed)

        time_decay = F.softplus(self.time_distance_weight).view(1, self.num_heads, 1, self.num_points)
        time_bias = self.time_distance_bias.view(1, self.num_heads, 1, self.num_points)
        return time_bias - time_decay * relative_elapsed

    def forward(self, x, src_key_padding_mask=None, time_delta=None):
        batch_size, seq_len, _ = x.shape

        q = self._reshape_heads(self.q_proj(x))
        k = self._reshape_heads(self.k_proj(x))
        v = self._reshape_heads(self.v_proj(x))

        temporal_context = self._build_temporal_context(x, time_delta)

        offsets = torch.tanh(self.offset_proj(temporal_context))
        offsets = offsets.view(batch_size, seq_len, self.num_heads, self.num_points)
        offsets = offsets.permute(0, 2, 1, 3).contiguous()

        point_scores = self.point_score_proj(temporal_context)
        point_scores = point_scores.view(batch_size, seq_len, self.num_heads, self.num_points)
        point_scores = point_scores.permute(0, 2, 1, 3).contiguous()

        query_positions = torch.arange(seq_len, device=x.device, dtype=x.dtype)
        query_positions = query_positions.view(1, 1, seq_len, 1)
        anchor_positions = self.anchor_positions.to(x.dtype).view(1, 1, 1, self.num_points)
        sample_positions = query_positions - anchor_positions + offsets * self.offset_scale

        if src_key_padding_mask is not None:
            valid_lengths = (~src_key_padding_mask).sum(dim=-1).clamp(min=1)
            max_valid_idx = (valid_lengths - 1).view(batch_size, 1, 1, 1).to(x.dtype)
            max_causal_idx = torch.minimum(query_positions, max_valid_idx)
        else:
            max_causal_idx = query_positions

        sample_positions = sample_positions.clamp(min=0.0)
        sample_positions = torch.minimum(sample_positions, max_causal_idx)

        sampled_k = self._gather_with_linear_interp(k, sample_positions)
        sampled_v = self._gather_with_linear_interp(v, sample_positions)
        temporal_scores = self._compute_temporal_score(sample_positions, time_delta, x.dtype)

        attn_scores = (q.unsqueeze(3) * sampled_k).sum(dim=-1) * self.scale
        attn_scores = attn_scores + point_scores + temporal_scores
        attn_scores = attn_scores + self.point_bias.view(1, self.num_heads, 1, self.num_points)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        out = torch.sum(attn_weights.unsqueeze(-1) * sampled_v, dim=3)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        out = self.out_proj(out)

        if src_key_padding_mask is not None:
            out = out.masked_fill(src_key_padding_mask.unsqueeze(-1), 0.0)

        return out


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        hidden_size,
        dropout=0.1,
        num_points=8,
        max_distance=128,
        offset_scale=8.0
    ):
        super().__init__()
        self.self_attn = DeformableTemporalSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_points=num_points,
            max_distance=max_distance,
            offset_scale=offset_scale,
            dropout=dropout
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.linear1 = nn.Linear(embed_dim, hidden_size)
        self.activation = nn.GELU()
        self.ffn_dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_size, embed_dim)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, time_delta=None):
        del src_mask

        attn_out = self.self_attn(
            src,
            src_key_padding_mask=src_key_padding_mask,
            time_delta=time_delta
        )
        src = self.norm1(src + self.dropout1(attn_out))

        ffn_out = self.linear2(self.ffn_dropout(self.activation(self.linear1(src))))
        src = self.norm2(src + self.dropout2(ffn_out))

        if src_key_padding_mask is not None:
            src = src.masked_fill(src_key_padding_mask.unsqueeze(-1), 0.0)

        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        hidden_size,
        num_layers,
        dropout=0.1,
        num_points=8,
        max_distance=128,
        offset_scale=8.0
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            DeformableTransformerEncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                hidden_size=hidden_size,
                dropout=dropout,
                num_points=num_points,
                max_distance=max_distance,
                offset_scale=offset_scale
            )
            for _ in range(num_layers)
        ])

    def forward(self, src, mask=None, src_key_padding_mask=None, time_delta=None):
        output = src
        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                time_delta=time_delta
            )
        return output


class UserHabitExpert(nn.Module):
    def __init__(self, user_dim, poi_dim, output_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.user_proj = nn.Linear(user_dim + poi_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_dim)
        self.act = nn.GELU()

    def forward(self, user_embed, poi_embeds, src_key_padding_mask=None):
        if src_key_padding_mask is not None:
            valid_mask = (~src_key_padding_mask).unsqueeze(-1).to(poi_embeds.dtype)
            pooled_poi = (poi_embeds * valid_mask).sum(dim=1)
            denom = valid_mask.sum(dim=1).clamp_min(1.0)
            pooled_poi = pooled_poi / denom
        else:
            pooled_poi = poi_embeds.mean(dim=1)

        habit_state = torch.cat([user_embed, pooled_poi], dim=-1)
        habit_state = self.user_proj(habit_state)
        habit_state = self.dropout(self.act(habit_state))
        habit_state = self.norm(self.out_proj(habit_state))
        return habit_state.unsqueeze(1).expand(-1, poi_embeds.size(1), -1)


class AdaptiveExpertRouter(nn.Module):
    def __init__(self, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3)
        )

    def forward(self, valid_lengths, time_delta=None, seq_len=None):
        batch_size = valid_lengths.size(0)
        if seq_len is None:
            if time_delta is not None:
                seq_len = time_delta.size(1)
            else:
                seq_len = int(valid_lengths.max().item())

        device = valid_lengths.device
        dtype = torch.float32

        len_norm = valid_lengths.to(dtype) / max(seq_len, 1)
        position_ids = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1)
        denom = (valid_lengths.clamp_min(1) - 1).unsqueeze(1).to(dtype)
        pos_norm = torch.where(
            valid_lengths.unsqueeze(1) > 1,
            position_ids / denom.clamp_min(1.0),
            torch.zeros_like(position_ids)
        )

        if time_delta is None:
            local_delta = torch.zeros(batch_size, seq_len, device=device, dtype=dtype)
            mean_delta = torch.zeros(batch_size, 1, device=device, dtype=dtype)
            max_delta = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        else:
            delta = torch.clamp(time_delta.to(dtype), min=0.0)
            valid_mask = (position_ids < valid_lengths.unsqueeze(1)).to(dtype)
            local_delta = torch.log1p(delta)
            mean_delta = (delta * valid_mask).sum(dim=1, keepdim=True) / valid_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            max_delta = (delta * valid_mask).amax(dim=1, keepdim=True)
            mean_delta = torch.log1p(mean_delta)
            max_delta = torch.log1p(max_delta)

        features = torch.stack([
            len_norm.unsqueeze(1).expand(-1, seq_len),
            pos_norm,
            local_delta,
            mean_delta.expand(-1, seq_len),
            max_delta.expand(-1, seq_len),
        ], dim=-1)

        route_logits = self.mlp(features)
        route_weights = torch.softmax(route_logits, dim=-1)

        valid_mask = (position_ids < valid_lengths.unsqueeze(1)).unsqueeze(-1)
        route_weights = route_weights * valid_mask.to(route_weights.dtype)
        return route_weights


class DeltaBucketizer(nn.Module):
    """
    Bucketize delta time in hours.
    Default buckets are designed for intervals within one day:
        0,
        (0, 0.5],
        (0.5, 1],
        (1, 2],
        (2, 4],
        (4, 8],
        (8, 12],
        (12, 24],
        >24
    """
    def __init__(self, boundaries=None):
        super().__init__()
        if boundaries is None:
            boundaries = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0]
        self.register_buffer(
            "boundaries",
            torch.tensor(boundaries, dtype=torch.float)
        )

    def forward(self, delta_t):
        if delta_t is None:
            return None
        # torch.bucketize returns indices in [0, len(boundaries)]
        # shape preserved
        return torch.bucketize(delta_t, self.boundaries)


class IntervalAdaptiveGate(nn.Module):
    """
    Stronger alternative to exp(-rate * delta).

    gate = sigmoid(MLP([proj(log1p(delta)), bucket_emb]))
    out  = gate * rotated_x + (1 - gate) * base_x
    """
    def __init__(
        self,
        embed_dim,
        bucket_num=9,
        bucket_emb_dim=16,
        hidden_dim=None,
        dropout=0.1
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim

        self.bucket_emb = nn.Embedding(bucket_num, bucket_emb_dim)
        self.cont_proj = nn.Linear(1, bucket_emb_dim)

        self.mlp = nn.Sequential(
            nn.Linear(bucket_emb_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, base_x, rotated_x, delta_t, delta_bucket=None):
        """
        base_x:     [B, L, D] or [B, D]
        rotated_x:  same shape as base_x
        delta_t:    [B, L] or [B]
        delta_bucket: same shape as delta_t
        """
        if delta_t is None:
            # if delta not available, default to a pure average-style learned-free fallback
            gate = 0.5
            return gate * rotated_x + (1.0 - gate) * base_x

        if delta_t.dim() == 1:
            delta_t = delta_t.unsqueeze(-1)  # [B, 1] before proj path
            is_2d_case = True
        else:
            is_2d_case = False

        delta_log = torch.log1p(torch.clamp(delta_t, min=0.0))

        cont_feat = self.cont_proj(delta_log.unsqueeze(-1) if delta_log.dim() == 1 else delta_log.unsqueeze(-1) if delta_log.dim() == 2 else delta_log)

        # normalize shape to [..., E]
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

        out = gate * rotated_x + (1.0 - gate) * base_x
        return out


class RotaryAdaptiveFusion(nn.Module):
    """
    For sequence embeddings before Transformer.
    Replace exponential damping with interval-adaptive gated fusion.
    """
    def __init__(
        self,
        embed_dim,
        rot_dim,
        hidden_dim=None,
        gate_hidden_dim=None,
        bucket_emb_dim=16,
        dropout=0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.rot_dim = rot_dim

        self.hour_logit = nn.Parameter(torch.tensor(0.7))
        self.day_logit = nn.Parameter(torch.tensor(0.3))

        self.bucketizer = DeltaBucketizer()

        self.hour_gate = IntervalAdaptiveGate(
            embed_dim=embed_dim,
            bucket_num=9,
            bucket_emb_dim=bucket_emb_dim,
            hidden_dim=gate_hidden_dim if gate_hidden_dim is not None else embed_dim,
            dropout=dropout
        )
        self.day_gate = IntervalAdaptiveGate(
            embed_dim=embed_dim,
            bucket_num=9,
            bucket_emb_dim=bucket_emb_dim,
            hidden_dim=gate_hidden_dim if gate_hidden_dim is not None else embed_dim,
            dropout=dropout
        )

    def forward(
        self,
        x,
        hour_t,
        day_t,
        rotate_fn,
        device,
        hour_delta=None,
        day_delta=None
    ):
        hour_rot = rotate_fn(x, hour_t, self.rot_dim, device)
        day_rot = rotate_fn(x, day_t, self.rot_dim, device)

        hour_bucket = self.bucketizer(hour_delta) if hour_delta is not None else None
        day_bucket = self.bucketizer(day_delta) if day_delta is not None else None

        hour_out = self.hour_gate(x, hour_rot, hour_delta, hour_bucket)
        day_out = self.day_gate(x, day_rot, day_delta, day_bucket)

        weights = torch.softmax(torch.stack([self.hour_logit, self.day_logit]), dim=0)
        out = weights[0] * hour_out + weights[1] * day_out
        return out


class RotaryAdaptiveFusionBatch(nn.Module):
    """
    For target-time transformation after Transformer.
    Replace exponential damping with interval-adaptive gated fusion.
    """
    def __init__(
        self,
        embed_dim,
        rot_dim,
        hidden_dim=None,
        gate_hidden_dim=None,
        bucket_emb_dim=16,
        dropout=0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.rot_dim = rot_dim

        self.hour_logit = nn.Parameter(torch.tensor(0.7))
        self.day_logit = nn.Parameter(torch.tensor(0.3))

        self.bucketizer = DeltaBucketizer()

        self.hour_gate = IntervalAdaptiveGate(
            embed_dim=embed_dim,
            bucket_num=9,
            bucket_emb_dim=bucket_emb_dim,
            hidden_dim=gate_hidden_dim if gate_hidden_dim is not None else embed_dim,
            dropout=dropout
        )
        self.day_gate = IntervalAdaptiveGate(
            embed_dim=embed_dim,
            bucket_num=9,
            bucket_emb_dim=bucket_emb_dim,
            hidden_dim=gate_hidden_dim if gate_hidden_dim is not None else embed_dim,
            dropout=dropout
        )

    def forward(
        self,
        x,
        hour_t,
        day_t,
        rotate_batch_fn,
        device,
        hour_delta=None,
        day_delta=None
    ):
        hour_rot = rotate_batch_fn(x, hour_t, self.rot_dim, device)
        day_rot = rotate_batch_fn(x, day_t, self.rot_dim, device)

        hour_bucket = self.bucketizer(hour_delta) if hour_delta is not None else None
        day_bucket = self.bucketizer(day_delta) if day_delta is not None else None

        hour_out = self.hour_gate(x, hour_rot, hour_delta, hour_bucket)
        day_out = self.day_gate(x, day_rot, day_delta, day_bucket)

        weights = torch.softmax(torch.stack([self.hour_logit, self.day_logit]), dim=0)
        out = weights[0] * hour_out + weights[1] * day_out
        return out


class ROTAN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = args.device

        self.user_embed_dim = args.user_embed_dim
        self.poi_embed_dim = args.poi_embed_dim
        self.time_embed_dim = args.time_embed_dim
        self.gps_embed_dim = args.gps_embed_dim

        self.gps_embed_model = nn.Embedding(
            num_embeddings=4096, embedding_dim=self.gps_embed_dim
        )
        self.user_embed_model = nn.Embedding(
            num_embeddings=args.num_users, embedding_dim=self.user_embed_dim
        )
        self.poi_embed_model = nn.Embedding(
            num_embeddings=args.num_pois, embedding_dim=self.poi_embed_dim
        )

        self.user_fused_dim = self.user_embed_dim + self.poi_embed_dim
        assert self.user_fused_dim % 2 == 0, \
            "user_embed_dim + poi_embed_dim must be even for complex rotation."

        self.user_rot_dim = self.user_fused_dim // 2

        self.time_embed_model_user = OriginTime2Vec('sin', self.user_rot_dim)
        self.time_embed_model_user_tgt = OriginTime2Vec('sin', self.user_rot_dim)

        self.time_embed_model_user_day = OriginTime2Vec('sin', self.user_rot_dim)
        self.time_embed_model_user_day_tgt = OriginTime2Vec('sin', self.user_rot_dim)

        self.n_head = args.transformer_nhead
        self.dropout = args.transformer_dropout
        self.n_layers = args.transformer_nlayers
        self.hidden_size = args.transformer_nhid
        self.use_deformable_attention = getattr(args, 'use_deformable_attention', True)
        self.deformable_num_points = getattr(args, 'deformable_num_points', 8)
        self.deformable_max_distance = getattr(args, 'deformable_max_distance', 128)
        self.deformable_offset_scale = getattr(args, 'deformable_offset_scale', 8.0)
        self.local_window_size = getattr(args, 'local_window_size', 10)
        self.local_encoder_layers = getattr(args, 'local_encoder_layers', max(1, self.n_layers // 2))
        self.habit_hidden_dim = getattr(args, 'habit_hidden_dim', self.user_fused_dim)
        self.route_hidden_dim = getattr(args, 'route_hidden_dim', max(64, self.user_fused_dim // 2))

        self.pos_encoder1 = RightPositionalEncoding(self.user_fused_dim, self.dropout)

        local_encoder_layer = TransformerEncoderLayer(
            self.user_fused_dim,
            self.n_head,
            self.hidden_size,
            self.dropout,
            batch_first=True
        )
        self.local_encoder = TransformerEncoder(local_encoder_layer, self.local_encoder_layers)

        if self.use_deformable_attention:
            self.global_encoder = DeformableTransformerEncoder(
                embed_dim=self.user_fused_dim,
                num_heads=self.n_head,
                hidden_size=self.hidden_size,
                num_layers=self.n_layers,
                dropout=self.dropout,
                num_points=self.deformable_num_points,
                max_distance=self.deformable_max_distance,
                offset_scale=self.deformable_offset_scale
            )
        else:
            encoder_layers1 = TransformerEncoderLayer(
                self.user_fused_dim,
                self.n_head,
                self.hidden_size,
                self.dropout,
                batch_first=True
            )
            self.global_encoder = TransformerEncoder(encoder_layers1, self.n_layers)

        self.habit_expert = UserHabitExpert(
            user_dim=self.user_embed_dim,
            poi_dim=self.poi_embed_dim,
            output_dim=self.user_fused_dim,
            hidden_dim=self.habit_hidden_dim,
            dropout=self.dropout
        )
        self.expert_router = AdaptiveExpertRouter(
            hidden_dim=self.route_hidden_dim,
            dropout=self.dropout
        )
        self.expert_fusion_norm = nn.LayerNorm(self.user_fused_dim)
        self.expert_dropout = nn.Dropout(self.dropout)

        self.history_time_fusion = RotaryAdaptiveFusion(
            embed_dim=self.user_fused_dim,
            rot_dim=self.user_rot_dim,
            hidden_dim=self.user_fused_dim,
            gate_hidden_dim=self.user_fused_dim,
            bucket_emb_dim=16,
            dropout=self.dropout
        )

        self.target_time_fusion = RotaryAdaptiveFusionBatch(
            embed_dim=self.user_fused_dim,
            rot_dim=self.user_rot_dim,
            hidden_dim=self.user_fused_dim,
            gate_hidden_dim=self.user_fused_dim,
            bucket_emb_dim=16,
            dropout=self.dropout
        )

        self.decoder_poi1 = nn.Linear(
            self.user_fused_dim + self.poi_embed_dim,
            args.num_pois
        )

        self.criterion_poi = nn.CrossEntropyLoss(ignore_index=0)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder_poi1.bias.data.zero_()
        self.decoder_poi1.weight.data.uniform_(-initrange, initrange)

    def _build_causal_mask(self, seq_len):
        mask = torch.triu(
            torch.ones((seq_len, seq_len), device=self.device, dtype=torch.bool),
            diagonal=1
        )
        return mask

    def _build_local_causal_mask(self, seq_len, window_size):
        row_idx = torch.arange(seq_len, device=self.device).unsqueeze(1)
        col_idx = torch.arange(seq_len, device=self.device).unsqueeze(0)
        invalid = (col_idx > row_idx) | ((row_idx - col_idx) >= window_size)
        return invalid

    def _get_time_delta(self, batch_data, key='time_delta', target=False):
        if not target:
            if key in batch_data:
                return batch_data[key].to(torch.float)
            return None
        else:
            if 'y_POI_id' in batch_data and key in batch_data['y_POI_id']:
                return batch_data['y_POI_id'][key].to(torch.float)
            return None

    def compute_poi_prob(
        self,
        src1,
        src_mask,
        target_hour,
        target_day,
        poi_embeds,
        src_key_mask,
        user_ids,
        history_delta=None,
        target_delta=None
    ):
        src1 = src1 * math.sqrt(self.user_fused_dim)
        src1 = self.pos_encoder1(src1)
        seq_len = src1.size(1)
        valid_lengths = seq_len - src_key_mask.sum(dim=-1)
        local_mask = self._build_local_causal_mask(seq_len, self.local_window_size)

        local_out = self.local_encoder(
            src=src1,
            mask=local_mask,
            src_key_padding_mask=src_key_mask
        )
        if self.use_deformable_attention:
            global_out = self.global_encoder(
                src=src1,
                mask=src_mask,
                src_key_padding_mask=src_key_mask,
                time_delta=history_delta
            )
        else:
            global_out = self.global_encoder(
                src=src1,
                mask=src_mask,
                src_key_padding_mask=src_key_mask
            )

        user_embed = self.user_embed_model(user_ids)
        habit_out = self.habit_expert(
            user_embed,
            poi_embeds,
            src_key_padding_mask=src_key_mask
        )

        route_weights = self.expert_router(
            valid_lengths=valid_lengths,
            time_delta=history_delta,
            seq_len=seq_len
        )
        fused_seq = (
            route_weights[..., 0:1] * habit_out +
            route_weights[..., 1:2] * local_out +
            route_weights[..., 2:3] * global_out
        )
        src1 = self.expert_fusion_norm(src1 + self.expert_dropout(fused_seq))
        src1 = src1.masked_fill(src_key_mask.unsqueeze(-1), 0.0)

        src1 = self.target_time_fusion(
            src1,
            target_hour,
            target_day,
            rotate_batch_fn=rotate_batch,
            device=self.device,
            hour_delta=target_delta,
            day_delta=target_delta
        )

        src1 = torch.cat((src1, poi_embeds), dim=-1)
        out_poi_prob = self.decoder_poi1(src1)
        return out_poi_prob

    def handle_sequence(self, batch_data):
        poi_id = batch_data['POI_id']
        norm_time = batch_data['norm_time']
        day_time = batch_data['day_time']
        seq_len = batch_data['mask']

        user_id = batch_data['user_id'].unsqueeze(dim=1).expand(
            poi_id.shape[0], poi_id.shape[1]
        )

        y_poi_id = batch_data['y_POI_id']['POI_id']
        y_norm_time = batch_data['y_POI_id']['norm_time']
        y_day_time = batch_data['y_POI_id']['day_time']

        B, L = poi_id.size()
        y_poi_seq = torch.zeros((B, L), dtype=torch.long, device=self.device)
        y_norm_time_seq = torch.zeros((B, L), dtype=torch.float, device=self.device)
        y_day_time_seq = torch.zeros((B, L), dtype=torch.float, device=self.device)

        has_delta = 'time_delta' in batch_data and 'time_delta' in batch_data['y_POI_id']
        if has_delta:
            time_delta = batch_data['time_delta']
            y_time_delta = batch_data['y_POI_id']['time_delta']
            time_delta_seq = torch.zeros((B, L), dtype=torch.float, device=self.device)

        for i in range(B):
            end = seq_len[i].item()
            if end <= 0:
                continue

            y_poi_seq[i, :end] = torch.cat(
                (poi_id[i, 1:end], y_poi_id[i].unsqueeze(dim=-1)), dim=-1
            )
            y_norm_time_seq[i, :end] = torch.cat(
                (norm_time[i, 1:end], y_norm_time[i].unsqueeze(dim=-1)), dim=-1
            )
            y_day_time_seq[i, :end] = torch.cat(
                (day_time[i, 1:end], y_day_time[i].unsqueeze(dim=-1)), dim=-1
            )

            if has_delta:
                time_delta_seq[i, :end] = torch.cat(
                    (time_delta[i, 1:end], y_time_delta[i].unsqueeze(dim=-1)), dim=-1
                )

        batch_data['user_id'] = user_id
        batch_data['y_POI_id']['POI_id'] = y_poi_seq
        batch_data['y_POI_id']['norm_time'] = y_norm_time_seq
        batch_data['y_POI_id']['day_time'] = y_day_time_seq

        if has_delta:
            batch_data['y_POI_id']['time_delta'] = time_delta_seq

        lengths = batch_data['mask']
        indices = torch.arange(user_id.shape[1], device=self.device).unsqueeze(0).expand(
            user_id.shape[0], user_id.shape[1]
        )
        padding_mask = (indices >= lengths.unsqueeze(1)).to(torch.bool)
        batch_data['mask'] = padding_mask
        batch_data['seq_len'] = lengths

        return batch_data

    def get_predict(self, batch_data):
        batch_data = self.handle_sequence(batch_data)

        seq_len = batch_data['POI_id'].size(1)
        src_mask = self._build_causal_mask(seq_len)

        x1, batch_target_time, batch_target_day, poi_embeds_padded, target_delta = \
            self.get_rotation_and_loss(batch_data, src_mask, batch_data['mask'])
        history_delta = self._get_time_delta(batch_data, key='time_delta', target=False)

        y_poi = batch_data['y_POI_id']['POI_id']

        y_pred_poi = self.compute_poi_prob(
            x1,
            src_mask,
            batch_target_time,
            batch_target_day,
            poi_embeds_padded,
            batch_data['mask'],
            batch_data['user_id'][:, 0],
            history_delta=history_delta,
            target_delta=target_delta
        )

        return y_pred_poi, y_poi

    def forward(self, batch_data):
        y_pred_poi, y_poi = self.get_predict(batch_data)
        loss_poi = self.criterion_poi(y_pred_poi.transpose(1, 2), y_poi)
        return loss_poi

    def predict(self, batch_data):
        y_pred_poi, _ = self.get_predict(batch_data)

        padding_mask = batch_data['mask']
        valid_len = y_pred_poi.shape[1] - torch.sum(padding_mask, dim=-1)

        batch_indices = torch.arange(valid_len.shape[0], device=self.device)
        y_pred = y_pred_poi[batch_indices, valid_len - 1]
        return y_pred

    def get_rotation_and_loss(self, batch_data, mask, src_key_mask):
        u_id = batch_data['user_id']
        poi_id = batch_data['POI_id']
        time = batch_data['norm_time'].to(torch.float)
        day_time = batch_data['day_time'].to(torch.float)

        target_time = batch_data['y_POI_id']['norm_time'].to(torch.float)
        target_day_time = batch_data['y_POI_id']['day_time'].to(torch.float)

        time_delta = self._get_time_delta(batch_data, key='time_delta', target=False)
        target_time_delta = self._get_time_delta(batch_data, key='time_delta', target=True)

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
            day_delta=time_delta
        )

        seq_embedding3 = user_next_times
        seq_embedding4 = user_next_day_times

        return seq_embedding1, seq_embedding3, seq_embedding4, poi_embeds, target_time_delta
