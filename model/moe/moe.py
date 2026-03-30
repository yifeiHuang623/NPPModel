# coding: utf-8
from __future__ import print_function
from __future__ import division

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class Embedding(nn.Module):
    def __init__(self, vocab_size, num_units, zeros_pad=True, scale=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_units = num_units
        self.zeros_pad = zeros_pad
        self.scale = scale
        self.lookup_table = nn.Parameter(torch.Tensor(vocab_size, num_units))
        nn.init.xavier_normal_(self.lookup_table.data)
        if self.zeros_pad:
            self.lookup_table.data[0, :].fill_(0)

    def forward(self, inputs):
        if inputs.dtype not in (torch.int32, torch.int64):
            inputs = inputs.long()
        padding_idx = 0 if self.zeros_pad else -1
        outputs = F.embedding(
            inputs,
            self.lookup_table,
            padding_idx,
            None,
            2,
            False,
            False,
        )
        if self.scale:
            outputs = outputs * (self.num_units ** 0.5)
        return outputs


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TemporalAdapterExpert(nn.Module):
    def __init__(self, embed_dim, user_dim, context_dim, hidden_dim, dropout=0.1):
        super().__init__()
        cond_dim = embed_dim * 2 + context_dim + user_dim + 2
        self.down = nn.Linear(embed_dim, hidden_dim)
        self.up = nn.Linear(hidden_dim, embed_dim)
        self.gate = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, token_state, history_intent, router_context, user_embed, history_delta, silent_score):
        user_expand = user_embed.unsqueeze(1).expand(-1, token_state.size(1), -1)
        history_expand = history_intent.unsqueeze(1).expand(-1, token_state.size(1), -1)
        cond = torch.cat([
            token_state,
            history_expand,
            router_context,
            user_expand,
            torch.log1p(torch.clamp(history_delta.unsqueeze(-1), min=0.0)),
            silent_score.unsqueeze(-1),
        ], dim=-1)
        hidden = self.activation(self.down(token_state))
        gated = hidden * self.gate(cond)
        return self.up(self.dropout(gated))


class MemoryBiasExpert(nn.Module):
    def __init__(self, embed_dim, user_dim, context_dim, hidden_dim, dropout=0.1):
        super().__init__()
        input_dim = embed_dim + user_dim + context_dim + 2
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, shared_summary, user_embed, context_summary, mean_delta, max_delta):
        features = torch.cat([
            shared_summary,
            user_embed,
            context_summary,
            torch.log1p(torch.clamp(mean_delta.unsqueeze(-1), min=0.0)),
            torch.log1p(torch.clamp(max_delta.unsqueeze(-1), min=0.0)),
        ], dim=-1)
        return self.mlp(features)


class ConditionalTimeRouter(nn.Module):
    def __init__(
        self,
        embed_dim,
        user_dim,
        context_dim,
        route_hidden_dim,
        num_experts=4,
        routing_mode="soft",
        gumbel_tau=1.0,
        top_k=2,
        restart_gap_hours=24.0,
        short_horizon_hours=3.0,
        medium_horizon_hours=12.0,
        long_horizon_hours=72.0,
        time_prior_scale=2.0,
        position_prior_scale=1.5,
        scene_prior_scale=1.0,
        user_prior_scale=0.5,
        token_refine_scale=0.1,
        dropout=0.1,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.routing_mode = routing_mode
        self.gumbel_tau = gumbel_tau
        self.top_k = max(1, min(int(top_k), num_experts))
        self.restart_gap_hours = restart_gap_hours
        self.short_horizon_hours = max(float(short_horizon_hours), 1e-3)
        self.medium_horizon_hours = max(float(medium_horizon_hours), self.short_horizon_hours)
        self.long_horizon_hours = max(float(long_horizon_hours), self.medium_horizon_hours)
        self.time_prior_scale = float(time_prior_scale)
        self.position_prior_scale = float(position_prior_scale)
        self.scene_prior_scale = float(scene_prior_scale)
        self.user_prior_scale = float(user_prior_scale)
        self.token_refine_scale = float(token_refine_scale)
        scene_templates = torch.tensor(
            [
                [0.65, 0.10, 0.20, 0.05],  # short-session
                [0.30, 0.35, 0.30, 0.05],  # long-history
                [0.25, 0.10, 0.20, 0.45],  # restart-prone
            ],
            dtype=torch.float,
        )
        self.register_buffer("scene_templates", scene_templates)
        self.current_tau = float(gumbel_tau)

        self.time_proj = nn.Linear(4, embed_dim)
        self.user_proj = nn.Linear(user_dim, embed_dim)
        self.context_proj = nn.Linear(context_dim, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 4, route_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(route_hidden_dim, num_experts),
        )
        self.user_prior_mlp = nn.Sequential(
            nn.Linear(user_dim, route_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(route_hidden_dim // 2, num_experts),
        )
        self.scene_mlp = nn.Sequential(
            nn.Linear(4, route_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(route_hidden_dim // 2, 3),
        )
        self.token_refine_mlp = nn.Sequential(
            nn.Linear(embed_dim + context_dim + 1, route_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(route_hidden_dim // 2, num_experts),
            nn.Tanh(),
        )
        self.reset_bias_scale = nn.Parameter(torch.tensor(1.0))

    def _build_time_prior(self, history_delta, src_key_padding_mask):
        log_delta = torch.log1p(torch.clamp(history_delta, min=0.0))
        batch_size, seq_len = history_delta.shape
        valid_len = (~src_key_padding_mask).sum(dim=-1, keepdim=True).clamp_min(1)
        pos = torch.arange(seq_len, device=history_delta.device, dtype=history_delta.dtype).unsqueeze(0).expand(batch_size, -1)
        rel_pos = pos / (valid_len.to(history_delta.dtype) - 1.0).clamp_min(1.0)
        center_score = 1.0 - torch.abs(2.0 * rel_pos - 1.0)
        prefix_score = 1.0 - rel_pos
        suffix_score = rel_pos
        seq_scale = (valid_len.to(history_delta.dtype) / max(seq_len, 1)).expand(-1, seq_len)

        short_center = math.log1p(self.short_horizon_hours)
        medium_center = math.log1p(self.medium_horizon_hours)
        long_center = math.log1p(self.long_horizon_hours)
        sigma = max(math.log1p(self.medium_horizon_hours) - math.log1p(self.short_horizon_hours), 1e-3)

        short_time = -((log_delta - short_center) ** 2) / (2 * sigma ** 2)
        medium_time = -((log_delta - medium_center) ** 2) / (2 * sigma ** 2)
        long_time = -((log_delta - long_center) ** 2) / (2 * sigma ** 2)
        restart_time = torch.relu(history_delta - self.restart_gap_hours) / max(self.restart_gap_hours, 1.0)

        short_score = self.position_prior_scale * suffix_score + self.time_prior_scale * short_time
        medium_score = self.position_prior_scale * center_score + 0.5 * self.time_prior_scale * medium_time
        long_score = self.position_prior_scale * prefix_score + 0.5 * seq_scale + 0.25 * self.time_prior_scale * long_time
        restart_score = self.time_prior_scale * restart_time + 0.25 * prefix_score

        return torch.stack(
            [short_score, medium_score, long_score, restart_score], dim=-1
        )

    def _build_scene_prior(self, history_delta, src_key_padding_mask):
        valid_mask = (~src_key_padding_mask).to(history_delta.dtype)
        valid_len = valid_mask.sum(dim=-1).clamp_min(1.0)
        mean_delta = (history_delta * valid_mask).sum(dim=-1) / valid_len
        max_delta = (history_delta * valid_mask).amax(dim=-1)
        seq_norm = valid_len / history_delta.size(1)
        scene_features = torch.stack([
            seq_norm,
            torch.log1p(mean_delta),
            torch.log1p(max_delta),
            (max_delta > self.restart_gap_hours).to(history_delta.dtype),
        ], dim=-1)
        scene_logits = self.scene_mlp(scene_features)
        scene_probs = torch.softmax(scene_logits, dim=-1)
        sample_prior = scene_probs @ self.scene_templates.to(scene_probs.dtype)
        return sample_prior, valid_len

    def forward(
        self,
        token_state,
        history_intent,
        router_context,
        user_embed,
        history_delta,
        src_key_padding_mask,
    ):
        if history_delta is None:
            history_delta = torch.zeros(token_state.size(0), token_state.size(1), device=token_state.device, dtype=token_state.dtype)
        history_delta = torch.nan_to_num(history_delta, nan=0.0, posinf=0.0, neginf=0.0)

        silent_score = F.relu(history_delta - self.restart_gap_hours)
        valid_mask = (~src_key_padding_mask).to(token_state.dtype)
        valid_len = valid_mask.sum(dim=-1).clamp_min(1.0)
        shared_summary = (token_state * valid_mask.unsqueeze(-1)).sum(dim=1) / valid_len.unsqueeze(-1)
        context_summary = (router_context * valid_mask.unsqueeze(-1)).sum(dim=1) / valid_len.unsqueeze(-1)
        time_features = torch.stack([
            torch.log1p(torch.clamp(history_delta, min=0.0)),
            torch.sqrt(torch.clamp(history_delta, min=0.0) + 1e-8),
            (history_delta > self.restart_gap_hours).to(token_state.dtype),
            torch.log1p(silent_score),
        ], dim=-1)

        sample_prior, valid_len = self._build_scene_prior(history_delta, src_key_padding_mask)

        max_delta = (history_delta * valid_mask).amax(dim=-1)
        seq_time_features = torch.stack([
            torch.log1p((history_delta * valid_mask).sum(dim=-1) / valid_len),
            torch.log1p(max_delta),
            (max_delta > self.restart_gap_hours).to(token_state.dtype),
            valid_len / max(history_delta.size(1), 1),
        ], dim=-1)

        seq_input = torch.cat([
            shared_summary,
            history_intent,
            self.context_proj(context_summary),
            self.time_proj(seq_time_features) + self.user_proj(user_embed),
        ], dim=-1)

        seq_logits = self.mlp(seq_input)
        seq_logits = seq_logits + self.user_prior_scale * self.user_prior_mlp(user_embed)
        seq_logits = seq_logits + self.scene_prior_scale * torch.log(sample_prior.clamp_min(1e-8))
        seq_logits = torch.nan_to_num(seq_logits, nan=0.0, posinf=30.0, neginf=-30.0)

        if self.routing_mode == "gumbel" and self.training:
            seq_weights = F.gumbel_softmax(seq_logits, tau=self.current_tau, hard=False, dim=-1)
        else:
            seq_weights = torch.softmax(seq_logits, dim=-1)
        seq_weights = torch.nan_to_num(seq_weights, nan=0.0, posinf=0.0, neginf=0.0)
        if self.top_k < self.num_experts:
            top_values, top_indices = torch.topk(seq_weights, self.top_k, dim=-1)
            sparse_weights = torch.zeros_like(seq_weights).scatter(-1, top_indices, top_values)
            seq_weights = sparse_weights / sparse_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        token_refine_input = torch.cat(
            [
                token_state,
                router_context,
                torch.log1p(torch.clamp(history_delta, min=0.0)).unsqueeze(-1),
            ],
            dim=-1,
        )
        token_refine_logits = self.token_refine_scale * self.token_refine_mlp(token_refine_input)
        route_logits = torch.log(seq_weights.clamp_min(1e-8)).unsqueeze(1) + token_refine_logits
        route_logits = route_logits + self._build_time_prior(history_delta, src_key_padding_mask)
        route_logits[..., -1] = route_logits[..., -1] + self.reset_bias_scale * torch.log1p(silent_score)
        route_logits = torch.nan_to_num(route_logits, nan=0.0, posinf=30.0, neginf=-30.0)
        route_weights = torch.softmax(route_logits, dim=-1)
        route_weights = torch.nan_to_num(route_weights, nan=0.0, posinf=0.0, neginf=0.0)
        route_weights = route_weights * valid_mask.unsqueeze(-1)

        mean_route = seq_weights.mean(dim=0)
        target_usage = (sample_prior * valid_len.unsqueeze(-1)).sum(dim=0) / valid_len.sum().clamp_min(1.0)
        load_balance_loss = torch.sum((mean_route - target_usage) ** 2)
        return route_weights, seq_weights, load_balance_loss, torch.log1p(silent_score), sample_prior


class moe(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = args.device
        self.loss = "WeightedProbBinaryCELoss"

        self.num_pois = args.num_pois
        self.num_users = args.num_users
        self.num_times = args.num_times
        self.num_regions = args.num_regions

        user_dim = args.user_dim
        loc_dim = args.loc_dim
        time_dim = args.time_dim
        reg_dim = args.region_dim
        nhead = args.nhead
        nlayers = args.nlayers
        dropout = args.dropout

        self.embed_dim = user_dim + loc_dim + time_dim + reg_dim
        self.context_dim = loc_dim + time_dim + reg_dim

        self.emb_loc = Embedding(self.num_pois, loc_dim, zeros_pad=True, scale=True)
        self.emb_reg = Embedding(self.num_regions, reg_dim, zeros_pad=True, scale=True)
        self.emb_time = Embedding(self.num_times + 1, time_dim, zeros_pad=True, scale=True)
        self.emb_user = Embedding(self.num_users, user_dim, zeros_pad=True, scale=True)

        self.pos_encoder = PositionalEncoding(self.embed_dim, dropout)
        shared_layer = TransformerEncoderLayer(self.embed_dim, nhead, self.embed_dim, dropout, batch_first=True)
        self.shared_encoder = TransformerEncoder(shared_layer, nlayers)

        expert_hidden_dim = getattr(args, "expert_hidden_dim", max(self.embed_dim // 4, 32))
        self.short_expert = TemporalAdapterExpert(
            self.embed_dim, user_dim, self.context_dim, expert_hidden_dim, dropout=dropout
        )
        self.medium_expert = TemporalAdapterExpert(
            self.embed_dim, user_dim, self.context_dim, expert_hidden_dim, dropout=dropout
        )
        self.memory_expert = MemoryBiasExpert(
            self.embed_dim, user_dim, self.context_dim, expert_hidden_dim, dropout=dropout
        )
        self.time_skip_expert = TemporalAdapterExpert(
            self.embed_dim, user_dim, self.context_dim, expert_hidden_dim, dropout=dropout
        )
        self.router = ConditionalTimeRouter(
            embed_dim=self.embed_dim,
            user_dim=user_dim,
            context_dim=self.context_dim,
            route_hidden_dim=getattr(args, "router_hidden_dim", self.embed_dim),
            num_experts=4,
            routing_mode=getattr(args, "routing_mode", "soft"),
            gumbel_tau=getattr(args, "gumbel_tau", 1.0),
            top_k=getattr(args, "top_k", 2),
            restart_gap_hours=getattr(args, "restart_gap_hours", 24.0),
            short_horizon_hours=getattr(args, "short_horizon_hours", 3.0),
            medium_horizon_hours=getattr(args, "medium_horizon_hours", 12.0),
            long_horizon_hours=getattr(args, "long_horizon_hours", 72.0),
            time_prior_scale=getattr(args, "time_prior_scale", 2.0),
            position_prior_scale=getattr(args, "position_prior_scale", 1.5),
            scene_prior_scale=getattr(args, "scene_prior_scale", 1.0),
            user_prior_scale=getattr(args, "user_prior_scale", 0.5),
            token_refine_scale=getattr(args, "token_refine_scale", 0.1),
            dropout=dropout,
        )

        self.output_dropout = nn.Dropout(dropout)
        self.router_loss_weight = getattr(args, "router_loss_weight", 0.0001)
        self.expert_residual_scale = nn.Parameter(torch.tensor(getattr(args, "expert_residual_scale", 0.3)))
        self.memory_bias_scale = nn.Parameter(torch.tensor(getattr(args, "memory_bias_scale", 0.05)))
        self.label_smoothing = getattr(args, "label_smoothing", 0.0)
        self.expert_warmup_epochs = max(int(getattr(args, "expert_warmup_epochs", 3)), 1)
        self.router_tau_start = float(getattr(args, "router_tau_start", getattr(args, "gumbel_tau", 1.0)))
        self.router_tau_end = float(getattr(args, "router_tau_end", getattr(args, "gumbel_tau", 1.0)))
        self.current_expert_scale = 0.0
        self.current_memory_scale = 0.0
        self.segment_context_proj = nn.Linear(self.context_dim * 2, self.context_dim)

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)
        self.decoder = nn.Linear(self.embed_dim, self.num_pois)

        self.last_router_loss = None
        self.last_route_usage = None
        self.last_route_max = None
        self.last_route_entropy = None

    def set_epoch(self, epoch_idx, total_epochs):
        warmup_progress = min((epoch_idx + 1) / max(self.expert_warmup_epochs, 1), 1.0)
        self.current_expert_scale = float(self.expert_residual_scale.detach().item()) * warmup_progress
        self.current_memory_scale = float(self.memory_bias_scale.detach().item()) * warmup_progress

        if total_epochs <= 1:
            tau = self.router_tau_end
        else:
            progress = epoch_idx / max(total_epochs - 1, 1)
            tau = self.router_tau_start + (self.router_tau_end - self.router_tau_start) * progress
        self.router.current_tau = float(tau)

    @staticmethod
    def _generate_square_mask_(sz, device):
        mask = (torch.triu(torch.ones(sz, sz).to(device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        return mask

    def _encode_inputs(self, loc, region, user_id, time):
        loc = loc.long()
        region = region.long()
        user_id = user_id.long()
        time = time.long()
        loc_emb = self.emb_loc(loc)
        reg_emb = self.emb_reg(region)
        user_emb = self.emb_user(user_id)
        time_emb = self.emb_time(time)
        src = torch.cat([loc_emb, reg_emb, user_emb, time_emb], dim=-1)
        src = src * math.sqrt(src.size(-1))
        return self.pos_encoder(src), loc_emb, reg_emb, time_emb, user_emb

    def _run_moe(self, src, src_square_mask, src_binary_mask, batch_data, loc_emb, reg_emb, time_emb):
        shared = self.shared_encoder(src, mask=src_square_mask, src_key_padding_mask=src_binary_mask)
        shared = torch.nan_to_num(shared, nan=0.0, posinf=0.0, neginf=0.0)
        shared = shared.masked_fill(src_binary_mask.unsqueeze(-1), 0.0)

        valid_len = src.size(1) - src_binary_mask.sum(dim=-1)
        safe_valid_len = valid_len.clamp_min(1)
        batch_indices = torch.arange(valid_len.size(0), device=self.device)
        history_intent = shared[batch_indices, safe_valid_len - 1]
        user_embed = self.emb_user(batch_data["user_id"].to(self.device).long())

        router_context = torch.cat([loc_emb, time_emb, reg_emb], dim=-1)
        valid_token_mask = (~src_binary_mask).unsqueeze(-1).to(shared.dtype)
        shared_summary = (shared * valid_token_mask).sum(dim=1) / valid_token_mask.sum(dim=1).clamp_min(1.0)
        context_summary = (router_context * valid_token_mask).sum(dim=1) / valid_token_mask.sum(dim=1).clamp_min(1.0)
        segment_summary = context_summary.unsqueeze(1).expand(-1, shared.size(1), -1)
        segment_context = self.segment_context_proj(torch.cat([router_context, segment_summary], dim=-1))

        history_delta = batch_data.get("time_delta", None)
        history_delta = history_delta.to(self.device) if history_delta is not None else None
        if history_delta is not None:
            history_delta = torch.nan_to_num(history_delta, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            history_delta = torch.zeros(shared.size(0), shared.size(1), device=shared.device, dtype=shared.dtype)
        valid_mask_2d = (~src_binary_mask).to(history_delta.dtype)
        valid_len_float = valid_mask_2d.sum(dim=-1).clamp_min(1.0)
        mean_delta = (history_delta * valid_mask_2d).sum(dim=-1) / valid_len_float
        max_delta = (history_delta * valid_mask_2d).amax(dim=-1)

        route_weights, seq_weights, router_loss, silent_score, sample_prior = self.router(
            shared,
            history_intent,
            router_context,
            user_embed,
            history_delta,
            src_key_padding_mask=src_binary_mask,
        )
        short_out = self.short_expert(shared, history_intent, router_context, user_embed, history_delta, silent_score)
        medium_out = self.medium_expert(shared, history_intent, segment_context, user_embed, history_delta, silent_score)
        reset_out = self.time_skip_expert(shared, history_intent, router_context, user_embed, history_delta, silent_score)
        memory_bias_hidden = self.memory_expert(shared_summary, user_embed, context_summary, mean_delta, max_delta)
        memory_weight = seq_weights[:, 2]

        residual_route = torch.stack(
            [route_weights[..., 0], route_weights[..., 1], route_weights[..., 3]],
            dim=-1,
        )
        residual_route = residual_route / residual_route.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        fused = (
            residual_route[..., 0:1] * short_out +
            residual_route[..., 1:2] * medium_out +
            residual_route[..., 2:3] * reset_out
        )
        fused = shared + self.current_expert_scale * self.output_dropout(fused)
        fused = torch.nan_to_num(fused, nan=0.0, posinf=0.0, neginf=0.0)
        fused = fused.masked_fill(src_binary_mask.unsqueeze(-1), 0.0)

        valid_token_count = (~src_binary_mask).sum().clamp_min(1).to(route_weights.dtype)
        self.last_router_loss = router_loss
        effective_route = torch.stack(
            [
                residual_route[..., 0],
                residual_route[..., 1],
                seq_weights[:, 2].unsqueeze(1).expand_as(residual_route[..., 0]),
                residual_route[..., 2],
            ],
            dim=-1,
        )
        self.last_route_usage = torch.stack(
            [
                (residual_route[..., 0] * valid_mask_2d).sum() / valid_token_count,
                (residual_route[..., 1] * valid_mask_2d).sum() / valid_token_count,
                seq_weights[:, 2].mean(),
                (residual_route[..., 2] * valid_mask_2d).sum() / valid_token_count,
            ]
        ).detach()
        seq_route_max = seq_weights.max(dim=-1).values
        seq_route_entropy = -(seq_weights.clamp_min(1e-8) * seq_weights.clamp_min(1e-8).log()).sum(dim=-1)
        self.last_route_max = seq_route_max.mean().detach()
        self.last_route_entropy = seq_route_entropy.mean().detach()
        return fused, memory_bias_hidden, memory_weight

    def calculate_loss(self, batch_data):
        user = batch_data["user_id"].to(self.device).long()
        loc = batch_data["POI_id"].to(self.device).long()
        time = batch_data["time_id"].to(self.device).long()
        region = batch_data["region_id"].to(self.device).long()
        ds = batch_data["mask"].to(self.device)

        user_id = user.unsqueeze(dim=1).expand(loc.shape[0], loc.shape[1])
        indices = torch.arange(user_id.shape[1], device=self.device).unsqueeze(0).expand(user_id.shape[0], user_id.shape[1])
        src_binary_mask = (indices >= ds.unsqueeze(1)).to(torch.bool)
        src_square_mask = self._generate_square_mask_(loc.shape[1], self.device)

        B, L = loc.shape
        y_poi = torch.full((B, L), 0, dtype=torch.long, device=self.device)
        for i in range(B):
            end = ds[i].item()
            y_poi[i, :end] = torch.cat(
                (
                    batch_data["POI_id"][i, 1:end].to(self.device).long(),
                    batch_data["y_POI_id"]["POI_id"][i].unsqueeze(dim=-1).to(self.device).long(),
                ),
                dim=-1,
            )

        src, loc_emb, reg_emb, time_emb, _ = self._encode_inputs(loc, region, user_id, time)
        fused, memory_bias_hidden, memory_weight = self._run_moe(
            src, src_square_mask, src_binary_mask, batch_data, loc_emb, reg_emb, time_emb
        )
        logits = self.decoder(fused)
        memory_logits = self.decoder(memory_bias_hidden).unsqueeze(1)
        logits = logits + self.current_memory_scale * memory_weight.unsqueeze(1).unsqueeze(2) * memory_logits
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)

        token_loss = F.cross_entropy(
            logits.transpose(1, 2),
            y_poi,
            ignore_index=0,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        valid_target_mask = y_poi.ne(0)
        valid_target_count = valid_target_mask.sum()
        if valid_target_count.item() == 0:
            loss_ce = logits.new_zeros(())
        else:
            loss_ce = token_loss.masked_select(valid_target_mask).mean()
        if self.last_router_loss is None:
            return loss_ce
        return loss_ce + self.router_loss_weight * self.last_router_loss

    def forward(self, src_loc, src_reg, src_user, src_time, src_square_mask, src_binary_mask, batch_data):
        src, loc_emb, reg_emb, time_emb, _ = self._encode_inputs(src_loc, src_reg, src_user, src_time)
        fused, memory_bias_hidden, memory_weight = self._run_moe(
            src, src_square_mask, src_binary_mask, batch_data, loc_emb, reg_emb, time_emb
        )
        logits = self.decoder(fused)
        memory_logits = self.decoder(memory_bias_hidden).unsqueeze(1)
        logits = logits + self.current_memory_scale * memory_weight.unsqueeze(1).unsqueeze(2) * memory_logits
        return torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)

    def predict(self, batch_data):
        user = batch_data["user_id"].to(self.device).long()
        loc = batch_data["POI_id"].to(self.device).long()
        time = batch_data["time_id"].to(self.device).long()
        region = batch_data["region_id"].to(self.device).long()
        ds = batch_data["mask"].to(self.device)

        user_id = user.unsqueeze(dim=1).expand(loc.shape[0], loc.shape[1])
        indices = torch.arange(user_id.shape[1], device=self.device).unsqueeze(0).expand(user_id.shape[0], user_id.shape[1])
        src_binary_mask = (indices >= ds.unsqueeze(1)).to(torch.bool)
        src_square_mask = self._generate_square_mask_(loc.shape[1], self.device)

        logits = self.forward(
            loc,
            region,
            user_id,
            time,
            src_square_mask,
            src_binary_mask,
            batch_data,
        )
        batch_indices = torch.arange(user_id.shape[0], device=self.device)
        result = logits[batch_indices, ds - 1]
        result[:, 0] = -1e9
        tie_break = 1e-8 * torch.arange(result.size(-1), device=result.device, dtype=result.dtype)
        return result + tie_break.unsqueeze(0)
