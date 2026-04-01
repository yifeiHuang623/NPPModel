import math
import torch
import torch.nn as nn


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


class OriginTime2Vec(nn.Module):
    def __init__(self, activation, out_dim):
        super(OriginTime2Vec, self).__init__()
        if activation != "sin":
            raise ValueError(f"Unsupported activation: {activation}")
        self.l1 = SineActivation(1, out_dim)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        elif x.dim() == 2:
            x = x.unsqueeze(-1)
        return self.l1(x)


class StateQueryEncoder(nn.Module):
    def __init__(self, user_dim, poi_dim, time_dim, hidden_dim, dropout=0.1):
        super().__init__()
        input_dim = user_dim + poi_dim + time_dim * 4
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, user_embed, poi_embed, cur_time, cur_day, tgt_time, tgt_day):
        x = torch.cat([user_embed, poi_embed, cur_time, cur_day, tgt_time, tgt_day], dim=-1)
        return self.net(x)


class AdaptiveStateMixer(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.order_scale = nn.Parameter(torch.tensor(1.0))
        self.time_scale = nn.Parameter(torch.tensor(1.0))
        self.value_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, query_state, memory_state, padding_mask, time_seq):
        _, seq_len, hidden_dim = query_state.shape
        q = self.query_proj(query_state)
        k = self.key_proj(memory_state)
        scores = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(hidden_dim)

        pos = torch.arange(seq_len, device=query_state.device, dtype=query_state.dtype)
        order_dist = (pos.view(1, seq_len, 1) - pos.view(1, 1, seq_len)).clamp_min(0.0)
        order_bias = -torch.nn.functional.softplus(self.order_scale) * torch.log1p(order_dist)
        scores = scores + order_bias

        time_seq = time_seq.to(query_state.dtype)
        time_gap = (time_seq.unsqueeze(2) - time_seq.unsqueeze(1)).clamp_min(0.0)
        time_bias = -torch.nn.functional.softplus(self.time_scale) * torch.log1p(time_gap)
        scores = scores + time_bias
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=query_state.device, dtype=torch.bool),
            diagonal=0,
        )
        invalid = causal_mask.unsqueeze(0) | padding_mask.unsqueeze(1)
        scores = scores.masked_fill(invalid, float("-inf"))

        all_invalid = torch.isinf(scores).all(dim=-1, keepdim=True)
        scores = scores.masked_fill(all_invalid, 0.0)
        attn = torch.softmax(scores, dim=-1)
        attn = attn.masked_fill(invalid, 0.0)

        v = self.value_proj(memory_state)
        retrieved = torch.matmul(attn, v)
        retrieved = self.out_proj(retrieved)
        return retrieved, attn


class BiasAwareRoutingHead(nn.Module):
    def __init__(self, hidden_dim, num_pois, dropout=0.1):
        super().__init__()
        self.preference_memory = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.dynamic_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.routing_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.dynamic_head = nn.Linear(hidden_dim, num_pois)
        self.preference_head = nn.Linear(hidden_dim, num_pois)

    def _build_preference_context(self, seq_repr, padding_mask):
        valid = (~padding_mask).to(seq_repr.dtype)
        prefix_sum = torch.cumsum(seq_repr * valid.unsqueeze(-1), dim=1)
        prefix_cnt = torch.cumsum(valid, dim=1)
        pref_ctx = prefix_sum / prefix_cnt.clamp_min(1.0).unsqueeze(-1)
        pref_ctx = pref_ctx.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return self.preference_memory(pref_ctx)

    def forward(self, seq_repr, padding_mask):
        seq_repr = seq_repr.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        dynamic_state = self.dynamic_proj(seq_repr)
        preference_state = self._build_preference_context(seq_repr, padding_mask)
        gate = torch.sigmoid(self.routing_gate(torch.cat([dynamic_state, preference_state], dim=-1)))
        gate = gate.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        dynamic_logits = self.dynamic_head(dynamic_state)
        preference_logits = self.preference_head(preference_state)
        routed_logits = gate * dynamic_logits + (1.0 - gate) * preference_logits
        return routed_logits.masked_fill(padding_mask.unsqueeze(-1), 0.0)


class ROTAN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = args.device

        self.user_embed_dim = args.user_embed_dim
        self.poi_embed_dim = args.poi_embed_dim
        self.dropout = args.transformer_dropout
        self.hidden_size = args.transformer_nhid

        self.user_embed_model = nn.Embedding(num_embeddings=args.num_users, embedding_dim=self.user_embed_dim)
        self.poi_embed_model = nn.Embedding(num_embeddings=args.num_pois, embedding_dim=self.poi_embed_dim)

        time_dim = max(self.poi_embed_dim // 2, 4)
        self.time_embed_model = OriginTime2Vec("sin", time_dim)
        self.day_embed_model = OriginTime2Vec("sin", time_dim)
        self.target_time_embed_model = OriginTime2Vec("sin", time_dim)
        self.target_day_embed_model = OriginTime2Vec("sin", time_dim)

        self.query_encoder = StateQueryEncoder(
            user_dim=self.user_embed_dim,
            poi_dim=self.poi_embed_dim,
            time_dim=time_dim,
            hidden_dim=self.hidden_size,
            dropout=self.dropout,
        )
        self.retriever = AdaptiveStateMixer(hidden_dim=self.hidden_size, dropout=self.dropout)
        self.direct_head = nn.Linear(self.hidden_size, args.num_pois)
        self.retrieve_head = nn.Linear(self.hidden_size, args.num_pois)
        self.routing_head = BiasAwareRoutingHead(
            hidden_dim=self.hidden_size * 2,
            num_pois=args.num_pois,
            dropout=self.dropout,
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, 1),
        )
        self.criterion_poi = nn.CrossEntropyLoss(ignore_index=0)

    def _build_causal_mask(self, seq_len):
        return torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=self.device),
            diagonal=1,
        )

    def _encode_states(self, batch_data, target_time, target_day):
        u_id = batch_data["user_id"]
        poi_id = batch_data["POI_id"]
        cur_time = batch_data["norm_time"].to(torch.float)
        cur_day = batch_data["day_time"].to(torch.float)
        tgt_time = target_time.to(torch.float)
        tgt_day = target_day.to(torch.float)

        user_embed = self.user_embed_model(u_id)
        poi_embed = self.poi_embed_model(poi_id)

        cur_time_embed = self.time_embed_model(cur_time)
        cur_day_embed = self.day_embed_model(cur_day)
        tgt_time_embed = self.target_time_embed_model(tgt_time)
        tgt_day_embed = self.target_day_embed_model(tgt_day)

        query_state = self.query_encoder(
            user_embed,
            poi_embed,
            cur_time_embed,
            cur_day_embed,
            tgt_time_embed,
            tgt_day_embed,
        )
        return query_state, cur_time

    def compute_poi_prob(self, src1, src_mask, target_hour, target_day, poi_embeds, src_key_mask, target_delta=None):
        del src_mask, poi_embeds, target_delta
        query_state, cur_time = self._encode_states(src1, target_hour, target_day)
        retrieved, _ = self.retriever(query_state, query_state, src_key_mask, cur_time)

        direct_logits = self.direct_head(query_state)
        retrieve_logits = self.retrieve_head(retrieved)
        gate = torch.sigmoid(self.fusion_gate(torch.cat([query_state, retrieved], dim=-1)))
        gate = gate.masked_fill(src_key_mask.unsqueeze(-1), 0.0)

        logits = gate * direct_logits + (1.0 - gate) * retrieve_logits
        fused_state = torch.cat([query_state, retrieved], dim=-1)
        routed_logits = self.routing_head(fused_state, src_key_mask)
        logits = logits + routed_logits
        logits = logits.masked_fill(src_key_mask.unsqueeze(-1), 0.0)
        return logits

    def handle_sequence(self, batch_data):
        poi_id = batch_data["POI_id"]
        norm_time = batch_data["norm_time"]
        day_time = batch_data["day_time"]
        seq_len = batch_data["mask"]

        user_id = batch_data["user_id"].unsqueeze(dim=1).expand(
            poi_id.shape[0], poi_id.shape[1]
        )

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

        batch_data["user_id"] = user_id
        batch_data["y_POI_id"]["POI_id"] = y_poi_seq
        batch_data["y_POI_id"]["norm_time"] = y_norm_time_seq
        batch_data["y_POI_id"]["day_time"] = y_day_time_seq

        if has_delta:
            batch_data["y_POI_id"]["time_delta"] = time_delta_seq

        lengths = batch_data["mask"]
        indices = torch.arange(user_id.shape[1], device=self.device).unsqueeze(0).expand(
            user_id.shape[0], user_id.shape[1]
        )
        padding_mask = (indices >= lengths.unsqueeze(1)).to(torch.bool)
        batch_data["mask"] = padding_mask
        batch_data["seq_len"] = lengths
        return batch_data

    def get_predict(self, batch_data):
        batch_data = self.handle_sequence(batch_data)

        seq_len = batch_data["POI_id"].size(1)
        src_mask = self._build_causal_mask(seq_len)
        y_poi = batch_data["y_POI_id"]["POI_id"]

        y_pred_poi = self.compute_poi_prob(
            batch_data,
            src_mask,
            batch_data["y_POI_id"]["norm_time"],
            batch_data["y_POI_id"]["day_time"],
            None,
            batch_data["mask"],
            target_delta=batch_data["y_POI_id"].get("time_delta"),
        )

        return y_pred_poi, y_poi

    def forward(self, batch_data):
        y_pred_poi, y_poi = self.get_predict(batch_data)
        loss_poi = self.criterion_poi(y_pred_poi.transpose(1, 2), y_poi)
        return loss_poi

    def predict(self, batch_data):
        y_pred_poi, _ = self.get_predict(batch_data)

        padding_mask = batch_data["mask"]
        valid_len = y_pred_poi.shape[1] - torch.sum(padding_mask, dim=-1)
        batch_indices = torch.arange(valid_len.shape[0], device=self.device)
        y_pred = y_pred_poi[batch_indices, valid_len - 1]
        return y_pred
