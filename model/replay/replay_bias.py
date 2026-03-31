import torch
import torch.nn as nn
import numpy as np
from enum import Enum
from math import pi

from .replay_utils import generate_tensor_of_distribution


class Rnn(Enum):
    RNN = 0
    GRU = 1
    LSTM = 2

    @staticmethod
    def from_string(name):
        if name == "rnn":
            return Rnn.RNN
        if name == "gru":
            return Rnn.GRU
        if name == "lstm":
            return Rnn.LSTM
        raise ValueError(f"{name} not supported in --rnn")


class RnnFactory:
    def __init__(self, rnn_type_str):
        self.rnn_type = Rnn.from_string(rnn_type_str)

    def is_lstm(self):
        return self.rnn_type in [Rnn.LSTM]

    def create(self, hidden_size):
        if self.rnn_type == Rnn.RNN:
            return nn.RNN(hidden_size, hidden_size, batch_first=True)
        if self.rnn_type == Rnn.GRU:
            return nn.GRU(hidden_size, hidden_size, batch_first=True)
        if self.rnn_type == Rnn.LSTM:
            return nn.LSTM(hidden_size, hidden_size, batch_first=True)


class User_Week_Distribution(nn.Module):
    def __init__(self, stamp_num):
        super().__init__()
        self.stamp_num = stamp_num
        self.sigma = nn.Parameter(torch.ones(self.stamp_num).view(self.stamp_num, 1))

    def forward(self, x):
        self.sigma.data = torch.abs(self.sigma.data)
        learned_weight = 1 / torch.sqrt(2 * pi * (self.sigma ** 2)) * torch.exp(-(x ** 2) / (2 * (self.sigma ** 2)))
        weight_sum = torch.sum(learned_weight, dim=1, keepdim=True)
        return learned_weight / weight_sum


class BiasAwareRoutingHead(nn.Module):
    def __init__(self, hidden_dim, num_pois, dropout=0.1):
        super().__init__()
        self.dynamic_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.preference_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.dynamic_head = nn.Linear(hidden_dim, num_pois)
        self.preference_head = nn.Linear(hidden_dim, num_pois)

    def _prefix_preference(self, seq_repr, padding_mask):
        valid = (~padding_mask).to(seq_repr.dtype)
        prefix_sum = torch.cumsum(seq_repr * valid.unsqueeze(-1), dim=1)
        prefix_cnt = torch.cumsum(valid, dim=1)
        pref = prefix_sum / prefix_cnt.clamp_min(1.0).unsqueeze(-1)
        pref = pref.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return self.preference_proj(pref)

    def forward(self, seq_repr, padding_mask):
        seq_repr = seq_repr.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        dynamic_state = self.dynamic_proj(seq_repr)
        preference_state = self._prefix_preference(seq_repr, padding_mask)
        gate = torch.sigmoid(self.gate(torch.cat([dynamic_state, preference_state], dim=-1)))
        gate = gate.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        dynamic_logits = self.dynamic_head(dynamic_state)
        preference_logits = self.preference_head(preference_state)
        routed_logits = gate * dynamic_logits + (1.0 - gate) * preference_logits
        routed_logits = routed_logits.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return routed_logits


class replay_bias(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.input_size = args.num_pois
        self.user_count = args.num_users
        self.hidden_size = args.hidden_dim
        self.loc_count = args.num_pois
        self.device = args.device

        self.f_t = lambda delta_t, user_len: ((torch.cos(delta_t * 2 * np.pi / 86400) + 1) / 2) * torch.exp(
            -(delta_t / 86400 * args.lambda_t)
        )
        self.f_s = lambda delta_s, user_len: torch.exp(-(delta_s * args.lambda_s))
        self.week_matrix = generate_tensor_of_distribution(168).to(self.device)
        self.week_weight_index = torch.tensor([x - 84 for x in range(168)]).repeat(168, 1).to(args.device)

        self.encoder = nn.Embedding(self.input_size, self.hidden_size)
        self.user_encoder = nn.Embedding(self.user_count, self.hidden_size)
        self.week_encoder = nn.Embedding(24 * 7, self.hidden_size // 2)

        rnn = RnnFactory(args.rnn)
        self.rnn = rnn.create(self.hidden_size)
        self.fc = nn.Linear(3 * self.hidden_size - self.hidden_size // 2, self.input_size)
        self.fcpt = nn.Linear(2 * self.hidden_size - self.hidden_size // 2, self.hidden_size)
        self.week_distribution = User_Week_Distribution(168)
        self.routing_head = BiasAwareRoutingHead(
            hidden_dim=3 * self.hidden_size - self.hidden_size // 2,
            num_pois=self.input_size,
            dropout=getattr(args, "routing_dropout", 0.1),
        )

        mu = 0.0
        sd = 1.0 / self.hidden_size
        mem = torch.randn(self.user_count, self.hidden_size) * sd + mu
        self.register_buffer("memory", mem)

        self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=0)

    def handle_sequence(self, batch_data):
        poi_id, time, time_delta = batch_data["POI_id"], batch_data["timestamps"], batch_data["time_slot"]
        seq_len = batch_data["mask"]
        y_poi_id, y_time, y_time_delta = (
            batch_data["y_POI_id"]["POI_id"],
            batch_data["y_POI_id"]["timestamps"],
            batch_data["y_POI_id"]["time_slot"],
        )

        bsz, seq_len_total = batch_data["POI_id"].size()
        y_poi_seq = torch.full((bsz, seq_len_total), 0, dtype=torch.long, device=self.device)
        y_time_seq = torch.full((bsz, seq_len_total), 0, dtype=torch.long, device=self.device)
        y_time_delta_seq = torch.full((bsz, seq_len_total), 0, dtype=torch.float, device=self.device)

        for i in range(bsz):
            end = seq_len[i].item()
            y_poi_seq[i, :end] = torch.cat((poi_id[i, 1:end], y_poi_id[i].unsqueeze(dim=-1)), dim=-1)
            y_time_seq[i, :end] = torch.cat((time[i, 1:end], y_time[i].unsqueeze(dim=-1)), dim=-1)
            y_time_delta_seq[i, :end] = torch.cat((time_delta[i, 1:end], y_time_delta[i].unsqueeze(dim=-1)), dim=-1)

        batch_data["y_POI_id"]["POI_id"] = y_poi_seq
        batch_data["y_POI_id"]["timestamps"] = y_time_seq
        batch_data["y_POI_id"]["time_slot"] = y_time_delta_seq
        return batch_data

    def _build_padding_mask(self, lengths, seq_len):
        indices = torch.arange(seq_len, device=lengths.device).unsqueeze(0).expand(lengths.shape[0], seq_len)
        return indices >= lengths.unsqueeze(1)

    def forward(self, batch_data):
        coordinates = torch.stack([batch_data["latitude"], batch_data["longitude"]], dim=-1)
        x, t, t_slot, s, y_t_slot, active_user, lengths = (
            batch_data["POI_id"],
            batch_data["timestamps"],
            batch_data["time_slot"],
            coordinates,
            batch_data["y_POI_id"]["time_slot"].long(),
            batch_data["user_id"],
            batch_data["mask"],
        )

        batch_size, seq_len = x.size()
        h = self.memory[active_user].unsqueeze(dim=0)

        week_weight = self.week_distribution(self.week_weight_index).view(168, 168)
        new_week_weight1 = week_weight.index_select(0, t_slot.view(-1)).view(batch_size, seq_len, 168, 1)
        new_week_weight2 = week_weight.index_select(0, y_t_slot.view(-1)).view(batch_size, seq_len, 168, 1)

        w_t1 = self.week_matrix.index_select(0, t_slot.view(-1)).view(batch_size, seq_len, -1)
        w_t1 = self.week_encoder(w_t1).permute(0, 1, 3, 2)
        w_t1 = torch.matmul(w_t1, new_week_weight1).squeeze(dim=-1)
        t_emb1 = w_t1

        w_t2 = self.week_matrix.index_select(0, y_t_slot.view(-1)).view(batch_size, seq_len, -1)
        w_t2 = self.week_encoder(w_t2).permute(0, 1, 3, 2)
        w_t2 = torch.matmul(w_t2, new_week_weight2).squeeze(dim=-1)
        t_emb2 = w_t2

        x_emb = self.encoder(x)
        poi_time = self.fcpt(torch.cat((x_emb, t_emb1), dim=-1))
        out, h = self.rnn(poi_time, h)
        out = out.transpose(0, 1)

        T, U, H = out.shape
        device = out.device
        dtype = out.dtype
        t = t.transpose(0, 1).to(device=device, dtype=dtype)
        s = s.transpose(0, 1).to(device=device, dtype=dtype)
        t_emb2 = t_emb2.transpose(0, 1).to(device=device, dtype=dtype)

        dist_t = torch.clamp(t[:, None, :] - t[None, :, :], min=0)
        dist_s = torch.norm(s[:, None, :, :] - s[None, :, :, :], dim=-1)

        a = self.f_t(dist_t, U).to(dtype)
        b = self.f_s(dist_s, U).to(dtype)
        w = a * b + 1e-10

        tril_mask = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))
        w = w.masked_fill(~tril_mask[:, :, None], 0.0)

        num = torch.einsum("tju,juh->tuh", w, out)
        den = w.sum(dim=1).unsqueeze(-1).clamp_min(1e-10)
        out_w = num / den

        p_u = self.user_encoder(active_user).reshape(U, H).to(device=device, dtype=dtype)
        p_u_exp = p_u.unsqueeze(0).expand(T, U, H)

        out_pu = torch.cat([out_w, p_u_exp, t_emb2], dim=-1)
        y_linear = self.fc(out_pu)

        seq_repr = out_pu.transpose(0, 1)
        padding_mask = self._build_padding_mask(lengths, seq_len)
        routed_logits = self.routing_head(seq_repr, padding_mask)
        y_linear = y_linear.transpose(0, 1) + routed_logits

        end = batch_data["end"]
        end_mask = end.to(torch.bool)
        with torch.no_grad():
            self.memory[active_user[end_mask]] = h.squeeze(0)[end_mask]

        return y_linear

    def calculate_loss(self, batch_data):
        batch_data = self.handle_sequence(batch_data)
        y = batch_data["y_POI_id"]["POI_id"]
        out = self.forward(batch_data).transpose(1, 2)
        return self.cross_entropy_loss(out, y)

    def predict(self, batch_data):
        batch_data = self.handle_sequence(batch_data)
        y_pred_poi = self.forward(batch_data)
        batch_indices = torch.arange(batch_data["mask"].shape[0], device=batch_data["mask"].device)
        y_pred = y_pred_poi[batch_indices, batch_data["mask"] - 1]
        return y_pred
