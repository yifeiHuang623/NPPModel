import math
import torch
from torch import nn
from torch.nn import Module
from torch_geometric.nn import GatedGraphConv
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from .iPCM_utils import softmax
import numpy as np

class POIGraph(Module):
    def __init__(self, n_nodes, hidden_size):
        super(POIGraph, self).__init__()
        self.hidden_size = hidden_size
        self.n_nodes = n_nodes
        self.embedding = nn.Embedding(self.n_nodes, self.hidden_size)
        self.ggnn = GatedGraphConv(self.hidden_size, num_layers=2)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, inputs, A):
        hidden = self.embedding(inputs)
        hidden = self.ggnn(hidden, A)
        return hidden

    def getembedding(self, inputs):
        return self.embedding(inputs)
   
class UserEmbeddings(nn.Module):
    def __init__(self, num_users, embedding_dim):
        super(UserEmbeddings, self).__init__()
        self.embedding_dim = embedding_dim
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=embedding_dim,)
        self.linear_1 = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.embedding_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
    def forward(self, user_idx, mean_poi_embeddings):
        embed = self.user_embedding(user_idx)
        embed = embed + mean_poi_embeddings
        embed = self.leaky_relu(self.linear_1(embed))
        return embed
    
class FuseEmbeddings(nn.Module):
    def __init__(self, embed_dim1, embed_dim2):
        super(FuseEmbeddings, self).__init__()
        embed_dim = embed_dim1 + embed_dim2
        self.fuse_embed = nn.Linear(embed_dim, embed_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, embed1, embed2):
        x = self.fuse_embed(torch.cat((embed1, embed2), -1))
        x = self.leaky_relu(x)
        return x

def t2v(tau, f, out_features, w, b, w0, b0, arg=None):
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    # TODO dim=0
    return torch.cat([v1, v2], -1)

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

class Time2Vec(nn.Module):
    def __init__(self, out_dim):
        super(Time2Vec, self).__init__()
        self.sin = SineActivation(1, out_dim)

    def forward(self, x):
        x = self.sin(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, num_poi, num_regions, num_times, embed_size, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        encoder_layers = TransformerEncoderLayer(embed_size, nhead, nhid, dropout, batch_first = True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embed_size = embed_size
        self.decoder_poi = nn.Linear(embed_size, num_poi)
        self.decoder_time = nn.Linear(embed_size, num_times)
        self.decoder_region = nn.Linear(embed_size, num_regions)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder_poi.bias.data.zero_()
        self.decoder_poi.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask, src_key_padding_mask):
        src = src * math.sqrt(self.embed_size)
        src = self.pos_encoder(src)
        x = self.transformer_encoder(src, src_mask, src_key_padding_mask)
        out_poi = self.decoder_poi(x)
        out_time = self.decoder_time(x)
        out_region = self.decoder_region(x)
        return out_poi, out_time, out_region

class iPCM(nn.Module):
    def __init__(self, args, graph_dict) -> None:
        super().__init__()
        
        self.num_pois = args.num_pois
        self.num_users = args.num_users
        self.num_regions = args.num_regions
        self.num_times = args.num_times
        
        self.device = args.device
        self.poi2region = args.poi2region
        
        self.time_embed_model = Time2Vec(args.time_embed_dim).to(self.device)
        self.embed_fuse_model = FuseEmbeddings(args.poi_embed_dim + args.user_embed_dim, args.time_embed_dim + args.region_embed_dim).to(self.device)
        dim = args.poi_embed_dim + args.user_embed_dim + args.time_embed_dim + args.region_embed_dim
        self.seq_model = TransformerModel(self.num_pois,
                                    self.num_regions,
                                    self.num_times,
                                        dim,
                                        args.transformer_nhead,
                                        args.transformer_nhid,
                                        args.transformer_nlayers,
                                        dropout=args.transformer_dropout,
                                    ).to(self.device)
        
        self.poi_embed_model = POIGraph(self.num_pois, args.poi_embed_dim).to(self.device)
        self.user_embed_model = UserEmbeddings(self.num_users, args.user_embed_dim).to(self.device)
        self.region_embed_model = POIGraph(self.num_regions, args.region_embed_dim).to(self.device)
        
        self.poi_edge_idx = torch.LongTensor(graph_dict["poi_edge_index"]).to(self.device)
        self.region_edge_idx = torch.LongTensor(graph_dict["region_edge_index"]).to(self.device)
        self.user_poi = torch.FloatTensor(graph_dict["user_poi"]).to(self.device)
        self.user_time_poi = graph_dict["user_time_poi"]
        
        self.criterion_poi = nn.CrossEntropyLoss(ignore_index=0)
        self.criterion_region = nn.CrossEntropyLoss(ignore_index=0)
        self.criterion_time = nn.CrossEntropyLoss(ignore_index=0)
        
        self.time_period_mapping = args.time_period_mapping
        
    def get_predict(self, batch_data):
        user_id, poi_id, mask = batch_data['user_id'], batch_data['POI_id'], batch_data['mask']
        time_period, region_id = batch_data['time_period'], batch_data['region_id']
        
        poi_embeddings = self.poi_embed_model(torch.arange(self.num_pois).to(self.device), self.poi_edge_idx)
        region_embeddings = self.region_embed_model(torch.arange(self.num_regions).to(self.device), self.region_edge_idx)
        user_embeddings = self.user_embed_model(torch.arange(self.num_users).to(self.device), torch.mm(self.user_poi, poi_embeddings))
        
        indices = torch.arange(poi_id.shape[1]).unsqueeze(0).expand(poi_id.shape[0], poi_id.shape[1]).to(self.device)
        mask = (indices < mask.unsqueeze(1)).to(torch.bool)
        
        user_id = user_id.unsqueeze(dim=-1).expand(poi_id.shape[0], poi_id.shape[1])
        user_embed = user_embeddings[user_id[mask]]
        poi_embed = poi_embeddings[poi_id[mask]]
        region_embed = region_embeddings[region_id[mask]]

        time_embed = self.time_embed_model(time_period[mask].unsqueeze(dim=-1).to(torch.float)).squeeze(dim=-1)
        
        user_poi_cat = torch.cat((user_embed, poi_embed), dim=-1)
        region_time_cat = torch.cat((time_embed, region_embed), dim=-1)
        fused_emb_mask = self.embed_fuse_model(user_poi_cat, region_time_cat)
        
        fused_emb = torch.zeros(poi_id.shape[0], poi_id.shape[1], fused_emb_mask.shape[-1]).to(self.device)
        indices = mask.nonzero(as_tuple=True)
        fused_emb[indices[0], indices[1]] = fused_emb_mask
        
        src_mask = (torch.triu(torch.ones(poi_id.size(1), poi_id.size(1))) == 1).transpose(0, 1)
        src_mask = src_mask.float().masked_fill(src_mask == 0, float('-inf')).masked_fill(src_mask == 1, float(0.0)).to(self.device)
        src_key_mask = ~mask
        y_pred_poi, y_pred_time, y_pred_region = self.seq_model(fused_emb, src_mask, src_key_mask)
        return y_pred_poi, y_pred_time, y_pred_region
        
    def forward(self, batch_data):
        y_poi_id, y_time_period, y_region = batch_data['y_POI_id']['POI_id'], batch_data['y_POI_id']['time_period'], batch_data['y_POI_id']['region_id']
        
        B, L = batch_data['POI_id'].size()
        y_poi_seq = torch.full((B, L), 0, dtype=torch.long, device=self.device)
        y_timestamp_seq = torch.full((B, L), 0, dtype=torch.long, device=self.device)
        y_region_seq = torch.full((B, L), 0, dtype=torch.long, device=self.device)
        valid_len = batch_data['mask']
        
        for i in range(B):
            end = valid_len[i].item()
            if end > 1:  
                y_poi_seq[i, :end] = torch.cat((batch_data['POI_id'][i, 1:end], y_poi_id[i].unsqueeze(dim=-1)), dim=-1)
                y_timestamp_seq[i, :end] = torch.cat((batch_data['time_period'][i, 1:end], y_time_period[i].unsqueeze(dim=-1)), dim=-1)
                y_region_seq[i, :end] = torch.cat((batch_data['region_id'][i, 1:end], y_region[i].unsqueeze(dim=-1)), dim=-1)
        
        y_pred_poi, y_pred_time, y_pred_region = self.get_predict(batch_data)
        
        loss_poi = self.criterion_poi(y_pred_poi.transpose(1, 2), y_poi_seq)
        loss_time = self.criterion_time(y_pred_time.transpose(1, 2), y_timestamp_seq)
        loss_region = self.criterion_region(y_pred_region.transpose(1, 2), y_region_seq)
        loss = loss_poi + loss_region + loss_time
        
        return loss
    
    def predict(self, batch_data):
        user_id, poi_id, mask = batch_data['user_id'], batch_data['POI_id'], batch_data['mask']
        user_id = user_id.unsqueeze(dim=1).expand(poi_id.shape[0], poi_id.shape[1])
        indices = torch.arange(user_id.shape[1]).unsqueeze(0).expand(user_id.shape[0], user_id.shape[1]).to(self.device)
        mask = (indices < mask.unsqueeze(1)).to(torch.bool)
        
        y_pred_poi, y_pred_time, y_pred_region = self.get_predict(batch_data)
        
        y_pred_poi_np = y_pred_poi.detach().cpu().numpy()
        y_pred_time_np = y_pred_time.detach().cpu().numpy()
        y_pred_region_np = y_pred_region.detach().cpu().numpy()
        
        total_batch_pred_pois = self.adjust_pred_pro(y_pred_poi_np, y_pred_region_np, user_id, y_pred_time_np, mask)
        total_batch_pred_pois = total_batch_pred_pois[np.arange(user_id.shape[0]), mask.sum(dim=-1).cpu().numpy()-1]
        
        batch_indices = torch.arange(batch_data['mask'].shape[0])
        y_pred_time = y_pred_time[batch_indices, batch_data['mask'] - 1]
        max_indices = torch.argmax(y_pred_time, dim=1)  
        y_pred_time = [self.time_period_mapping[int(i)] * 24 for i in max_indices]
        y_truth_time = [self.time_period_mapping[int(i)] * 24 for i in batch_data['y_POI_id']['time_period']]
    
        return total_batch_pred_pois, y_pred_time, y_truth_time
        
    def adjust_pred_pro(
        self,
        y_pred_poi: np.ndarray,
        y_pred_region: np.ndarray,
        batch_seq_users: torch.Tensor,
        y_pred_time: np.ndarray,
        mask: torch.Tensor,
    ) -> np.ndarray:
        y_pred_poi_adjusted = np.zeros_like(y_pred_poi)

        for i in range(batch_seq_users.shape[0]):
            useridx = batch_seq_users[i][0]
            traj_i_input = (mask[i] == 1).sum().item()
            j = int(traj_i_input) - 1
            region_pro = y_pred_region[i][j]
            time_pro = y_pred_time[i][j]
            poi_pro = y_pred_poi[i, j, :]
            pro = np.zeros(self.num_pois)

            regions = region_pro.argsort()[-20:]
            time_sort = time_pro.argsort()
            times1 = time_sort[-1]
            times2 = time_sort[-2]
            times3 = time_sort[-3]
            times4 = time_sort[-4]
            times5 = time_sort[-5]

            
            tmp_user_time_poi = self.user_time_poi[useridx][times1] + self.user_time_poi[useridx][times2] + self.user_time_poi[useridx][times3] + self.user_time_poi[useridx][times4] + self.user_time_poi[useridx][times5]
            index = np.where(tmp_user_time_poi != 0)
            for idx in index[0]:
                if self.poi2region[idx] in regions:
                    pro[idx] = 1.0
            
            poi_pro = softmax(poi_pro)
            y_pred_poi_adjusted[i, j, :] = pro + poi_pro
        return y_pred_poi_adjusted

        
        
        