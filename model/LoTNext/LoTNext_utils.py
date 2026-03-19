import torch
import torch.nn as nn
import numpy as np
from scipy.sparse import coo_matrix
import scipy.sparse as sp
from torch_geometric.nn import GCNConv
from torch.nn import init

import torch.nn.functional as F
from math import radians, cos, sin, asin, sqrt
import scipy.sparse as sp

def sparse_matrix_to_tensor(graph):
    graph = coo_matrix(graph)
    vaules = graph.data
    indices = np.vstack((graph.row, graph.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(vaules)
    shape = graph.shape
    graph = torch.sparse_coo_tensor(i, v, torch.Size(shape))

    return graph

def calculate_random_walk_matrix(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()

    return random_walk_mx  # D^-1 W

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r

class DenoisingGCNNet(nn.Module):
    def __init__(self, user_dim, item_dim, out_channels):
        super(DenoisingGCNNet, self).__init__()
        self.attention_layer = AttentionLayer(user_dim, item_dim)
        self.denoising_layer = DenoisingLayer()
        self.gcn_layer = GCNLayer(user_dim, out_channels)  

    def forward(self, user_embeddings, item_embeddings, edge_index):
        # Compute attention weights for each edge
        edge_weights = self.attention_layer(user_embeddings, item_embeddings, edge_index)
        
        # Filter edges to create a denoised graph
        denoised_edge_index, denoised_edge_weights = self.denoising_layer(edge_weights, edge_index)
        
        # Combine user and item embeddings for GCN input
        gcn_input = torch.cat([user_embeddings, item_embeddings], dim=0)
        
        # Apply GCN on the denoised graph
        gcn_output = self.gcn_layer(gcn_input, denoised_edge_index)
        
        return gcn_output, denoised_edge_index, denoised_edge_weights
    
class AttentionLayer(nn.Module):
    def __init__(self, user_dim, item_dim):
        super(AttentionLayer, self).__init__()
        self.attention_fc = nn.Sequential(
            nn.Linear(user_dim + item_dim, 32),  # Suppose the hidden layer size is 128
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1)
        )
        for layer in self.attention_fc:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                init.zeros_(layer.bias)
    
    def forward(self, user_embeddings, item_embeddings, edge_index):
        # Extract edges' user and item nodes' embeddings
        user_indices = edge_index[0]
        item_indices = edge_index[1]
        user_feats = user_embeddings[user_indices]
        item_feats = item_embeddings[item_indices]

        # Concatenate user and item embeddings to compute edge weights
        edge_feats = torch.cat([user_feats, item_feats], dim=1)
        edge_weights = torch.sigmoid(self.attention_fc(edge_feats)).squeeze()

        return edge_weights

class DenoisingLayer(nn.Module):
    def __init__(self):
        super(DenoisingLayer, self).__init__()

    def forward(self, edge_weights, edge_index, threshold=0.8):
        # Apply a threshold to filter edges
        mask = edge_weights > threshold
        if mask.sum() == 0:
            # 如果所有的权重都低于阈值，保留一个最大权重的边避免孤立节点
            mask[edge_weights.argmax()] = True
        denoised_edge_index = edge_index[:, mask]
        denoised_edge_weights = edge_weights[mask]

        return denoised_edge_index, denoised_edge_weights

class GCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNLayer, self).__init__()
        self.conv1 = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        return x
    
    

class EncoderLayer(nn.Module):
    def __init__(
        self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads
    ):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads
        )
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm1 = nn.LayerNorm(hidden_size)
        self.ffn_norm2 = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None, epoch=0, mask=None):
        # y = self.self_attention_norm(x)
        y = self.self_attention(x, x, x, attn_bias, epoch, mask=1)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm1(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        x = self.ffn_norm2(x)
        return x
    
class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x

def mask_value(epoch, T, v_min=-100, v_max=-1e9):
    return -10 ** ( np.log10(-v_min) + (np.log10(-v_max) - np.log10(-v_min)) * (epoch / T) )

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None, current_epoch=0, mask=True):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias
        if mask is not None:
            seq_len = x.size(-1)
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            mask = mask.unsqueeze(0).unsqueeze(1) 
            mask = mask.expand(batch_size, 1, seq_len, seq_len).to(x.device)
            # x = x.masked_fill(mask, -100)
            x = x.masked_fill(mask, mask_value(current_epoch, 100))

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x

class Time2Vec(nn.Module):
    def __init__(self, activation, batch_size, seq_len, out_dim):
        super(Time2Vec, self).__init__()
        if activation == "sin":
            self.l1 = SineActivation1(seq_len, out_dim)
            # self.l1 = SineActivation(batch_size, seq_len, out_dim)
        elif activation == "cos":
            self.l1 = CosineActivation(seq_len, out_dim)

    def forward(self, x):
        x = self.l1(x)
        return x
    
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

class SineActivation1(nn.Module):
    def __init__(self, seq_len, out_features):
        super(SineActivation1, self).__init__()
        self.out_features = out_features
        self.l1 = nn.Linear(1, out_features-1, bias=True)
        self.l2 = nn.Linear(1, 1, bias=True)
        self.f = torch.sin

    def forward(self, tau):
        v1 = self.l1(tau.unsqueeze(-1))
        v1 = self.f(v1)
        v2 = self.l2(tau.unsqueeze(-1))
        return torch.cat([v1, v2], -1)
    
def t2v(tau, f, out_features, w, b, w0, b0):
    # tau [batch_size, seq_len], w [seq_len, out_features - 1], b [out_features - 1]
    # w0 [seq_len, 1], b0 [1]
    v1 = f(tau.unsqueeze(-1) * w + b)  # [batch_size, seq_len, out_features - 1]
    v2 = tau.unsqueeze(-1) * w0 + b0   # [batch_size, seq_len, 1]
    return torch.cat([v1, v2], -1)  # [batch_size, seq_len, out_features]

class FuseEmbeddings(nn.Module):
    def __init__(self, user_embed_dim, poi_embed_dim):
        super(FuseEmbeddings, self).__init__()
        embed_dim = user_embed_dim + poi_embed_dim
        self.fuse_embed = nn.Linear(embed_dim, embed_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, user_embed, poi_embed):
        x = self.fuse_embed(torch.cat((user_embed, poi_embed), 2))
        x = self.leaky_relu(x)
        return x