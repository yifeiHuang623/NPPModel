import torch
import numpy as np
from utils import *
import pandas as pd
from sklearn.cluster import KMeans

def generate_tensor_of_distribution(time):
    list1=[]
    temp=[i for i in range(time)]
    for i in range(time):
        if i == time//2:
            list1.append(temp)
        elif i<time//2:
            list1.append(temp[-(time//2-i):]+temp[:-(time//2-i)])
        else :
            list1.append(temp[(i-time//2):]+temp[:(i-time//2)])
    return torch.tensor(list1)

def rotate_batch(head, relation, hidden):
    pi = 3.14159265358979323846
    device = head.device    
    re_head, im_head = torch.chunk(head, 2, dim=2)

    #Make phases of relations uniformly distributed in [-pi, pi]
    embedding_range = torch.nn.Parameter(
                    torch.Tensor([(24.0 + 2.0) / hidden]), 
                    requires_grad=False
            ).to(device)

    phase_relation = relation/(embedding_range/pi)

    re_relation = torch.cos(phase_relation)
    im_relation = torch.sin(phase_relation)


    re_score = re_head * re_relation - im_head * im_relation
    im_score = re_head * im_relation + im_head * re_relation

    score = torch.cat([re_score, im_score], dim = 2)
    return score

def build_region_id(raw_df: pd.DataFrame, num_clusters: int=4000) -> dict:
    ldf = raw_df[['longitude', 'latitude']]
    data = np.array(ldf)
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(data)
    labels = kmeans.labels_
    
    poi2region = {poi_id: labels[i] for i, poi_id in enumerate(raw_df['POI_id'])}
    return poi2region