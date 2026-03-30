import math
from nltk import ngrams
import torch
import torch.nn as nn
import numpy as np

from sklearn.cluster import KMeans
import pandas as pd

def build_region_id(raw_df: pd.DataFrame, num_clusters: int=300) -> dict:
    ldf = raw_df[['longitude', 'latitude']]
    data = np.array(ldf)
    region_id = KMeans(n_clusters=num_clusters, max_iter=1000).fit_predict(data).tolist()
    region_id_dict = {k: v for k, v in zip(raw_df['POI_id'], region_id)}
    return region_id_dict