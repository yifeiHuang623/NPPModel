import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from datetime import datetime

def build_time_period(raw_df: pd.DataFrame) -> list:
    """
    Build an adaptive time-period mapping for each record in raw_df.
    Returns a list: [time_period_value].
    """
    
    num_time_slices  = 72
    size_time_slices = 24 * 60 / num_time_slices
    num_times        = 20

    # Category index map
    cat_ids          = list(set(raw_df['POI_catid'].tolist()))
    cat_id2idx_dict  = dict(zip(cat_ids, range(len(cat_ids))))

    # Build category-time matrix
    cat_time = np.zeros((len(cat_ids), num_time_slices))
    for _, row in raw_df.iterrows():
        if row['POI_catid'] in cat_id2idx_dict.keys():
            c = cat_id2idx_dict[row['POI_catid']]
            t_str = datetime.fromtimestamp(row['timestamps'])
            t = int(t_str.hour * (60 / size_time_slices)) + \
                int(t_str.minute / size_time_slices)
            cat_time[c][t] += 1

    # Select top-50 categories
    cat_timesum = np.sum(cat_time, axis=1)
    top50       = cat_timesum.argsort()[-50:][::-1]
    time_cat    = cat_time[top50].T
    time_cat_sum = np.sum(time_cat, axis=1, keepdims=True)
    for i in range(num_time_slices):
        time_cat[i] = time_cat[i] / time_cat_sum[i]

    # Helper functions for DP
    def get_class_ave(samples, start, end):
        class_ave = [0.0 for _ in range(len(samples[0]))]
        class_ave = np.array(class_ave)
        for i in range(start, end):
            class_ave += samples[i]
        class_ave = class_ave / (end - start)
        return class_ave

    def get_class_diameter(samples, start, end):
        class_diameter = 0.0
        class_ave = get_class_ave(samples, start, end)
        for i in range(start, end):
            tem = samples[i] - class_ave
            for each in tem:
                class_diameter += each * each
        return class_diameter
        
    def get_split_loss(samples, sample_num, split_class_num):
        split_loss_result = np.zeros((sample_num + 1, split_class_num + 1))
        split_loss_result1 = np.zeros((sample_num + 1, split_class_num + 1),dtype=int)

        for n in range(1, sample_num + 1):
            split_loss_result[n, 1] = get_class_diameter(samples, 0, n)

        for k in range(2, split_class_num + 1):
            for n in range(k, sample_num + 1):
                loss = []
                mi = 10000
                flag = k - 1
                for j in range(k - 1, n):
                    if(split_loss_result[j, k - 1] + get_class_diameter(samples, j, n) < mi):
                        flag = j
                        mi = split_loss_result[j, k - 1] + get_class_diameter(samples, j, n)
                split_loss_result[n, k] = mi
                split_loss_result1[n, k] = flag

        return split_loss_result, split_loss_result1

    # Run DP and retrieve splits
    _, s1 = get_split_loss(time_cat, len(time_cat), len(time_cat))

    num = num_times
    last = num_time_slices
    roots = []
    while(num > 1):
        roots.append(s1[last][num])
        last = s1[last][num]
        num -= 1
    roots = roots[::-1]
    roots = [0] + roots + [num_time_slices]
    time2idx_dic = {}
    for i in range(len(roots)-1):
        left = roots[i]
        right = roots[i+1]
        for j in range(left, right):
            time2idx_dic[j] = (roots[i] + roots[i+1]) / (2.0 * num_time_slices)

    # Create dict: row_index -> time_period
    result = []
    for i in range(raw_df.shape[0]):
        time = datetime.fromtimestamp(raw_df.loc[i, 'timestamps'])
        data = time2idx_dic[int(time.hour * (60 / size_time_slices)) + int(time.minute / size_time_slices)]
        result.append(data)

    return result

def build_region_id(raw_df: pd.DataFrame, num_clusters: int=300) -> dict:
    ldf = raw_df[['longitude', 'latitude']]
    data = np.array(ldf)
    region_id = KMeans(n_clusters=num_clusters, max_iter=1000).fit_predict(data).tolist()
    region_id_dict = {k: v for k, v in zip(raw_df['POI_id'], region_id)}
    return region_id_dict

def softmax(x):
    x -= np.max(x)
    x = np.exp(x) / np.sum(np.exp(x))
    return x


def construct_graph(args, train_df):
    user_poi = np.zeros((args.num_users, args.num_pois))
    users = set(train_df['user_id'])
    user_time_poi = np.zeros((args.num_users, args.num_times, args.num_pois))
    for user in users:
        user_df = train_df[train_df['user_id'] == user]
        for i, row in user_df.iterrows():
            poi_idx = row['POI_id']
            user_poi[int(user)][int(poi_idx)] += 1
            user_time_poi[int(user)][int(row['time_period'])][int(poi_idx)] = 1
    user_poi_sum = np.sum(user_poi, axis=1)
    for i in range(1, user_poi.shape[0]):
        if user_poi_sum[i] != 0:
            user_poi[i] = user_poi[i] / user_poi_sum[i]

    poi_edge_index = [[],[]]
    region_edge_index = [[],[]]
    for traj_id in set(train_df['trajectory_id'].tolist()):
        traj_df0 = train_df[train_df['trajectory_id'] == traj_id]
        traj_df0 = traj_df0.reset_index(drop=True)
        for i in range(traj_df0.shape[0] - 1):
            poi_edge_index[0].append(traj_df0.loc[i, 'POI_id'])
            poi_edge_index[1].append(traj_df0.loc[i + 1, 'POI_id'])
            region_edge_index[0].append(traj_df0.loc[i, 'region_id'])
            region_edge_index[1].append(traj_df0.loc[i + 1, 'region_id'])
            
    return {
        'user_poi': user_poi,
        'user_time_poi': user_time_poi,
        'poi_edge_index': poi_edge_index,
        'region_edge_index': region_edge_index
    }