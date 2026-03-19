import networkx as nx
import dgl
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from datetime import datetime

def build_region_id(raw_df: pd.DataFrame, num_clusters: int=300) -> list:
    ldf = raw_df[['longitude', 'latitude']]
    data = np.array(ldf)
    region_id = KMeans(n_clusters=num_clusters, max_iter=1000).fit_predict(data).tolist()
    return region_id

def add_true_node(tree, trajectory, index, parent_node_id, nary):
    for i in range(nary - 1, 0, -1):
        if index - i >= 0:
            node_id = tree.number_of_nodes()
            node = trajectory[index - i]
            tree.add_node(node_id, x=node['features'], time=node['time'], y=node['labels'], mask=1, mask2=0, type=2)
            tree.add_edge(node_id, parent_node_id)
        else:  # empty node
            node_id = tree.number_of_nodes()
            tree.add_node(node_id, x=[0] * 4, time=0, y=[-1] * 3, mask=0, mask2=0, type=-1)
            tree.add_edge(node_id, parent_node_id)

    sub_parent_node_id = tree.number_of_nodes()
    tree.add_node(sub_parent_node_id, x=[0] * 4, time=0, y=[-1] * 3, mask=0, mask2=0, type=-1)
    tree.add_edge(sub_parent_node_id, parent_node_id)

    if index - (nary - 1) > 0:
        add_true_node(tree, trajectory, index - (nary - 1), sub_parent_node_id, nary)
        tree.add_node(sub_parent_node_id, x=[0] * 4, time=0, y=trajectory[index - (nary - 1)]['labels'], mask=0,
                      mask2=0, type=-1)


def add_period_node(tree, trajectory, nary):
    node_id = tree.number_of_nodes()
    period_label = trajectory[len(trajectory) - 1]['labels'] if len(trajectory) > 0 else [-1] * 3
    tree.add_node(node_id, x=[0] * 4, time=0, y=period_label, mask=0, mask2=1, type=1)

    if len(trajectory) > 0:
        add_true_node(tree, trajectory, len(trajectory), node_id, nary)

    return node_id

def add_day_node(tree, trajectory, labels, index, nary):
    node_id = tree.number_of_nodes()
    tree.add_node(node_id, x=[0] * 4, time=0, y=labels[index], mask=0, mask2=1, type=0)
    if index > 0:  # recursion
        child_node_id = add_day_node(tree, trajectory, labels, index - 1, nary)
        tree.add_edge(child_node_id, node_id)
    else:
        fake_node_id = tree.number_of_nodes()
        tree.add_node(fake_node_id, x=[0] * 4, time=0, y=[-1] * 3, mask=0, mask2=0, type=-1)
        tree.add_edge(fake_node_id, node_id)

    day_trajectory = trajectory[index]
    for i in range(len(day_trajectory)):  # Four time periods， 0-6， 6-12， 12-18， 18-24
        period_node_id = add_period_node(tree, day_trajectory[i], nary)
        tree.add_edge(period_node_id, node_id)

    return node_id

def construct_MobilityTree(trajectory, labels, nary):
    tree = nx.DiGraph()
    add_day_node(tree, trajectory, labels, len(trajectory) - 1, nary)# optional

    dgl_tree = dgl.from_networkx(tree, node_attrs=['x', 'time', 'y', 'mask', 'mask2', 'type'])
    return dgl_tree

def handle_data(batch_data, n_time_slot, test):
    # TODO 训练集/验证集/测试集是否有不同
    trajectories_list, labels_list = [], []
    
    for idx in range(len(batch_data['user_id'])):
        cur_tim = datetime.fromtimestamp(batch_data['timestamps'][idx][0])
        cur_day_of_year = pd.Timestamp(cur_tim).day_of_year
        
        trajectory = []
        label = []
        trajectory.append([[] for _ in range(n_time_slot)])
            
        total_seq_len = batch_data['mask'][idx]
    
        for seq_id in range(total_seq_len):
            pid = batch_data['POI_id'][idx][seq_id].item()
            cid = batch_data['POI_catid'][idx][seq_id].item()
            tim = datetime.fromtimestamp(batch_data['timestamps'][idx][seq_id].item())
            region_id = batch_data['region_id'][idx][seq_id].item()
            
            if seq_id == (total_seq_len - 1):
                y_pid = batch_data['y_POI_id']['POI_id'][idx].item()
                y_cid = batch_data['y_POI_id']['POI_catid'][idx].item()
                y_tim = datetime.fromtimestamp(batch_data['y_POI_id']['timestamps'][idx].item())
                y_region = batch_data['y_POI_id']['region_id'][idx].item()
            else:
                y_pid = batch_data['POI_id'][idx][seq_id+1].item()
                y_cid = batch_data['POI_catid'][idx][seq_id+1].item()
                y_tim = datetime.fromtimestamp(batch_data['timestamps'][idx][seq_id+1].item())
                y_region = batch_data['region_id'][idx][seq_id+1].item()
            
            user_id = batch_data['user_id'][idx].item()
            features = [user_id, pid, cid, region_id]
            tim_info = tim.hour * 4 + int(tim.minute / 15)
            
            label_item = [y_pid, y_cid, y_region]
            if test and (seq_id != (total_seq_len - 1)):
                label_item = [-1, -1, -1]
            
            checkin = {'features': features, 'time': tim_info, 'labels': label_item}
            
            if pd.Timestamp(y_tim).day_of_year != pd.Timestamp(tim).day_of_year or seq_id == total_seq_len - 1:
                label.append(label_item)
            if pd.Timestamp(tim).day_of_year == cur_day_of_year:
                trajectory[-1][int(tim.hour / (24 / n_time_slot))].append(checkin)
            else:
                cur_day_of_year = pd.Timestamp(tim).day_of_year
                trajectory.append([[] for _ in range(n_time_slot)])
                trajectory[-1][int(tim.hour / (24 / n_time_slot))].append(checkin)
                
        trajectories_list.append(trajectory)
        labels_list.append(label)
    
    return trajectories_list, labels_list