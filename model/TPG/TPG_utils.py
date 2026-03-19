import math
from nltk import ngrams
import torch
import torch.nn as nn
import numpy as np
from sklearn.neighbors import BallTree
from tqdm import tqdm
from collections import defaultdict
from collections import Counter, OrderedDict
from torchtext.vocab import vocab

EarthRadius = 6378137
MinLatitude = -85.05112878
MaxLatitude = 85.05112878
MinLongitude = -180
MaxLongitude = 180


def clip(n, min_value, max_value):
    return min(max(n, min_value), max_value)


def map_size(level_of_detail):
    return 256 << level_of_detail


def latlon2pxy(latitude, longitude, level_of_detail, swin_type):
    latitude = clip(latitude, MinLatitude, MaxLatitude)
    longitude = clip(longitude, MinLongitude, MaxLongitude)

    x = (longitude + 180) / 360
    sin_latitude = math.sin(latitude * math.pi / 180)
    y = 0.5 - math.log((1 + sin_latitude) / (1 - sin_latitude)) / (4 * math.pi)

    size = map_size(level_of_detail)
    pixel_x = int(clip(x * size + 0.5, 0, size - 1))
    pixel_y = int(clip(y * size + 0.5, 0, size - 1))
    interval = 64
    if swin_type == "cross":
        # [original, 上，下，左，右]
        return [pixel_x, pixel_x, pixel_x, pixel_x+interval, pixel_x-interval], \
                [pixel_y, pixel_y-interval, pixel_y+interval, pixel_y, pixel_y]
    elif swin_type == "grid":
        # [original, 左上，右上，左下，右下]
        return [pixel_x, pixel_x+interval, pixel_x-interval, pixel_x+interval, pixel_x-interval], \
                [pixel_y, pixel_y-interval, pixel_y-interval, pixel_y+interval, pixel_y+interval]
    elif swin_type == "mix":
        # [original, 上，下，左，右，左上，右上，左下，右下]
        return [pixel_x, pixel_x, pixel_x, pixel_x+interval, pixel_x-interval, pixel_x+interval, pixel_x-interval, pixel_x+interval, pixel_x-interval], \
                [pixel_y, pixel_y-interval, pixel_y+interval, pixel_y, pixel_y, pixel_y-interval, pixel_y-interval, pixel_y+interval, pixel_y+interval]
    else:
        raise ValueError("swin type {} is not available!".format(swin_type))

def txy2quadkey(tile_x, tile_y, level_of_detail):
    quadkey_list = []
    
    for x, y in zip(tile_x, tile_y):
        quadkey = []
        for i in range(level_of_detail, 0, -1):
            digit = 0
            mask = 1 << (i - 1)
            if (x & mask) != 0:
                digit += 1
            if (y & mask) != 0:
                digit += 2
            quadkey.append(str(digit))
        quadkey_list.append(''.join(quadkey))
    return quadkey_list


def pxy2txy(pixel_x, pixel_y):
    tile_x = []
    tile_y = []
    for x, y in zip(pixel_x, pixel_y):
        tile_x.append(x // 256)
        tile_y.append(y // 256)
    return tile_x, tile_y


def latlon2quadkey(lat, lon, level, swin_type):
    """
    经纬度 to quadkey 转换函数
    """
    pixel_x, pixel_y = latlon2pxy(lat, lon, level, swin_type)
    tile_x, tile_y = pxy2txy(pixel_x, pixel_y)
    return txy2quadkey(tile_x, tile_y, level)

def build_region_id(poi_id_seqs, lat_seqs, lon_seqs, length=9):
    region_quadkey_bigrams_map, all_quadkey_bigrams = {}, []
    for _ in range(length):
        all_quadkey_bigrams.append([])
    for poi_id, lat, lon in zip(poi_id_seqs, lat_seqs, lon_seqs):
        regions = latlon2quadkey(float(lat), float(lon), 17, 'mix')
        region_quadkey_bigrams = []
        for idx, region_quadkey in enumerate(regions):
            region_quadkey_bigram = ' '.join([''.join(x) for x in ngrams(region_quadkey, 6)])
            region_quadkey_bigram = region_quadkey_bigram.split()
            all_quadkey_bigrams[idx].append(region_quadkey_bigram)
            region_quadkey_bigrams.append(region_quadkey_bigram)
        region_quadkey_bigrams_map[poi_id] = region_quadkey_bigrams
    # add padding
    region_quadkey_bigrams_map[0] = region_quadkey_bigrams_map[1]
    return region_quadkey_bigrams_map, all_quadkey_bigrams

class KNNSampler(nn.Module):
    def __init__(self, query_sys, user_visited_locs, user_visited_times, num_nearest=100, exclude_visited=False, train=True):
        nn.Module.__init__(self)
        self.query_sys = query_sys
        self.num_nearest = num_nearest
        self.user_visited_locs = user_visited_locs
        self.user_visited_times = user_visited_times
        self.user_visited_locs2times = self.locs2times(user_visited_locs, user_visited_times)
        self.exclude_visited = exclude_visited
        self.sampler = "RandomSampler"
        self.clip = False
        self.nearby_times_dic = {}
        self.nearby_times_dic[0] = []
        self.train = train
        for i in range(1, 170):
            self.nearby_times_dic[i] = self.nearby_times_sampler(i)
    
    def locs2times(self, user_visited_locs, user_visited_times):
        locs2times = {}
        for user in user_visited_locs.keys():
            locs = user_visited_locs[user]
            times = user_visited_times[user]
            locs2times[user] = {}
            for loc, time in zip(locs, times):
                locs2times[user][loc] = time
        return locs2times
    
    def nearby_times_sampler(self, time):
        weekday = (time - 1) // 24
        hour = (time - 1) % 24
        if weekday == 7:
            weekday = 6
            hour = 24

        nearby_times = set()
        # Workday
        if 0 <= weekday <= 4:
            for i in range(5):
                time_temp = i * 24 + hour + 1
                for j in range(3):
                    a = time_temp-j
                    b = time_temp+j
                    if a < 1:
                        a += 169
                    if b > 169:
                        b -= 169
                    nearby_times.add(a)
                    nearby_times.add(b)
        # Weekend
        else:
            for i in range(5, 7):
                time_temp = i * 24 + hour + 1
                for j in range(4):
                    a = time_temp-j
                    b = time_temp+j
                    if a < 1:
                        a += 169
                    if b > 169:
                        b -= 169
                    nearby_times.add(a)
                    nearby_times.add(b)
        
        nearby_times.remove(time)
        return list(nearby_times)
    
    def forward(self, trg_seq, k, user, **kwargs):
        """
            基于query_sys从候选集中随机采样k个作为负样例
        """
        neg_samples = []
        times_neg_samples = []
        for idx in range(len(trg_seq['POI_id'])):
            trg_loc = trg_seq['POI_id'][idx]
            trg_time = trg_seq['time_id'][idx]
            nearby_locs = list(filter((trg_loc).__ne__, self.user_visited_locs[user])) if self.sampler == "HardSampler" else \
                            self.query_sys.get_knn(trg_loc, k=self.num_nearest)
            assert len(nearby_locs) > 0
            if self.sampler == "HardSampler":
                nearby_times = self.user_visited_times[user]
            elif self.sampler == "KNNSamper":
                nearby_times = self.nearby_times_dic[trg_time]
            else:
                nearby_times = [i for i in range(1, 170) if i != trg_time]
            locs2times = self.user_visited_locs2times[user]
            if not self.exclude_visited:
                samples = np.random.choice(nearby_locs, size=k, replace=True)
                times_samples = np.array([locs2times[loc] for loc in samples]) if self.sampler == "HardSampler" else \
                                    np.random.choice(nearby_times, size=k, replace=True)
            else:
                samples = []
                times_samples = []
                for _ in range(k):
                    sample = np.random.choice(nearby_locs)
                    times_sample = np.random.choice(nearby_times)
                    while sample in self.user_visited_locs[user]:
                        sample = np.random.choice(nearby_locs)
                    while times_sample in self.user_visited_times[user]:
                        times_sample = np.random.choice(nearby_times)
                    samples.append(sample)
                    times_samples.append(times_sample)
            neg_samples.append(samples)
            times_neg_samples.append(times_samples)
        neg_samples = torch.tensor(np.array(neg_samples), dtype=torch.long)
        times_neg_samples = torch.tensor(np.array(times_neg_samples), dtype=torch.long)
        probs = torch.ones((neg_samples.size(0), (k+1)*(k+1)-1), dtype=torch.float32) if self.clip else \
                    torch.ones_like(neg_samples, dtype=torch.float32)
        times_probs = torch.ones((times_neg_samples.size(0), (k+1)*(k+1)-1), dtype=torch.float32) if self.clip else \
                    torch.ones_like(times_neg_samples, dtype=torch.float32)
        if self.sampler == "NonSampler":
            return neg_samples, probs, None, None
        return neg_samples, probs, times_neg_samples, times_probs
    
    
class LocQuerySystem:
    def __init__(self):
        self.coordinates = []
        self.tree = None
        self.knn = None
        self.knn_results = None
        self.radius = None
        self.radius_results = None

    def build_tree(self, poi_seqs, lat, lon):
        """
        构建KNN(基于BallTree实现)，用于sampler中的采样操作
        """
        self.coordinates = np.zeros((len(set(poi_seqs)) + 1, 2), dtype=np.float64)
        for poi_id, lat, lon in zip(poi_seqs, lat, lon):
            self.coordinates[poi_id] = [lat, lon]
        self.tree = BallTree(
            self.coordinates,
            leaf_size=1,
            metric='haversine'
        )

    def prefetch_knn(self, k=100):
        self.knn = k
        self.knn_results = np.zeros((self.coordinates.shape[0], k), dtype=np.int32)
        for idx, gps in tqdm(enumerate(self.coordinates), total=len(self.coordinates), leave=True):
            trg_gps = gps.reshape(1, -1)
            _, knn_locs = self.tree.query(trg_gps, k + 1)
            knn_locs = knn_locs[0, 1:]
            knn_locs += 1
            self.knn_results[idx] = knn_locs

    def prefetch_radius(self, radius=10.0):
        self.radius = radius
        self.radius_results = {}
        radius /= 6371000/1000
        for idx, gps in tqdm(enumerate(self.coordinates), total=len(self.coordinates), leave=True):
            trg_gps = gps.reshape(1, -1)
            nearby_locs = self.tree.query_radius(trg_gps, r=radius)
            nearby_locs = nearby_locs[0]
            nearby_locs = np.delete(nearby_locs, np.where(nearby_locs == idx))
            nearby_locs += 1
            self.radius_results[idx + 1] = nearby_locs

    def get_knn(self, trg_loc, k=100):
        if self.knn is not None and k <= self.knn:
            return self.knn_results[trg_loc - 1][:k]
        trg_gps = self.coordinates[trg_loc - 1].reshape(1, -1)
        _, knn_locs = self.tree.query(trg_gps, k + 1)
        knn_locs = knn_locs[0, 1:]
        return knn_locs

    def get_radius(self, trg_loc, r=10.0):
        if r == self.radius:
            return self.radius_results[trg_loc]
        r /= 6371000/1000
        trg_gps = self.coordinates[trg_loc - 1].reshape(1, -1)
        nearby_locs = self.tree.query_radius(trg_gps, r=r)
        nearby_locs = nearby_locs[0]
        nearby_locs = np.delete(nearby_locs, np.where(nearby_locs == trg_loc - 1))
        nearby_locs += 1
        return nearby_locs

    def radius_stats(self, radius=10):
        radius /= 6371000/1000
        num_nearby_locs = []
        for gps in tqdm(self.coordinates, total=len(self.coordinates), leave=True):
            trg_gps = gps.reshape(1, -1)
            count = self.tree.query_radius(trg_gps, r=radius, count_only=True)[0]
            num_nearby_locs.append(count)
        num_nearby_locs = np.array(num_nearby_locs, dtype=np.int32)
        max_loc_idx = np.argsort(-num_nearby_locs)[0]
        print("max #nearby_locs: {:d}, at loc {:d}".format(num_nearby_locs[max_loc_idx], max_loc_idx + 1))
        
        
def get_visited_locs_times(raw_df):
    # 不知道为什么是原代码中这么复杂的计算方式.....，而且原代码不是说的训练时Randomsampler吗，怎么又变成KNNSampler了
    print("get_visited_locs_times...")
    user_visited_locs = defaultdict(list)
    user_visited_times = defaultdict(list)
    for u in range(len(raw_df)):
        user = raw_df['user_id'][u]
        user_visited_locs[user].append(raw_df['POI_id'][u])
        user_visited_times[user].append(raw_df['time_id'][u])
                
        assert len(user_visited_locs[user]) == len(user_visited_times[user])
    return user_visited_locs, user_visited_times

class QuadkeyField:
    def __init__(self):
        self.vocab = {}
        self.idx_to_token = {}
        self.unk_token = '<unk>'
        self.pad_token = '<pad>'
        
    def build_vocab(self, data):
        # 收集所有token
        all_tokens = set()
        for seq in data:
            for item in seq:
                if isinstance(item, str):
                    tokens = item.split()
                else:
                    tokens = item
                all_tokens.update(tokens)
        
        # 构建词汇表
        self.vocab = {self.pad_token: 0, self.unk_token: 1}
        for idx, token in enumerate(sorted(all_tokens), 2):
            self.vocab[token] = idx
            
        # 反向映射
        self.idx_to_token = {v: k for k, v in self.vocab.items()}
    
    def encode(self, text):
        if isinstance(text, str):
            tokens = text.split()
        else:
            tokens = text
        return [self.vocab.get(token, self.vocab[self.unk_token]) for token in tokens]
    
    def decode(self, indices):
        return [self.idx_to_token.get(idx, self.unk_token) for idx in indices]
    
    def numericalize(self, data):
        if isinstance(data, list):
            return np.array([self.encode(item) for item in data], dtype=np.int64)
        else:
            return np.array(self.encode(data), dtype=np.int64)