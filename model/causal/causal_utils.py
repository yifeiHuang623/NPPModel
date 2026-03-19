import math
from itertools import chain, tee
import torch
from torch import nn
from itertools import product
import pandas as pd
import numpy as np
import torch.nn.functional as F

EarthRadius = 6378137
MinLatitude = -85.05112878
MaxLatitude = 85.05112878
Minlnggitude = -180
Maxlnggitude = 180

def clip(n, minValue, maxValue):
    return min(max(n, minValue), maxValue)

def map_size(levelOfDetail):
    return 256 << levelOfDetail

def latlng2pxy(latitude, lnggitude, levelOfDetail):
    latitude = clip(latitude, MinLatitude, MaxLatitude)
    lnggitude = clip(lnggitude, Minlnggitude, Maxlnggitude)

    x = (lnggitude + 180) / 360
    sinLatitude = math.sin(latitude * math.pi / 180)
    y = 0.5 - math.log((1 + sinLatitude) / (1 - sinLatitude)) / (4 * math.pi)

    mapSize = map_size(levelOfDetail)
    pixelX = int(clip(x * mapSize + 0.5, 0, mapSize - 1))
    pixelY = int(clip(y * mapSize + 0.5, 0, mapSize - 1))
    return pixelX, pixelY

def txy2quadkey(tileX, tileY, levelOfDetail):
    quadKey = []
    for i in range(levelOfDetail, 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        if (tileX & mask) != 0:
            digit += 1
        if (tileY & mask) != 0:
            digit += 2
        quadKey.append(str(digit))

    return ''.join(quadKey)

def pxy2txy(pixelX, pixelY):
    tileX = pixelX // 256
    tileY = pixelY // 256
    return tileX, tileY

def latlng2quadkey(lat,lng,level):
    pixelX, pixelY = latlng2pxy(lat, lng, level)
    tileX, tileY = pxy2txy(pixelX, pixelY)
    return txy2quadkey(tileX, tileY,level)

def ngrams(sequence, n, **kwargs):
    """
    Return the ngrams generated from a sequence of items, as an iterator.
    For example:

        >>> from nltk.util import ngrams
        >>> list(ngrams([1,2,3,4,5], 3))
        [(1, 2, 3), (2, 3, 4), (3, 4, 5)]

    Wrap with list for a list version of this function.  Set pad_left
    or pad_right to true in order to get additional ngrams:

        >>> list(ngrams([1,2,3,4,5], 2, pad_right=True))
        [(1, 2), (2, 3), (3, 4), (4, 5), (5, None)]
        >>> list(ngrams([1,2,3,4,5], 2, pad_right=True, right_pad_symbol='</s>'))
        [(1, 2), (2, 3), (3, 4), (4, 5), (5, '</s>')]
        >>> list(ngrams([1,2,3,4,5], 2, pad_left=True, left_pad_symbol='<s>'))
        [('<s>', 1), (1, 2), (2, 3), (3, 4), (4, 5)]
        >>> list(ngrams([1,2,3,4,5], 2, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))
        [('<s>', 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, '</s>')]


    :param sequence: the source data to be converted into ngrams
    :type sequence: sequence or iter
    :param n: the degree of the ngrams
    :type n: int
    :param pad_left: whether the ngrams should be left-padded
    :type pad_left: bool
    :param pad_right: whether the ngrams should be right-padded
    :type pad_right: bool
    :param left_pad_symbol: the symbol to use for left padding (default is None)
    :type left_pad_symbol: any
    :param right_pad_symbol: the symbol to use for right padding (default is None)
    :type right_pad_symbol: any
    :rtype: sequence or iter
    """
    sequence = pad_sequence(sequence, n, **kwargs)

    # Creates the sliding window, of n no. of items.
    # `iterables` is a tuple of iterables where each iterable is a window of n items.
    iterables = tee(sequence, n)

    for i, sub_iterable in enumerate(iterables):  # For each window,
        for _ in range(i):  # iterate through every order of ngrams
            next(sub_iterable, None)  # generate the ngrams within the window.
    return zip(*iterables)  # Unpack and flattens the iterables.

def pad_sequence(
    sequence,
    n,
    pad_left=False,
    pad_right=False,
    left_pad_symbol=None,
    right_pad_symbol=None,
):
    """
    Returns a padded sequence of items before ngram extraction.

        >>> list(pad_sequence([1,2,3,4,5], 2, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))
        ['<s>', 1, 2, 3, 4, 5, '</s>']
        >>> list(pad_sequence([1,2,3,4,5], 2, pad_left=True, left_pad_symbol='<s>'))
        ['<s>', 1, 2, 3, 4, 5]
        >>> list(pad_sequence([1,2,3,4,5], 2, pad_right=True, right_pad_symbol='</s>'))
        [1, 2, 3, 4, 5, '</s>']

    :param sequence: the source data to be padded
    :type sequence: sequence or iter
    :param n: the degree of the ngrams
    :type n: int
    :param pad_left: whether the ngrams should be left-padded
    :type pad_left: bool
    :param pad_right: whether the ngrams should be right-padded
    :type pad_right: bool
    :param left_pad_symbol: the symbol to use for left padding (default is None)
    :type left_pad_symbol: any
    :param right_pad_symbol: the symbol to use for right padding (default is None)
    :type right_pad_symbol: any
    :rtype: sequence or iter
    """
    sequence = iter(sequence)
    if pad_left:
        sequence = chain((left_pad_symbol,) * (n - 1), sequence)
    if pad_right:
        sequence = chain(sequence, (right_pad_symbol,) * (n - 1))
    return sequence

def rotate(head,relation,hidden,device):
    pi = 3.14159265358979323846
    re_head, im_head = torch.chunk(head, 2, dim=-1)

    #Make phases of relations uniformly distributed in [-pi, pi]
    embedding_range = nn.Parameter(
                    torch.Tensor([(24.0 + 2.0) / hidden]), 
                    requires_grad=False
            ).to(device)

    phase_relation = relation/(embedding_range/pi)

    re_relation = torch.cos(phase_relation)
    im_relation = torch.sin(phase_relation)


    re_score = re_head * re_relation - im_head * im_relation
    im_score = re_head * im_relation + im_head * re_relation

    score = torch.cat([re_score, im_score], dim = -1)
    return score

def rotate_batch(head,relation,hidden,device):
    pi = 3.14159265358979323846
        
    re_head, im_head = torch.chunk(head, 2, dim=2)

    #Make phases of relations uniformly distributed in [-pi, pi]
    embedding_range = nn.Parameter(
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

def get_all_permutations_dict(length):
    characters = ['0', '1', '2', '3']

    # 生成所有可能的长度为6的字符串
    all_permutations = [''.join(p) for p in product(characters, repeat=length)]

    premutation_dict = dict(zip(all_permutations,range(len(all_permutations))))

    return premutation_dict

def get_norm_time96(time):
    time = pd.to_datetime(time, unit='s')
    hour = time.hour
    minute = time.minute
    
    ans = minute//15 + 4*hour
    
    return ans / 96

def get_day_norm7(time):
    time = pd.to_datetime(time, unit='s')
    day_number = time.dayofweek
    return day_number / 7

def get_time_slot_id(time):
    time = pd.to_datetime(time, unit='s')
    minute = time.minute
    hour = time.hour
    day_number = time.dayofweek
    
    if minute <= 30:
        ans = 2*hour
    else:
        ans = 2*hour + 1
    
    if day_number >= 5 :
        return ans+48
    else:
        return ans 
    
def get_ngrams_of_quadkey(quadkey,n,permutations_dict):
    region_quadkey_bigram = ' '.join([''.join(x) for x in ngrams(quadkey, n)])
    region_quadkey_bigram = region_quadkey_bigram.split()
    region_quadkey_bigram = [permutations_dict[each] for each in region_quadkey_bigram]
    return region_quadkey_bigram
    
def get_quad_keys(lat, lon, permutations_dict, quadkey_len=25, ngrams=6):
    quadkey = latlng2quadkey(lat, lon, quadkey_len)
    quadkey = get_ngrams_of_quadkey(quadkey, ngrams, permutations_dict)
    return quadkey

def _reshape_for_pairwise_rotation(x, dim):
    """
    x: [B, L, D]
    取前 dim 维做二维成对旋转，要求 dim 为偶数
    返回:
        x_rot_even: [B, L, dim/2]
        x_rot_odd:  [B, L, dim/2]
        x_pass:     [B, L, D-dim]
    """
    assert dim % 2 == 0, f"rotary dim must be even, got {dim}"
    assert x.size(-1) >= dim, f"input dim {x.size(-1)} < rotary dim {dim}"

    x_rot = x[..., :dim]
    x_pass = x[..., dim:]

    x_rot_even = x_rot[..., 0::2]
    x_rot_odd = x_rot[..., 1::2]
    return x_rot_even, x_rot_odd, x_pass


def _get_theta_from_phase(phase, dim):
    """
    phase: [B, L, P]
    我们只取前 dim 维，并使用偶数位作为每个二维对的角度。
    若 phase 的最后一维小于 dim，会报错。
    返回:
        theta: [B, L, dim/2]
    """
    assert phase.size(-1) >= dim, f"phase dim {phase.size(-1)} < rotary dim {dim}"
    phase_rot = phase[..., :dim]
    theta = phase_rot[..., 0::2]
    return theta


def rotate(x, phase, dim, device=None):
    """
    标准旋转，不带阻尼
    x: [B, L, D]
    phase: [B, L, dim] or larger
    """
    x_even, x_odd, x_pass = _reshape_for_pairwise_rotation(x, dim)
    theta = _get_theta_from_phase(phase, dim)

    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    y_even = x_even * cos_theta - x_odd * sin_theta
    y_odd = x_even * sin_theta + x_odd * cos_theta

    y = torch.stack([y_even, y_odd], dim=-1).flatten(-2)
    return torch.cat([y, x_pass], dim=-1)


def rotate_batch(x, phase, dim, device=None):
    """
    与 rotate 接口保持一致，便于兼容你原来的代码。
    """
    return rotate(x, phase, dim, device=device)


def damped_rotate(x, phase, dim, decay_rate, delta_t=None, device=None):
    """
    阻尼旋转：
        y = exp(-softplus(decay_rate) * delta_t) * Rot(theta) * x

    参数:
        x: [B, L, D]
        phase: [B, L, dim] or larger
        dim: rotary dim, must be even
        decay_rate:
            - 标量参数: shape []
            - 或 shape [dim/2]
        delta_t:
            - None: 无阻尼，退化成普通 rotate
            - [B, L]
            - [B, L, 1]
            - [B, L, dim/2]
    """
    x_even, x_odd, x_pass = _reshape_for_pairwise_rotation(x, dim)
    theta = _get_theta_from_phase(phase, dim)

    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    if delta_t is None:
        decay = 1.0
    else:
        if delta_t.dim() == 2:
            delta_t = delta_t.unsqueeze(-1)

        decay_rate_pos = F.softplus(decay_rate)

        if decay_rate_pos.dim() == 0:
            decay = torch.exp(-decay_rate_pos * delta_t)  # [B, L, 1] or [B, L, dim/2]
        else:
            decay = torch.exp(-decay_rate_pos.view(1, 1, -1) * delta_t)

    y_even = decay * (x_even * cos_theta - x_odd * sin_theta)
    y_odd = decay * (x_even * sin_theta + x_odd * cos_theta)

    y = torch.stack([y_even, y_odd], dim=-1).flatten(-2)
    return torch.cat([y, x_pass], dim=-1)


def damped_rotate_batch(x, phase, dim, decay_rate, delta_t=None, device=None):
    """
    batch 版本，接口与原 rotate_batch 对齐
    """
    return damped_rotate(x, phase, dim, decay_rate, delta_t=delta_t, device=device)