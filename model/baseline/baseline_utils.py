import numpy as np
import pandas as pd

# -----------------------------
# Helpers
# -----------------------------
def _require_cols(df, cols, name="df"):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}. Has: {list(df.columns)}")

def _to_datetime(s):
    # robust parse
    return pd.to_datetime(s, errors="coerce", utc=True, unit="s")

def _haversine_km(lat1, lon1, lat2, lon2):
    """
    Vectorized haversine distance in kilometers.
    Accepts numpy arrays / pandas Series.
    """
    lat1 = np.deg2rad(lat1.astype(float))
    lon1 = np.deg2rad(lon1.astype(float))
    lat2 = np.deg2rad(lat2.astype(float))
    lon2 = np.deg2rad(lon2.astype(float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    R = 6371.0088
    return R * c

def _js_divergence(p, q, eps=1e-12):
    """
    Jensen–Shannon divergence between two discrete distributions p and q.
    p, q are 1D numpy arrays (non-negative). We'll normalize them.
    Returns JSD in [0, ln(2)] using natural log.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def _kl(a, b):
        return np.sum(a * np.log(a / b))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)

def _freq_table(series, top_k=None, other_label="__OTHER__"):
    """
    Return frequency table (counts) for a Series.
    If top_k is set, keep top_k categories and map rest to __OTHER__.
    """
    s = series.astype("object")
    s = series.astype("object").astype(str)
    if top_k is not None:
        top = s.value_counts().head(top_k).index
        s = s.where(s.isin(top), other_label)
    return s.value_counts(dropna=False).sort_index()

def _align_counts(a_counts, b_counts):
    """
    Align two value_count Series to same index and return arrays.
    """
    idx = a_counts.index.union(b_counts.index)
    a = a_counts.reindex(idx, fill_value=0).to_numpy()
    b = b_counts.reindex(idx, fill_value=0).to_numpy()
    return idx, a, b

# -----------------------------
# 1) New entities ratios
# -----------------------------
def check_new_entities(train_df, test_df, user_col="user_id", item_col="POI_id"):
    _require_cols(train_df, [user_col, item_col], "train_df")
    _require_cols(test_df, [user_col, item_col], "test_df")

    train_users = set(train_df[user_col].dropna().unique())
    train_items = set(train_df[item_col].dropna().unique())

    test_users = test_df[user_col]
    test_items = test_df[item_col]

    new_user_mask = ~test_users.isin(train_users)
    new_item_mask = ~test_items.isin(train_items)

    # (u,i) pair novelty vs train
    train_pairs = set(zip(train_df[user_col].astype("object"), train_df[item_col].astype("object")))
    test_pairs = list(zip(test_df[user_col].astype("object"), test_df[item_col].astype("object")))
    new_pair_mask = np.array([p not in train_pairs for p in test_pairs], dtype=bool)

    out = {
        "train_n_users": len(train_users),
        "train_n_items": len(train_items),
        "test_n_rows": len(test_df),
        "test_new_user_rows": int(new_user_mask.sum()),
        "test_new_user_row_ratio": float(new_user_mask.mean()) if len(test_df) else np.nan,
        "test_new_item_rows": int(new_item_mask.sum()),
        "test_new_item_row_ratio": float(new_item_mask.mean()) if len(test_df) else np.nan,
        "test_new_pair_rows": int(new_pair_mask.sum()),
        "test_new_pair_row_ratio": float(new_pair_mask.mean()) if len(test_df) else np.nan,
        "test_unique_new_users": int(test_users[new_user_mask].nunique(dropna=True)),
        "test_unique_new_items": int(test_items[new_item_mask].nunique(dropna=True)),
    }
    return out

# -----------------------------
# 2) Time distribution drift
# -----------------------------
def check_time_drift(
    train_df,
    test_df,
    time_col="timestamp",
    tz="UTC",
    bins_hour=None,
    top_k_other=None
):
    """
    Computes train vs test distributions for hour-of-day and day-of-week.
    Returns counts + JSD.

    bins_hour: if provided, list of bin edges for hour (0-24), e.g. [0,6,12,18,24]
    top_k_other: if provided, keep only top_k categories for any high-cardinality categorical; not used here.
    """
    _require_cols(train_df, [time_col], "train_df")
    _require_cols(test_df, [time_col], "test_df")

    tr = train_df.copy()
    te = test_df.copy()
    tr[time_col] = _to_datetime(tr[time_col])
    te[time_col] = _to_datetime(te[time_col])

    # drop rows with invalid timestamps
    tr = tr.dropna(subset=[time_col])
    te = te.dropna(subset=[time_col])

    # convert to tz (both already UTC because we parse utc=True); allow conversion if desired
    if tz and tz.upper() != "UTC":
        tr[time_col] = tr[time_col].dt.tz_convert(tz)
        te[time_col] = te[time_col].dt.tz_convert(tz)

    tr_hour = tr[time_col].dt.hour
    te_hour = te[time_col].dt.hour
    tr_dow = tr[time_col].dt.dayofweek  # Mon=0
    te_dow = te[time_col].dt.dayofweek

    if bins_hour is not None:
        tr_hour_cat = pd.cut(tr_hour, bins=bins_hour, right=False, include_lowest=True)
        te_hour_cat = pd.cut(te_hour, bins=bins_hour, right=False, include_lowest=True)
        tr_hour_counts = tr_hour_cat.value_counts().sort_index()
        te_hour_counts = te_hour_cat.value_counts().sort_index()
    else:
        tr_hour_counts = tr_hour.value_counts().sort_index()
        te_hour_counts = te_hour.value_counts().sort_index()

    tr_dow_counts = tr_dow.value_counts().sort_index()
    te_dow_counts = te_dow.value_counts().sort_index()

    _, a, b = _align_counts(tr_hour_counts, te_hour_counts)
    jsd_hour = _js_divergence(a, b) if a.sum() and b.sum() else np.nan

    _, a, b = _align_counts(tr_dow_counts, te_dow_counts)
    jsd_dow = _js_divergence(a, b) if a.sum() and b.sum() else np.nan

    return {
        "n_train_valid_time": int(len(tr)),
        "n_test_valid_time": int(len(te)),
        "hour_counts_train": tr_hour_counts / len(tr_hour),
        "hour_counts_test": te_hour_counts / len(te_hour),
        "dow_counts_train": tr_dow_counts / len(tr_dow),
        "dow_counts_test": te_dow_counts / len(te_dow),
        "jsd_hour": float(jsd_hour) if jsd_hour == jsd_hour else np.nan,
        "jsd_dow": float(jsd_dow) if jsd_dow == jsd_dow else np.nan,
    }

# -----------------------------
# 3) Spatial drift: grid + user centroid drift
# -----------------------------
def add_grid_id(df, lat_col="lat", lon_col="lon", grid_km=1.0, lat0=0.0):
    """
    Adds a coarse grid id based on lat/lon.
    Uses simple equirectangular approx: 1 deg lat ~ 110.574 km
    and 1 deg lon ~ 111.320*cos(lat0) km.
    For large areas, this is approximate; good enough for drift checks.
    """
    _require_cols(df, [lat_col, lon_col], "df")
    out = df.copy()

    km_per_deg_lat = 110.574
    km_per_deg_lon = 111.320 * np.cos(np.deg2rad(lat0))

    dlat = grid_km / km_per_deg_lat
    dlon = grid_km / km_per_deg_lon if km_per_deg_lon > 1e-9 else grid_km / 111.320

    lat_bin = np.floor(out[lat_col].astype(float) / dlat).astype("int64")
    lon_bin = np.floor(out[lon_col].astype(float) / dlon).astype("int64")
    out["grid_id"] = lat_bin.astype(str) + "_" + lon_bin.astype(str)
    return out

def _top_mass(counts: pd.Series, k: int):
    if len(counts) == 0:
        return np.nan
    p = (counts / counts.sum()).sort_values(ascending=False)
    return float(p.head(k).sum())

def check_spatial_drift(
    train_df,
    test_df,
    user_col="user_id",
    lat_col="lat",
    lon_col="lon",
    grid_km=5.0,
    top_k_grids=5000,
    top_n_report=20,
    drift_thresholds_km=(1, 5, 10, 20, 50),
):
    _require_cols(train_df, [user_col, lat_col, lon_col], "train_df")
    _require_cols(test_df, [user_col, lat_col, lon_col], "test_df")

    lat0 = float(pd.concat([train_df[lat_col], test_df[lat_col]], ignore_index=True).dropna().median())
    tr = add_grid_id(train_df, lat_col, lon_col, grid_km=grid_km, lat0=lat0)
    te = add_grid_id(test_df, lat_col, lon_col, grid_km=grid_km, lat0=lat0)

    # frequency (cap)
    tr_counts = _freq_table(tr["grid_id"], top_k=top_k_grids)
    te_counts = _freq_table(te["grid_id"], top_k=top_k_grids)

    # coverage / novelty
    tr_grids = set(tr["grid_id"].dropna().unique())
    te_grids = set(te["grid_id"].dropna().unique())
    unseen_in_train = te_grids - tr_grids
    # 行级别“落在未见grid”的比例（比unique更直观）
    te_unseen_row_frac = float(te["grid_id"].isin(unseen_in_train).mean()) if len(te) else np.nan

    # JSD
    _, a, b = _align_counts(tr_counts, te_counts)
    jsd_grid = _js_divergence(a, b) if a.sum() and b.sum() else np.nan

    # report top-N grids with biggest probability change
    p_tr = (tr_counts / tr_counts.sum()) if tr_counts.sum() else tr_counts
    p_te = (te_counts / te_counts.sum()) if te_counts.sum() else te_counts
    idx = p_tr.index.union(p_te.index)
    p_tr = p_tr.reindex(idx, fill_value=0.0)
    p_te = p_te.reindex(idx, fill_value=0.0)
    delta = (p_te - p_tr).sort_values(key=lambda x: x.abs(), ascending=False)

    top_idx = delta.head(top_n_report).index  # 关键：先固定TopN索引

    top_delta = pd.DataFrame({
        "p_train": p_tr.reindex(top_idx).to_numpy(),
        "p_test":  p_te.reindex(top_idx).to_numpy(),
        "delta":   delta.reindex(top_idx).to_numpy(),
    }, index=top_idx)

    # user centroid drift summary + threshold rates
    tr_cent = tr.groupby(user_col, as_index=True)[[lat_col, lon_col]].mean()
    te_cent = te.groupby(user_col, as_index=True)[[lat_col, lon_col]].mean()
    common = tr_cent.index.intersection(te_cent.index)

    if len(common):
        d = _haversine_km(
            tr_cent.loc[common, lat_col].to_numpy(),
            tr_cent.loc[common, lon_col].to_numpy(),
            te_cent.loc[common, lat_col].to_numpy(),
            te_cent.loc[common, lon_col].to_numpy(),
        )
        drift = pd.Series(d, index=common, name="centroid_drift_km")
        q = drift.quantile([0.5, 0.75, 0.9, 0.95, 0.99]).to_dict()
        thr = {f"frac_users_drift_gt_{t}km": float((drift > t).mean()) for t in drift_thresholds_km}
    else:
        drift = pd.Series(dtype=float)
        q = {}
        thr = {}

    return {
        "grid_km": float(grid_km),
        "lat0_used": float(lat0),

        "n_rows_train": int(len(tr)),
        "n_rows_test": int(len(te)),
        "n_unique_grids_train": int(len(tr_grids)),
        "n_unique_grids_test": int(len(te_grids)),
        "n_unique_grids_test_unseen_in_train": int(len(unseen_in_train)),
        "frac_rows_test_in_unseen_grid": te_unseen_row_frac,

        "jsd_grid": float(jsd_grid) if jsd_grid == jsd_grid else np.nan,
        "top50_mass_train": _top_mass(tr_counts, 50),
        "top50_mass_test": _top_mass(te_counts, 50),

        "top_grid_probability_deltas": top_delta,  # 只给Top-N差异
        "n_common_users_for_centroid": int(len(common)),
        "user_centroid_drift_quantiles_km": q,
        **thr,
    }

# -----------------------------
# 4) Popularity drift: POI frequency + rank correlation
# -----------------------------
def check_popularity_drift(
    train_df,
    test_df,
    item_col="POI_id",
    top_k=1000
):
    """
    Compare POI popularity counts between train and test.
    Reports:
    - head share (top 1%, 5%, 10%) separately for train and test
    - JSD between count distributions (capped to top_k + OTHER)
    - Spearman rank correlation on common POIs within top_k by union
    """
    _require_cols(train_df, [item_col], "train_df")
    _require_cols(test_df, [item_col], "test_df")

    tr_counts_full = train_df[item_col].value_counts()
    te_counts_full = test_df[item_col].value_counts()

    def head_share(counts, frac):
        if len(counts) == 0:
            return np.nan
        k = max(1, int(np.ceil(frac * len(counts))))
        return float(counts.head(k).sum() / counts.sum())

    head = {
        "train_head_share_1pct": head_share(tr_counts_full, 0.01),
        "train_head_share_5pct": head_share(tr_counts_full, 0.05),
        "train_head_share_10pct": head_share(tr_counts_full, 0.10),
        "test_head_share_1pct": head_share(te_counts_full, 0.01),
        "test_head_share_5pct": head_share(te_counts_full, 0.05),
        "test_head_share_10pct": head_share(te_counts_full, 0.10),
    }

    # JSD on capped categories to keep dimension bounded
    tr_counts = _freq_table(train_df[item_col], top_k=top_k)
    te_counts = _freq_table(test_df[item_col], top_k=top_k)
    _, a, b = _align_counts(tr_counts, te_counts)
    jsd_poi = _js_divergence(a, b) if a.sum() and b.sum() else np.nan

    # Spearman on common POIs (use union of top_k from each)
    tr_top = set(tr_counts_full.head(top_k).index.astype("object"))
    te_top = set(te_counts_full.head(top_k).index.astype("object"))
    common = list(tr_top.union(te_top))

    if len(common) < 3:
        spearman = np.nan
    else:
        # ranks: higher count => higher rank (1 is most popular)
        tr_rank = tr_counts_full.reindex(common).fillna(0).rank(ascending=False, method="average")
        te_rank = te_counts_full.reindex(common).fillna(0).rank(ascending=False, method="average")
        spearman = float(tr_rank.corr(te_rank, method="spearman"))

    return {
        "n_train_items": int(train_df[item_col].nunique(dropna=True)),
        "n_test_items": int(test_df[item_col].nunique(dropna=True)),
        "head_shares": head,
        "poi_counts_train_top": tr_counts_full.head(20),
        "poi_counts_test_top": te_counts_full.head(20),
        "jsd_poi_popularity": float(jsd_poi) if jsd_poi == jsd_poi else np.nan,
        "spearman_rank_top_union": spearman,
    }

# -----------------------------
# One-shot report
# -----------------------------
def drift_report(
    train_df,
    test_df,
    user_col="user_id",
    item_col="POI_id",
    lat_col="latitude",
    lon_col="longitude",
    time_col="timestamps",
    grid_km=5.0,
):
    """
    Returns a dict of results. Most entries are scalars or pandas Series.
    """
    report = {}
    report["new_entities"] = check_new_entities(train_df, test_df, user_col=user_col, item_col=item_col)
    report["time_drift"] = check_time_drift(train_df, test_df, time_col=time_col)
    report["spatial_drift"] = check_spatial_drift(
        train_df, test_df,
        user_col=user_col, lat_col=lat_col, lon_col=lon_col,
        grid_km=grid_km
    )
    report["popularity_drift"] = check_popularity_drift(train_df, test_df, item_col=item_col)
    return report

import json
import pandas as pd
import numpy as np

class PandasEncoder(json.JSONEncoder):
    def default(self, obj):
        # 处理 Pandas 类型
        if isinstance(obj, pd.Series):
            return obj.to_dict()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        if isinstance(obj, pd.Index):
            return obj.tolist()
        
        # 处理 NumPy 类型
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        
        # 处理其他不可序列化类型
        if isinstance(obj, (set, frozenset)):
            return list(obj)
        
        return super().default(obj)