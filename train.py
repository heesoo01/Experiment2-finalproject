"""
train.py
Geometry-aware GradientBoosting for RTT indoor localization.

This file is for model training and validation.
It reads DH_FR1.mat, evaluates the selected algorithm on a fixed
train/validation split, and saves model.pkl trained on all available data.
"""

from __future__ import annotations

import argparse
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import scipy.io as sio
from scipy.optimize import least_squares
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

RANDOM_STATE = 42
MODEL_PATH = "model.pkl"


def load_project_mat(mat_path: str = "DH_FR1.mat") -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if not os.path.exists(mat_path):
        alt = "InF_DH_FR1.mat"
        if os.path.exists(alt):
            mat_path = alt
        else:
            raise FileNotFoundError(f"MAT file not found: {mat_path}")

    data = sio.loadmat(mat_path, squeeze_me=False)
    if "d_hat" not in data:
        raise KeyError("MAT file must contain 'd_hat'.")

    d_hat = np.asarray(data["d_hat"], dtype=float)

    if "BS_positions" in data:
        p_bs = np.asarray(data["BS_positions"], dtype=float)
    elif "p_bs" in data:
        p_bs = np.asarray(data["p_bs"], dtype=float)
    else:
        raise KeyError("MAT file must contain 'BS_positions' or 'p_bs'.")

    p = np.asarray(data["p"], dtype=float) if "p" in data else None

    if p_bs.shape[0] != 2 and p_bs.shape[1] == 2:
        p_bs = p_bs.T
    if d_hat.shape[0] != p_bs.shape[1] and d_hat.shape[1] == p_bs.shape[1]:
        d_hat = d_hat.T
    if p is not None and p.shape[0] != 2 and p.shape[1] == 2:
        p = p.T

    return d_hat, p_bs, p


def metric_summary(p_hat: np.ndarray, p_true: np.ndarray) -> Dict[str, float]:
    err = np.linalg.norm(p_hat - p_true, axis=0)
    return {
        "N": int(err.size),
        "MAE": float(np.mean(err)),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "Within2m": float(np.mean(err <= 2.0) * 100.0),
        "Within3m": float(np.mean(err <= 3.0) * 100.0),
        "Median": float(np.median(err)),
    }


def print_metric_table(title: str, method: str, metrics: Dict[str, float]) -> None:
    print(f"\n========== {title} ==========")
    print("| Method | N | MAE(m) | RMSE(m) | <=2m(%) | <=3m(%) | Median(m) |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    print(
        f"| {method} | {metrics['N']} | {metrics['MAE']:.4f} | {metrics['RMSE']:.4f} | "
        f"{metrics['Within2m']:.2f} | {metrics['Within3m']:.2f} | {metrics['Median']:.4f} |"
    )


def train_validation_split(n: int, train_size: int = 600, random_state: int = RANDOM_STATE) -> Tuple[np.ndarray, np.ndarray]:
    if train_size <= 0 or train_size >= n:
        raise ValueError("train_size must be between 1 and n-1.")
    rng = np.random.default_rng(random_state)
    idx = np.arange(n)
    rng.shuffle(idx)
    return idx[:train_size], idx[train_size:]


def true_distance_matrix(p_bs: np.ndarray, p: np.ndarray) -> np.ndarray:
    anchors = p_bs.T
    return np.linalg.norm(p.T[:, None, :] - anchors[None, :, :], axis=2).T


def robust_scale(x: np.ndarray, default: float = 1.0) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return default
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(max(1.4826 * mad, 1e-6))


def get_bounds(p_bs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    xy_min = np.min(p_bs, axis=1)
    xy_max = np.max(p_bs, axis=1)
    span = xy_max - xy_min
    margin = np.maximum(20.0, 0.25 * span)
    return xy_min - margin, xy_max + margin


def compute_anchor_stats(d_train: np.ndarray, p_bs: np.ndarray, p_train: np.ndarray) -> Dict[str, np.ndarray]:
    gt = true_distance_matrix(p_bs, p_train)
    residual = d_train - gt
    m = d_train.shape[0]
    bias = np.zeros(m)
    sigma = np.ones(m)

    for i in range(m):
        valid = np.isfinite(residual[i]) & np.isfinite(d_train[i]) & (d_train[i] > 0)
        if np.any(valid):
            bias[i] = float(np.median(residual[i, valid]))
            sigma[i] = robust_scale(residual[i, valid], default=1.0)
        else:
            bias[i] = 0.0
            sigma[i] = 10.0

    reliability = 1.0 / (sigma ** 2 + 1.0)
    reliability = reliability / (np.max(reliability) + 1e-12)
    return {"bias": bias, "sigma": sigma, "reliability": reliability}


def sanitize_distances(d: np.ndarray, p_bs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    d = np.asarray(d, dtype=float).ravel()
    lb, ub = get_bounds(p_bs)
    anchors = p_bs.T
    corners = np.array([
        [lb[0], lb[1]], [lb[0], ub[1]], [ub[0], lb[1]], [ub[0], ub[1]]
    ])
    max_possible = max(np.max(np.linalg.norm(c - anchors, axis=1)) for c in corners)

    finite_positive = np.isfinite(d) & (d > 0)
    valid = finite_positive & (d < max_possible * 1.25)
    replacement = float(np.nanmedian(d[finite_positive])) if np.any(finite_positive) else float(max_possible / 2.0)

    d_clean = d.copy()
    d_clean[~finite_positive] = replacement
    d_clean = np.clip(d_clean, 1e-3, max_possible * 1.10)
    return d_clean, valid.astype(float)


def weighted_centroid(
    p_bs: np.ndarray,
    d: np.ndarray,
    valid: np.ndarray,
    power: float = 2.0,
    reliability: np.ndarray | None = None,
) -> np.ndarray:
    anchors = p_bs.T
    w = valid / (np.power(np.maximum(d, 1e-3), power) + 1e-6)
    if reliability is not None:
        w = w * np.asarray(reliability, dtype=float)
    if np.sum(w) <= 1e-12:
        return np.mean(p_bs, axis=1)
    return (w[:, None] * anchors).sum(axis=0) / np.sum(w)


def refine_position(start: np.ndarray, p_bs: np.ndarray, d: np.ndarray, valid: np.ndarray, reliability: np.ndarray) -> np.ndarray:
    anchors = p_bs.T
    lb, ub = get_bounds(p_bs)
    w = np.clip(valid * reliability, 0.02, 1.0)
    sqrt_w = np.sqrt(w)

    def residual_func(z: np.ndarray) -> np.ndarray:
        pred = np.linalg.norm(z - anchors, axis=1)
        return sqrt_w * (pred - d)

    try:
        res = least_squares(
            residual_func,
            x0=np.clip(start, lb, ub),
            bounds=(lb, ub),
            loss="soft_l1",
            f_scale=5.0,
            max_nfev=80,
        )
        return res.x.astype(float)
    except Exception:
        return np.clip(start, lb, ub).astype(float)


def stats7(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    return np.array([
        np.mean(x), np.std(x), np.median(x), np.min(x), np.max(x),
        np.percentile(x, 25), np.percentile(x, 75)
    ], dtype=float)


def nearest_anchor_features(p_bs: np.ndarray, d: np.ndarray, k: int = 3) -> np.ndarray:
    anchors = p_bs.T
    order = np.argsort(d)
    near = order[:k]
    far = order[-k:][::-1]
    feats: List[float] = []
    for idx in near:
        feats.extend([d[idx], anchors[idx, 0], anchors[idx, 1], float(idx)])
    for idx in far:
        feats.extend([d[idx], anchors[idx, 0], anchors[idx, 1], float(idx)])
    return np.array(feats, dtype=float)


def build_feature_one(d_raw: np.ndarray, p_bs: np.ndarray, anchor_stats: Dict[str, np.ndarray]) -> np.ndarray:
    bias = anchor_stats["bias"]
    sigma = anchor_stats["sigma"]
    reliability = anchor_stats["reliability"]

    d_clean, valid = sanitize_distances(d_raw, p_bs)
    d_corr = np.maximum(d_clean - bias, 1e-3)
    correction = d_corr - d_clean

    c_raw_p1 = weighted_centroid(p_bs, d_clean, valid, power=1.0)
    c_raw_p2 = weighted_centroid(p_bs, d_clean, valid, power=2.0)
    c_corr_p1 = weighted_centroid(p_bs, d_corr, valid, power=1.0, reliability=reliability)
    c_corr_p2 = weighted_centroid(p_bs, d_corr, valid, power=2.0, reliability=reliability)

    p0 = refine_position(c_corr_p2, p_bs, d_corr, valid, reliability)
    anchors = p_bs.T
    pred_dist = np.linalg.norm(p0 - anchors, axis=1)
    residual = pred_dist - d_corr
    abs_res = np.abs(residual)
    rel_weighted_res = residual * reliability

    consistency = np.array([
        np.mean(abs_res),
        np.std(residual),
        np.max(abs_res),
        np.median(abs_res),
        np.mean(abs_res * reliability),
        np.sum(valid),
        np.mean(reliability * valid),
    ], dtype=float)

    pred_c_raw = np.linalg.norm(c_raw_p2 - anchors, axis=1)
    pred_c_corr = np.linalg.norm(c_corr_p2 - anchors, axis=1)
    centroid_res_stats = np.concatenate([
        stats7(pred_c_raw - d_clean),
        stats7(pred_c_corr - d_corr),
    ])

    return np.concatenate([
        d_clean,
        d_corr,
        correction,
        valid,
        bias,
        sigma,
        reliability,
        stats7(d_clean),
        stats7(d_corr),
        stats7(correction),
        c_raw_p1,
        c_raw_p2,
        c_corr_p1,
        c_corr_p2,
        p0,
        residual,
        rel_weighted_res,
        stats7(residual),
        stats7(abs_res),
        consistency,
        centroid_res_stats,
        nearest_anchor_features(p_bs, d_clean, k=3),
        nearest_anchor_features(p_bs, d_corr, k=3),
    ]).astype(float)


def build_feature_matrix(d_hat: np.ndarray, p_bs: np.ndarray, anchor_stats: Dict[str, np.ndarray]) -> np.ndarray:
    return np.vstack([build_feature_one(d_hat[:, u], p_bs, anchor_stats) for u in range(d_hat.shape[1])])


def make_gb_model(random_state: int = RANDOM_STATE) -> MultiOutputRegressor:
    base = GradientBoostingRegressor(
        n_estimators=350,
        learning_rate=0.035,
        max_depth=2,
        min_samples_leaf=6,
        subsample=0.85,
        random_state=random_state,
    )
    return MultiOutputRegressor(base)


def train_model(d_hat: np.ndarray, p_bs: np.ndarray, p: np.ndarray) -> Tuple[MultiOutputRegressor, Dict[str, np.ndarray], int]:
    anchor_stats = compute_anchor_stats(d_hat, p_bs, p)
    X = build_feature_matrix(d_hat, p_bs, anchor_stats)
    model = make_gb_model()
    model.fit(X, p.T)
    return model, anchor_stats, int(X.shape[1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="DH_FR1.mat")
    parser.add_argument("--train_size", type=int, default=600)
    args = parser.parse_args()

    d_hat, p_bs, p = load_project_mat(args.data)
    if p is None:
        raise ValueError("train.py requires ground-truth variable 'p'.")

    n = d_hat.shape[1]
    train_idx, valid_idx = train_validation_split(n, train_size=args.train_size, random_state=RANDOM_STATE)

    print("========== Data Split ==========")
    print(f"Total samples      : {n}")
    print(f"Train samples      : {len(train_idx)}")
    print(f"Validation samples : {len(valid_idx)}")

    d_train, p_train = d_hat[:, train_idx], p[:, train_idx]
    d_valid, p_valid = d_hat[:, valid_idx], p[:, valid_idx]

    val_model, val_anchor_stats, n_features = train_model(d_train, p_bs, p_train)
    X_valid = build_feature_matrix(d_valid, p_bs, val_anchor_stats)
    pred_valid = val_model.predict(X_valid).T
    validation_metrics = metric_summary(pred_valid, p_valid)
    method = "Geometry-aware GradientBoosting"
    print_metric_table("Validation Result", method, validation_metrics)

    final_model, final_anchor_stats, final_n_features = train_model(d_hat, p_bs, p)
    payload = {
        "algorithm": method,
        "model": final_model,
        "anchor_stats": final_anchor_stats,
        "p_bs": p_bs,
        "n_features": final_n_features,
        "validation_metrics": validation_metrics,
        "validation_train_size": int(len(train_idx)),
        "validation_size": int(len(valid_idx)),
        "random_state": RANDOM_STATE,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(payload, f)
    print(f"\nSaved final model trained on all {n} samples: {MODEL_PATH}")
    print(f"Feature dimension: {final_n_features}")


if __name__ == "__main__":
    main()
