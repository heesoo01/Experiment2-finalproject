"""
train.py
RWML: Robust Weighted Machine Learning Residual Correction

제출 안정성을 위해 common_rwml.py에 있던 학습용 공통 함수들을 모두 이 파일 안에 포함한 standalone 버전입니다.
이 파일은 DH_FR1.mat을 읽어 Ridge, RandomForest, GradientBoosting 후보 모델을 비교하고 model.pkl을 저장합니다.
"""

from __future__ import annotations

"""
RWML common utilities
- Robust WLS 초기 위치 추정
- 머신러닝 feature 생성

이 파일은 main.py와 train.py가 공통으로 사용한다.
과제 제출 시 루트 폴더에 함께 두는 것을 권장한다.
"""

import os
from typing import Dict, Tuple

import numpy as np
from scipy.optimize import least_squares


RANDOM_STATE = 42


def load_project_mat(mat_path: str = "DH_FR1.mat") -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """MAT 파일에서 d_hat, p_bs, p를 안전하게 읽는다.

    업로드 데이터에는 기지국 좌표명이 BS_positions로 들어 있고,
    과제 README에는 p_bs로 안내되어 있으므로 두 이름을 모두 지원한다.
    """
    import scipy.io as sio

    if not os.path.exists(mat_path):
        # 사용자가 받은 파일명이 InF_DH_FR1.mat인 경우도 지원
        alt = "InF_DH_FR1.mat"
        if os.path.exists(alt):
            mat_path = alt
        else:
            raise FileNotFoundError(f"MAT file not found: {mat_path}")

    data = sio.loadmat(mat_path, squeeze_me=False)
    d_hat = np.asarray(data["d_hat"], dtype=float)

    if "p_bs" in data:
        p_bs = np.asarray(data["p_bs"], dtype=float)
    elif "BS_positions" in data:
        p_bs = np.asarray(data["BS_positions"], dtype=float)
    else:
        raise KeyError("MAT file must contain either 'p_bs' or 'BS_positions'.")

    p = np.asarray(data["p"], dtype=float) if "p" in data else None

    # shape 보정: d_hat=(18,N), p_bs=(2,18), p=(2,N)이 되도록 한다.
    if p_bs.shape[0] != 2 and p_bs.shape[1] == 2:
        p_bs = p_bs.T
    if d_hat.shape[0] != p_bs.shape[1] and d_hat.shape[1] == p_bs.shape[1]:
        d_hat = d_hat.T
    if p is not None and p.shape[0] != 2 and p.shape[1] == 2:
        p = p.T

    return d_hat, p_bs, p


def get_search_bounds(p_bs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """기지국 좌표 기반으로 위치 탐색 범위를 자동 설정한다."""
    xy_min = np.min(p_bs, axis=1)
    xy_max = np.max(p_bs, axis=1)
    span = xy_max - xy_min
    margin = np.maximum(20.0, 0.25 * span)
    return xy_min - margin, xy_max + margin


def sanitize_distances(d: np.ndarray, p_bs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """NaN, inf, 음수, 지나치게 큰 거리값을 처리한다.

    반환값:
    - d_clean: 최적화에 사용할 안전한 거리값
    - valid: 물리적으로 비교적 타당한 앵커 여부
    - max_plausible: 탐색 범위 기준 최대 가능 거리 근사값
    """
    d = np.asarray(d, dtype=float).ravel()
    lb, ub = get_search_bounds(p_bs)

    corners = np.array(
        [
            [lb[0], lb[1]],
            [lb[0], ub[1]],
            [ub[0], lb[1]],
            [ub[0], ub[1]],
        ]
    )
    anchors = p_bs.T
    max_plausible = max(np.max(np.linalg.norm(c - anchors, axis=1)) for c in corners)

    finite_positive = np.isfinite(d) & (d > 0)
    valid = finite_positive & (d < max_plausible * 1.25)

    if np.any(finite_positive):
        replacement = float(np.nanmedian(d[finite_positive]))
    else:
        replacement = float(max_plausible / 2.0)

    d_clean = d.copy()
    d_clean[~finite_positive] = replacement
    d_clean = np.clip(d_clean, 1e-3, max_plausible * 1.10)
    return d_clean, valid, float(max_plausible)


def linear_initial_guess(d: np.ndarray, p_bs: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """선형화된 trilateration으로 초기값을 만든다."""
    d = np.asarray(d, dtype=float).ravel()
    anchors = p_bs.T
    finite = np.isfinite(d) & (d > 0)
    idx = np.where(finite)[0]
    if len(idx) < 3:
        return np.mean(p_bs, axis=1)

    # 가장 가까운 앵커를 기준 앵커로 사용하면 수치적으로 비교적 안정적이다.
    ref = idx[np.argmin(d[idx])]
    ar = anchors[ref]
    dr = d[ref]

    A, b, w = [], [], []
    for i in idx:
        if i == ref:
            continue
        ai = anchors[i]
        di = d[i]
        A.append(2.0 * (ar - ai))
        b.append(di**2 - dr**2 + np.dot(ar, ar) - np.dot(ai, ai))
        w.append(1.0 if weights is None else max(float(weights[i]), 1e-6))

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    w = np.sqrt(np.asarray(w, dtype=float))

    try:
        x = np.linalg.lstsq(A * w[:, None], b * w, rcond=None)[0]
    except Exception:
        x = np.mean(p_bs, axis=1)

    lb, ub = get_search_bounds(p_bs)
    return np.clip(x, lb, ub)


def robust_wls(d_raw: np.ndarray, p_bs: np.ndarray, max_iter: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Robust WLS로 초기 위치 p0를 계산한다.

    잔차가 큰 앵커의 영향력을 반복적으로 낮추는 IRWLS 구조다.
    """
    d_clean, valid, _ = sanitize_distances(d_raw, p_bs)
    anchors = p_bs.T
    lb, ub = get_search_bounds(p_bs)

    # 거리 분포만으로 1차 가중치를 만든다. 너무 큰 거리값은 일단 낮게 본다.
    w0 = np.ones_like(d_clean, dtype=float)
    w0[~valid] = 0.2
    med = np.median(d_clean[valid]) if np.any(valid) else np.median(d_clean)
    mad = np.median(np.abs(d_clean[valid] - med)) if np.any(valid) else np.median(np.abs(d_clean - med))
    scale = 1.4826 * mad + 1e-6
    w0 *= 1.0 / (1.0 + np.maximum(0.0, (d_clean - med) / (3.0 * scale)) ** 2)
    w0 = np.clip(w0, 0.03, 1.0)

    x = linear_initial_guess(d_clean, p_bs, w0)
    w = w0.copy()

    for _ in range(max_iter):
        sqrt_w = np.sqrt(np.maximum(w, 1e-6))

        def residual_func(z: np.ndarray) -> np.ndarray:
            pred = np.linalg.norm(z - anchors, axis=1)
            return sqrt_w * (pred - d_clean)

        try:
            result = least_squares(
                residual_func,
                x,
                bounds=(lb, ub),
                loss="soft_l1",
                f_scale=4.0,
                max_nfev=80,
                xtol=1e-7,
                ftol=1e-7,
                gtol=1e-7,
            )
            x = result.x
        except Exception:
            x = np.clip(x, lb, ub)

        raw_residual = np.linalg.norm(x - anchors, axis=1) - d_clean
        s = 1.4826 * np.median(np.abs(raw_residual - np.median(raw_residual))) + 1e-6
        c = 4.685 * s + 1e-6
        u = np.abs(raw_residual) / c

        # Tukey biweight 형태의 잔차 기반 가중치
        w_robust = (1.0 - u**2) ** 2
        w_robust[u >= 1.0] = 0.05
        w = np.clip(0.1 * w0 + 0.9 * w_robust, 0.03, 1.0)

    final_residual = np.linalg.norm(x - anchors, axis=1) - d_clean
    return x.astype(float), final_residual.astype(float), w.astype(float), d_clean.astype(float), valid.astype(float)


def build_feature_vector(d_raw: np.ndarray, p_bs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """한 명의 사용자 샘플에서 ML 입력 feature와 초기 위치 p0를 만든다."""
    p0, residual, weights, d_clean, valid = robust_wls(d_raw, p_bs)

    anchors = p_bs.T
    center = np.mean(p_bs, axis=1)
    pred_dist = np.linalg.norm(p0 - anchors, axis=1)
    abs_residual = np.abs(residual)

    dist_stats = np.array(
        [
            np.mean(d_clean),
            np.std(d_clean),
            np.median(d_clean),
            np.min(d_clean),
            np.max(d_clean),
            np.percentile(d_clean, 25),
            np.percentile(d_clean, 75),
        ],
        dtype=float,
    )
    residual_stats = np.array(
        [
            np.mean(residual),
            np.std(residual),
            np.median(residual),
            np.max(abs_residual),
            np.mean(abs_residual),
            np.percentile(abs_residual, 75),
            np.percentile(abs_residual, 90),
        ],
        dtype=float,
    )

    extra = np.array(
        [
            p0[0],
            p0[1],
            p0[0] - center[0],
            p0[1] - center[1],
            np.sum(valid),
            np.min(d_clean),
            np.max(d_clean),
        ],
        dtype=float,
    )

    feature = np.concatenate(
        [
            d_clean,          # 18개 정제 거리값
            residual,         # 18개 잔차
            weights,          # 18개 앵커 신뢰도
            valid,            # 18개 유효 앵커 표시
            dist_stats,       # 거리 통계량 7개
            residual_stats,   # 잔차 통계량 7개
            extra,            # 초기 위치/중심 차이/유효 수 등 7개
            pred_dist,        # p0 기준 예측 거리 18개
        ]
    ).astype(float)

    return feature, p0


def build_feature_matrix(d_hat: np.ndarray, p_bs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """전체 사용자에 대해 feature matrix와 robust WLS 초기 위치를 계산한다."""
    num_user = d_hat.shape[1]
    features = []
    p0_all = np.zeros((2, num_user), dtype=float)

    for u in range(num_user):
        f, p0 = build_feature_vector(d_hat[:, u], p_bs)
        features.append(f)
        p0_all[:, u] = p0

    return np.vstack(features), p0_all


def metric_summary(p_hat: np.ndarray, p_true: np.ndarray) -> Dict[str, float]:
    """평가 지표 계산."""
    err = np.linalg.norm(p_hat - p_true, axis=0)
    return {
        "MAE": float(np.mean(err)),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "Within_2m_%": float(np.mean(err <= 2.0) * 100.0),
        "Median": float(np.median(err)),
        "P90": float(np.percentile(err, 90)),
    }



# -----------------------------------------------------------------------------
# train.py execution part
# -----------------------------------------------------------------------------

import pickle
import time

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MAT_PATH = "DH_FR1.mat"
MODEL_PATH = "model.pkl"


def evaluate_model(model, X_valid: np.ndarray, p0_valid: np.ndarray, p_valid: np.ndarray) -> Dict[str, float]:
    """ML 모델의 delta 예측 성능을 위치 오차 기준으로 평가한다."""
    delta_hat = model.predict(X_valid).T
    p_hat = p0_valid + delta_hat
    return metric_summary(p_hat, p_valid)


def main() -> None:
    start_time = time.time()
    d_hat, p_bs, p = load_project_mat(MAT_PATH)
    if p is None:
        raise ValueError("train.py requires ground-truth variable 'p' in the MAT file.")

    print("[1] Data loaded")
    print(f"    d_hat shape: {d_hat.shape}")
    print(f"    p_bs shape : {p_bs.shape}")
    print(f"    p shape    : {p.shape}")

    print("[2] Building robust WLS initial positions and ML features...")
    X, p0_all = build_feature_matrix(d_hat, p_bs)
    y = (p - p0_all).T  # ML target: delta = p - p0

    baseline = metric_summary(p0_all, p)
    print("\n[Baseline] Robust WLS only")
    print(
        f"    MAE={baseline['MAE']:.4f}, "
        f"RMSE={baseline['RMSE']:.4f}, "
        f"Within2m={baseline['Within_2m_%']:.2f}%"
    )

    indices = np.arange(d_hat.shape[1])
    train_idx, valid_idx = train_test_split(indices, test_size=0.25, random_state=RANDOM_STATE)

    X_train, X_valid = X[train_idx], X[valid_idx]
    y_train = y[train_idx]
    p0_valid = p0_all[:, valid_idx]
    p_valid = p[:, valid_idx]

    # 복잡한 딥러닝 대신, 작은 데이터에서 설명 가능하고 재현 가능한 모델들을 비교한다.
    candidates = {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=15.0)),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "GradientBoosting": MultiOutputRegressor(
            GradientBoostingRegressor(
                n_estimators=250,
                learning_rate=0.035,
                max_depth=2,
                min_samples_leaf=5,
                random_state=RANDOM_STATE,
            )
        ),
    }

    results = []
    best_name = None
    best_model = None
    best_mae = float("inf")

    print("\n[3] Training candidate ML residual models")
    for name, model in candidates.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        score = evaluate_model(model, X_valid, p0_valid, p_valid)
        elapsed = time.time() - t0
        results.append((name, score, elapsed))
        print(
            f"    {name:16s} | MAE={score['MAE']:.4f} | RMSE={score['RMSE']:.4f} | "
            f"Within2m={score['Within_2m_%']:.2f}% | time={elapsed:.2f}s"
        )
        if score["MAE"] < best_mae:
            best_mae = score["MAE"]
            best_name = name
            best_model = model

    # 최종 제출용 모델은 전체 700개 데이터로 다시 학습한다.
    print(f"\n[4] Best model: {best_name}. Re-training on all available data...")
    best_model.fit(X, y)

    payload = {
        "algorithm": "RWML: Robust Weighted Machine Learning Residual Correction",
        "model_name": best_name,
        "model": best_model,
        "n_features": int(X.shape[1]),
        "p_bs_train": p_bs,
        "baseline_valid": baseline,
        "valid_results": results,
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(payload, f)

    print(f"[5] Saved {MODEL_PATH}")
    print(f"[Done] Total train.py time: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()
