"""
train.py
RWML: Robust Weighted Machine Learning Residual Correction

실행 방법:
    python train.py

실행 결과:
    - validation 성능 출력
    - model.pkl 저장
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common_rwml import RANDOM_STATE, build_feature_matrix, load_project_mat, metric_summary


MAT_PATH = "DH_FR1.mat"
MODEL_PATH = "model.pkl"


def evaluate_model(model, X_valid, p0_valid, p_valid):
    """ML 모델의 delta 예측 성능을 위치 오차 기준으로 평가한다."""
    delta_hat = model.predict(X_valid).T
    p_hat = p0_valid + delta_hat
    return metric_summary(p_hat, p_valid)


def main():
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
    print(f"    MAE={baseline['MAE']:.4f}, RMSE={baseline['RMSE']:.4f}, Within2m={baseline['Within_2m_%']:.2f}%")

    indices = np.arange(d_hat.shape[1])
    train_idx, valid_idx = train_test_split(indices, test_size=0.25, random_state=RANDOM_STATE)

    X_train, X_valid = X[train_idx], X[valid_idx]
    y_train, y_valid = y[train_idx], y[valid_idx]
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
