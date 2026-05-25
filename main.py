"""
main.py
RWML: Robust Weighted Machine Learning Residual Correction

채점기는 main() 함수를 호출하고, 반환값 p_hat의 shape이 (2, num_user)인지 확인한다.
"""

from __future__ import annotations

import os
import pickle

import numpy as np

from common_rwml import build_feature_vector, load_project_mat, robust_wls


MAT_PATH = "DH_FR1.mat"
MODEL_PATH = "model.pkl"


def your_algorithm(d_one_user: np.ndarray, p_bs: np.ndarray, model_payload=None) -> np.ndarray:
    """한 명의 사용자에 대한 위치 추정 함수.

    1) Robust WLS로 초기 위치 p0 계산
    2) model.pkl이 있으면 ML이 delta를 예측
    3) 최종 p_hat = p0 + delta_hat 반환
    4) model.pkl이 없거나 오류가 나면 p0를 fallback으로 반환
    """
    if model_payload is None:
        p0, _, _, _, _ = robust_wls(d_one_user, p_bs)
        return p0

    try:
        feature, p0 = build_feature_vector(d_one_user, p_bs)
        model = model_payload["model"]
        expected = model_payload.get("n_features", feature.shape[0])

        if feature.shape[0] != expected:
            return p0

        delta = model.predict(feature.reshape(1, -1)).reshape(-1)
        if delta.size != 2 or not np.all(np.isfinite(delta)):
            return p0

        p_hat = p0 + delta
        if not np.all(np.isfinite(p_hat)):
            return p0
        return p_hat.astype(float)
    except Exception:
        p0, _, _, _, _ = robust_wls(d_one_user, p_bs)
        return p0


def main() -> np.ndarray:
    d_hat, p_bs, _ = load_project_mat(MAT_PATH)
    num_user = d_hat.shape[1]
    p_hat = np.zeros((2, num_user), dtype=float)

    model_payload = None
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                model_payload = pickle.load(f)
        except Exception:
            model_payload = None

    for u in range(num_user):
        p_hat[:, u] = your_algorithm(d_hat[:, u], p_bs, model_payload)

    return p_hat


if __name__ == "__main__":
    result = main()
    print("p_hat shape:", result.shape)
    print("first 3 predictions:")
    print(result[:, :3])
