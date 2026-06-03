# Geometry-aware GradientBoosting 기반 RTT Fingerprinting 위치 추정 알고리즘

이름: 12236649 채희수

## 1. 모티베이션 & 인트로

### 1.1 중간발표 SAFL 알고리즘과 기존 결과

중간발표 단계까지 우리 팀은 UWB와 Wi-Fi의 센서 합의도를 기반으로 가중치를 부여하는 SAFL 알고리즘을 설계하고 구현하였다. SAFL은 실내 측위 환경에서 발생하는 NLOS, 반사, 장애물, 센서 잡음, 이상 거리값의 영향을 줄이기 위한 규칙 기반 강건 측위 알고리즘이다. 전체 흐름은 물리적으로 불가능한 거리값을 제거하는 전처리, UWB와 Wi-Fi 거리값의 유사도 계산, 앵커별 RTT 거리값의 분산과 센서 유사도 기반 가중치 산출, Grid Search와 Huber Loss 기반 초기 위치 추정, 그리고 MAD와 Tukey Biweight 및 IRWLS를 이용한 최종 위치 정제로 구성된다.

기존 SAFL 알고리즘은 특정 validation dataset에서 MAE 1.24 m, RMSE 1.60 m, 2 m 이내 정확도 80.0%를 보였다. 이를 통해 센서 합의도 기반 가중치와 강건 최적화가 특정 실험 환경에서는 좋은 성능을 낼 수 있음을 확인하였다. 그러나 해당 결과는 UWB와 Wi-Fi가 분리되어 제공되고, 두 센서의 유사도를 직접 계산할 수 있는 조건에서 얻어진 결과이다.

### 1.2 Final Project 데이터셋 구조 차이

Final Project에서 제공된 DH_FR1.mat 데이터셋은 중간발표 데이터셋과 구조가 다르다. 이 데이터셋은 d_hat, BS_positions, p로 구성되며, d_hat은 18개 기지국이 측정한 RTT 기반 거리값이고, BS_positions는 18개 기지국의 2차원 좌표이며, p는 학습용 정답 사용자 위치이다. 이 데이터셋에는 UWB와 Wi-Fi가 명확히 분리된 센서 쌍으로 제공되지 않는다. 따라서 기존 SAFL의 핵심인 UWB-Wi-Fi 센서 합의도 계산을 그대로 적용하기 어렵다.

### 1.3 `d_hat`과 실제거리 차이 분석

나는 먼저 DH_FR1.mat의 d_hat이 실제 거리로 얼마나 해석 가능한지 확인하였다. 학습 데이터에서는 정답 위치 p와 기지국 좌표 BS_positions가 주어지므로, 각 사용자와 앵커 사이의 실제 기하학적 거리를 d_true,i = ||p - b_i||로 계산할 수 있다. 이 값과 d_hat을 비교한 결과, d_hat은 실제 거리와 큰 차이를 보였다.

| 항목 | 값 |
|---|---:|
| d_hat과 실제거리의 평균 절대 오차 | 약 16.17 m |
| d_hat과 실제거리의 중앙값 오차 | 약 10.15 m |
| 상위 10% 거리 오차 | 약 41.26 m 이상 |
| 최대 거리 오차 | 약 252 m |
| d_hat이 실제거리보다 큰 비율 | 약 81% |

이 결과를 통해 본 문제는 clean ranging 기반 삼변측량 문제가 아니라, NLOS와 bias가 강하게 포함된 noisy RTT fingerprinting 문제에 가깝다고 판단하였다. 즉 d_hat을 실제 거리로 믿고 WLS나 SAFL-baseline을 적용하는 방식은 성능이 제한될 수밖에 없다. 실제로 LS, WLS, Weighted Centroid와 같은 물리 기반 방법은 RTT 오차에 매우 민감하게 반응하였다. 반면 머신러닝 기반 방법은 d_hat을 실제 거리로 직접 해석하지 않고, 18개 RTT 거리 패턴과 앵커 geometry feature로부터 실제 위치를 학습할 수 있다는 장점이 있다.

### 1.4 최종 알고리즘 방향 도출

따라서 최종 알고리즘은 기존 SAFL을 그대로 사용하는 방식이 아니라, d_hat의 noisy RTT pattern과 BS_positions의 기하 정보를 함께 사용하는 Geometry-aware GradientBoosting 기반 위치 직접 예측 모델로 결정하였다. 이 모델은 d_hat을 clean distance로 가정하지 않고, RTT 거리값, 거리 통계량, weighted centroid, residual 통계량, 가까운 앵커 정보, 앵커별 bias 및 reliability 정보를 feature로 구성한다. 이후 GradientBoosting 회귀 모델이 이 feature들과 실제 위치 p 사이의 비선형 관계를 학습하여 최종 p_hat = (x, y)를 직접 예측한다.

### 1.5 후보 알고리즘 비교와 최종 선택

최종 알고리즘을 선택하기 위해 전체 700개 데이터를 600개의 학습 데이터와 100개의 validation 데이터로 나누어 비교하였다. 학습 데이터는 모델 학습과 앵커별 통계 추정에 사용하고, validation 데이터는 학습에 사용하지 않은 샘플에서 일반화 성능을 확인하기 위해 사용하였다.

| Method | N | MAE(m) | RMSE(m) | 2m 이내(%) | 3m 이내(%) | Median(m) |
|---|---:|---:|---:|---:|---:|---:|
| Weighted Centroid | 100 | 13.5659 | 15.1811 | 0.00 | 1.00 | 13.6066 |
| LS | 100 | 24.8011 | 26.1927 | 0.00 | 0.00 | 24.4256 |
| WLS | 100 | 18.0723 | 20.1280 | 1.00 | 1.00 | 17.0153 |
| SVR(RBF) | 100 | 약 5.7311 | 약 7.1226 | 약 18.00 | 약 29.00 | 약 4.6125 |
| ExtraTrees | 100 | 약 7.3005 | 약 8.4645 | 약 7.00 | 약 20.00 | 약 6.6118 |
| Grid NLOS-aware GB | 100 | 약 6.1790 | 약 7.2514 | 약 11.00 | 약 20.00 | 약 5.3310 |
| Distance Correction + GB | 100 | 약 6.0503 | 약 6.9647 | 약 8.00 | 약 18.00 | 약 5.8108 |
| Geometry-aware GradientBoosting | 100 | 5.0315 | 5.9563 | 15.00 | 27.00 | 4.3033 |

비교 결과, Geometry-aware GradientBoosting이 validation set에서 가장 낮은 MAE와 RMSE를 보였다. SVR은 일부 샘플에서 2 m 또는 3 m 이내 비율이 조금 높았지만, 평균 오차와 RMSE 기준에서는 GradientBoosting보다 불리하였다. 또한 SVR은 kernel, C, gamma, epsilon 설정에 민감하므로 hidden test 일반화와 제출 환경 재현성을 고려했을 때 최종 알고리즘으로는 GradientBoosting이 더 안정적이라고 판단하였다.

결론적으로 본 프로젝트에서는 Geometry-aware GradientBoosting 기반 RTT Fingerprinting 위치 추정 알고리즘을 최종 알고리즘으로 채택하였다. 이 알고리즘은 기존 SAFL의 문제의식인 이상 거리값과 앵커 신뢰도 문제를 유지하면서도, Final Project 데이터셋 구조에 맞게 UWB-Wi-Fi 합의도 대신 RTT pattern과 앵커 기하 정보를 머신러닝 feature로 사용하는 방식이다.

## 2. 알고리즘 설명

이 장은 공지 요구사항에 맞추어, 설명만 읽어도 코드 구현 흐름을 이해할 수 있도록 입력·전처리·feature 생성·모델 학습·추론 과정을 순서대로 정리한다.

| 단계 | 내용 | 코드 구현과의 대응 |
|---|---|---|
| 2.1 | 입력/출력 정의 | `DH_FR1.mat`에서 `d_hat`, `BS_positions`를 읽고 `(2, num_user)` 형태의 `p_hat` 반환 |
| 2.2~2.4 | 거리 전처리 및 앵커 통계 | 비정상 거리값 보정, 앵커별 bias, sigma, reliability 계산 |
| 2.5~2.8 | geometry feature 생성 | weighted centroid, 후보 위치 `p0`, residual, 가까운/먼 앵커 정보 생성 |
| 2.9~2.10 | ML 학습 및 추론 | GradientBoosting을 학습하고 `model.pkl`로 저장한 뒤 `main.py`에서 로드 |
| 2.11 | 참고 논문과 차이 | 논문이 제안한 부분과 본 프로젝트에서 제안한 부분을 구분 |

### 2.1 입력과 출력 정의

최종 알고리즘의 목표는 각 사용자 u에 대해 18개 기지국의 RTT 거리 벡터 d_u와 기지국 좌표 행렬 B를 입력받아 2차원 위치 p_hat_u = (x_u, y_u)를 예측하는 것이다. 입력 데이터의 형태는 d_hat ∈ R^(18×N), BS_positions ∈ R^(2×18)이며, 학습 과정에서는 정답 위치 p ∈ R^(2×N)를 사용한다. hidden test에서는 정답 위치 p를 사용할 수 없으므로 main.py는 d_hat과 BS_positions, 그리고 train.py에서 저장한 model.pkl만 이용하여 p_hat을 반환한다. 반환되는 p_hat의 형태는 반드시 (2, num_user)이다.

### 2.2 거리값 전처리

첫 번째 단계는 거리값 전처리이다. 한 샘플의 거리 벡터를 d = [d_1, d_2, ..., d_18]이라고 할 때, 각 d_i는 양수이고 유한한 값이어야 한다. NaN, inf, 음수, 지나치게 큰 값은 실제 거리값으로 보기 어렵기 때문에 중앙값 기반 대체값으로 치환하거나 유효도 mask를 낮게 부여한다. 측위 가능한 공간은 BS_positions의 최소 좌표와 최대 좌표를 기준으로 잡고, 여기에 margin을 추가하여 정의한다. 이 공간의 모서리와 각 앵커 사이의 최대 거리를 이용하여 물리적으로 가능한 거리 범위를 근사하고, 이 범위를 크게 벗어나는 값은 낮은 신뢰도의 거리값으로 처리한다.

### 2.3 학습 데이터 기반 앵커 통계 계산

두 번째 단계는 학습 데이터 기반 앵커 통계 계산이다. 학습 데이터에서는 정답 위치 p가 주어지므로 i번째 앵커 좌표 b_i와 사용자 위치 p_u 사이의 실제 거리를 d_true,i,u = ||p_u - b_i||로 계산할 수 있다. 각 앵커의 RTT 오차는 e_i,u = d_hat,i,u - d_true,i,u로 정의한다. 학습 데이터 전체에서 e_i,u의 중앙값을 해당 앵커의 bias_i로 사용하고, MAD 기반 robust scale을 sigma_i로 사용한다. bias_i는 특정 앵커가 반복적으로 실제 거리보다 크게 또는 작게 측정되는 경향을 나타내며, sigma_i는 해당 앵커의 오차가 얼마나 불안정한지를 나타낸다.

앵커별 reliability는 sigma_i가 작을수록 커지도록 정의한다. 본 알고리즘에서는 reliability_i가 앵커의 평균적 안정성을 나타내는 feature이자, geometry 후보 위치를 계산할 때의 보조 가중치로 사용된다. 이 과정은 기존 SAFL의 분산 기반 신뢰도 아이디어를 Final Project 데이터셋 구조에 맞게 변형한 것이다. 다만 hidden test에서는 p를 알 수 없으므로, hidden test의 앵커별 bias와 sigma를 새로 계산하지 않고 train.py에서 학습 데이터로 계산하여 model.pkl에 저장한 값을 사용한다.

### 2.4 bias 보정 거리와 기본 거리 feature 생성

세 번째 단계는 bias 보정 거리와 기본 거리 feature 생성이다. 전처리된 거리값을 d_clean이라고 하면, 각 앵커에 대해 d_corr,i = max(d_clean,i - bias_i, ε)를 계산한다. d_corr는 d_hat의 고정적인 positive bias를 일부 줄인 거리 feature이다. 이때 d_corr를 실제 거리로 완전히 신뢰하는 것은 아니며, 원본 d_clean, bias 보정 거리 d_corr, 그리고 correction_i = d_corr,i - d_clean,i를 모두 feature로 사용한다. 이렇게 하면 모델은 원본 RTT pattern과 보정된 거리 pattern을 함께 비교할 수 있다.

### 2.5 거리 통계량 생성

네 번째 단계는 거리 통계량 생성이다. d_clean, d_corr, correction에 대해 평균, 표준편차, 중앙값, 최솟값, 최댓값, 25 percentile, 75 percentile을 계산한다. 이 통계량은 현재 샘플의 거리값들이 전체적으로 큰지, 특정 앵커가 과도하게 튀는지, 거리 분포가 얼마나 퍼져 있는지를 나타낸다. RTT fingerprinting 문제에서는 개별 거리값뿐 아니라 전체 거리 분포의 형태도 위치 추정에 중요한 정보가 된다.

### 2.6 `BS_positions` 기반 weighted centroid 생성

다섯 번째 단계는 BS_positions 기반 weighted centroid 생성이다. i번째 앵커 좌표를 b_i, 거리값을 d_i라고 할 때, 가까운 앵커일수록 사용자 위치에 더 큰 영향을 줄 수 있다고 보고 a_i = 1 / (d_i^q + ε) 형태의 가중치를 사용한다. 이때 q는 1 또는 2로 설정하여 서로 다른 민감도의 centroid를 만든다. Weighted centroid는 c = Σ_i a_i b_i / Σ_i a_i 로 계산된다. 본 알고리즘은 원본 거리 d_clean 기반 centroid와 bias 보정 거리 d_corr 기반 centroid를 모두 feature로 사용한다. 또한 d_corr 기반 centroid에는 앵커별 reliability를 함께 곱하여 안정적인 앵커의 영향이 더 커지도록 한다.

### 2.7 geometry-consistent 후보 위치 `p0` 계산

여섯 번째 단계는 geometry-consistent 후보 위치 p0 계산이다. Weighted centroid는 간단하고 빠르지만 RTT noise가 큰 경우 위치 후보가 크게 흔들릴 수 있다. 따라서 bias 보정 거리 d_corr와 reliability를 이용하여 soft-L1 loss 기반의 제한된 least-squares 정제를 수행한다. 후보 위치 x에 대해 예측 거리는 ||x - b_i||이고, 잔차는 r_i(x) = ||x - b_i|| - d_corr,i로 정의된다. reliability가 높은 앵커의 잔차를 더 중요하게 반영하고, 큰 잔차의 영향을 줄이기 위해 soft-L1 형태의 강건 손실을 사용한다. 이 과정을 통해 geometry-consistent 위치 후보 p0를 얻는다.

### 2.8 residual 및 geometry consistency feature 생성

일곱 번째 단계는 residual 및 geometry consistency feature 생성이다. p0에서 각 앵커까지의 예측 거리와 d_corr의 차이를 residual_i = ||p0 - b_i|| - d_corr,i로 계산한다. residual이 작으면 현재 후보 위치와 해당 앵커의 거리값이 비교적 잘 맞는다는 뜻이고, residual이 크면 해당 앵커의 RTT 값이 NLOS 또는 큰 bias를 포함할 가능성이 있다는 뜻이다. 본 알고리즘은 residual 18개, reliability가 곱해진 residual 18개, residual의 평균, 표준편차, 중앙값, 최댓값, 절대 residual 통계량을 feature로 사용한다. 또한 weighted centroid에서 각 앵커까지의 예측 거리와 d_clean 또는 d_corr의 차이도 통계량으로 추가한다. 이 과정은 현재 샘플의 RTT pattern이 앵커 geometry와 얼마나 일관적인지를 표현한다.

### 2.9 가까운 앵커와 먼 앵커 feature 생성

여덟 번째 단계는 가까운 앵커와 먼 앵커 feature 생성이다. d_clean과 d_corr 각각에 대해 가장 가까운 앵커 3개와 가장 먼 앵커 3개를 선택하고, 해당 앵커의 거리값, x좌표, y좌표, 앵커 index를 feature로 사용한다. 가까운 앵커는 사용자의 대략적인 위치를 나타내는 강한 단서가 될 수 있고, 먼 앵커는 큰 RTT bias나 NLOS 가능성을 나타낼 수 있다. 따라서 단순히 18개 거리값만 사용하는 것보다, 거리 순위와 앵커 좌표 정보를 함께 제공하는 것이 GradientBoosting 모델이 위치별 RTT 패턴을 학습하는 데 유리하다.

### 2.10 GradientBoosting 기반 위치 직접 예측

아홉 번째 단계는 GradientBoosting 기반 위치 직접 예측이다. 최종 feature vector φ_u는 d_clean, d_corr, correction, 유효 mask, bias, sigma, reliability, 거리 통계량, weighted centroid, p0, residual, residual 통계량, 가까운 앵커 및 먼 앵커 feature를 포함한다. 학습 데이터에서는 각 φ_u에 대해 정답 위치 p_u = (x_u, y_u)가 주어지므로, GradientBoosting 회귀 모델 f_GB를 학습하여 p_hat_u = f_GB(φ_u)가 되도록 한다. x좌표와 y좌표를 동시에 예측하기 위해 실제 구현에서는 GradientBoostingRegressor를 MultiOutputRegressor로 감싸 사용한다.

GradientBoosting은 여러 개의 약한 decision tree를 순차적으로 학습하면서 이전 단계에서 남은 오차를 줄이는 방식으로 동작한다. 첫 번째 tree는 전체적인 위치 경향을 학습하고, 이후의 tree들은 앞선 예측의 residual error를 보완한다. DH_FR1.mat 데이터셋에서는 d_hat이 실제 거리와 크게 다르기 때문에 선형 모델이나 순수 거리 기반 모델은 성능이 낮다. 반면 GradientBoosting은 특정 거리 패턴, 앵커 조합, residual 통계량, weighted centroid 위치가 특정 좌표와 연결되는 비선형 관계를 조건별로 학습할 수 있다.

### 2.11 validation과 최종 모델 저장

열 번째 단계는 validation과 최종 모델 저장이다. train.py는 먼저 전체 700개 중 600개를 학습 데이터, 100개를 validation 데이터로 나누어 선택한 알고리즘의 일반화 성능을 확인한다. 이때 validation 데이터는 모델 학습에 사용하지 않으며, MAE, RMSE, 2m 이내 비율, 3m 이내 비율, Median error를 출력한다. 검증 후 최종 제출용 model.pkl은 전체 700개 데이터로 다시 학습하여 저장한다. 이는 최종 hidden test에 대비하여 제공된 학습 데이터를 모두 활용하기 위한 절차이다. 다만 보고서에서 성능 수치로 사용하는 값은 전체 데이터 재평가 결과가 아니라, 학습에 사용하지 않은 validation 100개에서의 결과이다.


## 3. Agent AI 활용 방안


| 구분 | 내가 수행한 역할 | Agent AI를 활용한 역할 |
|---|---|---|
| 문제 제기 | SAFL이 UWB-WiFi 합의도에 의존하므로 Final Project 데이터셋에 그대로 적용하기 어렵다고 판단 | 문제점을 보고서 문장으로 정리하는 데 활용 |
| 데이터 분석 | `BS_positions`와 `p`로 실제거리를 계산하고 `d_hat`과 비교하여 noisy RTT fingerprinting 문제로 해석 | 분석 결과를 바탕으로 가능한 알고리즘 후보를 정리하는 데 활용 |
| 알고리즘 방향 | WLS/SAFL보다 RTT pattern과 geometry feature를 학습하는 구조가 적합하다고 판단 | Ridge, RandomForest, ExtraTrees, SVR, GradientBoosting, NLOS-aware 방식 등을 비교하는 데 활용 |
| 최종 선택 | validation 결과와 제출 재현성을 고려하여 Geometry-aware GradientBoosting을 선택 | 코드 구조, 결과 해석, 제출 규격 점검을 보조적으로 활용 |

본 프로젝트에서는 ChatGPT를 Agent AI 도구로 활용하였다. 다만 알고리즘의 핵심 문제 인식과 최종 방향 선택은 내가 수행하였다. 나는 중간발표 SAFL 알고리즘이 UWB와 Wi-Fi의 센서 합의도에 기반한다는 점을 먼저 확인하였고, Final Project 데이터셋에는 UWB와 Wi-Fi가 분리되어 제공되지 않기 때문에 기존 SAFL을 그대로 적용하기 어렵다는 문제를 파악하였다.

또한 나는 BS_positions와 p를 이용해 true distance를 계산하고, 이를 d_hat과 비교하여 d_hat이 실제 거리와 큰 차이를 보인다는 점을 확인하였다. 이 분석을 바탕으로 본 문제를 clean ranging 기반 삼변측량 문제가 아니라 noisy RTT fingerprinting 문제로 해석하였다. 따라서 규칙 기반 SAFL이나 WLS보다, d_hat의 패턴과 앵커 geometry feature를 함께 학습하는 머신러닝 접근이 더 적합하다고 판단하였다.

AI는 내가 제시한 문제점과 방향을 바탕으로 여러 후보 알고리즘을 빠르게 비교하고 정리하는 데 활용하였다. 구체적으로 Ridge, RandomForest, ExtraTrees, SVR, GradientBoosting, Distance Correction + GradientBoosting, NLOS-aware weighting, Geometry-aware GradientBoosting의 장단점을 비교하였다. 또한 각 모델이 작은 데이터셋, hidden test, 과적합 위험, 제출 환경 재현성 측면에서 어떤 장단점을 갖는지 검토하는 데 도움을 받았다.

최종적으로 나는 Geometry-aware GradientBoosting을 선택하였다. AI는 이 판단을 검토하고, 알고리즘 설명을 보고서 형식에 맞게 정리하고, main.py와 train.py가 제출 규격을 만족하는지 점검하는 보조 도구로 사용되었다. 따라서 본 프로젝트에서 AI는 답안을 대체한 도구가 아니라, 내가 정의한 문제점과 알고리즘 개선 방향을 구체화하고 구현 과정을 정리하기 위한 보조 도구로 활용되었다.

## 4. 결과 도출 & 디스커션


| 요구 항목 | 본 보고서에서의 처리 |
|---|---|
| 수치의 단순 비교를 넘은 해석 | `d_hat`과 실제거리 차이를 분석하고, 왜 물리 기반 baseline이 낮은 성능을 보였는지 설명하였다. |
| 본인의 사고와 구현 적합성 | SAFL을 그대로 적용하지 않고, RTT pattern과 geometry feature를 학습하는 구조로 바꾼 이유를 설명하였다. |
| baseline 비교의 fairness | 동일 데이터셋, 동일 validation split, 동일 평가지표를 사용했으며, 물리 기반 baseline의 목적을 명확히 밝혔다. |
| 알고리즘 장점과 단점 | Geometry-aware GradientBoosting의 장점, 한계, hidden test 위험을 구분해 작성하였다. |
| future work | 반복 평가, 구역별 local model, 추가 센서 기반 NLOS classifier 확장을 제안하였다. |
| 자체 평가 방식의 fairness | validation set을 학습에 사용하지 않았고, 전체 데이터 재평가 결과를 일반화 성능으로 쓰지 않았음을 명시하였다. |

본 프로젝트에서는 동일한 DH_FR1.mat 데이터셋에서 여러 알고리즘을 비교하였다. 비교의 공정성을 위해 같은 train/validation split을 사용하고, 모든 모델은 동일한 validation set에서 평가하였다. 평가지표는 MAE, RMSE, 2m 이내 비율, 3m 이내 비율, Median error를 사용하였다. MAE는 평균적인 위치 오차를 나타내고, RMSE는 큰 오차에 더 민감하다. 2m 이내 비율과 3m 이내 비율은 실내 측위에서 사용자가 체감할 수 있는 정확도 범위를 나타내는 지표로 사용하였다.

물리 기반 baseline들은 낮은 성능을 보였다. Weighted Centroid, LS, WLS는 모두 d_hat을 실제 거리와 유사한 값으로 가정한다. 그러나 데이터 분석 결과 d_hat은 실제 거리와 평균적으로 약 16 m 수준의 차이를 보였고, 일부 샘플에서는 매우 큰 outlier가 존재하였다. 따라서 거리 기반 최적화 알고리즘은 입력값의 오차를 그대로 위치 오차로 전파할 수밖에 없었다.

| Method | N | MAE(m) | RMSE(m) | 2m 이내(%) | 3m 이내(%) | Median(m) |
|---|---:|---:|---:|---:|---:|---:|
| Weighted Centroid | 100 | 13.5659 | 15.1811 | 0.00 | 1.00 | 13.6066 |
| LS | 100 | 24.8011 | 26.1927 | 0.00 | 0.00 | 24.4256 |
| WLS | 100 | 18.0723 | 20.1280 | 1.00 | 1.00 | 17.0153 |
| Geometry-aware GradientBoosting | 100 | 5.0315 | 5.9563 | 15.00 | 27.00 | 4.3033 |

위 결과에서 Geometry-aware GradientBoosting은 물리 기반 baseline보다 MAE와 RMSE를 크게 낮췄다. 이는 d_hat을 실제 거리로 직접 해석하는 방식보다, d_hat의 패턴과 BS_positions 기반 feature를 함께 학습하는 방식이 DH_FR1.mat 데이터셋에 더 적합하다는 것을 보여준다. 특히 LS와 WLS가 크게 실패한 것은 이 문제가 단순 거리 기반 삼변측량 문제가 아니라 noisy RTT fingerprinting 문제라는 해석을 뒷받침한다.

다른 비선형 머신러닝 모델도 검토하였다. SVR(RBF)은 일부 샘플에서 2m 또는 3m 이내 비율이 더 높게 나타났지만, 전체 MAE와 RMSE는 GradientBoosting보다 좋지 않았다. ExtraTrees는 noisy pattern을 흡수할 수 있는 tree ensemble이라는 장점이 있지만, 본 실험에서는 d_hat만을 직접 입력한 구조가 geometry feature를 포함한 GradientBoosting보다 낮은 성능을 보였다. Distance Correction + GradientBoosting은 d_hat과 true distance의 차이를 먼저 보정하는 구조였으나, 보정된 거리들이 항상 하나의 위치와 기하학적으로 일관되지는 않아 큰 성능 개선으로 이어지지 않았다. Grid NLOS-aware 방식도 residual이 큰 앵커의 가중치를 낮추는 과정을 포함했지만, 특정 앵커 몇 개만 문제가 아니라 대부분 앵커가 큰 bias와 noise를 가지는 데이터 특성 때문에 제한적인 성능을 보였다.

최종 알고리즘의 장점은 세 가지이다. 첫째, d_hat을 실제 거리로 무리하게 가정하지 않고 RTT fingerprinting feature로 사용한다. 둘째, BS_positions를 이용하여 weighted centroid, geometry-consistent 후보 위치, residual 통계량, 가까운 앵커 정보를 만들어 앵커 배치 정보를 적극적으로 반영한다. 셋째, GradientBoosting을 사용하여 RTT pattern과 실제 위치 사이의 비선형 관계를 학습할 수 있다.

한계도 존재한다. validation 결과에서 Geometry-aware GradientBoosting은 물리 기반 baseline보다 크게 개선되었지만, 2m 이내 비율은 15% 수준으로 높지 않다. 이는 d_hat이 실제 거리와 크게 다르고, 비슷한 RTT pattern을 가진 샘플들이 실제 공간에서는 멀리 떨어져 있을 수 있기 때문이다. 즉 현재 입력 정보인 18개 RTT 거리값만으로 hidden test에서 1~2 m 수준의 정밀 측위를 안정적으로 달성하기는 어려울 수 있다. RSSI, SNR, CIR, LOS/NLOS label, 시간 연속 정보 같은 추가 feature가 제공된다면 더 높은 정확도를 기대할 수 있다.

또한 train.py에서는 validation으로 최종 구조를 확인한 뒤 전체 700개 데이터로 최종 model.pkl을 다시 학습한다. 이 과정은 hidden test에 대비하여 제공된 데이터를 최대한 활용하기 위한 절차이다. 그러나 전체 700개에 대해 다시 평가한 결과는 학습 데이터가 포함된 값이므로 보고서의 일반화 성능 지표로 사용하지 않는다. 보고서에서는 학습에 사용하지 않은 validation 100개에서의 결과를 기준으로 알고리즘의 성능을 판단한다.

향후 개선 방향으로는 세 가지를 고려할 수 있다. 첫째, 현재는 전체 700개 데이터 중 600개를 학습에 사용하고 100개를 검증에 사용하는 방식으로 성능을 확인하였다. 그러나 검증 데이터 100개가 어떤 샘플로 구성되느냐에 따라 결과가 달라질 수 있다. 예를 들어 우연히 어려운 샘플이 검증 데이터에 많이 포함되면 성능이 낮게 나오고, 상대적으로 쉬운 샘플이 포함되면 성능이 높게 나올 수 있다. 따라서 한 번의 데이터 분할 결과만으로 알고리즘 성능을 판단하기보다는, 데이터를 여러 번 다른 방식으로 나누어 반복적으로 평가하고 그 평균 성능을 확인하는 과정이 필요하다. 이렇게 하면 특정 검증 데이터 구성에만 유리한 모델인지, 아니면 전체 데이터에 대해 안정적으로 동작하는 모델인지 더 공정하게 판단할 수 있다. 둘째, 현재 알고리즘은 전체 실내 공간을 하나의 GradientBoosting 모델로 학습한다. 하지만 RTT 오차는 위치에 따라 다르게 나타날 수 있으므로, 향후에는 대략적인 위치 후보를 기준으로 공간을 몇 개의 구역으로 나눈 뒤 각 구역에 맞는 모델을 따로 학습하는 방법을 고려할 수 있다. 셋째, 현재 데이터셋에는 18개 RTT 거리값만 제공되기 때문에 어떤 앵커가 LOS 환경이고 어떤 앵커가 NLOS 환경인지 명확히 구분하기 어렵다. 만약 RSSI, SNR, CIR, LOS/NLOS label과 같은 추가 센서 정보가 제공된다면, 먼저 이상 앵커나 NLOS 가능성이 높은 앵커를 판별한 뒤 위치를 예측하는 구조로 확장할 수 있다.

## 5. Reference

### 5.1 참고문헌 목록

[1] Lim DZ, Yeo M, Dahan A, Tahayori B, Kok HK, Abbasi-Rad M, et al. “Development of a machine learning-based real-time location system to streamline acute endovascular intervention in acute stroke: a proof-of-concept study.” Journal of NeuroInterventional Surgery, 2022;14:799-803. PubMed: https://pubmed.ncbi.nlm.nih.gov/34426539/

[2] scikit-learn Developers. “GradientBoostingRegressor.” scikit-learn documentation. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html

[3] scikit-learn Developers. “MultiOutputRegressor.” scikit-learn documentation. https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputRegressor.html

[4] scikit-learn Developers. “SVR.” scikit-learn documentation. https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html

[5] scikit-learn Developers. “RandomForestRegressor.” scikit-learn documentation. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

[6] scikit-learn Developers. “ExtraTreesRegressor.” scikit-learn documentation. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html

[7] scikit-learn Developers. “Ridge.” scikit-learn documentation. https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html

[8] IBM. “릿지 회귀란 무엇인가요?” IBM Think. https://www.ibm.com/kr-ko/think/topics/ridge-regression

[9] SciPy Developers. “scipy.optimize.least_squares.” SciPy documentation. https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html

### 5.2 참고문헌에서 참고한 부분

본 프로젝트에서 참고문헌을 사용한 목적은 기존 알고리즘을 그대로 복제하기 위한 것이 아니라, 실내 위치 추정 문제에서 어떤 접근이 가능한지 확인하고, 본 데이터셋에 맞는 모델을 선택하기 위한 근거를 마련하기 위한 것이었다.

| 참고문헌 | 참고한 부분 | 본 프로젝트에서 활용한 방식 |
|---|---|---|
| [1] Lim et al. | WiFi fingerprinting 기반 real-time location system에서 RandomForest와 SVM 같은 전통적 머신러닝 모델이 실내 위치 관련 문제에 활용될 수 있음을 확인하였다. | RTT 측위 문제도 거리값 자체를 직접 공식에 대입하기보다, 신호 또는 거리 패턴을 학습하는 머신러닝 문제로 볼 수 있다는 방향 설정에 참고하였다. |
| [2] GradientBoostingRegressor | 여러 decision tree를 순차적으로 학습하면서 이전 단계의 오차를 줄이는 회귀 모델이라는 점을 확인하였다. | RTT pattern과 실제 위치 사이의 비선형 관계를 학습하는 최종 회귀 모델로 사용하였다. |
| [3] MultiOutputRegressor | 하나의 회귀 모델을 여러 출력 변수에 적용할 수 있음을 확인하였다. | 위치 좌표가 `x`, `y` 두 개이므로 GradientBoostingRegressor를 MultiOutputRegressor로 감싸 `p_hat = (x, y)`를 동시에 예측하도록 구현하였다. |
| [4] SVR | RBF kernel을 이용해 비선형 회귀를 수행할 수 있고, `C`, `gamma`, `epsilon` 설정에 민감하다는 점을 확인하였다. | SVR을 후보 모델로 비교하였으나, 평균 MAE와 RMSE가 GradientBoosting보다 좋지 않고 튜닝 민감도가 높아 최종 모델에서는 제외하였다. |
| [5] RandomForestRegressor | 여러 decision tree의 평균을 이용해 비선형 패턴을 안정적으로 학습할 수 있음을 확인하였다. | tree ensemble 계열 baseline으로 비교하였다. 다만 본 실험에서는 Geometry-aware GradientBoosting보다 낮은 성능을 보여 최종 모델로 선택하지 않았다. |
| [6] ExtraTreesRegressor | 랜덤성이 큰 여러 tree를 평균내어 비선형 회귀를 수행할 수 있음을 확인하였다. | noisy RTT pattern을 직접 흡수할 수 있는 후보로 테스트하였지만, d_hat만 직접 입력하는 구조에서는 geometry feature를 포함한 GradientBoosting보다 성능이 낮았다. |
| [7], [8] Ridge Regression | L2 정규화를 통해 선형 회귀의 과적합을 줄일 수 있는 안정적인 선형 모델임을 확인하였다. | 선형 baseline으로 참고하였다. 그러나 RTT 오차가 비선형적으로 발생하기 때문에 최종 모델로는 부적합하다고 판단하였다. |
| [9] SciPy least_squares | 제한 조건이 있는 least-squares 최적화와 soft-L1 loss를 사용할 수 있음을 확인하였다. | 최종 위치를 직접 계산하기 위한 주 알고리즘이 아니라, geometry-consistent 후보 위치 `p0`를 만들고 residual feature를 생성하기 위한 보조 단계로 활용하였다. |

### 5.3 참고문헌이 제안한 부분과 본 프로젝트의 차이

가장 직접적으로 참고한 Lim et al.의 연구는 WiFi fingerprinting 데이터를 이용하여 병원 내 위치 또는 구역을 예측하는 real-time location system을 다룬다. 해당 연구는 WiFi 신호 패턴을 기반으로 위치 관련 판단을 수행하며, RandomForest와 SVM이 높은 성능을 보였다는 점을 보고하였다. 즉 참고 논문 [1]이 보여준 핵심은 실내 위치 추정 문제에서 신호 패턴을 머신러닝 모델이 학습할 수 있다는 가능성이다.

반면 본 프로젝트는 WiFi fingerprinting 구역 분류 문제가 아니라, 18개 기지국의 RTT 거리 추정값 `d_hat`과 앵커 좌표 `BS_positions`를 이용하여 연속적인 2차원 좌표 `p_hat = (x, y)`를 예측하는 회귀 문제이다. 따라서 참고 논문의 알고리즘을 그대로 복제하지 않았고, 실내 위치 추정에서 머신러닝 모델이 유효할 수 있다는 아이디어만 참고하였다.

내가 제안한 부분은 `DH_FR1.mat` 데이터셋의 구조를 먼저 분석하고, 그 구조에 맞게 알고리즘을 재설계한 점이다. 학습 데이터에는 정답 위치 `p`가 있으므로, `BS_positions`와 `p`를 이용해 실제 기하학적 거리 `d_true = ||p - b_i||`를 계산할 수 있다. 이를 `d_hat`과 비교한 결과, `d_hat`은 실제 거리와 큰 차이를 보였고 평균 절대 거리 오차가 약 16.17 m 수준으로 나타났다. 이 분석을 통해 본 문제를 clean distance 기반 삼변측량 문제가 아니라, NLOS와 bias가 포함된 noisy RTT fingerprinting 문제로 해석하였다.

또한 기존 중간발표 SAFL 알고리즘은 UWB와 Wi-Fi의 센서 합의도를 계산하여 앵커별 신뢰도를 부여하는 구조였지만, Final Project 데이터셋에는 UWB와 Wi-Fi가 분리된 센서 쌍으로 제공되지 않는다. 따라서 기존 SAFL의 핵심인 UWB-Wi-Fi sensor agreement를 그대로 적용할 수 없다고 판단하였다. 대신 SAFL의 문제의식인 이상 거리값 처리, 앵커 신뢰도, 잔차 기반 판단을 `d_hat`과 `BS_positions` 기반 feature 생성 과정으로 재해석하였다.

최종적으로 본 프로젝트에서 제안한 알고리즘은 Geometry-aware GradientBoosting 기반 RTT Fingerprinting 위치 추정 알고리즘이다. 이 알고리즘은 `d_hat`을 실제 거리로 그대로 신뢰하지 않고, 원본 RTT 거리값, bias 보정 거리값, 거리 통계량, weighted centroid, geometry-consistent 후보 위치, residual 통계량, 가까운 앵커 및 먼 앵커 정보를 feature로 구성한다. 이후 GradientBoosting 모델이 이러한 feature와 실제 위치 좌표 사이의 비선형 관계를 학습하여 최종 위치를 직접 예측한다.

### 5.4 차이 요약

| 구분 | 참고문헌 또는 기존 접근 | 본 프로젝트에서 제안한 접근 |
|---|---|---|
| 문제 유형 | WiFi fingerprinting 기반 위치 또는 구역 예측, 혹은 일반적인 회귀 모델 설명 | RTT 거리 추정값 `d_hat`과 앵커 좌표를 이용한 2차원 좌표 회귀 문제 |
| 데이터 해석 | 신호 패턴 또는 일반적인 feature를 이용한 위치 추정 가능성 확인 | `d_hat`이 실제 거리와 크게 다르다는 점을 직접 분석하고 noisy RTT fingerprinting 문제로 재정의 |
| 기존 SAFL과의 관계 | UWB-WiFi sensor agreement 기반 규칙 알고리즘 | UWB/WiFi 분리 정보가 없으므로 sensor agreement를 그대로 쓰지 않고, SAFL의 신뢰도·잔차 아이디어를 geometry feature로 변환 |
| 모델 선택 | RandomForest, SVM, Ridge, GradientBoosting 등 일반 ML 모델의 가능성 확인 | 여러 후보를 같은 validation 조건에서 비교한 뒤 Geometry-aware GradientBoosting을 최종 선택 |
| 구현 방식 | 참고문헌의 모델 구조 또는 공식 문서의 기본 사용법 | `d_hat`, `d_corr`, anchor bias/reliability, weighted centroid, residual feature, nearest/farthest anchor feature를 구성한 뒤 MultiOutput GradientBoosting으로 `x`, `y` 직접 예측 |
| 차별점 | 실내 위치 추정에 ML이 활용될 수 있음을 보여줌 | `DH_FR1.mat`의 RTT 오차 특성을 분석하고, 해당 데이터셋에 맞춘 feature engineering과 직접 좌표 회귀 구조를 제안 |

정리하면, 참고문헌에서 얻은 것은 실내 위치 추정 문제에서 머신러닝 모델이 활용될 수 있다는 근거와 각 회귀 모델의 기본 개념 및 구현 방법이다. 반면 본인이 제안한 핵심은 `DH_FR1.mat` 데이터셋에서 `d_hat`과 실제 거리의 차이를 분석하고, 기존 SAFL의 센서 합의도 구조가 해당 데이터셋에 그대로 적용되기 어렵다는 점을 바탕으로, RTT pattern과 앵커 geometry feature를 함께 사용하는 Geometry-aware GradientBoosting 위치 예측 구조를 설계한 것이다.
