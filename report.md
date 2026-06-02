# Geometry-aware GradientBoosting 기반 RTT Fingerprinting 위치 추정 알고리즘

이름: 12236649 채희수

## 1. 모티베이션 & 인트로

중간발표 단계까지 우리 팀은 UWB와 Wi-Fi의 센서 합의도를 기반으로 가중치를 부여하는 SAFL 알고리즘을 설계하고 구현하였다. SAFL은 실내 측위 환경에서 발생하는 NLOS, 반사, 장애물, 센서 잡음, 이상 거리값의 영향을 줄이기 위한 규칙 기반 강건 측위 알고리즘이다. 전체 흐름은 물리적으로 불가능한 거리값을 제거하는 전처리, UWB와 Wi-Fi 거리값의 유사도 계산, 앵커별 RTT 거리값의 분산과 센서 유사도 기반 가중치 산출, Grid Search와 Huber Loss 기반 초기 위치 추정, 그리고 MAD와 Tukey Biweight 및 IRWLS를 이용한 최종 위치 정제로 구성된다.

기존 SAFL 알고리즘은 특정 validation dataset에서 MAE 1.24 m, RMSE 1.60 m, 2 m 이내 정확도 80.0%를 보였다. 이를 통해 센서 합의도 기반 가중치와 강건 최적화가 특정 실험 환경에서는 좋은 성능을 낼 수 있음을 확인하였다. 그러나 해당 결과는 UWB와 Wi-Fi가 분리되어 제공되고, 두 센서의 유사도를 직접 계산할 수 있는 조건에서 얻어진 결과이다.

Final Project에서 제공된 DH_FR1.mat 데이터셋은 중간발표 데이터셋과 구조가 다르다. 이 데이터셋은 d_hat, BS_positions, p로 구성되며, d_hat은 18개 기지국이 측정한 RTT 기반 거리값이고, BS_positions는 18개 기지국의 2차원 좌표이며, p는 학습용 정답 사용자 위치이다. 이 데이터셋에는 UWB와 Wi-Fi가 명확히 분리된 센서 쌍으로 제공되지 않는다. 따라서 기존 SAFL의 핵심인 UWB-Wi-Fi 센서 합의도 계산을 그대로 적용하기 어렵다.

나는 먼저 DH_FR1.mat의 d_hat이 실제 거리로 얼마나 해석 가능한지 확인하였다. 학습 데이터에서는 정답 위치 p와 기지국 좌표 BS_positions가 주어지므로, 각 사용자와 앵커 사이의 실제 기하학적 거리를 d_true,i = ||p - b_i||로 계산할 수 있다. 이 값과 d_hat을 비교한 결과, d_hat은 실제 거리와 큰 차이를 보였다.

| 항목 | 값 |
|---|---:|
| d_hat과 실제거리의 평균 절대 오차 | 약 16.17 m |
| d_hat과 실제거리의 중앙값 오차 | 약 10.15 m |
| 상위 10% 거리 오차 | 약 41.26 m 이상 |
| 최대 거리 오차 | 약 252 m |
| d_hat이 실제거리보다 큰 비율 | 약 81% |

이 결과를 통해 본 문제는 clean ranging 기반 삼변측량 문제가 아니라, NLOS와 bias가 강하게 포함된 noisy RTT fingerprinting 문제에 가깝다고 판단하였다. 즉 d_hat을 실제 거리로 믿고 WLS나 SAFL-baseline을 적용하는 방식은 성능이 제한될 수밖에 없다. 실제로 LS, WLS, Weighted Centroid와 같은 물리 기반 방법은 RTT 오차에 매우 민감하게 반응하였다. 반면 머신러닝 기반 방법은 d_hat을 실제 거리로 직접 해석하지 않고, 18개 RTT 거리 패턴과 앵커 geometry feature로부터 실제 위치를 학습할 수 있다는 장점이 있다.

따라서 최종 알고리즘은 기존 SAFL을 그대로 사용하는 방식이 아니라, d_hat의 noisy RTT pattern과 BS_positions의 기하 정보를 함께 사용하는 Geometry-aware GradientBoosting 기반 위치 직접 예측 모델로 결정하였다. 이 모델은 d_hat을 clean distance로 가정하지 않고, RTT 거리값, 거리 통계량, weighted centroid, residual 통계량, 가까운 앵커 정보, 앵커별 bias 및 reliability 정보를 feature로 구성한다. 이후 GradientBoosting 회귀 모델이 이 feature들과 실제 위치 p 사이의 비선형 관계를 학습하여 최종 p_hat = (x, y)를 직접 예측한다.

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

최종 알고리즘의 목표는 각 사용자 u에 대해 18개 기지국의 RTT 거리 벡터 d_u와 기지국 좌표 행렬 B를 입력받아 2차원 위치 p_hat_u = (x_u, y_u)를 예측하는 것이다. 입력 데이터의 형태는 d_hat ∈ R^(18×N), BS_positions ∈ R^(2×18)이며, 학습 과정에서는 정답 위치 p ∈ R^(2×N)를 사용한다. hidden test에서는 정답 위치 p를 사용할 수 없으므로 main.py는 d_hat과 BS_positions, 그리고 train.py에서 저장한 model.pkl만 이용하여 p_hat을 반환한다. 반환되는 p_hat의 형태는 반드시 (2, num_user)이다.

첫 번째 단계는 거리값 전처리이다. 한 샘플의 거리 벡터를 d = [d_1, d_2, ..., d_18]이라고 할 때, 각 d_i는 양수이고 유한한 값이어야 한다. NaN, inf, 음수, 지나치게 큰 값은 실제 거리값으로 보기 어렵기 때문에 중앙값 기반 대체값으로 치환하거나 유효도 mask를 낮게 부여한다. 측위 가능한 공간은 BS_positions의 최소 좌표와 최대 좌표를 기준으로 잡고, 여기에 margin을 추가하여 정의한다. 이 공간의 모서리와 각 앵커 사이의 최대 거리를 이용하여 물리적으로 가능한 거리 범위를 근사하고, 이 범위를 크게 벗어나는 값은 낮은 신뢰도의 거리값으로 처리한다.

두 번째 단계는 학습 데이터 기반 앵커 통계 계산이다. 학습 데이터에서는 정답 위치 p가 주어지므로 i번째 앵커 좌표 b_i와 사용자 위치 p_u 사이의 실제 거리를 d_true,i,u = ||p_u - b_i||로 계산할 수 있다. 각 앵커의 RTT 오차는 e_i,u = d_hat,i,u - d_true,i,u로 정의한다. 학습 데이터 전체에서 e_i,u의 중앙값을 해당 앵커의 bias_i로 사용하고, MAD 기반 robust scale을 sigma_i로 사용한다. bias_i는 특정 앵커가 반복적으로 실제 거리보다 크게 또는 작게 측정되는 경향을 나타내며, sigma_i는 해당 앵커의 오차가 얼마나 불안정한지를 나타낸다.

앵커별 reliability는 sigma_i가 작을수록 커지도록 정의한다. 본 알고리즘에서는 reliability_i가 앵커의 평균적 안정성을 나타내는 feature이자, geometry 후보 위치를 계산할 때의 보조 가중치로 사용된다. 이 과정은 기존 SAFL의 분산 기반 신뢰도 아이디어를 Final Project 데이터셋 구조에 맞게 변형한 것이다. 다만 hidden test에서는 p를 알 수 없으므로, hidden test의 앵커별 bias와 sigma를 새로 계산하지 않고 train.py에서 학습 데이터로 계산하여 model.pkl에 저장한 값을 사용한다.

세 번째 단계는 bias 보정 거리와 기본 거리 feature 생성이다. 전처리된 거리값을 d_clean이라고 하면, 각 앵커에 대해 d_corr,i = max(d_clean,i - bias_i, ε)를 계산한다. d_corr는 d_hat의 고정적인 positive bias를 일부 줄인 거리 feature이다. 이때 d_corr를 실제 거리로 완전히 신뢰하는 것은 아니며, 원본 d_clean, bias 보정 거리 d_corr, 그리고 correction_i = d_corr,i - d_clean,i를 모두 feature로 사용한다. 이렇게 하면 모델은 원본 RTT pattern과 보정된 거리 pattern을 함께 비교할 수 있다.

네 번째 단계는 거리 통계량 생성이다. d_clean, d_corr, correction에 대해 평균, 표준편차, 중앙값, 최솟값, 최댓값, 25 percentile, 75 percentile을 계산한다. 이 통계량은 현재 샘플의 거리값들이 전체적으로 큰지, 특정 앵커가 과도하게 튀는지, 거리 분포가 얼마나 퍼져 있는지를 나타낸다. RTT fingerprinting 문제에서는 개별 거리값뿐 아니라 전체 거리 분포의 형태도 위치 추정에 중요한 정보가 된다.

다섯 번째 단계는 BS_positions 기반 weighted centroid 생성이다. i번째 앵커 좌표를 b_i, 거리값을 d_i라고 할 때, 가까운 앵커일수록 사용자 위치에 더 큰 영향을 줄 수 있다고 보고 a_i = 1 / (d_i^q + ε) 형태의 가중치를 사용한다. 이때 q는 1 또는 2로 설정하여 서로 다른 민감도의 centroid를 만든다. Weighted centroid는 c = Σ_i a_i b_i / Σ_i a_i 로 계산된다. 본 알고리즘은 원본 거리 d_clean 기반 centroid와 bias 보정 거리 d_corr 기반 centroid를 모두 feature로 사용한다. 또한 d_corr 기반 centroid에는 앵커별 reliability를 함께 곱하여 안정적인 앵커의 영향이 더 커지도록 한다.

여섯 번째 단계는 geometry-consistent 후보 위치 p0 계산이다. Weighted centroid는 간단하고 빠르지만 RTT noise가 큰 경우 위치 후보가 크게 흔들릴 수 있다. 따라서 bias 보정 거리 d_corr와 reliability를 이용하여 soft-L1 loss 기반의 제한된 least-squares 정제를 수행한다. 후보 위치 x에 대해 예측 거리는 ||x - b_i||이고, 잔차는 r_i(x) = ||x - b_i|| - d_corr,i로 정의된다. reliability가 높은 앵커의 잔차를 더 중요하게 반영하고, 큰 잔차의 영향을 줄이기 위해 soft-L1 형태의 강건 손실을 사용한다. 이 과정을 통해 geometry-consistent 위치 후보 p0를 얻는다.

일곱 번째 단계는 residual 및 geometry consistency feature 생성이다. p0에서 각 앵커까지의 예측 거리와 d_corr의 차이를 residual_i = ||p0 - b_i|| - d_corr,i로 계산한다. residual이 작으면 현재 후보 위치와 해당 앵커의 거리값이 비교적 잘 맞는다는 뜻이고, residual이 크면 해당 앵커의 RTT 값이 NLOS 또는 큰 bias를 포함할 가능성이 있다는 뜻이다. 본 알고리즘은 residual 18개, reliability가 곱해진 residual 18개, residual의 평균, 표준편차, 중앙값, 최댓값, 절대 residual 통계량을 feature로 사용한다. 또한 weighted centroid에서 각 앵커까지의 예측 거리와 d_clean 또는 d_corr의 차이도 통계량으로 추가한다. 이 과정은 현재 샘플의 RTT pattern이 앵커 geometry와 얼마나 일관적인지를 표현한다.

여덟 번째 단계는 가까운 앵커와 먼 앵커 feature 생성이다. d_clean과 d_corr 각각에 대해 가장 가까운 앵커 3개와 가장 먼 앵커 3개를 선택하고, 해당 앵커의 거리값, x좌표, y좌표, 앵커 index를 feature로 사용한다. 가까운 앵커는 사용자의 대략적인 위치를 나타내는 강한 단서가 될 수 있고, 먼 앵커는 큰 RTT bias나 NLOS 가능성을 나타낼 수 있다. 따라서 단순히 18개 거리값만 사용하는 것보다, 거리 순위와 앵커 좌표 정보를 함께 제공하는 것이 GradientBoosting 모델이 위치별 RTT 패턴을 학습하는 데 유리하다.

아홉 번째 단계는 GradientBoosting 기반 위치 직접 예측이다. 최종 feature vector φ_u는 d_clean, d_corr, correction, 유효 mask, bias, sigma, reliability, 거리 통계량, weighted centroid, p0, residual, residual 통계량, 가까운 앵커 및 먼 앵커 feature를 포함한다. 학습 데이터에서는 각 φ_u에 대해 정답 위치 p_u = (x_u, y_u)가 주어지므로, GradientBoosting 회귀 모델 f_GB를 학습하여 p_hat_u = f_GB(φ_u)가 되도록 한다. x좌표와 y좌표를 동시에 예측하기 위해 실제 구현에서는 GradientBoostingRegressor를 MultiOutputRegressor로 감싸 사용한다.

GradientBoosting은 여러 개의 약한 decision tree를 순차적으로 학습하면서 이전 단계에서 남은 오차를 줄이는 방식으로 동작한다. 첫 번째 tree는 전체적인 위치 경향을 학습하고, 이후의 tree들은 앞선 예측의 residual error를 보완한다. DH_FR1.mat 데이터셋에서는 d_hat이 실제 거리와 크게 다르기 때문에 선형 모델이나 순수 거리 기반 모델은 성능이 낮다. 반면 GradientBoosting은 특정 거리 패턴, 앵커 조합, residual 통계량, weighted centroid 위치가 특정 좌표와 연결되는 비선형 관계를 조건별로 학습할 수 있다.

열 번째 단계는 validation과 최종 모델 저장이다. train.py는 먼저 전체 700개 중 600개를 학습 데이터, 100개를 validation 데이터로 나누어 선택한 알고리즘의 일반화 성능을 확인한다. 이때 validation 데이터는 모델 학습에 사용하지 않으며, MAE, RMSE, 2m 이내 비율, 3m 이내 비율, Median error를 출력한다. 검증 후 최종 제출용 model.pkl은 전체 700개 데이터로 다시 학습하여 저장한다. 이는 최종 hidden test에 대비하여 제공된 학습 데이터를 모두 활용하기 위한 절차이다. 다만 보고서에서 성능 수치로 사용하는 값은 전체 데이터 재평가 결과가 아니라, 학습에 사용하지 않은 validation 100개에서의 결과이다.


## 3. Agent AI 활용 방안

본 프로젝트에서는 ChatGPT를 Agent AI 도구로 활용하였다. 다만 알고리즘의 핵심 문제 제기, 해결 방향 판단, 최종 알고리즘 선택은 내가 직접 수행하였다. 나는 먼저 중간발표에서 사용한 SAFL 알고리즘이 UWB와 Wi-Fi의 센서 합의도를 기반으로 동작한다는 점을 확인하였다. 이후 Final Project에서 제공된 DH_FR1.mat 데이터셋에는 UWB와 Wi-Fi가 분리된 센서 쌍으로 존재하지 않고, 18개 앵커의 RTT 거리값 d_hat만 제공된다는 점을 파악하였다. 이로 인해 기존 SAFL의 핵심인 UWB-Wi-Fi sensor agreement를 그대로 적용하기 어렵다고 판단하였다.

또한 나는 BS_positions와 정답 위치 p를 이용하여 각 앵커와 사용자 사이의 실제 기하학적 거리 d_true = ||p - b_i||를 계산하고, 이를 d_hat과 비교하였다. 그 결과 d_hat이 실제 거리와 큰 차이를 보였고, 전체적으로 실제 거리보다 크게 측정되는 경향이 있음을 확인하였다. 이 분석을 바탕으로 본 문제를 clean ranging 기반 삼변측량 문제가 아니라, NLOS와 bias가 포함된 noisy RTT fingerprinting 문제로 해석하였다. 따라서 단순 WLS, SAFL-baseline, 거리 기반 최적화만으로는 성능 향상에 한계가 있으며, d_hat의 거리 패턴과 앵커 geometry 정보를 함께 학습하는 머신러닝 접근이 필요하다고 판단하였다.

Agent AI는 내가 제기한 문제와 방향을 바탕으로 여러 알고리즘 후보를 빠르게 정리하고 비교하는 보조 역할을 수행하였다. 구체적으로 Ridge, RandomForest, ExtraTrees, SVR, GradientBoosting, Distance Correction + GradientBoosting, Residual Learning, NLOS-aware weighting, Grid NLOS-aware 방식, Geometry-aware GradientBoosting 등의 후보 구조를 제안하고 각 방법의 장단점을 비교하는 데 활용하였다. 또한 각 알고리즘이 작은 데이터셋, hidden test, 과적합 위험, 실행 시간 제한, 제출 환경 재현성 측면에서 어떤 위험과 장점을 갖는지 검토하는 데 도움을 받았다.

알고리즘 구현 과정에서도 Agent AI를 보조적으로 활용하였다. 나는 최종적으로 d_hat을 실제 거리로 그대로 신뢰하지 않고, RTT pattern과 BS_positions 기반 geometry feature를 함께 사용하는 방향이 적합하다고 판단하였다. 이에 대해 AI는 main.py와 train.py의 코드 구조를 정리하고, validation 결과 출력 방식, p_hat의 shape, hidden test에서 정답 p를 사용하지 않는 구조, model.pkl 저장 및 로드 방식 등을 점검하는 데 도움을 주었다. 또한 여러 후보 모델의 validation 결과를 비교하여, 단순히 train 성능이 좋은 모델이 아니라 학습에 사용하지 않은 validation set에서 더 안정적인 모델을 선택할 수 있도록 결과 해석을 보조하였다.

최종적으로 나는 Geometry-aware GradientBoosting 기반 RTT Fingerprinting 위치 추정 알고리즘을 선택하였다. 이 선택은 AI가 자동으로 결정한 것이 아니라, 내가 데이터셋 구조, d_hat과 실제 거리의 차이, 기존 SAFL 적용의 한계, 여러 후보 알고리즘의 validation 성능을 종합하여 판단한 결과이다. AI는 후보 알고리즘을 제안하고, 코드 구현과 결과 해석을 보조하고, 보고서 문장을 정리하는 도구로 활용되었다. 따라서 본 프로젝트에서 Agent AI는 답안을 대체한 도구가 아니라, 내가 정의한 문제점과 해결 방향을 구체화하고 구현 과정을 효율적으로 정리하기 위한 보조 도구로 사용되었다.


## 4. 결과 도출 & 디스커션

본 프로젝트의 결과 도출 과정에서는 단순히 여러 알고리즘의 수치를 나열하는 것보다, 각 알고리즘이 Final Project 데이터셋의 특성과 얼마나 잘 맞는지를 확인하는 데 초점을 두었다. 처음에는 RTT 거리값과 앵커 좌표가 주어지므로 LS, WLS, Weighted Centroid와 같은 거리 기반 알고리즘이 자연스러운 baseline이 될 수 있다고 생각하였다. 그러나 DH_FR1.mat 데이터셋을 분석한 결과, d_hat은 실제 기하학적 거리와 큰 차이를 보였다. 학습 데이터에서는 정답 위치 p와 앵커 좌표 BS_positions가 주어지므로 실제 거리 d_true = ||p - b_i||를 계산할 수 있는데, 이를 d_hat과 비교했을 때 평균 절대 거리 오차가 약 16.17 m 수준으로 나타났다. 따라서 본 문제는 clean distance를 이용한 단순 삼변측량 문제가 아니라, NLOS와 bias가 포함된 noisy RTT fingerprinting 문제에 가깝다고 판단하였다.

이러한 문제 인식에 따라 나는 기존 SAFL 알고리즘을 그대로 사용하는 방식이 Final Project 데이터셋에는 적합하지 않다고 보았다. 중간발표에서 사용한 SAFL은 UWB와 Wi-Fi의 센서 합의도를 계산하여 앵커별 신뢰도를 부여하는 구조였지만, Final Project 데이터셋에서는 UWB와 Wi-Fi가 분리된 센서 쌍으로 제공되지 않는다. 따라서 기존 SAFL의 sensor agreement를 그대로 계산할 수 없고, SAFL을 최종 위치 계산기로 사용하는 방식은 데이터셋 구조와 맞지 않는다. 다만 SAFL의 문제의식인 이상 거리값 처리, 앵커 신뢰도, 잔차 기반 판단은 여전히 유효하므로, 이를 feature 설계와 baseline 설정에 참고하였다.

본인의 구현 과정에서는 먼저 물리 기반 baseline을 설정하였다. Weighted Centroid는 가까운 앵커가 사용자 위치에 더 큰 영향을 준다는 단순한 가정을 사용한 기준 알고리즘이고, LS와 WLS는 거리값과 앵커 좌표를 이용하여 위치를 계산하는 대표적인 기하 기반 방법이다. 이 baseline들은 머신러닝 모델처럼 정답 위치를 학습하지 않으므로 GradientBoosting과 모델 복잡도 측면에서 완전히 동일한 조건은 아니다. 그러나 모든 알고리즘을 동일한 DH_FR1.mat 데이터셋과 동일한 validation set에서 평가하였고, baseline은 d_hat을 실제 거리처럼 해석했을 때 어느 정도 성능이 나오는가를 확인하기 위한 기준으로 사용하였다. 따라서 딥러닝 모델과 단순 삼각측량을 무작정 비교한 것이 아니라, Final Project 문제에서 거리 기반 해법이 유효한지 확인하기 위한 baseline 비교로 설정하였다.

동일한 train/validation split에서 비교한 결과는 다음과 같다.

| Method | N | MAE(m) | RMSE(m) | 2m 이내(%) | 3m 이내(%) | Median(m) |
|---|---:|---:|---:|---:|---:|---:|
| Weighted Centroid | 100 | 13.5659 | 15.1811 | 0.00 | 1.00 | 13.6066 |
| LS | 100 | 24.8011 | 26.1927 | 0.00 | 0.00 | 24.4256 |
| WLS | 100 | 18.0723 | 20.1280 | 1.00 | 1.00 | 17.0153 |
| SVR(RBF) | 100 | 약 5.7311 | 약 7.1226 | 약 18.00 | 약 29.00 | 약 4.6125 |
| ExtraTrees | 100 | 약 7.3005 | 약 8.4645 | 약 7.00 | 약 20.00 | 약 6.6118 |
| Distance Correction + GB | 100 | 약 6.0503 | 약 6.9647 | 약 8.00 | 약 18.00 | 약 5.8108 |
| Grid NLOS-aware GB | 100 | 약 6.1790 | 약 7.2514 | 약 11.00 | 약 20.00 | 약 5.3310 |
| Geometry-aware GradientBoosting | 100 | 5.0315 | 5.9563 | 15.00 | 27.00 | 4.3033 |

물리 기반 baseline의 결과는 전반적으로 낮게 나타났다. Weighted Centroid, LS, WLS는 모두 d_hat이 실제 거리와 어느 정도 일치한다는 가정에 의존한다. 그러나 실제 데이터에서는 d_hat이 NLOS, 반사, bias로 인해 실제 거리와 크게 달랐기 때문에, 이러한 거리 기반 해법은 입력 오차를 위치 오차로 그대로 전파하였다. 특히 LS와 WLS는 여러 앵커의 거리값을 동시에 만족하는 위치를 찾는 방식이므로, 일부 거리값이 크게 왜곡되면 전체 위치 추정 결과도 크게 흔들렸다. 이 결과를 통해 Final Project 데이터셋에서는 단순 거리 기반 측위보다 RTT pattern을 직접 학습하는 접근이 더 적합하다고 판단하였다.

여러 머신러닝 모델도 함께 검토하였다. Ridge는 선형 모델이므로 구현과 해석은 쉽지만, RTT 거리 패턴과 실제 위치 사이의 비선형 관계를 충분히 표현하기 어렵다. RandomForest와 ExtraTrees는 여러 decision tree를 이용해 비선형 패턴을 학습할 수 있지만, 본 실험에서는 Geometry feature를 함께 사용한 GradientBoosting보다 낮은 성능을 보였다. SVR(RBF)은 2m 또는 3m 이내 비율에서 일부 장점을 보였지만, 평균 오차와 RMSE 기준에서는 GradientBoosting보다 불리하였다. 또한 SVR은 C, gamma, epsilon과 같은 하이퍼파라미터에 민감하여 hidden test와 제출 환경의 재현성을 고려했을 때 최종 모델로 선택하기에는 부담이 있다고 판단하였다.

본인이 최종적으로 선택한 Geometry-aware GradientBoosting은 d_hat을 실제 거리로 직접 신뢰하지 않고, RTT fingerprinting feature로 해석한다. 구체적으로 원본 RTT 거리값, bias 보정 거리값, 거리 통계량, weighted centroid, geometry-consistent 후보 위치, residual 통계량, 가까운 앵커와 먼 앵커 정보 등을 feature로 구성하였다. 이 feature들은 단순히 거리값을 나열하는 것이 아니라, 현재 샘플의 RTT pattern이 앵커 geometry와 얼마나 일관적인지, 어떤 앵커 조합이 위치 예측에 중요한지, 특정 거리값이 전체 분포에서 얼마나 튀는지를 모델이 학습할 수 있도록 설계한 것이다. GradientBoosting은 여러 개의 decision tree를 순차적으로 학습하면서 이전 단계의 예측 오차를 줄이는 구조이므로, RTT pattern과 실제 위치 사이의 비선형 관계를 학습하는 데 적합하다고 판단하였다.

비교가 공정했는지에 대해서는 다음과 같이 평가할 수 있다. 첫째, 모든 알고리즘은 동일한 DH_FR1.mat 데이터셋과 동일한 validation split에서 평가하였다. 둘째, 학습 데이터와 validation 데이터를 분리하여, 모델이 이미 본 데이터에서 좋은 성능을 내는 상황을 최종 성능으로 사용하지 않았다. 셋째, 물리 기반 baseline은 머신러닝 모델보다 학습 능력이 없으므로 모델 용량 측면에서는 불리하지만, 이 비교의 목적은 거리값을 실제 거리로 해석하는 방식이 유효한가를 확인하는 것이었다. 따라서 baseline 비교는 최종 성능 경쟁이라기보다는 문제 구조를 판단하기 위한 기준으로 사용하였다. 넷째, 머신러닝 모델끼리는 같은 입력 데이터와 같은 validation 조건에서 비교했으므로 상대적인 성능 비교가 비교적 공정하다고 볼 수 있다.

다만 본 평가 방식에도 한계는 있다. 전체 700개 중 600개를 학습에 사용하고 100개를 validation으로 사용한 단일 split 결과는 validation set의 구성에 따라 성능이 달라질 수 있다. 실제로 validation 데이터 100개가 어떤 샘플로 구성되느냐에 따라 결과가 다르게 나타날 수 있으므로, 단일 validation 결과가 항상 hidden test 성능을 정확히 대표한다고 보기는 어렵다. 따라서 본 프로젝트에서는 validation 결과를 알고리즘 선택을 위한 기준으로 사용하되, 최종 제출용 model.pkl은 제공된 전체 700개 데이터를 다시 학습하여 저장하였다. 이는 hidden test에 대비하여 주어진 학습 정보를 최대한 활용하기 위한 절차이다. 단, 전체 700개에 대해 다시 평가한 결과는 학습 데이터가 포함된 값이므로 일반화 성능 지표로 사용하지 않았다.

Geometry-aware GradientBoosting의 장점은 세 가지이다. 첫째, d_hat을 clean distance로 가정하지 않고 noisy RTT fingerprinting feature로 활용하므로, WLS나 LS보다 데이터셋 특성에 더 적합하다. 둘째, BS_positions를 이용해 weighted centroid, geometry-consistent 후보 위치, residual 통계량 등을 생성하므로 앵커 배치 정보를 모델에 반영할 수 있다. 셋째, scikit-learn 기반 GradientBoosting과 MultiOutputRegressor로 구현할 수 있어 추가 외부 라이브러리 의존성이 작고, 제출 환경에서 재현성이 높다.

반면 단점도 존재한다. 첫째, 2m 이내 비율은 15% 수준으로 높지 않다. 이는 d_hat이 실제 거리와 크게 다르고, 비슷한 RTT pattern을 가진 샘플이 실제 공간에서는 서로 멀리 떨어져 있을 수 있기 때문으로 해석된다. 둘째, GradientBoosting은 학습 데이터의 분포를 기반으로 예측하므로 hidden test의 공간 분포나 NLOS 패턴이 학습 데이터와 크게 다르면 성능이 떨어질 수 있다. 셋째, 본 알고리즘은 시간 연속 정보, RSSI, SNR, CIR, LOS/NLOS label과 같은 추가 정보를 사용하지 못하므로, RTT 거리값만으로 구분하기 어려운 위치에서는 오차가 남을 수 있다.

향후 개선 방향으로는 세 가지를 고려할 수 있다. 첫째, 현재는 전체 700개 데이터 중 600개를 학습에 사용하고 100개를 검증에 사용하는 방식으로 성능을 확인하였다. 그러나 검증 데이터 100개가 어떤 샘플로 구성되느냐에 따라 결과가 달라질 수 있다. 예를 들어 우연히 어려운 샘플이 검증 데이터에 많이 포함되면 성능이 낮게 나오고, 상대적으로 쉬운 샘플이 포함되면 성능이 높게 나올 수 있다. 따라서 한 번의 데이터 분할 결과만으로 알고리즘 성능을 판단하기보다는, 데이터를 여러 번 다른 방식으로 나누어 반복적으로 평가하고 그 평균 성능을 확인하는 과정이 필요하다. 이렇게 하면 특정 검증 데이터 구성에만 유리한 모델인지, 아니면 전체 데이터에 대해 안정적으로 동작하는 모델인지 더 공정하게 판단할 수 있다.

둘째, 현재 알고리즘은 전체 실내 공간을 하나의 GradientBoosting 모델로 학습한다. 하지만 RTT 오차는 위치에 따라 다르게 나타날 수 있다. 예를 들어 특정 구역에서는 한 앵커의 RTT 값이 자주 크게 튀고, 다른 구역에서는 다른 앵커의 오차가 커질 수 있다. 따라서 향후에는 대략적인 위치 후보를 기준으로 공간을 몇 개의 구역으로 나눈 뒤, 각 구역에 맞는 모델을 따로 학습하는 방법을 고려할 수 있다. 이 방식은 위치별로 달라지는 RTT bias나 NLOS 패턴을 더 세밀하게 반영할 수 있다는 장점이 있다.

셋째, 현재 데이터셋에는 18개 RTT 거리값만 제공되기 때문에, 어떤 앵커가 LOS 환경이고 어떤 앵커가 NLOS 환경인지 명확히 구분하기 어렵다. 만약 RSSI, SNR, CIR, LOS/NLOS label과 같은 추가 센서 정보가 제공된다면, 먼저 이상 앵커나 NLOS 가능성이 높은 앵커를 판별한 뒤 위치를 예측하는 구조로 확장할 수 있다. 이 경우 현재보다 더 정확하게 신뢰도 낮은 거리값의 영향을 줄일 수 있을 것으로 예상된다.

결론적으로 본 프로젝트의 자체 평가 방식은 동일 데이터셋, 동일 validation split, 동일 성능 지표를 사용했다는 점에서 기본적인 공정성을 갖는다. 물리 기반 baseline과 머신러닝 모델의 비교는 모델 복잡도 측면에서는 완전히 동일하지 않지만, Final Project 데이터셋에서 clean ranging 기반 해법이 적합한지 판단하기 위한 기준으로 의미가 있다. 최종적으로 Geometry-aware GradientBoosting은 물리 기반 baseline과 다른 머신러닝 후보보다 낮은 MAE와 RMSE를 보였고, d_hat을 실제 거리로 신뢰하지 않고 RTT pattern과 앵커 geometry를 함께 학습하는 접근이 본 데이터셋에 가장 적합하다고 판단하였다.


## 5. Reference

[1] Lim DZ, Yeo M, Dahan A, Tahayori B, Kok HK, Abbasi-Rad M, et al. “Development of a machine learning-based real-time location system to streamline acute endovascular intervention in acute stroke: a proof-of-concept study.” Journal of NeuroInterventional Surgery, 2022;14:799-803. PubMed: https://pubmed.ncbi.nlm.nih.gov/34426539/

[2] scikit-learn Developers. “GradientBoostingRegressor.” scikit-learn documentation. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html

[3] scikit-learn Developers. “MultiOutputRegressor.” scikit-learn documentation. https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputRegressor.html

[4] scikit-learn Developers. “SVR.” scikit-learn documentation. https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html

[5] IBM. “릿지 회귀란 무엇인가요?” IBM Think. https://www.ibm.com/kr-ko/think/topics/ridge-regression

본 프로젝트에서 가장 직접적으로 참고한 연구는 Lim et al.의 machine learning 기반 real-time location system 연구이다. 해당 연구는 WiFi fingerprinting 데이터를 이용하여 병원 내 위치 또는 구역을 예측하는 문제를 다루었으며, RandomForest와 SVM 같은 전통적 머신러닝 모델이 실내 위치 추정 문제에서 높은 성능을 보일 수 있음을 보고하였다. 나는 이 논문을 통해 실내 위치 추정 문제에서 거리 또는 신호 패턴을 직접 학습하는 머신러닝 접근이 유효할 수 있다는 점을 참고하였다.

그러나 Lim et al.의 연구와 본 프로젝트는 문제 설정이 다르다. 해당 논문은 WiFi fingerprinting 기반 real-time location system을 다루며, 주어진 신호 패턴을 바탕으로 특정 위치 또는 구역을 예측하는 문제에 가깝다. 반면 본 프로젝트는 18개 기지국의 RTT 거리 추정값 d_hat과 앵커 좌표 BS_positions를 이용하여 연속적인 2차원 좌표 p_hat = (x, y)를 예측하는 회귀 문제이다. 따라서 본 프로젝트에서는 해당 논문의 알고리즘을 그대로 복제하지 않았고, 실내 위치 추정 문제에서 RandomForest, SVM, GradientBoosting과 같은 전통적 머신러닝 모델을 후보로 검토할 수 있다는 근거로 참고하였다.

본인이 제안한 부분은 DH_FR1.mat 데이터셋의 특성을 직접 분석하고, 그 구조에 맞는 알고리즘으로 바꾼 점이다. 먼저 BS_positions와 정답 위치 p를 이용하여 실제 기하학적 거리 d_true = ||p - b_i||를 계산하였고, 이를 d_hat과 비교하였다. 그 결과 d_hat은 실제 거리와 큰 차이를 보였으며, 평균 절대 거리 오차가 약 16.17 m 수준으로 나타났다. 이 분석을 통해 본 문제를 clean distance 기반 삼변측량 문제가 아니라, NLOS와 bias가 강하게 포함된 noisy RTT fingerprinting 문제로 해석하였다.

이러한 데이터 분석을 바탕으로 본 프로젝트에서는 기존 중간발표 SAFL 알고리즘을 그대로 사용하지 않았다. 기존 SAFL은 UWB와 Wi-Fi의 센서 합의도를 계산하여 앵커별 신뢰도를 부여하는 구조였지만, Final Project 데이터셋에는 UWB와 Wi-Fi가 분리된 센서 쌍으로 제공되지 않는다. 따라서 기존 SAFL의 핵심인 UWB-Wi-Fi sensor agreement를 그대로 적용하기 어렵다고 판단하였다. 대신 SAFL의 문제의식인 이상 거리값, 앵커 신뢰도, 잔차 기반 판단은 유지하되, 이를 d_hat과 BS_positions 기반 feature 생성 과정으로 재해석하였다.

최종적으로 본 프로젝트에서 제안한 알고리즘은 Geometry-aware GradientBoosting 기반 RTT Fingerprinting 위치 추정 알고리즘이다. 이 알고리즘은 d_hat을 실제 거리로 그대로 신뢰하지 않고, 18개 RTT 거리값의 패턴과 앵커 geometry 정보를 함께 사용한다. 구체적으로 원본 RTT 거리값, bias 보정 거리값, 거리 통계량, weighted centroid, geometry-consistent 후보 위치, residual 통계량, 가까운 앵커 및 먼 앵커 정보를 feature로 구성하고, GradientBoosting 모델이 최종 위치 좌표를 직접 예측하도록 설계하였다.

scikit-learn의 GradientBoostingRegressor와 MultiOutputRegressor 문서는 본 알고리즘을 구현하기 위한 공식 라이브러리 사용 방법을 확인하는 데 참고하였다. GradientBoostingRegressor는 여러 decision tree를 순차적으로 학습하면서 이전 단계의 오차를 줄이는 모델이며, MultiOutputRegressor는 x좌표와 y좌표를 동시에 예측하기 위해 사용하였다. 즉, 참고문헌 [2]와 [3]은 알고리즘 아이디어의 출처라기보다는, 내가 제안한 위치 회귀 구조를 Python 코드로 구현하기 위한 공식 문서로 활용하였다.

SVR 문서는 후보 모델 검토 과정에서 참고하였다. SVR은 RBF kernel을 사용할 수 있어 비선형 회귀 문제에 적용 가능하며, 실제 실험에서도 일부 샘플에서는 2 m 또는 3 m 이내 비율이 비교적 높게 나타났다. 그러나 평균 MAE와 RMSE 기준에서는 Geometry-aware GradientBoosting보다 낮은 성능을 보였고, C, gamma, epsilon과 같은 하이퍼파라미터에 민감하다는 단점이 있었다. 따라서 본 프로젝트에서는 SVR을 최종 모델로 선택하지 않고, 비교 후보로만 사용하였다.

IBM의 Ridge Regression 자료는 선형 회귀 기반 baseline 모델을 이해하기 위해 참고하였다. Ridge Regression은 L2 정규화를 통해 과적합을 줄일 수 있는 안정적인 선형 모델이지만, 본 프로젝트의 RTT 데이터는 NLOS와 bias로 인해 실제 거리와 비선형적인 관계를 보였다. 따라서 Ridge는 최종 모델이 아니라 선형 baseline으로 사용하였고, 비선형 위치 오차 패턴을 학습하기에는 GradientBoosting이 더 적합하다고 판단하였다.

정리하면, 참고문헌에서 제안하거나 설명하는 부분은 실내 위치 추정 문제에서 머신러닝 모델이 활용될 수 있다는 점, 그리고 Ridge, SVR, GradientBoosting 같은 회귀 모델의 기본 개념과 구현 방식이다. 반면 본인이 제안한 부분은 DH_FR1.mat 데이터셋에서 d_hat과 실제 거리의 차이를 직접 분석하고, 기존 SAFL의 센서 합의도 구조가 해당 데이터셋에 그대로 적용되기 어렵다는 점을 파악한 뒤, RTT pattern과 앵커 geometry feature를 함께 사용하는 Geometry-aware GradientBoosting 위치 예측 구조를 설계한 것이다.
