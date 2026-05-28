# RWML: Robust Weighted Machine Learning Residual Correction

## 1. 모티베이션 & 인트로

본 프로젝트는 18개 기지국이 측정한 RTT 기반 거리값을 이용하여 사용자 위치를 2차원 좌표로 추정하는 문제를 다룬다. 입력 데이터는 각 사용자에 대한 18개 거리 측정값 d_hat과 18개 기지국의 좌표 p_bs로 구성되며, 최종 목표는 각 사용자에 대해 추정 위치 p_hat을 계산하는 것이다. 최종 제출 파일의 main.py는 사용자 수를 하드코딩하지 않고 d_hat의 열 개수에 따라 동적으로 처리해야 하며, 결과는 p_hat shape = (2, num_user) 형태로 반환되어야 한다.

RTT 기반 실내 측위는 이론적으로 여러 기지국을 중심으로 하는 원의 교점을 찾는 문제로 해석할 수 있다. 이상적인 환경에서는 각 기지국과 사용자 사이의 거리값이 정확하므로 여러 원이 하나의 위치 근처에서 만난다. 그러나 실제 실내 환경에서는 NLOS, 반사, 장애물, 다중 경로, 잡음으로 인해 일부 거리값이 실제 거리보다 크게 측정될 수 있다. 이 경우 단순 삼각측량이나 일반 최소제곱 방식은 큰 오차를 가진 거리값에 쉽게 끌려가며, 최종 위치 추정 결과가 실제 위치에서 크게 벗어날 수 있다.

기본적인 WLS 방식은 각 거리값에 서로 다른 가중치를 부여할 수 있다는 장점이 있지만, 가중치가 잔차나 거리 분포만으로 결정되는 경우 hidden test set에서 충분히 일반화된다고 보기 어렵다. 따라서 본 프로젝트에서는 물리 기반 알고리즘과 데이터 기반 보정 방식을 결합한 RWML 구조를 제안한다. RWML은 먼저 Robust WLS로 기하학적으로 타당한 초기 위치 p0를 계산하고, 이후 머신러닝 모델이 p0의 오차 보정량을 예측하여 최종 위치를 산출하는 방식이다.

본 알고리즘은 완전한 end-to-end 딥러닝 모델이 아니다. 데이터 수가 700명으로 제한되어 있고 hidden test set이 존재하므로, 복잡한 딥러닝 모델보다 설명 가능하고 재현 가능한 회귀 모델을 비교하는 방향을 선택하였다. 참고 논문인 Lim et al.의 WiFi fingerprinting 기반 RTLS 연구에서는 Random Forest와 Support Vector Machine 계열 모델이 실내 위치 관련 분류 문제에서 높은 정확도를 보였다는 점을 확인하였다. 다만 해당 논문은 병원 구역을 분류하는 WiFi fingerprinting 문제이고, 본 프로젝트는 RTT 거리값을 이용해 2차원 좌표를 회귀하는 문제이므로 직접적인 알고리즘 재현이 아니라 모델 선택 방향의 참고 자료로 활용하였다.

## 2. 알고리즘 설명

제안하는 RWML 알고리즘은 전처리, Robust WLS 초기 추정, feature 생성, 머신러닝 모델 비교, 머신러닝 잔차 보정, 최종 위치 산출의 순서로 동작한다. 핵심 아이디어는 머신러닝이 위치를 처음부터 직접 예측하지 않고, 물리 기반 초기 위치 p0의 오차 보정량만 학습하도록 만드는 것이다. 이를 통해 기하학적 해석 가능성을 유지하면서도 데이터에서 반복적으로 나타나는 residual pattern을 보정할 수 있다.

첫 번째 단계에서는 거리값을 전처리한다. d_hat에는 NaN, inf, 음수 거리값, 물리적으로 지나치게 큰 거리값이 포함될 수 있으므로, 계산 전에 이를 안정적으로 처리한다. 기지국 좌표의 최소값과 최대값을 기준으로 측위 가능한 탐색 영역을 설정하고, 해당 영역에서 가능한 최대 거리 범위를 근사한다. 유효하지 않은 거리값은 중앙값 기반 대체값으로 치환하거나 낮은 신뢰도 가중치를 부여한다. 이 과정은 특정 거리값 하나가 전체 위치 추정 결과를 크게 왜곡하는 것을 방지하기 위한 단계이다.

두 번째 단계에서는 Robust WLS를 이용하여 초기 위치 p0를 계산한다. 임의의 후보 위치 x와 i번째 기지국 좌표 b_i에 대해 예측 거리는 ||x - b_i||로 계산된다. 측정 거리 d_i와의 잔차는 r_i = ||x - b_i|| - d_i 이다. 일반 WLS는 sum w_i r_i^2을 최소화하지만, 본 알고리즘에서는 잔차가 큰 앵커가 위치 추정을 지배하지 못하도록 IRWLS 구조를 적용한다. 초기에는 거리 분포와 물리적 타당성에 따라 1차 가중치를 부여하고, 이후 추정 위치에서 계산된 잔차의 크기에 따라 Tukey biweight 형태의 신뢰도 가중치를 반복적으로 갱신한다. 잔차가 작은 앵커는 높은 가중치를 유지하고, 잔차가 큰 앵커는 NLOS 또는 이상치 가능성이 높다고 판단하여 영향력을 낮춘다.

세 번째 단계에서는 머신러닝 입력 feature를 구성한다. feature에는 정제된 18개 거리값, 18개 잔차값, 18개 앵커 가중치, 각 앵커의 유효 여부, 거리값의 평균·표준편차·중앙값·최소값·최대값, 잔차의 평균·표준편차·중앙값·최대 절댓값, Robust WLS 초기 위치 p0, 기지국 중심과 p0 사이의 차이, 유효 앵커 개수, p0 기준 각 앵커까지의 예측 거리 등이 포함된다. 즉, 머신러닝 모델은 단순히 RTT 거리값만 보는 것이 아니라, 현재 샘플의 거리 분포가 얼마나 불안정한지, 어떤 앵커가 이상치로 의심되는지, Robust WLS가 계산한 초기해가 공간적으로 어느 위치에 있는지를 함께 입력받는다.

네 번째 단계에서는 머신러닝 모델을 학습한다. 모델은 최종 위치 p를 직접 예측하지 않고, Robust WLS 초기 위치 p0의 오차 보정량을 학습한다. 학습 target은 delta = p - p0 이다. 즉, 모델은 feature를 입력받아 delta_x와 delta_y를 예측한다. 최종 위치는 p_hat = p0 + f_ML(feature)로 계산된다. 이 방식은 순수 블랙박스 위치 예측보다 과적합 위험이 낮고, 물리 기반 초기 추정값을 유지하면서 데이터 기반 보정 효과를 추가할 수 있다는 장점이 있다.

본 프로젝트에서는 작은 데이터셋에서 설명 가능하고 재현 가능한 모델을 비교하기 위해 Ridge Regression, Random Forest Regressor, Gradient Boosting Regressor를 후보 모델로 두었다. 세 모델은 모두 동일한 feature와 동일한 validation split에서 비교되며, validation MAE가 가장 낮고 RMSE 및 2m 이내 비율이 안정적인 모델을 최종 model.pkl로 저장한다.

Ridge Regression은 선형 회귀에 L2 정규화 항을 추가한 모델이다. 일반 선형 회귀는 feature 수가 많거나 feature 간 상관관계가 높은 경우 계수가 과도하게 커져 학습 데이터에 민감해질 수 있다. Ridge는 큰 계수에 페널티를 부여하여 계수 크기를 줄이고, 학습 데이터에 대한 과적합을 완화한다. 본 프로젝트의 feature에는 거리값, 잔차, 가중치, 예측 거리처럼 서로 상관관계가 있는 항목이 많기 때문에 Ridge를 단순하고 안정적인 기준 모델로 사용하였다. Ridge는 비선형 패턴을 강하게 잡지는 못하지만, hidden test 일반화 관점에서 baseline ML 모델로 의미가 있다.

Random Forest Regressor는 여러 개의 decision tree를 독립적으로 학습한 뒤 평균을 내는 ensemble 모델이다. 각 트리는 데이터와 feature의 일부를 사용해 서로 다른 예측 규칙을 만들고, 최종 결과는 여러 트리의 평균으로 결정된다. 이 구조는 단일 decision tree의 과적합을 줄이고, 잔차·거리 통계·앵커별 가중치 사이의 비선형 관계를 비교적 안정적으로 학습할 수 있다. 본 프로젝트에서는 특정 앵커의 잔차가 크거나 유효 앵커 수가 적을 때 발생하는 비선형 오차 패턴을 학습할 수 있는 후보 모델로 Random Forest를 사용하였다. 다만 트리 개수와 깊이가 커지면 학습 데이터에 과도하게 맞춰질 수 있으므로 max_depth와 min_samples_leaf를 제한하였다.

Gradient Boosting Regressor는 여러 개의 약한 decision tree를 순차적으로 학습하면서 이전 단계에서 남은 오차를 점진적으로 줄이는 boosting 모델이다. Random Forest가 여러 트리를 병렬적으로 만들고 평균을 내는 방식이라면, Gradient Boosting은 이전 모델이 틀린 부분을 다음 모델이 보완하는 방식으로 동작한다. 본 프로젝트의 target 자체가 Robust WLS의 residual correction, 즉 p0가 남긴 오차 delta이기 때문에 Gradient Boosting의 구조와 잘 맞는다. 다시 말해 Robust WLS가 먼저 큰 위치 구조를 잡고, Gradient Boosting이 남은 작은 systematic bias를 단계적으로 보정하는 역할을 수행한다. validation 결과 Gradient Boosting이 가장 낮은 MAE를 보여 최종 residual correction 모델로 선택하였다.

SVM 또는 SVR 계열 모델은 참고 논문에서 Random Forest와 함께 높은 정확도를 보인 모델이지만, 본 프로젝트의 최종 후보에는 포함하지 않았다. 그 이유는 본 과제가 2차원 좌표 오차 보정이라는 regression 문제이고, feature 수가 많으며, SVR은 kernel과 scaling, C, epsilon, gamma 등 하이퍼파라미터 선택에 민감하기 때문이다. 또한 10분 실행 제한과 hidden test 일반화를 고려하면, 튜닝 부담이 큰 SVR보다 Ridge, Random Forest, Gradient Boosting처럼 구현과 재현성이 안정적인 모델을 비교하는 편이 적절하다고 판단하였다.

다섯 번째 단계에서는 main.py 추론 과정이 수행된다. 채점 시에는 정답 p를 사용할 수 없으므로, main.py는 d_hat과 기지국 좌표만을 읽는다. 각 사용자에 대해 Robust WLS로 p0를 계산하고, 동일한 feature 생성 과정을 거친 뒤, train.py에서 저장된 model.pkl을 불러와 보정량을 예측한다. model.pkl이 존재하지 않거나 로드에 실패하는 경우에는 Robust WLS 결과 p0를 그대로 반환하도록 fallback을 구성하였다. 이를 통해 채점 환경에서 모델 파일 문제로 전체 실행이 실패하는 상황을 방지한다.

참고 논문과 본 알고리즘의 차이는 명확하다. Lim et al.의 연구는 WiFi fingerprinting 데이터를 사용하여 병원 내 구역을 분류하는 real-time location system을 제안하였고, Random Forest와 SVM의 높은 정확도를 보고하였다. 반면 본 프로젝트는 18개 기지국의 RTT 거리값과 기지국 좌표를 이용해 연속적인 2차원 좌표를 추정하는 회귀 문제이다. 따라서 본 알고리즘은 해당 논문의 RF/SVM 성능 결과를 그대로 재현하지 않고, 실내 위치 문제에서 tree-based ML 모델이 유효할 수 있다는 근거로 참고하였다. 본 프로젝트의 차별점은 Robust WLS로 물리 기반 초기 위치를 먼저 계산하고, 머신러닝은 최종 위치가 아니라 오차 보정량 delta만 학습하도록 설계했다는 점이다.

## 3. Agent AI 활용 방안

본 프로젝트에서는 ChatGPT를 알고리즘 아이디어 정리, 코드 구조 설계, 보고서 문장 정리, GitHub 제출 구조 점검에 활용하였다. 특히 단순 WLS 방식과 머신러닝 직접 예측 방식의 장단점을 비교하고, 최종적으로 물리 기반 초기 추정과 머신러닝 잔차 보정을 결합한 RWML 구조를 선택하는 데 보조적으로 사용하였다.

AI는 여러 후보 알고리즘의 장단점과 구현 흐름을 제안하는 역할을 수행하였다. 예를 들어 Ridge Regression, Random Forest, Gradient Boosting, SVR 등의 후보를 비교하고, 작은 데이터셋과 hidden test 환경에서는 복잡한 딥러닝보다 설명 가능한 회귀 모델이 더 안정적일 수 있다는 방향을 정리하는 데 활용하였다. 다만 최종 알고리즘의 방향 설정, 데이터 shape 확인, validation 결과 해석, hidden test 과적합 가능성 판단은 사용자가 직접 확인하는 방식으로 진행하였다. 따라서 본 프로젝트에서 Agent AI는 답안을 대체하는 도구가 아니라, 후보 설계안을 빠르게 비교하고 구현을 정리하기 위한 보조 도구로 활용되었다.

## 4. 결과 도출 & 디스커션

본 알고리즘은 Baseline WLS, Robust WLS, RWML의 순서로 성능을 비교하는 방식으로 평가한다. 평가지표는 MAE, RMSE, 2m 이내 비율, 실행시간을 사용한다. MAE는 평균적인 위치 오차를 나타내고, RMSE는 큰 오차에 더 민감하게 반응하므로 이상치 영향을 확인하는 데 유용하다. 2m 이내 비율은 실제 실내 측위 시스템에서 사용자가 체감할 수 있는 정확도 지표로 볼 수 있다.

| Method | MAE | RMSE | Within 2m (%) | Runtime |
|---|---:|---:|---:|---:|
| Baseline WLS | 22.1589 | 23.8616 | 0.00 | validation 기준 |
| Robust WLS | 15.5857 | 18.6215 | 8.00 | validation 기준 |
| RWML, GradientBoosting | 6.6253 | 8.5013 | 18.29 | train.py 기준 약 13.67s |

비교의 공정성을 위해 모든 알고리즘은 동일한 입력 데이터와 동일한 validation split에서 평가하였다. 단순 삼각측량과 복잡한 머신러닝 모델을 서로 다른 조건에서 비교하는 것은 공정하지 않으므로, 본 프로젝트에서는 Robust WLS를 물리 기반 baseline으로 두고, RWML이 해당 baseline의 오차를 얼마나 줄이는지에 초점을 맞춘다.

결과적으로 Baseline WLS는 일부 큰 RTT 오차에 민감하게 반응하여 MAE와 RMSE가 크게 나타났다. Robust WLS는 잔차 기반 가중치 갱신을 통해 이상치 앵커의 영향을 줄였고, Baseline WLS보다 오차가 감소하였다. 그러나 Robust WLS는 여전히 모든 보정을 잔차 기반 규칙에 의존하므로, 반복적으로 나타나는 비선형 bias를 충분히 보정하기 어렵다. RWML은 Robust WLS의 초기 위치 p0와 잔차 feature를 이용하여 delta를 학습함으로써 MAE와 RMSE를 추가로 줄였다. 이는 물리 기반 초기해와 머신러닝 보정 구조가 상호 보완적으로 작동했음을 의미한다.

RWML의 장점은 물리 기반 해석 가능성과 머신러닝 보정의 유연성을 동시에 가진다는 점이다. Robust WLS는 각 앵커의 잔차와 가중치를 통해 어떤 거리값이 위치 추정에 영향을 주었는지 설명할 수 있고, 머신러닝 모델은 거리 분포와 잔차 패턴을 이용해 반복적으로 나타나는 bias를 보정할 수 있다. 특히 Gradient Boosting은 이전 단계의 residual error를 순차적으로 보완하는 구조이므로, 본 프로젝트의 residual correction 문제와 잘 맞는다.

반면 한계도 존재한다. 제공된 학습 데이터 700명에 과적합될 가능성이 있으며, hidden test set의 공간 분포나 NLOS 패턴이 크게 달라지는 경우 보정량 예측이 흔들릴 수 있다. 또한 본 알고리즘은 시간 순서 정보를 사용하지 않기 때문에, 사용자의 이동 연속성이나 순간 튐을 직접적으로 보정하지 못한다. 향후에는 앵커별 NLOS 확률을 직접 예측하거나, 앵커별 신뢰도를 누적 학습하는 방식, 시간 연속 데이터가 주어질 경우 temporal filtering을 적용하는 방식으로 개선할 수 있다. 추가적으로 feature importance 분석을 통해 어떤 앵커의 잔차나 어떤 통계량이 보정량 예측에 크게 기여했는지를 확인하면 알고리즘의 해석 가능성을 더 높일 수 있다.

## 5. Reference

[1] Lim DZ, Yeo M, Dahan A, Tahayori B, Kok HK, Abbasi-Rad M, et al. “Development of a machine learning-based real-time location system to streamline acute endovascular intervention in acute stroke: a proof-of-concept study.” Journal of NeuroInterventional Surgery, 2022;14:799-803. PubMed: https://pubmed.ncbi.nlm.nih.gov/34426539/

[2] IBM. “릿지 회귀란 무엇인가요?” IBM Think. https://www.ibm.com/kr-ko/think/topics/ridge-regression

[3] scikit-learn Developers. “Ridge.” scikit-learn documentation. https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html

[4] scikit-learn Developers. “RandomForestRegressor.” scikit-learn documentation. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

[5] scikit-learn Developers. “GradientBoostingRegressor.” scikit-learn documentation. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html

본 보고서는 수업에서 제공된 RTT 기반 실내 측위 데이터와 일반적인 Weighted Least Squares, Robust Regression, Huber Loss, Tukey biweight, Machine Learning Residual Correction 개념을 바탕으로 작성하였다. 외부 연구는 본 과제 알고리즘을 직접 복제하기 위한 자료가 아니라, 실내 위치 추정 문제에서 Random Forest, SVM 등 전통적 머신러닝 모델이 활용될 수 있음을 확인하고, 작은 데이터셋에서 설명 가능한 회귀 모델을 비교하기 위한 참고 자료로 사용하였다.
