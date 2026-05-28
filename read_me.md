# RWML: Robust Weighted Machine Learning Residual Correction

## 1. 제출 파일

본 프로젝트 제출용 ZIP에는 아래 5개 파일이 포함되어 있습니다.

| 파일                 | 설명                                                                                                               |
| ------------------ | ---------------------------------------------------------------------------------------------------------------- |
| `main.py`          | RWML 추론 알고리즘. Robust WLS 초기 위치 + GradientBoosting residual correction. Standalone, 채점 환경에서 바로 실행 가능.             |
| `train.py`         | RWML 학습 알고리즘. Ridge, RandomForest, GradientBoosting 후보 모델 비교 및 GradientBoosting 학습 후 `model.pkl` 생성. Standalone. |
| `model.pkl`        | 학습된 GradientBoosting 모델. `main.py`에서 불러 residual correction 적용.                                                  |
| `report.md`        | 제출용 보고서. RWML 알고리즘 구조, 결과, 머신러닝 모델 후보 비교, Reference 포함.                                                          |
| `requirements.txt` | 필요 Python 패키지 명시.                                                                                                |

## 2. 실행 방법

### 2.1 모델 학습 (선택 사항)

```bash
python train.py
```

* 학습 데이터: `DH_FR1.mat` (로컬 테스트용, 제출 ZIP에는 포함하지 않음)
* 실행 후 `model.pkl` 생성
* Ridge, RandomForest, GradientBoosting 후보 모델 비교, validation 성능 기준 최종 모델 선택

### 2.2 RWML 추론

```bash
python main.py
```

* 입력: `DH_FR1.mat` (채점 환경에서 제공)
* 출력: `p_hat` shape `(2, num_user)`
* `model.pkl` 로드 실패 시 Robust WLS 초기 위치 p0 반환 (fallback)

## 3. 파일 구조

```text
project_root/
├─ main.py
├─ train.py
├─ model.pkl
├─ report.md
└─ requirements.txt
```

* 모든 코드와 알고리즘은 독립적으로 실행 가능
* 채점 기준에 맞게 main.py가 RWML 핵심 알고리즘 포함

## 4. 알고리즘 요약

1. **전처리**: 거리값 NaN, inf, 음수, 과도한 값 처리
2. **Robust WLS 초기 위치**: Tukey biweight 기반 IRWLS 적용
3. **Feature 생성**: 거리값, 잔차, 앵커 가중치, p0, 거리 통계량 등
4. **Residual ML correction**: Ridge, RandomForest, GradientBoosting 후보 모델 비교 → GradientBoosting 선택
5. **최종 위치**: `p_hat = p0 + delta_hat`
6. **Fallback**: `model.pkl` 없으면 `p0` 반환

## 5. Reference

1. Lim DZ, et al. "Development of a machine learning-based real-time location system to streamline acute endovascular intervention in acute stroke: a proof-of-concept study." Journal of NeuroInterventional Surgery, 2022;14:799-803. PubMed: [https://pubmed.ncbi.nlm.nih.gov/34426539/](https://pubmed.ncbi.nlm.nih.gov/34426539/)
2. IBM. "릿지 회귀란 무엇인가요?" IBM Think. [https://www.ibm.com/kr-ko/think/topics/ridge-regression](https://www.ibm.com/kr-ko/think/topics/ridge-regression)
3. scikit-learn Developers. "Ridge / RandomForestRegressor / GradientBoostingRegressor". [https://scikit-learn.org](https://scikit-learn.org)
