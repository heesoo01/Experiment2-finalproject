# RWML 실행 순서

## 1. 파일 배치
아래 파일들을 같은 폴더에 둔다.

- DH_FR1.mat
- common_rwml.py
- train.py
- main.py
- report.md

업로드 받은 파일명이 `InF_DH_FR1.mat`이면 `DH_FR1.mat`으로 이름을 바꾸는 것을 권장한다.
단, 현재 코드는 `InF_DH_FR1.mat`도 자동 인식한다.

## 2. 학습 실행
```bash
python train.py
```

성공하면 `model.pkl`이 생성된다.

## 3. 채점용 실행 확인
```bash
python main.py
```

정상이라면 `p_hat shape: (2, 700)`처럼 출력된다.

## 4. GitHub 업로드 파일
최종 repo 루트에는 최소한 다음 파일이 있어야 한다.

- main.py
- train.py
- common_rwml.py
- report.md
- model.pkl
- requirements.txt

`DH_FR1.mat`은 과제 제출 repo에 올려도 되는지 수업 지시를 확인해야 한다.
일반적으로 채점기는 자체 mat 파일을 넣어 실행하므로, 코드 제출용 repo에는 데이터 파일을 제외하는 편이 안전하다.
