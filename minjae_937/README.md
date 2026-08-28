# LG Aimers 제구 성공 확률 예측 — 937.858점 제출본

DACON Public 리더보드에서 **937.8580380511점**을 기록한 추론 제출본입니다.

이 폴더는 이후 실험본이 아니라 해당 점수를 기록한 파일을 그대로 보존합니다.

## 모델 구성

- CatBoost depth 8
- 최근 시즌 가중치(`season decay=0.45`) CatBoost seed 123
- HistGradientBoosting
- 공식 학습 데이터의 팀 ID Target Encoding을 사용하는 1-tree XGBoost 보조 신호
- 가중 소프트보팅 후 Logit 확률 보정

앙상블 가중치는 다음과 같습니다.

```text
CatBoost depth 8           0.14395068
HistGradientBoosting       0.19208417
최근 시즌 가중 CatBoost     0.21364048
XGBoost 보조 신호           0.45032467
```

최종 확률 보정값:

```text
slope     = 2.0
intercept = -0.13282943247529952
```

## 제출 구조

```text
submit.zip
├── model/
│   ├── cat_d8.cbm
│   ├── cat_decay045_seed123.cbm
│   ├── histboost.joblib
│   └── xgboost.joblib
├── script.py
└── requirements.txt
```

`submit.zip`을 DACON 코드 제출 화면에 그대로 업로드하면 됩니다.

압축을 풀어 실행할 때 평가 서버가 다음 파일을 추가합니다.

```text
data/test.csv
data/sample_submission.csv
output/
```

실행 명령:

```bash
python script.py
```

결과는 `output/submission.csv`에 생성됩니다.

## 재현 정보

- 245,789행 로컬 추론 시간: 약 7초
- 제출 ZIP SHA-256: `8A2D547D4CDD85D58D4C953174EBBE9B675FFA938CB8192694341131E9F507A0`
- 추가 패키지: `catboost==1.2.10`, `xgboost==3.0.5`

## 규칙 준수

- 대회 공식 학습 데이터만 사용
- 외부 데이터 및 외부 API 미사용
- Target Encoding 통계는 공식 학습 데이터로만 계산
- 테스트 데이터 행 간 집계 및 보정 미사용
- 각 테스트 행을 독립적으로 추론

