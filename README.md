# AIS Ship Type and Route Prediction System

AIS 데이터를 활용해 선박 종류 분류, 항로 분석, 이상 탐지, 정박지 분석, 미래 좌표 예측을 연결한 프로젝트입니다. 결과는 Flask + Leaflet 웹 지도와 발표용 figure로 확인합니다.

## 핵심 설계

- 선종 분류는 MMSI 기준 group split으로 같은 선박이 train/test에 동시에 들어가는 누수를 줄입니다.
- 항로 분류기는 기본적으로 `처음 6시간` 관측 특징과 초기 궤적 signature만 사용합니다. `end_lat`, `end_lon`, 전체 `duration_hours`, 전체 이동거리, 전체 위경도 범위는 classifier 입력에서 제외했습니다.
- 전체 궤적 signature는 항로 라벨링(KMeans), 이상 거리 계산, 보고용 CSV에만 사용합니다.
- 미래 위치 예측은 MMSI GroupShuffleSplit holdout으로 평가하고, RandomForest, 현재 위치 유지, 등속 직선(dead-reckoning), RF+DR 앙상블을 함께 저장합니다.
- 선종 `predicted_shiptype_probability`는 MMSI group holdout에서 학습한 isotonic calibration을 거친 confidence입니다. 항로 probability는 calibration 전 confidence입니다.

## 핵심 파일

- `shipml/type_anal/ship_type_model.py`: 선종 모델 공통 학습/추론 유틸리티
- `shipml/type_anal/train_ship_type_classifier_group_split.py`: MMSI group split 기반 선종 평가 및 저장
- `shipml/type_anal/tune_ship_type_classifier_group_split.py`: nested 구조의 선종 RandomForest 튜닝
- `shipml/type_anal/export_ship_type_classifier_reports.py`: class metrics, confusion matrix, feature importance 산출
- `shipml/route_anal/ship_route_anal.py`: 항로 라벨링, 초기 항로 분류, 이상 탐지, 정박지 분석
- `shipml/route_anal/train_future_position_regressor.py`: horizon별 미래 좌표 예측 및 baseline 비교
- `shipml/type_anal/add_ship_type_predictions_to_routes.py`: 항로 예측 결과에 선종 예측 결합
- `shipml/reports/export_project_model_figures.py`: 발표용 모델 결과 figure 생성
- `shipml/web/app.py`: Flask + Leaflet 웹 지도

## 데이터 스키마

항로/미래 위치 AIS 입력 필수 컬럼:

- `MMSI`, `Timestamp`, `Latitude`, `Longitude`

항로/미래 위치 AIS 선택 컬럼:

- `SOG`, `COG`, `Width`, `Length`, `Draught`, `shiptype`

선종 분류 입력:

- target: `shiptype`
- group: `mmsi`
- 주요 feature: `navigationalstatus`, `sog`, `cog`, `heading`, `width`, `length`, `draught`, `cog_sin`, `cog_cos`, `heading_sin`, `heading_cos`

## 실행 순서

```powershell
C:\Users\green\anaconda3\envs\myproject\python.exe shipml\type_anal\train_ship_type_classifier_group_split.py --models random_forest xgboost --compare-random-split
C:\Users\green\anaconda3\envs\myproject\python.exe shipml\type_anal\tune_ship_type_classifier_group_split.py
C:\Users\green\anaconda3\envs\myproject\python.exe shipml\type_anal\export_ship_type_classifier_reports.py
C:\Users\green\anaconda3\envs\myproject\python.exe shipml\route_anal\ship_route_anal.py
C:\Users\green\anaconda3\envs\myproject\python.exe shipml\route_anal\train_future_position_regressor.py
C:\Users\green\anaconda3\envs\myproject\python.exe shipml\type_anal\add_ship_type_predictions_to_routes.py --model shipml\type_anal\outputs\ship_type_classifier_group_split.joblib --metrics shipml\type_anal\outputs\ship_type_classifier_group_split_metrics.json
C:\Users\green\anaconda3\envs\myproject\python.exe shipml\reports\export_project_model_figures.py
C:\Users\green\anaconda3\envs\myproject\python.exe -m flask --app shipml.web.app:app run --host 127.0.0.1 --port 5000 --no-reload
```

웹 접속:

```text
http://127.0.0.1:5000
```

## 주요 산출물

- `shipml/type_anal/outputs/ship_type_classifier_group_split_metrics.json`
- `shipml/type_anal/outputs/ship_type_classifier_tuned_group_split_metrics.json`
- `shipml/type_anal/outputs/ship_type_classifier_class_metrics.csv`
- `shipml/type_anal/outputs/ship_type_classifier_confusion_matrix.csv`
- `shipml/type_anal/outputs/ship_type_classifier_confusion_pairs.csv`
- `shipml/route_anal/outputs/run_summary.json`
- `shipml/route_anal/outputs/route_early_window_metrics.csv`
- `shipml/route_anal/outputs/route_predictions.csv`
- `shipml/route_anal/outputs/route_predictions_with_types.csv`
- `shipml/route_anal/outputs/future_position_regressor_metrics.json`
- `shipml/route_anal/outputs/future_position_forecast.csv`
- `shipml/reports/figures/*.png`
- `shipml/reports/project_upgrade_report.md`

## 현재 기준 성능

- 선종 분류 대표 지표: RandomForest, MMSI group split 기준 Accuracy 0.8453, Macro F1 0.7378, train/test MMSI overlap 0
- 선종 confidence calibration: ECE 0.0324 → 0.0103
- 항로 분류: KMeans route label 12개, 초기 6시간 관측 feature 기준 Accuracy 0.8507, Macro F1 0.8504
- 항로 엄격 평가: KMeans를 train fold에서만 fit한 holdout Accuracy 0.8258, Macro F1 0.8302
- 항로 시간 평가: 과거 vessel train, 이후 vessel test 기준 Accuracy 0.7379, Macro F1 0.7366
- 항로 관측창별 Accuracy: 1h 0.8214 / 3h 0.8287 / 6h 0.8507 / 12h 0.8858
- 미래 좌표 RF+DR 앙상블: MMSI GroupShuffleSplit 기준 평균 오차 1h 1.55km / 2h 3.16km / 3h 4.62km, train/test MMSI overlap 0
- 미래 좌표 RandomForest 단독: 1h 2.31km / 2h 3.42km / 3h 4.78km
- 현재 위치 유지 baseline: 1h 2.97km / 2h 5.64km / 3h 8.13km
- dead-reckoning baseline: 1h 1.56km / 2h 3.78km / 3h 6.26km

미래 위치 모델은 RF와 dead-reckoning을 horizon별로 섞는 앙상블로 배포합니다. 검증 가중치는 1h RF 0.10, 2h RF 0.65, 3h RF 0.85입니다.

## 해석 포인트

- 항로 classifier 입력은 초기 관측 구간으로 제한되어 있지만, 항로 label 자체는 전체 궤적 signature 기반 KMeans cluster입니다.
- 항로 이상 탐지는 전체 궤적이 충분히 관측된 뒤의 post-analysis 성격이 강합니다. 실시간 조기 이상 탐지라고 표현하면 과장입니다.
- class imbalance 때문에 선종 accuracy만 보면 안 됩니다. `ship_type_classifier_class_metrics.csv`와 confusion pair를 함께 보세요.
- 항로 probability는 calibration 전 confidence입니다. 선종 probability는 보정됐지만 데이터 drift가 생기면 calibration도 다시 해야 합니다.

## 다음 업그레이드 후보

- 항로 라벨 품질 개선: KMeans pseudo-label 대신 실제 항로/항만 OD 라벨이나 해역 graph 기반 라벨 도입
- 미래 위치 모델 고도화: 칼만필터, IMM, sequence model, 해안선/항로망 제약을 비교
- 항로 probability calibration 추가: route classifier confidence도 별도 holdout으로 보정
- class imbalance 대응 강화: rare class 재가중, Towing/HSC/Pleasure 등 소수 class 별도 분석
- 운영 모니터링: 최근 데이터에서 calibration drift, route distribution drift, horizon별 오차 drift 추적
- QGIS/웹 지도 QA: 예측 항로 중심선과 실제 대표 track의 차이, 육지 관통 여부, 정박지 cluster 품질을 별도 표로 관리

## 환경 설정

Conda 환경을 새로 만들 때:

```powershell
conda env create -f environment.yml
conda activate shipml
```

이미 만든 Python 환경에 패키지만 설치할 때:

```powershell
C:\Users\green\anaconda3\envs\myproject\python.exe -m pip install -r requirements.txt
```

## 대용량 모델 파일

`.joblib` 모델 파일은 Git에 올리지 않고 GitHub Release에 보관합니다.

Release tag:

```text
shipml-model-artifacts-v1
```

업로드:

```powershell
$env:GITHUB_TOKEN="ghp_..."
.\scripts\upload_model_artifacts_to_github_release.ps1
```

다운로드:

```powershell
.\scripts\download_model_artifacts_from_github_release.ps1
```
