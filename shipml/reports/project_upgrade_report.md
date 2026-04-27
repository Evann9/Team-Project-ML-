# ShipML Upgrade Report

## 한 줄 요약

선박 종류 분류, 항로 분석, 이상 탐지, 정박지 분석, 미래 위치 예측을 하나의 AIS 분석 파이프라인으로 묶고, 누수 방지 평가와 baseline 비교를 산출물에 남겼다.

## 완료한 업그레이드

1. 항로 classifier를 초기 관측 기반으로 제한했다.
   - 기본 관측창: 처음 6시간
   - classifier 제외: `point_count`, `end_lat`, `end_lon`, 전체 `duration_hours`, 전체 위경도 범위, 전체 이동거리
   - 전체 궤적은 route label 생성, 이상 거리 계산, 보고용 산출물에만 사용

2. 항로 관측창별 성능표를 추가했다.
   - 산출물: `shipml/route_anal/outputs/route_early_window_metrics.csv`
   - Accuracy: 1h 0.8214 / 3h 0.8287 / 6h 0.8507 / 12h 0.8858
   - Macro F1: 1h 0.8152 / 3h 0.8211 / 6h 0.8504 / 12h 0.8853

3. 항로 KMeans까지 train fold 안에서만 학습하는 엄격 평가를 추가했다.
   - 일반 holdout: Accuracy 0.8507, Macro F1 0.8504
   - strict route catalog holdout: Accuracy 0.8258, Macro F1 0.8302
   - temporal route holdout: Accuracy 0.7379, Macro F1 0.7366

4. 미래 위치 예측에 RF+dead-reckoning 앙상블을 도입했다.
   - RF+DR 앙상블: 1h 1.55km / 2h 3.16km / 3h 4.62km
   - RandomForest 단독: 1h 2.31km / 2h 3.42km / 3h 4.78km
   - Dead-reckoning baseline: 1h 1.56km / 2h 3.78km / 3h 6.26km
   - 배포 가중치: 1h RF 0.10, 2h RF 0.65, 3h RF 0.85

5. 미래 위치 시간 기준 검증을 추가했다.
   - 산출물: `future_position_regressor_metrics.json`의 `temporal_holdout`
   - Ensemble temporal error: 1h 1.31km / 2h 2.40km / 3h 3.41km

6. 선종 confidence calibration을 추가했다.
   - 모델 선택 기준을 macro F1 우선으로 변경해 class imbalance에 더 맞췄다.
   - 선택 모델: RandomForest
   - MMSI group split: Accuracy 0.8453, Macro F1 0.7378
   - Calibration ECE: 0.0324 -> 0.0103

7. 웹 지도에 검증 요약을 추가했다.
   - 항로 일반/strict/temporal Accuracy
   - 초기 관측 시간, route class 수, anomaly count
   - 미래 위치 앙상블 오차와 dead-reckoning baseline 비교

## 발표에서 말할 수 있는 강점

- 같은 MMSI가 train/test에 동시에 들어가지 않도록 선종과 미래 위치 평가를 설계했다.
- 항로 예측은 전체 궤적을 몰래 쓰지 않고 초기 관측 feature만 사용한다.
- 미래 위치 모델은 baseline과 비교했으며, RF 단독 약점을 dead-reckoning 앙상블로 줄였다.
- 선종 모델은 accuracy뿐 아니라 macro F1, class metrics, confusion pair, calibration까지 제시한다.
- 웹 지도는 단순 시각화가 아니라 모델 검증 지표와 함께 결과를 확인하는 데모다.

## 남은 한계

- 항로 label은 실제 정답 항로가 아니라 KMeans 기반 pseudo-label이다.
- 항로 이상 탐지는 전체 궤적 관측 이후의 post-analysis 성격이 강하다.
- 항로 probability는 아직 calibration 전 confidence다.
- 시간 기준 평가는 데이터 기간과 지역이 제한적이어서 운영 drift를 완전히 증명하지는 못한다.
- AIS 품질, 결측, class imbalance가 여전히 성능에 영향을 준다.

## 다음 개선 우선순위

1. 실제 OD/항만/항로망 기반 route label 도입
2. 항로 probability calibration
3. 칼만필터/IMM/sequence model과 미래 위치 예측 비교
4. 해안선과 항로망 제약을 반영한 미래 위치 후보 보정
5. 기간별 drift 모니터링 리포트 자동 생성
