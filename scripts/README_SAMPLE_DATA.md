# Sample Data Generation Guide

이 프로젝트는 **샘플 데이터로 시작**하는 것을 권장합니다. 전체 M5 데이터는 용량이 크고 전처리/학습 시간이 길 수 있습니다.

## 기본 생성 방법
```bash
python scripts/generate_sample_data.py
```

## 기본 설정값
- stores: 3
- items: 200
- days: 120

## 파라미터 변경
`generate_sample_data()` 함수 인자를 변경해 샘플링 규모를 조정할 수 있습���다.

예시:
```python
# scripts/generate_sample_data.py
# generate_sample_data(num_stores=5, num_items=300, days=180)
```

## 생성 파일
- data/sample/sales_sample.csv
- data/sample/calendar_sample.csv
- data/sample/prices_sample.csv

## 참고
샘플링 규모를 줄이면 학습/전처리 시간이 크게 단축됩니다. 이후 전체 데이터로 확장 가능하다는 점만 README에 언급하면 충분합니다.
