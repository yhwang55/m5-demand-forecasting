# M5 Demand Forecasting

M5 Forecasting (Walmart 5년치 일별 판매) 데이터 기반 수요예측 프로젝트 스켈레톤입니다.  
Baseline + LightGBM + Prophet 모델을 포함하며, 노트북 단계별 구성과 성능 비교, 가격 탄력성 분석, Streamlit 대시보드 MVP를 제공합니다.

## 프로젝트 구조
```
M5 Demand Forecasting/
├─ README.md
├─ data/
│  ├─ sample/                 # 샘플 데이터 (스토어/상품/캘린더)
│  ├─ raw/                    # 원본 데이터 (Kaggle M5)
│  └─ processed/              # 전처리 데이터
├─ notebooks/
│  ├─ 00_data_overview.ipynb
│  ├─ 01_baseline_model.ipynb
│  ├─ 02_lightgbm_model.ipynb
│  ├─ 03_prophet_model.ipynb
│  ├─ 04_model_comparison.ipynb
│  ├─ 05_price_elasticity.ipynb
│  └─ 06_dashboard_preview.ipynb
├─ src/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ data.py
│  ├─ features.py
│  ├─ metrics.py
│  ├─ pipeline.py
│  └─ models/
│     ├─ baseline.py
│     ├─ lightgbm_model.py
│     └─ prophet_model.py
├─ scripts/
│  ├─ generate_sample_data.py
│  ├─ train_baseline.py
│  ├─ train_lightgbm.py
│  ├─ train_prophet.py
│  └─ evaluate_models.py
├─ app/
│  └─ streamlit_app.py
├─ outputs/
│  ├─ models/
│  └─ reports/
├─ requirements.txt
└─ .gitignore
```

## 빠른 시작
```bash
pip install -r requirements.txt
python scripts/generate_sample_data.py
python scripts/train_baseline.py
```

## 노트북 흐름
1. **00_data_overview**: 데이터 구조 및 EDA
2. **01_baseline_model**: Baseline 모델
3. **02_lightgbm_model**: LightGBM
4. **03_prophet_model**: Prophet
5. **04_model_comparison**: RMSE/MAE/MAPE 비교
6. **05_price_elasticity**: 가격 탄력성 분석
7. **06_dashboard_preview**: Streamlit 대시보드 미리보기

## 데이터
- Kaggle M5 Forecasting 데이터셋 사용
- 샘플 데이터는 `scripts/generate_sample_data.py`로 생성 (3 stores, 200 items)

## 모델
- Baseline (최근값 기준)
- LightGBM
- Prophet

## 평가 지표
- RMSE / MAE / MAPE

## Streamlit 대시보드
```bash
streamlit run app/streamlit_app.py
```

---

## TODO
- 하이퍼파라미터 튜닝
- 피처 엔지니어링 확장
- 모델 앙상블