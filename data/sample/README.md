# Sample Data 안내

이 프로젝트는 **샘플 데이터로 시작**하는 것을 권장합니다. 전체 M5 데이터는 용량이 크고 전처리/학습 시간이 길 수 있습니다. 샘플링으로 1개월 내 프로젝트 완성이 가능하며, 추후 **확장 가능성**만 언급해도 충분히 어필할 수 있습니다.

## 데이터 출처 선택지 (2가지)

### 1) Zenodo 미러 사용 (라이선스 명확 — 추천)
- Zenodo에 공개된 **M5 Forecasting Accuracy dataset**는 *CC BY 4.0* 라이선스로 명시되어 있어 재사용/재배포 조건이 명확합니다. citeturn1view0
- ZIP 파일로 한 번에 다운로드 가능합니다. citeturn1view0
- Kaggle 데이터에서 가져온 것으로 설명되어 있습니다. citeturn1view0

**Zenodo 링크 (ZIP, CC BY 4.0)**
```
https://zenodo.org/records/12636070
```

### 2) Zenodo 개별 CSV 버전 (파일 단위 다운로드)
- calendar, sales_train_validation, sales_train_evaluation, sell_prices, sample_submission 등의 **개별 CSV**가 제공됩니다. citeturn3view0
- 기록 설명에 Kaggle 대회 데이터에서 가져왔다고 되어 있습니다. citeturn3view0

**Zenodo 링크 (개별 CSV)**
```
https://zenodo.org/records/10203108
```

> 참고: 이 레코드 페이지에는 라이선스가 명시적으로 보이지 않습니다. 사용 전 Zenodo 레코드의 권리/라이선스 정보를 확인하세요.

### 3) Kaggle 공식 대회 데이터
- Kaggle 대회 데이터는 대회 페이지에서 직접 받거나 Kaggle API로 다운로드 합니다.
- Kaggle API는 `kaggle competitions download` 명령으로 대회 데이터를 받을 수 있습니다. citeturn4search1
- 예시:
```
# Kaggle API 설치 후
kaggle competitions download -c m5-forecasting-accuracy
```
  (예시 명령은 일반적인 사용 패��이며, 실제 이용 전 Kaggle 규정/Rules를 확인하세요.) citeturn4search0

## 샘플링 권장 범위
- 카테고리 1~2개 선택
- 3~5개 스토어
- 200~500개 상품

이 정도 규모면 시계열 수가 1,000~3,000개 수준으로 관리 가능하며, 1개월 내 완성이 현실적입니다.