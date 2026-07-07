# 🏆 2024 한국정보기술학회 추계종합학술대회 우수논문상 수상

# 표정 인식 기반 향수 추천 AI
### 앙상블 기법(Random Forest + Cosine Similarity)을 활용한 감정 맞춤형 향수 추천 시스템

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv8-00FFFF?style=flat)
![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest%20%7C%20CosineSimilarity-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![License](https://img.shields.io/badge/Dataset_License-CC_BY_4.0-green?style=flat)

---

## Overview

본 프로젝트는 사용자의 **실시간 얼굴 표정을 분석**하여 현재 감정에 맞는 향수를 자동으로 추천하는 AI 시스템입니다.

웹캠이 10초간 얼굴 표정을 분석하고, 감지된 감정과 사용자가 입력한 성별·선호 향·상황 정보를 조합하여 **앙상블 추천 알고리즘**이 1,003개의 향수 데이터셋 중 최적의 향수 TOP 3를 추천합니다.

### 인식 대상 표정 클래스 (5종)

| 클래스 | 설명 | 추천 향 카테고리 |
|--------|------|----------------|
| `happy` | 기쁨 | Floral, Fruity, Citrus |
| `sad` | 슬픔 | Woody, Vanilla, Musk |
| `angry` | 분노 | Spicy, Fresh, Citrus |
| `nervous` | 긴장 | Lavender, Fresh, Floral |
| `neutral` | 무표정 | Woody, Fresh, Musk |

---

## Research Background

단순한 키워드 매칭 방식의 추천 시스템은 사용자의 현재 감정 상태를 반영하지 못한다는 한계가 있었습니다. 이를 극복하기 위해 두 가지 알고리즘을 결합한 앙상블 방식을 설계하였습니다.

- **Cosine Similarity**: 사용자의 감정·선호도를 벡터로 표현하고, 향수 특성 벡터와의 방향 유사도를 계산합니다.
- **Random Forest**: 향수의 성별·향·베이스 노트·농도 특성을 학습하여 예상 평점을 예측합니다.

사용자 입력값(성별, 선호 향, 상황, 감정)은 `LabelEncoder`로 숫자 벡터로 변환되며, 향수 데이터의 `base_note`는 복수값을 `explode`하여 각각 인코딩합니다.

---

## Algorithm Design

### 앙상블 추천 점수 계산

| 알고리즘 | 가중치 | 역할 |
|---------|--------|------|
| Cosine Similarity | **70%** | 사용자 감정 벡터 ↔ 향수 특성 벡터 유사도 |
| Random Forest | **30%** | 향수 특성 기반 평점 예측 |

```
최종 점수 = Cosine Similarity × 0.7 + RF 예측 평점 × 0.3
```

> 평점 3.5 이상인 향수만 후보에 포함하며, 최종 점수 기준 상위 3개를 추천합니다.

### 사용자 벡터 구성

```
user_vector = [성별(인코딩), 선호향(인코딩), 감정향 평균값, 상황별 농도 평균값]
```

### 상황 → 농도 매핑

| 상황 | 추천 농도 |
|------|---------|
| Everyday (일상) | EDT, EDC |
| Special Occasion (특별한 날) | EDP, Parfum |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          사용자 입력                             │
│         성별 / 선호 향(9종) / 상황(일상·특별)                    │
└───────────────────────────┬─────────────────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
┌───────────────────────┐       ┌───────────────────────────────┐
│    감정 인식 모듈       │       │        데이터 전처리           │
│                       │       │                               │
│  WebCam (OpenCV)      │       │  LabelEncoder                 │
│       ↓               │       │  - 성별 / 향 / 농도 → 숫자     │
│  YOLOv8 (Fine-tuned)  │       │  base_note explode 처리       │
│       ↓               │       │                               │
│  감정 클래스 반환       │       │  향수 데이터셋 (1,003개)       │
│  (happy/sad/angry      │       │                               │
│   nervous/neutral)    │       └───────────────┬───────────────┘
└───────────┬───────────┘                       │
            │                                   │
            └──────────────┬────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     사용자 벡터 생성                              │
│      [성별, 선호향, 감정향 평균값, 상황별 농도 평균값]             │
└───────────────────────────┬─────────────────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
┌───────────────────────┐       ┌───────────────────────────────┐
│   Cosine Similarity   │       │       Random Forest           │
│       (가중치 70%)     │       │         (가중치 30%)           │
│                       │       │                               │
│  사용자 벡터 ↔ 향수    │       │  향수 특성으로 학습 후         │
│  벡터 유사도 계산      │       │  예상 평점 예측               │
└───────────┬───────────┘       └───────────────┬───────────────┘
            │                                   │
            └──────────────┬────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│               최종 점수 = 유사도 × 0.7 + 예측평점 × 0.3         │
│                   평점 3.5 이상 향수만 후보 포함                  │
└───────────────────────────┬─────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    추천 결과 TOP 3 출력                           │
│              브랜드 / 향수명 / 유사도 점수 / 최종 점수            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| 분류 | 기술 |
|------|------|
| Language | Python 3.8+ |
| Computer Vision | OpenCV, Ultralytics YOLOv8 |
| Machine Learning | scikit-learn (Random Forest, Cosine Similarity, LabelEncoder) |
| Data Processing | Pandas, NumPy |
| Dataset | 향수 데이터셋 1,003개 (`preprocessing_perfumes_dataset1.csv`) |

---

## Project Structure

```
2ndPerfume_recommend/
├── main_raspberrypi.py          # [Main] 메인 실행 파일
├── app_pc_test.py               # PC 환경 테스트 버전
├── preprocessing_perfumes_dataset1.csv  # 향수 데이터셋 (1,003개)
├── best.pt                      # YOLOv8 커스텀 감정 인식 모델 (별도 배치 필요)
├── pc_modular/
│   ├── camera.py                # YOLOv8 감정 인식 모듈
│   ├── filter.py                # Apriori 기반 향수 필터 (프로토타입)
│   └── ui.py                    # UI 모듈 (PC 개발 버전)
└── research/
    ├── face_train.ipynb         # YOLOv8 파인튜닝 및 학습
    ├── face_image.ipynb         # 데이터셋 EDA
    └── model_comparison.ipynb  # 추천 알고리즘 비교 실험
```

---

## Installation

```bash
# 1. 저장소 클론
git clone https://github.com/taehyeooo/2ndPerfume_recommend.git
cd 2ndPerfume_recommend

# 2. 의존성 설치
pip install ultralytics opencv-python pandas numpy scikit-learn mlxtend
```

---

## Usage

### 감정 인식 테스트

```python
from pc_modular.camera import cameramood

# YOLOv8 모델 경로 지정 후 10초간 웹캠 감정 인식
# 반환값: 0=nan, 1=angry, 2=fear, 3=happy, 4=neutral, 5=sad
result = cameramood("best.pt")
print(result)
```

### 향수 추천 테스트

```python
from main_raspberrypi import load_and_prepare_data, recommend_perfume

data_exploded, le_dict = load_and_prepare_data()

recommendations = recommend_perfume(
    gender="Women",
    preferred_scent="Floral",
    mood="happy",
    situation="everyday",
    data_exploded=data_exploded,
    le_dict=le_dict
)

for perfume in recommendations:
    print(f"{perfume[3]['brand']} - {perfume[3]['name']} | Score: {perfume[1]:.4f}")
```

---

## Dataset

- **파일**: `preprocessing_perfumes_dataset1.csv`
- **크기**: 1,003개 향수
- **주요 컬럼**: `brand`, `name`, `new_price`, `concentration`, `department`, `scents`, `base_note`, `item_rating`
- **전처리**: LabelEncoder 인코딩, base_note explode 처리, 중복 제거
