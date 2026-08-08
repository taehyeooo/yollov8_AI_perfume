# 최종 라즈베리파이 배포 버전 코드
# 실행 전 best.pt(YOLOv8 감정 인식 모델)와 CSV 데이터셋이 같은 폴더에 있어야 합니다.
import sys
import os
import cv2
from ultralytics import YOLO
import pandas as pd
import numpy as np
from PyQt5 import QtWidgets, QtCore
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import time

# YOLOv8 감정 인식 모델 로드 (실행 파일과 같은 디렉토리의 best.pt 사용)
model_path = os.path.join(os.getcwd(), "best.pt")
model = YOLO(model_path)


# 향수 데이터 로드 및 전처리
def load_and_prepare_data():
    # 실행 파일과 같은 디렉토리의 CSV 데이터셋 로드
    data = pd.read_csv(os.path.join(os.getcwd(), "preprocessing_perfumes_dataset1.csv"), encoding='latin1')
    
    # 범주형 컬럼(성별/향/농도)을 숫자로 인코딩 (LabelEncoder)
    le_dict = {}
    for column in ['department', 'scents', 'concentration']:
        le_dict[column] = LabelEncoder()
        data[column] = le_dict[column].fit_transform(data[column])

    # base_note는 쉼표로 구분된 복수값 → 행 분리(explode)하여 각각 인코딩
    data['base_note_split'] = data['base_note'].str.split(', ')
    data_exploded = data.explode('base_note_split').reset_index(drop=True)
    le_dict['base_note_split'] = LabelEncoder()
    data_exploded['base_note_split'] = le_dict['base_note_split'].fit_transform(data_exploded['base_note_split'])

    # 같은 향수의 중복 행 제거
    data_exploded = data_exploded.drop_duplicates(subset=['name', 'department', 'scents', 'base_note_split', 'concentration'])

    return data_exploded, le_dict

# Random Forest 모델 학습 및 검증
# 이전에는 recommend_perfume() 안에서 요청이 들어올 때마다 매번 처음부터 재학습했고,
# 학습/검증 데이터 분리 없이 전체 데이터로만 fit()하고 있어 모델이 실제로 새로운 향수에도
# 일반화되는지 확인할 방법이 없었음. 여기서 한 번만 학습해 재사용하고, held-out 검증도 함께 수행함.
def train_recommendation_model(data_exploded):
    X = data_exploded[['department', 'scents', 'base_note_split', 'concentration']]
    y = data_exploded['item_rating']

    # 학습/검증 분리로 일반화 성능(R^2)을 먼저 확인
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    eval_model = RandomForestRegressor(n_estimators=100, random_state=42)
    eval_model.fit(X_train, y_train)
    val_r2 = eval_model.score(X_val, y_val)
    print(f"[Random Forest 검증] held-out R^2 = {val_r2:.3f} (검증 샘플 {len(X_val)}개)")

    # 검증을 마친 뒤, 실제 추천에 사용할 모델은 전체 데이터로 다시 학습해 데이터를 최대한 활용
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    return rf_model

# 감정 인식: 웹캠으로 10초간 얼굴 표정 분석 후 가장 많이 감지된 감정 반환
def detect_mood():
    cap = cv2.VideoCapture(0)
    # 키는 YOLOv8 모델이 실제로 반환하는 클래스명(anger/fear/happy/neutral/sad)과 정확히 일치해야 함
    mood_counts = {'neutral': 0, 'happy': 0, 'sad': 0, 'anger': 0, 'fear': 0}
    start_time = time.time()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture image")
            break

        # YOLOv8으로 현재 프레임의 감정 클래스 감지
        results = model(frame)
        if results and results[0].boxes:
            detected_class = results[0].names[int(results[0].boxes[0].cls)]
            mood_counts[detected_class] = mood_counts.get(detected_class, 0) + 1

        cv2.putText(frame, f"Detecting...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Mood Detection', frame)

        # ESC 키 또는 10초 경과 시 종료
        if cv2.waitKey(1) & 0xFF == 27 or time.time() - start_time > 10:
            break

    cap.release()
    cv2.destroyAllWindows()

    # 10초 동안 가장 많이 감지된 감정을 최종 결과로 반환
    final_mood = max(mood_counts, key=mood_counts.get)
    return final_mood

# 앙상블 향수 추천: Cosine Similarity(70%) + Random Forest(30%) 조합
def recommend_perfume(gender, preferred_scent, mood, situation, data_exploded, le_dict, rf_model):
    # 사용자 입력값을 인코딩된 숫자로 변환 (데이터셋에 없는 값이 들어와도 크래시하지 않도록 평균값으로 대체)
    if gender in le_dict['department'].classes_:
        gender_encoded = le_dict['department'].transform([gender])[0]
    else:
        gender_encoded = le_dict['department'].transform(le_dict['department'].classes_).mean()

    if preferred_scent in le_dict['scents'].classes_:
        scent_encoded = le_dict['scents'].transform([preferred_scent])[0]
    else:
        scent_encoded = le_dict['scents'].transform(le_dict['scents'].classes_).mean()

    # 감정별 어울리는 향 카테고리 매핑
    mood_scents = {
        'happy': ['Floral', 'Fruity', 'Citrus'],
        'sad': ['Woody', 'Vanilla', 'Musk'],
        'anger': ['Spicy', 'Fresh', 'Citrus'],
        'fear': ['Lavender', 'Fresh', 'Floral'],
        'neutral': ['Woody', 'Fresh', 'Musk']
    }.get(mood, [])

    # 감정에 해당하는 향들의 인코딩 평균값 계산
    mood_encoded = [le_dict['scents'].transform([s])[0] for s in mood_scents if s in le_dict['scents'].classes_]
    mood_avg = np.mean(mood_encoded) if mood_encoded else 0

    # 상황별 향수 농도 매핑 (일상: 가벼운 향 / 특별한 날: 진한 향)
    # 'PDT'는 데이터셋에 실제로 존재하는 고농도 표기(전에는 존재하지 않는 'Parfum'으로 되어 있었음)
    situation_concentration_mapping = {
        'everyday': ['EDT', 'EDC'],
        'special occasion': ['EDP', 'PDT']
    }

    # 데이터셋에 없는 값은 건너뛰어 안전하게 처리
    situation_encoded = [le_dict['concentration'].transform([c])[0] for c in situation_concentration_mapping[situation] if c in le_dict['concentration'].classes_]
    situation_avg = np.mean(situation_encoded) if situation_encoded else 0

    # 사용자 선호 벡터 생성 [성별, 선호향, 감정향, 상황농도]
    user_vector = np.array([gender_encoded, scent_encoded, mood_avg, situation_avg])

    # Random Forest 모델은 train_recommendation_model()에서 미리 학습해 전달받음 (요청마다 재학습하지 않음)
    similarity_scores = []
    for name, group in data_exploded.groupby('name'):
        perfume_vector = np.array([group['department'].iloc[0], group['scents'].iloc[0], group['base_note_split'].mean(), group['concentration'].iloc[0]])

        # Cosine Similarity: 사용자 벡터와 향수 벡터의 방향 유사도 계산
        similarity = cosine_similarity(user_vector.reshape(1, -1), perfume_vector.reshape(1, -1))[0][0]

        # 최종 점수 = Cosine Similarity(70%) + RF 예측 평점(30%)
        predicted_rating = rf_model.predict(perfume_vector.reshape(1, -1))[0]
        final_score = similarity * 0.7 + predicted_rating * 0.3

        # 평점 3.5 이상인 향수만 후보에 포함
        if group['item_rating'].iloc[0] >= 3.5:
            similarity_scores.append((name, final_score, similarity, group.iloc[0]))

    # 최종 점수 내림차순 정렬 후 TOP 3 반환
    recommended_perfumes = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[:3]

    return recommended_perfumes

# PyQt5 키오스크 앱: 4단계 스택 위젯으로 구성된 터치 UI
# 흐름: 성별 선택 → 선호 향 선택 → 상황 선택 → [감정 인식] → 추천 결과
class PerfumeRecommendationApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.data_exploded, self.le_dict = load_and_prepare_data()
        self.rf_model = train_recommendation_model(self.data_exploded)
        self.selected_gender = None
        self.selected_scent = None
        self.selected_situation = None

    def initUI(self):
        self.stackedWidget = QtWidgets.QStackedWidget(self)

        # 페이지 1: 성별 선택 (Men / Women)
        self.page1 = QtWidgets.QWidget()
        self.gender_layout = QtWidgets.QVBoxLayout(self.page1)
        self.gender_label = QtWidgets.QLabel("Select Gender:", self.page1)
        self.men_button = QtWidgets.QPushButton("Men", self.page1)
        self.women_button = QtWidgets.QPushButton("Women", self.page1)
        self.men_button.clicked.connect(lambda: self.set_gender('Men'))
        self.women_button.clicked.connect(lambda: self.set_gender('Women'))
        self.gender_layout.addWidget(self.gender_label)
        self.gender_layout.addWidget(self.men_button)
        self.gender_layout.addWidget(self.women_button)

        # 페이지 2: 선호 향 선택
        self.page2 = QtWidgets.QWidget()
        self.scent_layout = QtWidgets.QVBoxLayout(self.page2)
        self.scent_label = QtWidgets.QLabel("Preferred Scent:", self.page2)
        scents = ["Floral", "Woody", "Fresh", "Spicy", "Citrus", "Lavender", "Musk", "Vanilla", "Fruity"]
        for scent in scents:
            button = QtWidgets.QPushButton(scent, self.page2)
            button.clicked.connect(lambda _, s=scent: self.set_scent(s))
            self.scent_layout.addWidget(button)
        self.scent_layout.addWidget(self.scent_label)

        # 페이지 3: 상황 선택 (버튼으로 수정)
        self.page3 = QtWidgets.QWidget()
        self.situation_layout = QtWidgets.QVBoxLayout(self.page3)
        self.situation_label = QtWidgets.QLabel("Select Situation:", self.page3)
        everyday_button = QtWidgets.QPushButton("Everyday", self.page3)
        special_button = QtWidgets.QPushButton("Special Occasion", self.page3)
        everyday_button.clicked.connect(lambda: self.set_situation('everyday'))
        special_button.clicked.connect(lambda: self.set_situation('special occasion'))
        self.situation_layout.addWidget(self.situation_label)
        self.situation_layout.addWidget(everyday_button)
        self.situation_layout.addWidget(special_button)

        # 페이지 4: 추천 결과 표시
        self.page4 = QtWidgets.QWidget()
        self.result_layout = QtWidgets.QVBoxLayout(self.page4)
        self.result_label = QtWidgets.QLabel("Recommended Perfumes:", self.page4)
        self.result_layout.addWidget(self.result_label)

        # 페이지 추가
        self.stackedWidget.addWidget(self.page1)
        self.stackedWidget.addWidget(self.page2)
        self.stackedWidget.addWidget(self.page3)
        self.stackedWidget.addWidget(self.page4)

        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.main_layout.addWidget(self.stackedWidget)
        self.setLayout(self.main_layout)
        self.setWindowTitle('Perfume Recommendation System')

    def set_gender(self, gender):
        self.selected_gender = gender
        self.goto_page2()  # 성별 선택 후 다음 페이지로 이동

    def set_scent(self, scent):
        self.selected_scent = scent
        self.goto_page3()  # 선호 향기 선택 후 다음 페이지로 이동

    def set_situation(self, situation):
        self.selected_situation = situation
        self.detect_mood_and_recommend()  # 상황 선택 후 표정 인식 및 추천으로 이동

    def goto_page2(self):
        self.stackedWidget.setCurrentIndex(1)  # 페이지 2로 이동

    def goto_page3(self):
        self.stackedWidget.setCurrentIndex(2)  # 페이지 3으로 이동

    def detect_mood_and_recommend(self):
        # 기분 감지 및 결과 페이지로 이동
        detected_mood = detect_mood()

        # 향수 추천
        recommendations = recommend_perfume(self.selected_gender, self.selected_scent, detected_mood, self.selected_situation, self.data_exploded, self.le_dict, self.rf_model)

        # 추천 결과 표시
        result_text = f"Detected Mood: {detected_mood}\n\nRecommended Perfumes:\n"
        for perfume in recommendations:
            result_text += f"Brand: {perfume[3]['brand']}\n"
            result_text += f"Name: {perfume[3]['name']}\n"
            result_text += f"Scent: {self.le_dict['scents'].inverse_transform([perfume[3]['scents']])[0]}\n"
            result_text += f"Gender: {self.le_dict['department'].inverse_transform([perfume[3]['department']])[0]}\n"
            result_text += f"Base Note: {perfume[3]['base_note']}\n"
            result_text += f"New Price: {perfume[3]['new_price']}\n"
            result_text += f"Calculated Similarity: {perfume[2]:.4f}\n"
            result_text += f"Final Score: {perfume[1]:.4f}\n"
            result_text += "-" * 30 + "\n"

        self.result_label.setText(result_text)
        self.stackedWidget.setCurrentIndex(3)  # 페이지 4로 이동

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ex = PerfumeRecommendationApp()
    ex.show()
    sys.exit(app.exec_())
