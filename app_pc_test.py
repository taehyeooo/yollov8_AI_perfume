#라즈베리파이 버전
# 버튼으로 선택, 다 잘 되고 마지막 결과 화면도 잘 나옴 하지만 얼굴인식이 약함

import sys
import cv2
from ultralytics import YOLO
import pandas as pd
import numpy as np
from PyQt5 import QtWidgets, QtCore
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import time

# 모델 경로 설정 (YOLOv8)
model_path = "/home/jiwon/Desktop/perfume/runs/detect/train3/weights/best.pt"
model = YOLO(model_path)

# 향수 데이터 로드 및 전처리
def load_and_prepare_data():
    data = pd.read_csv("/home/jiwon/perfume/preprocessing_perfumes_dataset1.csv ", encoding='utf-8')
    
    le_dict = {}
    for column in ['department', 'scents', 'concentration']:
        le_dict[column] = LabelEncoder()
        data[column] = le_dict[column].fit_transform(data[column])

    data['base_note_split'] = data['base_note'].str.split(', ')
    data_exploded = data.explode('base_note_split').reset_index(drop=True)
    le_dict['base_note_split'] = LabelEncoder()
    data_exploded['base_note_split'] = le_dict['base_note_split'].fit_transform(data_exploded['base_note_split'])

    data_exploded = data_exploded.drop_duplicates(subset=['name', 'department', 'scents', 'base_note_split', 'concentration'])
    
    return data_exploded, le_dict

# 감정 인식 (카메라 사용)
def detect_mood():
    cap = cv2.VideoCapture(0)
    mood_counts = {'neutral': 0, 'happy': 0, 'sad': 0, 'angry': 0, 'nervous': 0}  # 기분별 카운트 초기화
    start_time = time.time()  # 시작 시간 기록

    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to capture image")
            break
        
        results = model(frame)
        if results and results[0].boxes:
            detected_class = results[0].names[int(results[0].boxes[0].cls)]
            mood_counts[detected_class] = mood_counts.get(detected_class, 0) + 1  # 감지된 기분 카운트 증가
        
        cv2.putText(frame, f"Detecting...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Mood Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == 27 or time.time() - start_time > 10:
            break

    cap.release()
    cv2.destroyAllWindows()

    final_mood = max(mood_counts, key=mood_counts.get)
    return final_mood

# 추천 시스템 로직
def recommend_perfume(gender, preferred_scent, mood, situation, data_exploded, le_dict):
    gender_encoded = le_dict['department'].transform([gender])[0]
    scent_encoded = le_dict['scents'].transform([preferred_scent])[0]
    
    mood_scents = {
        'happy': ['Floral', 'Fruity', 'Citrus'],
        'sad': ['Woody', 'Vanilla', 'Musk'],
        'angry': ['Spicy', 'Fresh', 'Citrus'],
        'nervous': ['Lavender', 'Fresh', 'Floral'],
        'neutral': ['Woody', 'Fresh', 'Musk']
    }.get(mood, [])

    mood_encoded = [le_dict['scents'].transform([s])[0] for s in mood_scents if s in le_dict['scents'].classes_]
    mood_avg = np.mean(mood_encoded) if mood_encoded else 0

    situation_concentration_mapping = {
        'everyday': ['EDT', 'EDC'],
        'special occasion': ['EDP', 'Parfum']
    }
    
    situation_encoded = [le_dict['concentration'].transform([c])[0] for c in situation_concentration_mapping[situation]]
    situation_avg = np.mean(situation_encoded) if situation_encoded else 0

    user_vector = np.array([gender_encoded, scent_encoded, mood_avg, situation_avg])

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    X = data_exploded[['department', 'scents', 'base_note_split', 'concentration']]
    y = data_exploded['item_rating']
    rf_model.fit(X, y)

    similarity_scores = []
    for name, group in data_exploded.groupby('name'):
        perfume_vector = np.array([group['department'].iloc[0], group['scents'].iloc[0], group['base_note_split'].mean(), group['concentration'].iloc[0]])
        similarity = cosine_similarity(user_vector.reshape(1, -1), perfume_vector.reshape(1, -1))[0][0]

        predicted_rating = rf_model.predict(perfume_vector.reshape(1, -1))[0]
        final_score = similarity * 0.7 + predicted_rating * 0.3

        if group['item_rating'].iloc[0] >= 3.5:
            similarity_scores.append((name, final_score, similarity, group.iloc[0]))

    recommended_perfumes = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[:3]

    return recommended_perfumes

# UI를 통한 사용자 입력 처리
class PerfumeRecommendationApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.data_exploded, self.le_dict = load_and_prepare_data()
        self.selected_gender = None
        self.selected_scent = None
        self.selected_situation = None

    def initUI(self):
        self.stackedWidget = QtWidgets.QStackedWidget(self)

        # 페이지 1: 성별 선택
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
        recommendations = recommend_perfume(self.selected_gender, self.selected_scent, detected_mood, self.selected_situation, self.data_exploded, self.le_dict)

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
