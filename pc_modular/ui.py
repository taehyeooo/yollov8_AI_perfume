# pc_modular/ui.py
# PC 환경용 키오스크 UI (개발/테스트 버전)
# camera.py, filter.py와 함께 실행해야 합니다.
#
# 개발 환경: Python 3.8, pandas, PyQt5

import os
import pandas as pd
import random
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import cv2
from ultralytics import YOLO
import time
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # 현재 모듈 폴더를 경로에 추가
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QCheckBox
from PyQt5.QtCore import QTimer, Qt
import camera
import filter

# 실행 파일 기준 상대 경로로 모델과 데이터셋 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
modeldir = os.path.join(BASE_DIR, "best.pt")
datadir = os.path.join(BASE_DIR, "preprocessing_perfumes_dataset1.csv")
cam = 0
perfume_vect = ['향수1','향수2','향수3','향수4','향수5']
perfume_vect_bot = []

class KioskApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # 현재 화면을 나타내는 변수
        self.current_screen = "initial"

        # 초기화면 설정
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Kiosk App')
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.initial_layout()
        self.user_info_layout()
        self.emotion_analysis_layout()
        self.analysis_wait_layout()
        self.perfume_recommendation_layout()
        self.perfume_selection_layout()
        self.perfume_output_wait_layout()

        self.show_initial_screen()

    def initial_layout(self):
        self.initial_widget = QWidget()
        self.initial_layout = QVBoxLayout()
        self.initial_widget.setLayout(self.initial_layout)

        self.initial_label = QLabel('터치하여 시작하세요')
        self.initial_layout.addWidget(self.initial_label)

        self.initial_layout.setAlignment(self.initial_label, Qt.AlignCenter)
        self.initial_label.mousePressEvent = self.show_user_info_screen

    def user_info_layout(self):
        self.user_info_widget = QWidget()
        self.user_info_layout = QVBoxLayout()
        self.user_info_widget.setLayout(self.user_info_layout)
        
        self.male_checkbox = QCheckBox('남성')
        self.female_checkbox = QCheckBox('여성')
        self.next_button = QPushButton('다음')
        self.next_button.clicked.connect(self.show_emotion_analysis_screen)
        
        self.user_info_layout.addWidget(self.male_checkbox)
        self.user_info_layout.addWidget(self.female_checkbox)
        self.user_info_layout.addWidget(self.next_button)

    def emotion_analysis_layout(self):
        self.emotion_analysis_widget = QWidget()
        self.emotion_analysis_layout = QVBoxLayout()
        self.emotion_analysis_widget.setLayout(self.emotion_analysis_layout)

        self.emotion_label = QLabel('오늘 기분을 이야기 해보세요!')
        self.emotion_analysis_layout.addWidget(self.emotion_label)

    def analysis_wait_layout(self):
        self.analysis_wait_widget = QWidget()
        self.analysis_wait_layout = QVBoxLayout()
        self.analysis_wait_widget.setLayout(self.analysis_wait_layout)

        self.analysis_wait_label = QLabel('분석 중입니다. 잠시만 기다려 주세요.')
        self.analysis_wait_layout.addWidget(self.analysis_wait_label)

    def perfume_recommendation_layout(self):
        self.perfume_recommendation_widget = QWidget()
        self.perfume_recommendation_layout = QVBoxLayout()
        self.perfume_recommendation_widget.setLayout(self.perfume_recommendation_layout)

        self.perfume_recommendation_buttons = []
        for i in range(5):
            global perfume_vect
            button_name = perfume_vect[i]
            button = QPushButton(button_name)
            button.clicked.connect(lambda checked, name=button_name: self.show_perfume_selection_screen(name))
            self.perfume_recommendation_buttons.append(button)
            self.perfume_recommendation_layout.addWidget(button)

    def perfume_selection_layout(self):
        self.perfume_selection_widget = QWidget()
        self.perfume_selection_layout = QVBoxLayout()
        self.perfume_selection_widget.setLayout(self.perfume_selection_layout)

        self.selected_perfume_label = QLabel()
        self.spray_button = QPushButton('시향')
        self.close_button = QPushButton('종료')
        self.spray_button.clicked.connect(self.show_perfume_output_wait_screen)
        self.close_button.clicked.connect(self.show_initial_screen)

        self.perfume_selection_layout.addWidget(self.selected_perfume_label)
        self.perfume_selection_layout.addWidget(self.spray_button)
        self.perfume_selection_layout.addWidget(self.close_button)

    def perfume_output_wait_layout(self):
        self.perfume_output_wait_widget = QWidget()
        self.perfume_output_wait_layout = QVBoxLayout()
        self.perfume_output_wait_widget.setLayout(self.perfume_output_wait_layout)

        self.perfume_output_wait_label = QLabel('향수를 출력하고 있습니다. 잠시만 기다려 주세요.')
        self.perfume_output_wait_layout.addWidget(self.perfume_output_wait_label)

    def show_initial_screen(self):
        self.current_screen = "initial"
        self.setCentralWidget(self.initial_widget)

    def show_user_info_screen(self, event):
        self.current_screen = "user_info"
        self.setCentralWidget(self.user_info_widget)

    def show_emotion_analysis_screen(self):
        #global cam
        self.current_screen = "emotion_analysis"
        self.setCentralWidget(self.emotion_analysis_widget)
        # 15초 후에 분석 대기 화면으로 넘어가게 설정
        QTimer.singleShot(100, self.show_analysis_wait_screen)

        #return cam

    def show_analysis_wait_screen(self):
        global cam
        global perfume_vect_bot
        cam =camera.cameramood(modeldir)
        self.current_screen = "analysis_wait"
        self.setCentralWidget(self.analysis_wait_widget)
        perfume_vec_test = filter.filtering_mood(datadir, cam)
        print(perfume_vec_test)
        
        # 향수 추천 버튼의 이름으로 사용할 향수 이름들을 가져옵니다.
        perfume_recommendations = perfume_vec_test[:5]  # 상위 5개 추천 향수
        
        # 향수 추천 버튼의 이름을 업데이트합니다.
        for i, button in enumerate(self.perfume_recommendation_buttons):
            button.setText(perfume_recommendations[i])
        
        perfume_vect_bot = perfume_recommendations
        print(perfume_vect_bot)
        # 10초 후에 향수추천 화면으로 넘어가게 설정
        QTimer.singleShot(3000, self.show_perfume_recommendation_screen)
        return perfume_vect_bot

    def show_perfume_recommendation_screen(self):
        self.current_screen = "perfume_recommendation"
        self.setCentralWidget(self.perfume_recommendation_widget)

    def show_perfume_selection_screen(self, perfume_name):
        global perfume_vect_bot
        global perfume_vect
        perfume_idx = perfume_vect.index(perfume_name)
        print(perfume_idx)
        print(perfume_vect_bot[perfume_idx])
        self.current_screen = "perfume_selection"
        self.selected_perfume_label.setText(perfume_vect_bot[perfume_idx]+"를 선택하셨습니다. 향수를 직접 시향해보고 싶으시면 시향 버튼을 아니면 종료 버튼을 눌러주세요")
        self.setCentralWidget(self.perfume_selection_widget)

    def show_perfume_output_wait_screen(self):
        self.current_screen = "perfume_output_wait"
        self.setCentralWidget(self.perfume_output_wait_widget)
        QTimer.singleShot(5000, self.show_initial_screen)  # 5초 후에 초기 화면으로 이동

if __name__ == '__main__':
    app = QApplication(sys.argv)
    kiosk = KioskApp()
    kiosk.show()
    sys.exit(app.exec_())
