# camera.py: YOLOv8 기반 실시간 감정 인식 모듈
# 웹캠으로 10초간 촬영하여 가장 많이 감지된 감정의 인덱스를 반환합니다.
# 감정 인덱스: 0=nan, 1=angry, 2=fear, 3=happy, 4=neutral, 5=sad

import cv2
from ultralytics import YOLO
import time

# 감정별 감지 횟수 카운터 (전역 변수)
angry, fear, happy, neutral, sad, nan = 0, 0, 0, 0, 0, 0

def count_mood(x):
    """YOLOv8이 감지한 감정 클래스 인덱스(x)에 해당하는 카운터를 1 증가시킵니다.
    클래스 매핑: 1=angry, 2=fear, 3=happy, 4=neutral, 5=sad, 0=인식 불가(nan)
    """
    global angry, fear, happy, neutral, sad, nan
    if(x==1):
        angry=angry+1
        return angry
    elif(x==2):
        fear=fear+1
        return fear
    elif(x==3):
        happy=happy+1
        return happy
    elif(x==4):
        neutral=neutral+1
        return neutral
    elif(x==5):
        sad=sad+1
        return sad
    else:
        nan=nan+1
        return nan

def print_results(res):
    """10초 감지 결과 전체 카운트를 출력합니다."""
    print("======result======")
    print("nan :",res[0])
    print("angry :",res[1])
    print("fear :",res[2])
    print("happy :",res[3])
    print("neutral :",res[4])
    print("sad :",res[5])
    print("======result======")

def print_output(x):
    """최종 감정 인덱스를 감정 이름 문자열로 출력합니다."""
    out = ['nan','anger', 'fear', 'happy', 'neutral', 'sad']
    if 0 <= x <= 5:
        print(out[x])
    else:
        print('error')


def cameramood(modeldir):
    """웹캠으로 10초간 얼굴 표정을 분석하여 가장 많이 감지된 감정의 인덱스를 반환합니다.

    Args:
        modeldir: YOLOv8 커스텀 감정 인식 모델(.pt 파일) 경로
    Returns:
        int: 감정 인덱스 (0=nan, 1=angry, 2=fear, 3=happy, 4=neutral, 5=sad)
    """
    # YOLOv8 커스텀 감정 인식 모델 로드
    model = YOLO(modeldir)

    # 웹캠(0번 장치) 열기
    cap = cv2.VideoCapture(0)

    if cap.isOpened():
        start = time.time()

    # 매 프레임마다 감정 감지 및 카운트 (10초 동안)
    while cap.isOpened():
        now = time.time()
        success, frame = cap.read()

        if success:
            # YOLOv8 추론 실행
            results = model(frame)
            if results[0].boxes:
                # 감지된 클래스 인덱스를 1-base(1~5)로 변환
                clss = results[0].boxes.cls.cpu().detach().numpy().tolist()[0]
                clss = int(clss+1)
            else:
                clss = int(0)  # 얼굴 미감지 시 nan(0) 처리

            print(clss)
            count_mood(clss)

            # 10초 경과 시 종료
            if now-start >= 10:
                break
        else:
            break

    # 10초간 감지 결과 집계
    res = [nan, angry, fear, happy, neutral, sad]
    output = res.index(max(res))  # 가장 많이 감지된 감정의 인덱스

    cap.release()
    cv2.destroyAllWindows()
    print_results(res)
    print("======output======")
    print("index :", output)
    print_output(output)
    print("======output======")
    return output



