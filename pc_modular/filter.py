# filter.py: Apriori 연관 규칙 기반 향수 필터링 모듈 (초기 개발 버전)
# 감정(cam 인덱스)에 따라 연관된 향수 ID 목록을 반환합니다.
# 참고: 최종 버전(main_raspberrypi.py)에서는 앙상블 알고리즘으로 대체되었습니다.

import pandas as pd
import random

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

def filtering_mood(datadir, cam):
    """Apriori 연관 규칙 알고리즘으로 감정에 맞는 향수를 추천합니다.

    Args:
        datadir: 향수 데이터셋 CSV 파일 경로
        cam: 감정 인덱스 (0=nan, 1=angry, 2=fear, 3=happy, 4=neutral, 5=sad)
    Returns:
        list: 추천 향수 ID 목록
    """
    pfdf = pd.read_csv(datadir)

    mood = ['anger', 'fear', 'happy', 'neutral', 'sad']
    res = []
    moodv = []

    # 각 향수에 무작위 감정 태그 부여 (시뮬레이션용)
    for i in range(0, len(pfdf)):
        moodv.append(random.choice(mood))
    pfdf['mood'] = moodv

    # cam=0(nan)이면 전체 향수 중 추천, 아니면 해당 감정 관련 향수만 필터링
    if(cam != 0):
        genx = pfdf[pfdf.mood.apply(lambda x: mood[cam - 1] in x)]
        pf = genx.groupby(['perfume_id']).sum()
    else:
        print("nan이 나와서 아무거나 추천")
        pf = pfdf.groupby(['perfume_id']).sum()

    pf = pf.drop(['mood'], axis=1)

    # Apriori 알고리즘으로 빈발 항목 집합 추출 (최소 지지도 1%)
    frequent_itemsets = apriori(pf, min_support=0.01, use_colnames=True)
    # 연관 규칙 생성 (최소 신뢰도 75%)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.75)

    # 리프트 > 1: 우연보다 의미 있는 연관 규칙만 유지
    rules = rules[(rules['lift'] > 1)]
    # 신뢰도 내림차순 정렬
    rules = rules.sort_values(by='confidence', ascending=False)

    # 중복 결과 제거 후 상위 5개 선택
    urg = rules.drop_duplicates(['consequents'], keep='first')
    urg = urg.head()

    # 감지된 감정이 선행 조건(antecedents)에 포함된 규칙으로 필터링
    urule = urg[urg.antecedents.apply(lambda x: mood[cam - 1] in x)]
    urule = urule.drop_duplicates(['consequents'], keep='first')
    urule = urule.head()

    if len(urule) >= 1:
        urule = [list(x)[0] for x in urule['consequents']]
        print("선택하신 " + mood[cam - 1] + " 관련 추천")
        for i in urule:
            print(i)
            res.append(i)
    elif len(urg) >= 1:
        urg = [list(x)[0] for x in urg['consequents']]
        print("선택하신 " + mood[cam - 1] + " 관련 추천")
        for i in urg:
            print(i)
            res.append(i)
    else:
        print("error")

    return res
