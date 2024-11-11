import numpy as np

MAX_COR = 33
MAX_HONEYBEE = 50
MAX_HORNET = 22
MAX_KILLERBEE = 8
MAX_HEALTH = 150

# make state
def process_obs():
    pass

def process_features(self, state, action):
    features = []
    # coord incode
    features.append(state.px / MAX_COR)
    features.append(state.py / MAX_COR)
    features.append(state.pd / MAX_COR)
    
    # hp incode
    features.append(1 if state.hp > 10 else 0)
    
    # killerbee incode TODO: nearest killerbee만 고르기
    for i in range(MAX_KILLERBEE):
        features.append(state.killerbee[i].x / MAX_COR)
        features.append(state.killerbee[i].y / MAX_COR)

    # hornet incode TODO: nearest hornet만 고르기
    for i in range(MAX_HORNET):
        features.append(state.hornet[i].x / MAX_COR)
        features.append(state.hornet[i].y / MAX_COR)
    
    # honeybee incode TODO: nearest honeybee만 고르기
    for i in range(MAX_HONEYBEE):
        features.append(state.honeybee[i].x / MAX_COR)
        features.append(state.honeybee[i].y / MAX_COR)
    
    # action incode
    for i in range(3):
        features.append(1.0 if i == action else 0.0)
    
    return np.array(features)