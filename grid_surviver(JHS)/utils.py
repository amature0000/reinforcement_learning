import numpy as np
from state import State
MAX_COR = 33.0
MAX_DIR = 3.0
MAX_FAR = 10.0
L = 0
R = 1
U = 2
D = 3
"""
MAX_HONEYBEE = 50.0
MAX_HORNET = 22.0
MAX_KILLERBEE = 8.0
MAX_HEALTH = 100.0
"""
    
def process_features(state:State, action):
    # 16 features(px, py, pd, kx, ky, hx, hy, bx, by, kdir, hdir, bdir, hp, a1, a2, a3)
    features = []
    # coord incode
    features.append(state.px / MAX_COR)
    features.append(state.py / MAX_COR)
    features.append(state.pd / MAX_DIR)
    
    # killerbee incode
    features.append(state.kx / MAX_COR)
    features.append(state.ky / MAX_COR)

    # hornet incode
    features.append(state.hx / MAX_COR)
    features.append(state.hy / MAX_COR)
    
    # honeybee incode
    features.append(state.bx / MAX_COR)
    features.append(state.by / MAX_COR)

    # dir incode
    features.append(1 - min(state.kdir / MAX_FAR, 1))
    features.append(1 - min(state.hdir / MAX_FAR, 1))
    features.append(1 - min(state.bdir / MAX_FAR, 1))
    
    # hp incode
    features.append(state.hp) # 1 if hp > 10, else 0

    # action incode
    for i in range(3):
        features.append(1.0 if i == action else 0.0)

    return np.array(features)

def process_reward(state:State, terminated):
    # touched all honeybee
    if state.bdir == None: return 1
    # dead
    if terminated: return -1
    # touched honeybee
    # TODO: implementation
    # basic movement
    return -0.01