def process_obs(self, obs):
    types = ['E', 'W', 'AL', 'AR', 'AU', 'AD', 'B', 'H', 'K']
    state = np.zeros((9, 34, 34))  # 9개의 채널로 초기화

    self.bee_cnt = 0
    self.hornet_cnt = 0
    self.killer_cnt = 0
    py, px = 0, 0

    for y in range(34):
        for x in range(34):
            # 에이전트 위치 저장
            if obs["grid"][y][x][0] == "A":
                py, px = y, x
            obj_type = obs["grid"][y][x]
            if obj_type in types:
                channel_idx = types.index(obj_type)
                state[channel_idx, y, x] = 1
            if obj_type == "B":
                self.bee_cnt += 1
                self.bee_distance += abs(py - y) + abs(px - x)
            elif obj_type == "K":
                self.killer_cnt += 1

    self.hp = obs["hit_points"]

    return state, self.hp, self.get_distance(py, px, obs["grid"])


def get_reward(self, obs, obs_, action):
    reward = -0.01

    if self.killer_cnt != 8 or self.hp <= 0:
        reward = -5.0
        self.done = True
        return reward

    reward += self.hp_reward[self.hp // 10]
    self.hp_reward[self.hp // 10] = 0.0

    if self.action_cnt >= 1200:
        self.done = True

    px = -1; py = -1; px2 = -1; px2 = -1
    for y in range(34):
        for x in range(34):
            if obs["grid"][y][x][0] == "A":
                py = y; px = x
            if obs_["grid"][y][x][0] == "A":
                py2 = y; px2 = x

    # min_dis = self.get_distance(py, px, obs["grid"])
    # min_dis2 = self.get_distance(py2, px2, obs_["grid"])

    # if min_dis > min_dis2:
    #     reward += -0.5

    if action == 2 and obs_["grid"][py][px][0] == "A":
        reward += -0.5

    if self.bee_reward[self.bee_cnt]:
        reward = 0.5
        self.bee_reward[self.bee_cnt] = 0

    return reward