# importing required libraries
import numpy as np
import random as rand
import re
import keras
with open('Q_matrix.txt') as f:
    shape = tuple(int(num) for num in re.findall(r'\d+', f.readline()))
Q = np.loadtxt('Q_matrix.txt').reshape(shape)
model = keras.models.load_model('temp_diff_re.h5')
# model1 = keras.models.load_model('my_model.h5')

# for num_q = 5 (2), arr_rate = 0.21 (0.4) and dep_rate = 0.5 gives a loss % of 6-7%.
# for num_q = 2, arr_rate = 0.21 and dep_rate = 0.5 gives a loss % of 5-6%.
num_q = 5
buff_size = 2
arr_rate = np.zeros(num_q)
arr_rate += 0.2
# arr_rate[1] = 0.4
dep_rate = np.zeros(num_q)
dep_rate += 0.5
# dep_rate[0] = 0.3
# dep_rate[1] = 0.7
num_iter = 200
time_steps = 1000
ratio = np.zeros(num_iter)
ratio1 = np.zeros(num_iter)
ratio2 = np.zeros(num_iter)

def lcqf(len_q, channel, num_q):
    action = -1
    tran_ind = []
    for index in range(num_q):
        if channel[index] == 1:
            tran_ind.append(index)
    if not tran_ind:
        return action
    rand.shuffle(tran_ind)
    max_len = len_q[tran_ind[0]]
    action = tran_ind[0]
    for index in tran_ind:
        if len_q[index] > max_len:
            action = index
    return action


def q_proj(len_q, buff_size, num_q):
    ret_q = np.zeros(num_q)
    lost_count = 0
    for i in range(num_q):
        if len_q[i] > buff_size:
            ret_q[i] = buff_size
            lost_count += 1
        else:
            ret_q[i] = len_q[i]
    return ret_q, lost_count


def random_open_action(channel, num_q):
    action = -1
    tran_ind = []
    for index in range(num_q):
        if channel[index] == 1:
            tran_ind.append(index)
    if not tran_ind:
        return action
    return rand.choice(tran_ind)


def random_action(num_q):
    return rand.randint(0, num_q - 1)


def DQN(state, num_q, model):
    action = np.argmax(model.predict(np.reshape(state, [1, 2*num_q]))[0])
    return action


def Q_value_iter(len_q, channel, num_q, Q):
    action = -1
    tran_ind = []
    for index in range(num_q):
        if channel[index] == 1:
            tran_ind.append(index)
    if not tran_ind:
        return action
    state = np.zeros(5)
    state[:2] = np.copy(len_q)
    state[2:4] = np.copy(channel)
    state = state.astype(int)
    max_value = np.min(Q)
    rand.shuffle(tran_ind)
    for index in tran_ind:
        state[4] = index
        state_tup = tuple(state)
        if Q[(state_tup)] >= max_value:
            max_value = Q[(state_tup)]
            action = index
    return action


for j in range(num_iter):
    np.random.seed()
    len_q = np.zeros(num_q)
    channel = np.zeros(num_q)
    total_arr = 0.0
    total_lost = 0.0
    len_q1 = np.zeros(num_q)
    total_lost1 = 0.0
    # len_q2 = np.zeros(num_q)
    # total_lost2 = 0.0
    for i in range(time_steps):
        action = lcqf(len_q, channel, num_q)
        state = np.concatenate((len_q1, channel))
        # state1 = np.concatenate((len_q2, channel))
        action1 = DQN(state, num_q, model)
        # action2 = DQN(state1, num_q, model1)
        # action1 = random_action(num_q)
        if action != -1 and channel[action] != 0:
            len_q[action] = max(0, len_q[action]-1)
        if action1 != -1 and channel[action1] != 0:
            len_q1[action1] = max(0, len_q1[action1]-1)
        # if action2 != -1 and channel[action2] != 0:
        #     len_q2[action2] = max(0, len_q2[action2]-1)
        arrivals = np.random.binomial(1, arr_rate)
        len_q += arrivals
        len_q1 += arrivals
        # len_q2 += arrivals
        channel = np.random.binomial(1, dep_rate)
        len_q, lost_count = q_proj(len_q, buff_size, num_q)
        len_q1, lost_count1 = q_proj(len_q1, buff_size, num_q)
        # len_q2, lost_count2 = q_proj(len_q2, buff_size, num_q)
        total_arr += sum(arrivals)
        total_lost += lost_count
        total_lost1 += lost_count1
        # total_lost2 += lost_count2
    ratio[j] = total_lost/total_arr
    ratio1[j] = total_lost1/total_arr
    # ratio2[j] = total_lost2/total_arr
loss_ratio = np.mean(ratio)
std_dev = np.std(ratio)
loss_ratio1 = np.mean(ratio1)
std_dev1 = np.std(ratio1)
# loss_ratio2 = np.mean(ratio2)
# std_dev2 = np.std(ratio2)

