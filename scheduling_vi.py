# importing required libraries
import numpy as np
import random as rand
import itertools

# for num_q = 5 (2), arr_rate = 0.21 (0.4) and dep_rate = 0.5 gives a loss % of 6-7%.
# for num_q = 2, arr_rate = 0.21 and dep_rate = 0.5 gives a loss % of 5-6%.
num_q = 2
buff_size = 2
arr_rate = np.zeros(num_q)
arr_rate += 0.2
# arr_rate[1] = 0.4
dep_rate = np.zeros(num_q)
dep_rate += 0.5
# dep_rate[1] = 0.7
# dep_rate[0] = 0.3
Q = np.ones((buff_size + 1, buff_size + 1, 2, 2, 2))
Q_prev = np.ones((buff_size + 1, buff_size + 1, 2, 2, 2))
q1_enum = np.zeros(buff_size+1).astype(int)
q2_enum = np.zeros(buff_size+1).astype(int)
c1_enum = np.zeros(2).astype(int)
c2_enum = np.zeros(2).astype(int)
act_enum = np.zeros(num_q).astype(int)
q1_arr = np.zeros(2).astype(int)
q2_arr = np.zeros(2).astype(int)
q1_chan = np.zeros(2).astype(int)
q2_chan = np.zeros(2).astype(int)
disc = 0.99
iter = 0
threshold = 1e-8
for i in range(buff_size+1):
    q1_enum[i] = i
    q2_enum[i] = i
for i in range(2):
    c1_enum[i] = i
    c2_enum[i] = i
    q1_arr[i] = i
    q2_arr[i] = i
    q1_chan[i] = i
    q2_chan[i] = i
for i in range(num_q):
    act_enum[i] = i


def reward(per, arr, num, buff):
    reward_val = 0
    for i in range(num):
        if per[i] == buff:
            reward_val += -arr[i]
    return reward_val


while(1):
    iter += 1
    for perm in itertools.product(q1_enum, q2_enum, c1_enum, c2_enum, act_enum):
        if perm[perm[4] + 2] == 1:
            q_temp = max(perm[perm[4]] - 1, 0)
        else:
            q_temp = perm[perm[4]]
        perm_new = np.asarray(perm[:4])
        perm_new[perm[4]] = q_temp
        perm_new = perm_new.astype(int)
        perm_new = tuple(perm_new)
        Q[perm] = reward(perm_new, arr_rate, num_q, buff_size)
        for comb in itertools.product(q1_arr, q2_arr, q1_chan, q2_chan):
            perm_next = np.asarray(perm_new[:4])
            perm_next[0] = min(buff_size, perm_new[0] + comb[0])
            perm_next[1] = min(buff_size, perm_new[1] + comb[1])
            perm_next[2] = comb[2]
            perm_next[3] = comb[3]
            perm_next = tuple(perm_next)
            Q[perm] += disc*(comb[0]*arr_rate[0] + (1 - comb[0])*(1 - arr_rate[0]))*(comb[1]*arr_rate[1] + (1 - comb[1])*(1 - arr_rate[1]))*(comb[2]*dep_rate[0] + (1 - comb[2])*(1 - dep_rate[0]))*(comb[3]*dep_rate[1] + (1 - comb[3])*(1 - dep_rate[1]))*max(Q_prev[(perm_next)])
    err = np.max(abs(Q - Q_prev))
    print iter, err
    if err < threshold:
        break
    Q_prev = np.copy(Q)
# np.savetxt('Q_matrix.txt', Q.flatten(), header=str(Q.shape))
