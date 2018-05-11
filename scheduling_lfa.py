# importing required libraries
import numpy as np
import random as rand
import time
import math

# starting time
start_time = time.time()
start_time1 = start_time

# for num_q = 5 (2), arr_rate = 0.21 (0.4) and dep_rate = 0.5 gives a loss % of 6-7%.
num_q = 2
buff_size = 5
arr_rate = np.zeros(num_q)
arr_rate += 0.4
dep_rate = np.zeros(num_q)
dep_rate += 0.5
num_episodes = 5000
time_steps = 400
ratio = np.zeros(num_episodes)
weights = np.zeros(2)
discount = 0.99;
epsilon_min = 0.01;
alpha_min = 0.001


def feature(state, action, num_q):
    ft = np.zeros(2)
    ft[0] = state[action]*state[action + num_q]/(10.0)
    ft[1] = 1
    return ft


def proj(weights, upper, lower):
    temp_weights = np.copy(weights)
    for i in range(len(weights)):
        if weights[i]>upper:
            temp_weights[i] = upper
        if weights[i]<lower:
            temp_weights[i] = lower
    return temp_weights


def choose_action_train(state, weights, epsilon, num_q):
    if np.random.random() < epsilon:
        return rand.randint(0, num_q - 1)
    else:
        return max_action(state, weights, num_q)


def max_action(state, weights, num_q):
    action = 0;
    temp_ini = feature(state, action, num_q)
    reward_max = np.dot(weights, temp_ini)
    for i in range(num_q):
        temp = feature(state, i, num_q)
        if np.dot(weights, temp) >= reward_max:
            action = i
            reward_max = np.dot(weights, temp)
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


for j in range(num_episodes):
    rand.seed()
    len_q = np.zeros(num_q)
    channel = np.zeros(num_q)
    state = np.zeros(2*num_q)
    state_new = np.zeros(2*num_q)
    total_arr = 0.0
    total_lost = 0.0
    epsilon = max(epsilon_min, 0.1*(1. - (j+1.)/num_episodes))
    alpha = max(alpha_min, 0.1*(1. - (j+1.)/num_episodes))
    # epsilon = max(epsilon_min, min(0.1, 1*(1.0 - math.log10((j + 1)/100.))))
    # alpha = max(alpha_min, min(0.01, 1*(1.0 - math.log10((j + 1)/100.))));
    gamma = discount;
    for i in range(time_steps):
        state[:num_q] = len_q
        state[num_q:] = channel
        action = choose_action_train(state, weights, epsilon, num_q)
        if channel[action] != 0:
            len_q[action] = max(0, len_q[action]-1)
        arrivals = np.random.binomial(1, arr_rate)
        len_q += arrivals
        channel = np.random.binomial(1, dep_rate)
        len_q, lost_count = q_proj(len_q, buff_size, num_q)
        state_new[:num_q] = len_q
        state_new[num_q:] = channel
        action_new = max_action(state_new, weights, num_q)
        weights = weights - alpha*(np.dot(weights, feature(state, action, num_q)) - (-lost_count + gamma*np.dot(weights, feature(state_new, action_new, num_q))))*feature(state, action, num_q)
        weights = proj(weights, 100, -100)
        total_arr += sum(arrivals)
        total_lost += lost_count
    ratio[j] = total_lost/total_arr
    if j%100 == 0:
        print j, weights, alpha, epsilon

