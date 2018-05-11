# importing required libraries
import numpy as np
import random as rand
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import sgd, adagrad, adam
from keras.models import load_model
import time
import math


# initializing the parameters
start_time = time.time()
num_q = 2
buff_size = 2
arr_rate = np.zeros(num_q)
arr_rate += 0.2
# arr_rate[1] = 0.4
dep_rate = np.zeros(num_q)
dep_rate += 0.5
# dep_rate[0] = 0.3
# dep_rate[1] = 0.7
num_episodes = 100
time_steps = 1000
ratio = np.zeros(num_episodes)
discount = 0.99
epsilon_min = 0.01
mc_sim = 1
mc_time = 500


# defining the feed-forward neural network model
model = Sequential()
model.add(Dense(8, input_dim=2*num_q, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(num_q, activation='linear'))
model.compile(loss='mse', optimizer=adam(lr=0.01))
#model = load_model('temp.h5')
a = model.get_weights()

def choose_action_train(state, num_action, model, epsilon):
    if np.random.random() < epsilon:
        return rand.randint(0, num_action - 1);
    else:
        return np.argmax(model.predict(np.reshape(state, [1, 2*num_action]))[0]);


def q_proj(len_q, buff_size, num_q):
    ret_q = np.zeros(num_q)
    lost_count = 0
    for i in range(num_q):
        if len_q[i]>buff_size:
            ret_q[i] = buff_size
            lost_count += 1
        else:
            ret_q[i] = len_q[i]
    return ret_q, lost_count


for j in range(num_episodes):
    np.random.seed()
    len_q = np.zeros(num_q)
    channel = np.zeros(num_q)
    state = np.concatenate((len_q, channel))
    total_arr = 0.0
    total_lost = 0.0
    epsilon = max(epsilon_min, 1*(1 - (j + 1.)/num_episodes))
    epsilon = 1
    gamma = discount
    x_data = []
    y_data = []
    action_data = []
    lost_count_ls = []
    print "Episode number :", j
    for i in range(time_steps):
        action = choose_action_train(state, num_q, model, epsilon)
        action_data.append(action)
        if channel[action] != 0:
            len_q[action] = max(0, len_q[action] - 1)
        arrivals = np.random.binomial(1, arr_rate)
        len_q += arrivals
        channel = np.random.binomial(1, dep_rate)
        len_q, lost_count = q_proj(len_q, buff_size, num_q)
        new_q = model.predict(np.reshape(state, [1, 2*num_q]))[0]
        x_data.append(state)
        state = np.concatenate((len_q, channel))
        lost_count_ls.append(lost_count)
        if mc_sim != 1:
            new_q[action] = -lost_count + gamma*(np.amax(model.predict(np.reshape(state, [1, 2*num_q]))[0]))
            y_data.append(new_q)
        else:
            new_q[action] = -lost_count
            y_data.append(new_q)
            for k in range(max(0, i - mc_time), i):
                y_data[k][action_data[k]] += math.pow(gamma, i - k)*(-lost_count)
            if i - mc_time - 1 >= 0:
                y_data[i - mc_time - 1] += math.pow(gamma, mc_time + 1)*(np.amax(model.predict(np.reshape(state, [1, 2*num_q]))[0]))
        total_arr += sum(arrivals)
        total_lost += lost_count
    model.fit(np.array(x_data), np.array(y_data), batch_size = 32, epochs = 1, verbose = 2)
    # model.train_on_batch(np.array(x_data), np.array(y_data))
    ratio[j] = total_lost/total_arr
    print total_lost, total_arr
    print "ratio: ", ratio[j]
print("--- %s seconds ---", (time.time() - start_time))