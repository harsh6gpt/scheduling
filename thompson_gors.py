import numpy as np
import scipy as sp
from scipy import optimize
import math
import matplotlib.pyplot as plt
import time
from scipy import stats


def truncated_beta_sample(alpha_in, beta_in, left_lim, right_lim):
    left_uni = stats.beta.cdf(left_lim, alpha_in, beta_in)
    right_uni = stats.beta.cdf(right_lim, alpha_in, beta_in)
    uni = np.random.uniform(left_uni, right_uni)
    return stats.beta.ppf(uni, alpha_in, beta_in)


def optimal_rate(rates_in, theta_in, num_rates_in):
    opt_index = 0
    max_val = 0
    for j in range(num_rates_in):
        if rates_in[j] * theta_in[j] > max_val:
            max_val = rates_in[j] * theta_in[j]
            opt_index = j
    return opt_index


def transmission(theta_in):
    np.random.seed()
    out = np.random.binomial(1, theta_in)
    return out


def proj(a, b, x):
    if x > b:
        return b
    elif x < a:
        return a
    else:
        return x


def kl_div(theta1, theta2):
    theta1 = proj(0, 1, theta1)
    theta2 = proj(0, 1, theta2)
    if theta1 == 0:
        if theta2 == 0:
            return 0
        else:
            return math.inf
    if theta2 == 0:
        return math.inf
    if theta1 == 1:
        if theta2 == 1:
            return 0
        else:
            return math.inf
    if theta2 == 1:
        return math.inf
    return theta1 * math.log(theta1 / theta2, 2) + (1 - theta1) * math.log((1 - theta1) / (1 - theta2), 2)


def index_gors(num_rates, rates, rates_count, reward_est, lead_count, grain, c_gors):
    curr_leader = np.argmax(reward_est)
    rate_ind_new = np.zeros(num_rates)
    for i in range(num_rates):
        rate_ind_new[i] = index_opt_gors(rates[i], reward_est[i], rates_count[i], lead_count, curr_leader, grain, c_gors)
    return rate_ind_new


def index_kl_ucb(num_rates, rates, rates_count, reward_est, c_kl, grain, time):
    rate_ind_new = np.zeros(num_rates)
    for i in range(num_rates):
        rate_ind_new[i] = index_opt_kl(rates[i], reward_est[i], rates_count[i], c_kl, grain, time)
    return rate_ind_new


def index_opt_kl(rate, reward_est, rate_count, c_kl, grain, time):
    q = reward_est
    rhs = math.log(time, 2) + c_kl*math.log(math.log(time, 2), 2)
    for i in range(math.ceil((rate - q)/grain)):
        q_new = q + grain*(i + 1)
        if rate_count*kl_div(reward_est/rate, q_new/rate) < rhs:
            q = q_new
        else:
            break
    return q


def index_opt_gors(rate, reward_est, rate_count, lead_count, curr_leader, grain, c_gors):
    q = reward_est
    rhs = math.log(lead_count[curr_leader], 2) + c_gors*math.log(math.log(lead_count[curr_leader], 2), 2)
    for i in range(math.ceil((rate - q)/grain)):
        q_new = q + grain*(i + 1)
        if rate_count*kl_div(reward_est/rate, q_new/rate) < rhs:
            q = q_new
        else:
            break
    return q


def index_opt(rate, reward_est, count_est, lead_count, curr_leader, c_gors):
    def fun(x):
        return -x

    cons = ({'type': 'ineq', 'fun': lambda x: x}, {'type': 'ineq', 'fun': lambda x: -x + rate},
            {'type': 'ineq', 'fun': lambda x: math.log(lead_count[curr_leader], 2)
                                              + c_gors * math.log(math.log(lead_count[curr_leader], 2), 2)
                                              - count_est * kl_div(reward_est / rate, x / rate)})
    res = sp.optimize.minimize(fun, 0, method="COBYLA", constraints=cons)
    return res.x


def theta_dec(theta_in):
    return all(sorted(theta_in, reverse=True) == theta_in)


def theta_unimodal(theta_in, rates_in):
    throughput = rates_in * theta_in
    max_ind = np.argmax(throughput)
    if not all(throughput[:max_ind + 1] == sorted(throughput[:max_ind + 1])):
        return False
    else:
        return all(sorted(throughput[max_ind:], reverse=True) == throughput[max_ind:])


def intersect_intervals(left_lim1, right_lim1, left_lim2, right_lim2):
    return max(left_lim1, left_lim2), min(right_lim1, right_lim2)


def local_min_in_theta(theta_val, index, rates_arr, num_comp):
    l_temp = 0
    r_temp = 0
    thru = theta_val[index]*rates_arr[index]
    for m in range(index + 1, num_comp):
        if theta_val[m] > 0:
            r_temp = theta_val[m]*rates_arr[m]
    for m in range(index - 1, -1, -1):
        if theta_val[m] > 0:
            l_temp = theta_val[m]*rates_arr[m]
    return min(l_temp, r_temp, thru) == thru


start_time = time.time()
num_iter = 100
tot_time = 10000
num_rates = 8
c_gors = 3
c_kl = 3
gamma = 3
grain = 1e-2
burnin_ts = 00
scenario = "lossy"  # "gradual"/"lossy"/"steep"/"custom"
theta = np.zeros(num_rates)
rates = np.zeros(num_rates)
# regret_ts = np.load("tuned-mts.npy")
# regret_ts1 = np.load("standard-mts.npy")
regret_ts = np.zeros(tot_time)
regret_cots = np.zeros(tot_time)
regret_gors = np.zeros(tot_time)
regret_kl = np.zeros(tot_time)
runtime_ts = np.zeros(num_iter)
runtime_gors = np.zeros(num_iter)
runtime_kl = np.zeros(num_iter)
runtime_cots = np.zeros(num_iter)
rates[0] = 6
rates[1] = 9
rates[2] = 12
rates[3] = 18
rates[4] = 24
rates[5] = 36
rates[6] = 48
rates[7] = 54
if scenario == "steep":
    theta[0] = 0.99
    theta[1] = 0.98
    theta[2] = 0.96
    theta[3] = 0.93
    theta[4] = 0.90
    theta[5] = 0.10
    theta[6] = 0.06
    theta[7] = 0.04
if scenario == "gradual":
    theta[0] = 0.95
    theta[1] = 0.90
    theta[2] = 0.80
    theta[3] = 0.65
    theta[4] = 0.45
    theta[5] = 0.25
    theta[6] = 0.15
    theta[7] = 0.10
if scenario == "lossy":
    theta[0] = 0.90
    theta[1] = 0.80
    theta[2] = 0.70
    theta[3] = 0.55
    theta[4] = 0.45
    theta[5] = 0.35
    theta[6] = 0.20
    theta[7] = 0.10
if scenario == "custom":
    theta[0] = 0.90
    theta[1] = 0.80
    theta[2] = 0.70
    theta[3] = 0.55
    theta[4] = 0.40
    theta[5] = 0.35
    theta[6] = 0.20
    theta[7] = 0.18
# theta[0] = 1.0
# theta[1] = 0.55
# theta[2] = 0.49
opt_ind = optimal_rate(rates, theta, num_rates)
for l in range(num_iter):
    print("Iteration Number: ", l)
    alpha = np.ones(num_rates)
    beta = np.ones(num_rates)
    alpha1 = np.ones(num_rates)
    beta1 = np.ones(num_rates)
    reward_ts = 0
    reward_cots = 0
    rates_count_ts = np.zeros(num_rates)
    rates_count_cots = np.zeros(num_rates)
    reward_gors = 0
    rates_count_gors = np.zeros(num_rates) + 1
    reward_est_gors = np.zeros(num_rates) + 1
    index_arr = np.zeros(num_rates)
    lead_count_gors = np.zeros(num_rates) + 2
    reward_kl = 0
    rates_count_kl = np.zeros(num_rates) + 1
    reward_est_kl = np.zeros(num_rates) + 1
    index_arr_kl = np.zeros(num_rates)
    start_cots = time.time()
    for i in range(tot_time):
        np.random.seed()
        # print("Tuned-MTS ", i, ": ", alpha, beta, rates_count_ts)
        # print("MTS ", i, ": ", alpha1, beta1, rates_count_ts1)
        # theta_check = False
        # if i > burnin_ts:
        #     while not theta_check:
        #         theta_sample = np.random.beta(alpha, beta)
        #         theta_check = theta_dec(theta_sample) \
        #             # and theta_unimodal(theta_sample, rates)
        # else:
        #     theta_sample = np.random.beta(alpha, beta)
        beta_param_sum = np.copy(alpha + beta)
        beta_param_sum_ind = sorted(range(len(beta_param_sum)), key=lambda x: beta_param_sum[x], reverse=True)
        theta_sample = np.zeros(num_rates)
        sample_left_cond = np.zeros(num_rates)
        sample_right_cond = np.zeros(num_rates)
        for beta_index in beta_param_sum_ind:
            min_r = 0
            max_l = 1
            min_r_uni = 0
            max_l_uni = 1
            l_uni_ind = -1
            r_uni_ind = -1
            if beta_index == beta_param_sum_ind[0]:
                theta_sample[beta_index] = np.random.beta(alpha[beta_index], beta[beta_index])
                continue
            for n in range(beta_index + 1, num_rates):
                if theta_sample[n] > 0:
                    min_r = theta_sample[n]
                    break
            for n in range(beta_index - 1, -1, -1):
                if theta_sample[n] > 0:
                    max_l = theta_sample[n]
                    break
            for n in range(beta_index + 1, num_rates):
                if theta_sample[n] > 0:
                    min_r_uni = theta_sample[n]
                    r_uni_ind = n
                    break
            for n in range(beta_index - 1, -1, -1):
                if theta_sample[n] > 0:
                    max_l_uni = theta_sample[n]
                    l_uni_ind = n
                    break
            if r_uni_ind == -1 or l_uni_ind == -1:
                temp_min = 0
            else:
                temp_min = min(min_r_uni*rates[r_uni_ind], max_l_uni*rates[l_uni_ind])/rates[beta_index]
            temp_max = max(min_r_uni*rates[r_uni_ind], max_l_uni*rates[l_uni_ind])/rates[beta_index]
            if min_r_uni*rates[r_uni_ind] > max_l_uni*rates[l_uni_ind]:
                temp_max_ind = r_uni_ind
            else:
                temp_max_ind = l_uni_ind
            theta_sample[beta_index] = temp_max + 0.01
            if not local_min_in_theta(theta_sample, temp_max_ind, rates, num_rates) :
                temp_max = 1
            left_l, right_l = intersect_intervals(min_r, max_l, temp_min, temp_max)
            theta_sample[beta_index] = truncated_beta_sample(alpha[beta_index], beta[beta_index], left_l, right_l)
        # print(theta_sample, rates*theta_sample, theta_unimodal(theta_sample, rates))
        rate_opt_ind = optimal_rate(theta_sample, rates, num_rates)
        rates_count_cots[rate_opt_ind] += 1
        outcome = transmission(theta[rate_opt_ind])
        if outcome == 1:
            alpha[rate_opt_ind] += 1
        else:
            beta[rate_opt_ind] += 1
        reward_cots += rates[rate_opt_ind] * outcome
        regret_cots[i] += rates[opt_ind] * theta[opt_ind] * (i + 1) - reward_cots
    end_cots = time.time()
    runtime_cots[l] = end_cots - start_cots
    start_ts = time.time()
    for n in range(tot_time):
        theta_sample1 = np.random.beta(alpha1, beta1)
        rate_opt_ind1 = optimal_rate(theta_sample1, rates, num_rates)
        rates_count_ts[rate_opt_ind1] += 1
        outcome = transmission(theta[rate_opt_ind1])
        if outcome == 1:
            alpha1[rate_opt_ind1] += 1
        else:
            beta1[rate_opt_ind1] += 1
        reward_ts += rates[rate_opt_ind1] * outcome
        regret_ts[n] += rates[opt_ind] * theta[opt_ind] * (n + 1) - reward_ts
    end_ts = time.time()
    runtime_ts[l] = end_ts - start_ts
    start_gors = time.time()
    for j in range(tot_time):
        np.random.seed()
        if j < 1 * num_rates:
            rate_opt_ind = j % num_rates
        else:
            index_arr = index_gors(num_rates, rates, rates_count_gors, reward_est_gors, lead_count_gors, grain, c_gors)
            curr_leader = np.argmax(reward_est_gors)
            temp_ind = curr_leader
            max_ind = index_arr[temp_ind]
            for k in range(max(curr_leader - 1, 0), min(curr_leader + 2, num_rates)):
                if index_arr[k] >= max_ind:
                    max_ind = index_arr[k]
                    temp_ind = k
            if (lead_count_gors[curr_leader] - 1) % gamma == 0 and lead_count_gors[curr_leader] - 1 > 1:
                rate_opt_ind = curr_leader
            else:
                rate_opt_ind = temp_ind
        rates_count_gors[rate_opt_ind] += 1
        outcome = transmission(theta[rate_opt_ind])
        reward = rates[rate_opt_ind] * outcome
        reward_est_gors[rate_opt_ind] = ((rates_count_gors[rate_opt_ind] - 1) / rates_count_gors[rate_opt_ind]) * \
                                        reward_est_gors[rate_opt_ind] + reward / (rates_count_gors[rate_opt_ind])
        reward_gors += reward
        curr_leader = np.argmax(reward_est_gors)
        lead_count_gors[curr_leader] += 1
        regret_gors[j] += rates[opt_ind] * theta[opt_ind] * (j + 1) - reward_gors
    end_gors = time.time()
    runtime_gors[l] = end_gors - start_gors
    start_kl = time.time()
    for j in range(tot_time):
        np.random.seed()
        if j < 1 * num_rates:
            rate_opt_ind = j % num_rates
        else:
            index_arr_kl = index_kl_ucb(num_rates, rates, rates_count_kl, reward_est_kl, c_kl, grain, j+1)
            rate_opt_ind = np.argmax(index_arr_kl)
        rates_count_kl[rate_opt_ind] += 1
        outcome = transmission(theta[rate_opt_ind])
        reward = rates[rate_opt_ind] * outcome
        reward_est_kl[rate_opt_ind] = ((rates_count_kl[rate_opt_ind] - 1) / rates_count_kl[rate_opt_ind]) * \
                                        reward_est_kl[rate_opt_ind] + reward / (rates_count_kl[rate_opt_ind])
        reward_kl += reward
        regret_kl[j] += rates[opt_ind] * theta[opt_ind] * (j + 1) - reward_kl
    end_kl = time.time()
    runtime_kl[l] = end_kl - start_kl
regret_ts = regret_ts / num_iter
regret_gors = regret_gors / num_iter
regret_cots = regret_cots / num_iter
regret_kl = regret_kl / num_iter
avgtime_ts = np.average(runtime_ts)
avgtime_cots = np.average(runtime_cots)
avgtime_kl = np.average(runtime_kl)
avgtime_gors = np.average(runtime_gors)
plt.plot(regret_cots, label="Co-TS")
plt.plot(regret_ts, label="MTS")
plt.plot(regret_kl, label="KL-UCB")
plt.plot(regret_gors, label="G-ORS")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Regret")
plt.title(scenario)
plt.show()
print("Total time taken: ", time.time() - start_time, " seconds")
