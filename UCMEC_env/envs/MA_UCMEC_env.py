import gym
from gym import spaces
from gym.utils import seeding
from os import path
import random
import numpy as np
import pandas as pd
import math
from stable_baselines3.common.env_checker import check_env
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import heapq
import cvxpy as cp
import matplotlib.pyplot as plt

# Initialization
random.seed(3)
M = 10  # number of users
N = 100  # number of APs
K = 4  # number of CPUs
P_max = 0.1  # maximum transmit power of user / pilot power

# locations of users and APs
locations_users = np.random.random_sample([M, 2]) * 900  # 2-D location of users
locations_aps = np.random.random_sample([N, 2]) * 900  # 2-D location of APs

# location of 4 CPUs
locations_cpu = np.zeros([4, 2])
locations_cpu[0, 0] = 300
locations_cpu[0, 1] = 300
locations_cpu[1, 0] = 600
locations_cpu[1, 1] = 300
locations_cpu[2, 0] = 300
locations_cpu[2, 1] = 600
locations_cpu[3, 0] = 600
locations_cpu[3, 1] = 600

# calculate distance between APs and users MxN matrix
distance_matrix = np.zeros([M, N])
distance_matrix_front = np.zeros([N, K])
for i in range(M):
    for j in range(N):
        distance_matrix[i, j] = math.sqrt((locations_users[i, 0] - locations_aps[j, 0]) ** 2
                                          + (locations_users[i, 1] - locations_aps[j, 1]) ** 2)

for i in range(N):
    for j in range(K):
        distance_matrix_front[i, j] = math.sqrt((locations_aps[i, 0] - locations_cpu[j, 0]) ** 2
                                                + (locations_aps[i, 1] - locations_cpu[j, 1]) ** 2)

# edge computing parameter
# user parameter
C_user = np.random.uniform(2e8, 5e8, [1, M])  # computing resource of users  in Hz
Task_size = np.random.uniform(100000, 200000, [1, M])  # task size in bit
Task_density = np.random.uniform(10000, 18000, [1, M])  # task density cpu cycles per bit
Task_max_delay = np.random.uniform(2, 5, [1, M])  # task max delay in second
cluster_size = 5  # AP cluster size

# edge server parameter
C_edge = np.random.uniform(10e9, 20e9, [K, 1])  # computing resource of edge server in CPU

# access channel
access_chan = np.zeros([M, N], dtype=complex)  # complex channel
bandwidth_a = 100e6  # bandwidth of access channel
kappa_1 = np.random.rand(1, N)  # parameter in Eq. (5)
kappa_2 = np.random.rand(1, M)  # parameter in Eq. (5)
f_carrier = 1.9e9  # carrier frequency in Hz
h_ap = 15  # antenna height of AP
h_user = 1.65  # antenna height of user
# L = 46.3 + 33.9 * np.log10(f_carrier / 1000) - 13.82 * np.log10(h_ap) - (
#        1.11 * np.log10(f_carrier / 1000) - 0.7) * h_user + 1.56 * np.log10(f_carrier / 1000) - 0.8
L= 140.7
d_0 = 10  # path-loss distance threshold
d_1 = 50  # path-loss distance threshold
PL = np.zeros([M, N])  # path-loss in dB
beta = np.zeros([M, N])  # large scale fading
gamma = np.zeros([M, N])
sigma_s = 8  # standard deviation of shadow fading (dB)
delta = 0.5  # parameter in Eq. (5)
mu = np.zeros([M, N])  # shadow fading parameter
h = np.zeros([M, N], dtype=complex)  # small scale fading
noise_access = 3.9810717055349565e-21 * bandwidth_a  # noise of access channel -> -174 dbm/Hz

for i in range(M):
    for j in range(N):
        # three slope path-loss model
        if distance_matrix[i, j] > d_1:
            PL[i, j] = -L - 35 * np.log10(distance_matrix[i, j] / 1000)
        elif d_0 <= distance_matrix[i, j] <= d_1:
            PL[i, j] = -L - 10 * np.log10((d_1 / 1000) ** 1.5 * (distance_matrix[i, j] / 1000) ** 2)
        else:
            PL[i, j] = -L - 10 * np.log10((d_1 / 1000) ** 1.5 * (d_0 / 1000) ** 2)

        # Eq. (5) shadow fading computation
        mu[i, j] = math.sqrt(delta) * kappa_1[0, j] + math.sqrt(1 - delta) * kappa_2[
            0, i]  # MxN matrix as Eq. (5)

        # Eq. (2) channel computation
        beta[i, j] = pow(10, PL[i, j] / 10) * pow(10, (sigma_s * mu[i, j]) / 10)
        h[i, j] = np.random.normal(loc=0, scale=0.5) + 1j * np.random.normal(loc=0, scale=0.5)
        access_chan[i, j] = np.sqrt(beta[i, j]) * h[i, j]

# fronthaul channel
front_chan = np.zeros([N, K])
bandwidth_f = 500e6  # bandwidth of fronthaul channel
epsilon = 6e-4  # blockage density
p_ap = 1  # transmit power of APs (30 dBm = 1 W)
alpha_los = 2.5  # path-loss exponent for LOS links
alpha_nlos = 4  # path-loss exponent for NLOS links
psi_los = 3  # Nakagami fading parameter for LOS links
psi_nlos = 2  # Nakagami fading parameter for NLOS links
noise_front = 1.380649 * 10e-23 * 290 * 9 * bandwidth_f  # fronthaul channel noise variance
Gt = 10  # total gain (10 dB)

P_los = np.zeros([N, K])  # probability of LOS links
link_type = np.zeros([N, K])  # type of fronthaul links
for i in range(N):
    for j in range(K):
        P_los[i, j] = np.exp(-epsilon * distance_matrix_front[i, j] / 1000)
        link_type[i, j] = np.random.choice([0, 1], p=[P_los[i, j], 1 - P_los[i, j]])  # 0 for LOS, 1 for NLOS
        if link_type[i, j] == 0:  # LOS link
            front_chan[i, j] = np.random.gamma(2, 1 / psi_los)  # Nakagami channel gain
        else:  # NLOS link
            front_chan[i, j] = np.random.gamma(2, 1 / psi_nlos)  # Nakagami channel gain

# pilot assignment
tau_p = 16  # length of pilot symbol
pilot_matrix = np.zeros([M, tau_p])
for i in range(M):
    pilot_index = random.randint(0, tau_p - 1)
    pilot_matrix[i, pilot_index] = 1

# channel estimation
access_chan_estimate = np.zeros([M, N], dtype=complex)
receive_pilot = np.zeros([N, tau_p], dtype=complex)
for i in range(N):
    for j in range(M):
        receive_pilot[i, :] = receive_pilot[i, :] + np.sqrt(p_ap) * access_chan[j, i] * pilot_matrix[j, :]

    # noise_access = np.random.normal(loc=0, scale=0.5, size=(1, tau_p)) + 1j * np.random.normal(loc=0, scale=0.5,
    #                                                                                            size=(1, tau_p))
    noise_access_list = noise_access * np.ones([1, tau_p])
    receive_pilot[i, :] = receive_pilot[i, :] + noise_access_list

for i in range(M):
    for j in range(N):
        access_chan_estimate[i, j] = 1 / np.sqrt(p_ap) * np.dot(pilot_matrix[i, :].conj().T, receive_pilot[i, :])

# AP cluster
cluster_matrix = np.zeros([M, N])  # AP serve user when value = 1, otherwise 0
max_h_index_list = np.zeros([M, N])  # sort access channel from high to low
ap_index_list = np.zeros([M, cluster_size])  # obtain the index of APs serve for each user
for i in range(M):
    for j in range(N):
        max_h_index_list[i, :] = np.abs(access_chan[i, :]).argsort()
        max_h_index_list[i, :] = max_h_index_list[i, :][::-1]
        ap_index_list[i, :] = max_h_index_list[i, 0:cluster_size]
        for k in range(cluster_size):
            cluster_matrix[i, int(ap_index_list[i, k])] = 1


class MA_UCMEC(gym.Env):
    def __init__(self, render: bool = False):
        # paremeter init
        self._render = render
        # action space: [omega_1,omega_2,...,omega_K,p]  K+1 continuous vector for each agent
        # a in [0,1], p in [p_min, p_max]
        omega_low = np.zeros(K)
        p_low = 0
        omega_high = np.ones(K)
        p_high = P_max
        action_low = np.append(omega_low, p_low)
        action_high = np.append(omega_high, p_high)
        self.action_space = spaces.Box(low=np.array(list(action_low) * M, dtype=np.float32),
                                       high=np.array(list(action_high) * M, dtype=np.float32), shape=((K + 1) * M,),
                                       dtype=np.float32)
        # state space: [r_1,r_2,...,r_M]  1xM continuous vector. -> uplink rate
        # r in [0, 10e8]
        r_low = np.zeros(M)
        r_high = np.ones(M) * 1e8
        self.observation_space = spaces.Box(low=r_low, high=r_high, shape=(M,), dtype=np.float32)
        # self.np_random = None
        self.step_num = 0

    def uplink_rate_cal(self, p):  # calculate the uplink transmit rate in Eq. (12)
        SINR_access = np.zeros([M, 1])
        uplink_rate_access = np.zeros([M, 1])
        SINR_access_deno_2_sum = 0
        for i in range(M):
            for j in range(N):
                if cluster_matrix[i, j] == 1:
                    SINR_access_deno_2_sum = SINR_access_deno_2_sum + p[i, 0] * abs(
                        access_chan_estimate[i, j].conjugate() * access_chan[i, j]) ** 2

        SINR_access_mole = np.zeros([M,1])
        SINR_access_deno_1 =np.zeros([M,1])
        SINR_access_deno_2 = np.zeros([M,1])
        for i in range(M):
            SINR_access_mole[i,0] = 0
            SINR_access_deno_1[i,0] = 0
            for j in range(N):
                if cluster_matrix[i, j] == 1:
                    SINR_access_mole[i,0] = SINR_access_mole[i,0] + abs(
                        access_chan_estimate[i, j].conjugate() * access_chan[i, j]) ** 2
                    SINR_access_deno_1[i,0] = SINR_access_deno_1[i,0] + noise_access * abs(
                        access_chan_estimate[i, j]) ** 2
            SINR_access_mole[i,0] = SINR_access_mole[i,0] * p[i, 0]
            SINR_access_deno_2[i,0] = SINR_access_deno_2_sum - SINR_access_mole[i,0]
            SINR_access[i, 0] = SINR_access_mole[i,0] / (SINR_access_deno_1[i,0] + SINR_access_deno_2[i,0])
            uplink_rate_access[i, 0] = bandwidth_a * np.log2(1 + SINR_access[i, 0])
        return uplink_rate_access

    def front_rate_cal(self, omega):
        chi = np.zeros([N, K])  # whether a AP transmit symbol to a CPU or not
        SINR_front = np.zeros([N, K])  # SINR in Eq. (7)
        front_rate = np.zeros([N, K])  # Eq. (12)
        front_rate_user = np.zeros([M, 1])
        I_sum = 0  # total sum
        for i in range(M):
            CPU_id = np.argmax(omega[i, :])
            for j in range(N):
                if cluster_matrix[i, j] == 1:  # This AP is belong to the cluster of user i
                    chi[j, CPU_id] = 1

        for i in range(N):
            for j in range(K):
                if chi[i, j] == 1:
                    if link_type[j, j] == 0:  # LOS link
                        I_sum = I_sum + p_ap * front_chan[i, j] * pow(distance_matrix_front[i, j] / 1000,
                                                                      -alpha_los)
                    else:
                        I_sum = I_sum + p_ap * front_chan[i, j] * pow(distance_matrix_front[i, j] / 1000,
                                                                      -alpha_nlos)
                else:
                    pass

        for i in range(N):
            for j in range(K):
                if chi[i, j] == 1:
                    if link_type[i, j] == 0:  # LOS link
                        SINR_front_mole = p_ap * front_chan[i, j] * Gt * pow(
                            distance_matrix_front[i, j] / 1000, -alpha_los)
                    else:
                        SINR_front_mole = p_ap * front_chan[i, j] * Gt * pow(
                            distance_matrix_front[i, j] / 1000,
                            -alpha_nlos)
                    SINR_front[i, j] = SINR_front_mole / (I_sum - SINR_front_mole / Gt + noise_front)
                    front_rate[i, j] = bandwidth_f * np.log2(1 + SINR_front[i, j])

        for i in range(M):
            front_rate_list = np.zeros([cluster_size, 1])
            n_flag = 0
            CPU_id = np.argmax(omega[i, :])
            for j in range(N):
                if cluster_matrix[i, j] == 1:
                    front_rate_list[n_flag, 0] = front_rate[j, CPU_id]
                    n_flag = n_flag + 1
            front_rate_user[i, 0] = np.min(front_rate_list)

        return front_rate_user

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def render(self, mode='human'):
        pass

    def reset(self):
        reset_state = np.random.uniform(low=0, high=1e6, size=(M,))
        return np.array(reset_state)

    def step(self, action):
        self.step_num += 1
        # obtain and clip the action
        omega_current = np.zeros([M, K])
        p_current = np.zeros([M, 1])
        for i in range(M):
            flag = K + 1
            action_current = action[i * flag:(i + 1) * flag]  # K+1 dimension
            omega_current[i, :] = np.clip(action_current[0:K], 0, 1)  # K dimension
            p_current[i, 0] = np.clip(action_current[K:K + 1], 0, P_max)  # 1 dimension

        uplink_rate_access = self.uplink_rate_cal(p_current)
        print(uplink_rate_access)
        front_rate_user = self.front_rate_cal(omega_current)
        # print(uplink_rate_access)
        # print(front_rate_user)
        obs = uplink_rate_access

        # local computing delay
        local_delay = np.zeros([M, 1])
        for i in range(M):
            local_proportion = 1 - np.max(omega_current[i, :])
            local_delay[i, 0] = local_proportion * Task_density[0, i] * Task_size[0, i] / C_user[0, i]

        # uplink delay
        uplink_delay = np.zeros([M, 1])
        for i in range(M):
            uplink_delay_list = np.zeros([cluster_size, 1])
            for j in range(cluster_size):
                uplink_delay_list[j, 0] = np.max(omega_current[i, :]) * Task_size[0, i] / uplink_rate_access[i, 0]
            uplink_delay[i, 0] = np.max(uplink_delay_list)
        # print(uplink_delay)

        # fronthaul delay
        front_delay = np.zeros([M, 1])
        offload_task_size = np.zeros([M, 1])
        for i in range(M):
            offload_task_size[i, 0] = np.max(omega_current[i, :]) * Task_size[0, i]
        for i in range(M):
            front_delay[i, 0] = np.sum(offload_task_size) / front_rate_user[i, 0]
        # print(front_delay)

        # processing delay calculation
        C = cp.Variable((M, K))
        # solve convex problem according to Eq. (24)
        proportion_list = np.zeros([M, 1])  # offloading proportion of each user
        for i in range(M):
            proportion_list[i, 0] = np.max(omega_current[i, :])
        task_mat = np.zeros([M, K])
        for i in range(M):
            CPU_id = np.argmax(omega_current[i, :])
            task_mat[i, CPU_id] = proportion_list[i, 0] * Task_size[0, i] * Task_density[0, i]

        # Each CPU solves a resource allocation optimization problem
        actual_C = np.zeros([M, 1])
        for i in range(K):
            serve_user_id = []
            serve_user_task = []
            _local_delay = []
            _front_delay = []
            _uplink_delay = []

            for j in range(M):
                if task_mat[j, i] != 0:
                    serve_user_id.append(j)
                    serve_user_task.append(task_mat[j, i])
                    _local_delay.append(local_delay[j, 0])
                    _front_delay.append(front_delay[j, 0])
                    _uplink_delay.append(uplink_delay[j, 0])
            if len(serve_user_id) == 0:
                continue
            C = cp.Variable(len(serve_user_id))
            _process_delay = cp.multiply(serve_user_task, cp.inv_pos(C))
            _local_delay = np.array(_local_delay)
            _front_delay = np.array(_front_delay)
            _uplink_delay = np.array(_uplink_delay)

            func = cp.Minimize(cp.sum(cp.maximum(_local_delay, _front_delay + _uplink_delay + _process_delay)))
            cons = [0 <= C, cp.sum(C) <= C_edge[i, 0]]
            prob = cp.Problem(func, cons)
            prob.solve(solver=cp.SCS, verbose=False)
            for k in range(len(serve_user_id)):
                _C = C.value
                actual_C[serve_user_id[k], 0] = _C[k]

        actual_process_delay = np.zeros([M, 1])
        for i in range(M):
            CPU_id = np.argmax(omega_current[i, :])
            actual_process_delay[i, 0] = task_mat[i, CPU_id] / actual_C[i, 0]
        print(actual_process_delay)
        '''
        process_delay = cp.max(cp.multiply(task_mat, cp.inv_pos(C)))  # Mx1
        func = cp.Minimize(cp.sum(cp.maximum(local_delay, front_delay + uplink_delay + process_delay)))
        # func = cp.Minimize(cp.sum(cp.maximum(local_delay, process_delay)))
        cons = [0 <= C]
        for i in range(K):
            cons += [cp.sum(C[:, i]) <= C_edge[i, 0]]

        prob = cp.Problem(func, cons)
        prob.solve(solver=cp.SCS, verbose=False)
        actual_C = C.value
        actual_process_delay = np.max(task_mat / actual_C, axis=1)
        # print(actual_process_delay)
        # print(C.value)
        '''

        # reward calculation
        reward = np.zeros([M, 1])
        for i in range(M):
            reward[i, 0] = - np.sum(np.maximum(local_delay, front_delay + uplink_delay + actual_process_delay)) / M
        if self.step_num > 36000:
            done = [1] * M
        else:
            done = [0] * M

        info = {}
        return tuple(obs), reward, done, info


if __name__ == "__main__":
    env = MA_UCMEC(render=False)
    # check_env(env)
    obs = env.reset()
    n_steps = 50
    for _ in range(n_steps):
        # Random action
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        if np.all(done):
            obs = env.reset()
        print(f"state: {obs} \n")
        print(f"action : {action}, reward : {reward}")
