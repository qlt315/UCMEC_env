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


class MA_UCMEC(gym.Env):
    def __init__(self, render: bool = False):
        # paremeter init
        self._render = render
        # action space: [omega_1,omega_2,...,omega_K,p]  K+1 continuous vector for each agent
        # a in {0,1,2,3,4}, p in {0, 1, 2, 3, 4} (totally 5 levels (p+1)/5*100 mW)
        self.action_space = spaces.MultiDiscrete([5, 5] * M_sim)
        # state space: [r_1(t-1),r_2(t-1),...,r_M(t-1)]  1xM continuous vector. -> uplink rate
        # r in [0, 10e8]
        r_low = np.zeros(M_sim)
        r_high = np.ones(M_sim) * 1e8
        self.observation_space = spaces.Box(low=r_low, high=r_high, shape=(M_sim,), dtype=np.float32)
        # self.np_random = None
        self.uplink_rate_access_b = np.zeros([M_sim, 1])
        self.step_num = 0
        self.cluster_matrix = self.cluster()

    def cluster(self):
        # AP cluster
        cluster_matrix = np.zeros([M_sim, N_sim])  # AP serve user when value = 1, otherwise 0
        max_h_index_list = np.zeros([M_sim, N_sim])  # sort access channel (large-scale fading) from high to low
        ap_index_list = np.zeros([M_sim, cluster_size])  # obtain the index of APs serve for each user
        for i in range(M_sim):
            for j in range(N_sim):
                max_h_index_list[i, :] = beta[i, 0:N_sim].argsort()
                max_h_index_list[i, :] = max_h_index_list[i, :][::-1]
                ap_index_list[i, :] = max_h_index_list[i, 0:cluster_size]
                for k in range(cluster_size):
                    cluster_matrix[i, int(ap_index_list[i, k])] = 1
        return cluster_matrix

    def uplink_rate_cal(self, p, omega):  # calculate the uplink transmit rate in Eq. (12)
        SINR_access = np.zeros([M_sim, 1])
        uplink_rate_access = np.zeros([M_sim, 1])
        SINR_access_mole = np.zeros([M_sim, 1])
        SINR_access_inter = np.zeros([M_sim, 1])
        SINR_access_noise = np.zeros([M_sim, 1])

        for i in range(M_sim):
            if omega[i] == 0:  # Local processing
                continue
            else:
                SINR_access_inter[i, 0] = 0  # Interference
                SINR_access_mole[i, 0] = 0  # Useful symbol
                SINR_access_noise[i, 0] = 0  # Noise
                for j in range(N_sim):
                    if self.cluster_matrix[i, j] == 1:
                        SINR_access_mole[i, 0] = SINR_access_mole[i, 0] + theta[i, j]
                        SINR_access_noise[i, 0] = noise_access * theta[i, j]
                SINR_access_mole[i, 0] = (SINR_access_mole[i, 0] ** 2) * p[i] * varsig

                for k in range(M_sim):
                    if k == i or omega[k] == 0:
                        continue
                    else:
                        for j in range(N_sim):
                            if self.cluster_matrix[i, j] == 1:
                                SINR_access_inter[i, 0] = SINR_access_inter[i, 0] + theta[i, j] * beta[k, j] * p[k]
                SINR_access[i, 0] = SINR_access_mole[i, 0] / (SINR_access_inter[i, 0] + SINR_access_noise[i, 0])
                uplink_rate_access[i, 0] = bandwidth_a * np.log2(1 + SINR_access[i, 0])
        print(SINR_access_mole)
        return uplink_rate_access

    def front_rate_cal(self, omega):
        chi = np.zeros([N_sim, K])  # whether a AP transmit symbol to a CPU or not
        SINR_front = np.zeros([N_sim, K])  # SINR in Eq. (7)
        front_rate = np.zeros([N_sim, K])  # Eq. (12)
        front_rate_user = np.zeros([M_sim, N_sim])
        I_sum = 0  # total sum of fronthaul interference
        for i in range(M_sim):
            if omega[i] == 0:
                continue
            CPU_id = int(omega[i] - 1)
            for j in range(N_sim):
                if self.cluster_matrix[i, j] == 1:  # This AP is belong to the cluster of user i
                    chi[j, CPU_id] = 1

        for i in range(N_sim):
            for j in range(K):
                if chi[i, j] == 1:
                    if link_type[j, j] == 0:  # LOS link
                        I_sum = I_sum + p_ap * pow(distance_matrix_front[i, j] / 1000, -alpha_los)
                    else:
                        I_sum = I_sum + p_ap * pow(distance_matrix_front[i, j] / 1000, -alpha_nlos)
                else:
                    pass

        for i in range(N_sim):
            for j in range(K):
                if chi[i, j] == 1:
                    if link_type[i, j] == 0:  # LOS link
                        SINR_front_mole = p_ap * G[i, j] * pow(distance_matrix_front[i, j] / 1000, -alpha_los)
                    else:
                        SINR_front_mole = p_ap * G[i, j] * pow(distance_matrix_front[i, j] / 1000, alpha_nlos)
                    SINR_front[i, j] = SINR_front_mole / (I_sum - SINR_front_mole / G[i, j] + noise_front)
                    front_rate[i, j] = bandwidth_f * np.log2(1 + SINR_front[i, j])

        for i in range(M_sim):
            if omega[i] == 0:
                pass
            else:
                CPU_id = int(omega[i] - 1)
                for j in range(N_sim):
                    if self.cluster_matrix[i, j] == 1:
                        front_rate_user[i, j] = front_rate[j, CPU_id]

        return front_rate_user

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def render(self, mode='human'):
        pass

    def reset(self):
        reset_state = np.random.uniform(low=0, high=1e6, size=(M_sim,))
        return np.array(reset_state)

    def step(self, action):
        self.step_num += 1
        # obtain and clip the action
        omega_current = np.zeros([M_sim])
        p_current = np.zeros([M_sim])
        p_level = P_max / 5

        if self.step_num == 1:  # The first step
            for i in range(M_sim):
                action_num = 2
                action_current = action[i * action_num:(i + 1) * action_num]  # 2 dimension action of the current user
                omega_current[i] = action_current[0]  # 1 dimension
                p_current[i] = (action_current[1] + 1) * p_level  # 1 dimension
            print("Chosen CPU ID:", omega_current)
            print("Power:", p_current)
            uplink_rate_access = self.uplink_rate_cal(p_current, omega_current)
            front_rate_user = self.front_rate_cal(omega_current)
            observation = uplink_rate_access  # Last time access uplink rate
            self.uplink_rate_access_b = uplink_rate_access

            print("Uplink Rate (Mbps):", uplink_rate_access / 10e6)
            print("Average Uplink Rate (Mbps):", np.sum(uplink_rate_access) / (np.count_nonzero(omega_current) * 10e6))
        else:

            observation = self.uplink_rate_access_b  # Last time access uplink rate
            for i in range(M_sim):
                action_num = 2
                action_current = action[i * action_num:(i + 1) * action_num]  # 2 dimension action of the current user
                omega_current[i] = action_current[0]  # 1 dimension
                p_current[i] = (action_current[1] + 1) * p_level  # 1 dimension
            print("Chosen CPU ID:", omega_current)
            print("Power:", p_current)

            uplink_rate_access = self.uplink_rate_cal(p_current, omega_current)
            front_rate_user = self.front_rate_cal(omega_current)
            self.uplink_rate_access_b = uplink_rate_access
            # print("Fronthaul Rate", front_rate_user)
            print("Uplink Rate (Mbps):", uplink_rate_access / 10e6)
            print("Average Uplink Rate (Mbps):", np.sum(uplink_rate_access) / (np.count_nonzero(omega_current) * 10e6))

        # local computing delay
        local_delay = np.zeros([M_sim, 1])
        for i in range(M_sim):
            if omega_current[i] == 0:
                local_delay[i, 0] = Task_density[0, i] * Task_size[0, i] / C_user[0, i]

        # uplink delay
        uplink_delay = np.zeros([M_sim, 1])
        for i in range(M_sim):
            if omega_current[i] != 0:
                uplink_delay_list = np.zeros([cluster_size, 1])
                for j in range(cluster_size):
                    uplink_delay_list[j, 0] = Task_size[0, i] / uplink_rate_access[i, 0]
                uplink_delay[i, 0] = np.max(uplink_delay_list)

        # fronthaul delay
        front_delay_matrix = np.zeros([M_sim, cluster_size])
        front_delay = np.zeros([M_sim, 1])
        for i in range(M_sim):
            if omega_current[i] != 0:
                for j in range(N_sim):
                    for k in range(cluster_size):
                        if self.cluster_matrix[i, j] == 1:
                            front_delay_matrix[i, k] = Task_size[0, i] / front_rate_user[i, j]
                front_delay[i, 0] = np.max(front_delay_matrix[i, :])
        # print("Fronthaul Delay", front_delay)

        # processing delay calculation
        # solve convex problem according to Eq. (24)
        task_mat = np.zeros([M_sim, K])
        for i in range(M_sim):
            if omega_current[i] != 0:
                CPU_id = int(omega_current[i] - 1)
                task_mat[i, CPU_id] = Task_size[0, i] * Task_density[0, i]

        # Each CPU solves a resource allocation optimization problem
        actual_C = np.zeros([M_sim, 1])
        for i in range(K):
            serve_user_id = []
            serve_user_task = []
            _local_delay = []
            _front_delay = []
            _uplink_delay = []

            for j in range(M_sim):
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

        actual_process_delay = np.zeros([M_sim, 1])
        for i in range(M_sim):
            if omega_current[i] != 0:
                CPU_id = int(omega_current[i] - 1)
                actual_process_delay[i, 0] = task_mat[i, CPU_id] / actual_C[i, 0]
        # print("Processing Delay:", actual_process_delay)
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
        print("Uplink Delay:", uplink_delay)
        print("Local Delay:", local_delay)
        print("Front Delay:", front_delay)
        print("Edge Processing Delay:", actual_process_delay)
        print("Offloading Delay:", front_delay + uplink_delay + actual_process_delay)
        reward = np.zeros([M_sim, 1])
        for i in range(M_sim):
            reward[i, 0] = - np.sum(np.maximum(local_delay, front_delay + uplink_delay + actual_process_delay)) / M_sim
        if self.step_num > 36000:
            done = [1] * M_sim
        else:
            done = [0] * M_sim

        info = {}
        return tuple(observation), reward, done, info


if __name__ == "__main__":
    # Initialization
    np.random.seed(3)
    M = 50  # number of users
    N = 200  # number of APs
    varsig = 32  # number of antennas of each AP
    K = 4  # number of CPUs
    P_max = 0.1  # maximum transmit power of user / pilot power

    M_sim = 10  # number of users for simulation
    N_sim = 50  # number of APs for simulation
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
    # Task_max_delay = np.random.uniform(2, 5, [1, M])  # task max delay in second
    cluster_size = 5  # AP cluster size

    # edge server parameter
    C_edge = np.random.uniform(10e9, 20e9, [K, 1])  # computing resource of edge server in CPU

    # access channel
    access_chan = np.zeros([M, N, varsig], dtype=complex)  # complex channel
    bandwidth_a = 2e6  # bandwidth of access channel
    kappa_1 = np.random.rand(1, N)  # parameter in Eq. (5)
    kappa_2 = np.random.rand(1, M)  # parameter in Eq. (5)
    f_carrier = 1.9e9  # carrier frequency in Hz
    h_ap = 15  # antenna height of AP
    h_user = 1.65  # antenna height of user
    # L = 46.3 + 33.9 * np.log10(f_carrier / 1000) - 13.82 * np.log10(h_ap) - (
    #        1.11 * np.log10(f_carrier / 1000) - 0.7) * h_user + 1.56 * np.log10(f_carrier / 1000) - 0.8
    L = 140.7
    d_0 = 10  # path-loss distance threshold
    d_1 = 50  # path-loss distance threshold
    PL = np.zeros([M, N])  # path-loss in dB
    beta = np.zeros([M, N])  # large scale fading
    gamma = np.zeros([M, N])
    sigma_s = 8  # standard deviation of shadow fading (dB)
    delta = 0.5  # parameter in Eq. (5)
    mu = np.zeros([M, N])  # shadow fading parameter
    h = np.zeros([M, N, varsig], dtype=complex)  # small scale fading
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
            for k in range(varsig):
                h[i, j, k] = np.random.normal(loc=0, scale=0.5) + 1j * np.random.normal(loc=0, scale=0.5)
                access_chan[i, j, k] = np.sqrt(beta[i, j]) * h[i, j, k]

    # fronthaul channel
    # front_chan = np.zeros([N, K])
    bandwidth_f = 2e9  # bandwidth of fronthaul channel
    epsilon = 6e-4  # blockage density
    p_ap = 1  # transmit power of APs (30 dBm = 1 W)
    alpha_los = 2.5  # path-loss exponent for LOS links
    alpha_nlos = 4  # path-loss exponent for NLOS links
    psi_los = 3  # Nakagami fading parameter for LOS links
    psi_nlos = 2  # Nakagami fading parameter for NLOS links
    noise_front = 1.380649 * 10e-23 * 290 * 9 * bandwidth_f  # fronthaul channel noise variance
    G = np.zeros([N, K])  # random antenna gain
    fai = math.pi / 6  # Main lobe beamwidth
    Gm = 63.1  # Directivity gain of main lobes
    Gs = 0.631  # Directivity gain of side lobes
    Gain = np.array([Gs * Gs, Gm * Gm, Gm * Gs])  # random antenna gain in Eq. (7)
    Gain_pro = np.array([(fai / (2 * math.pi)) ** 2, 2 * fai * (2 * math.pi - fai) / (2 * math.pi) ** 2,
                         ((2 * math.pi - fai) / (2 * math.pi)) ** 2])

    P_los = np.zeros([N, K])  # probability of LOS links
    link_type = np.zeros([N, K])  # type of fronthaul links
    for i in range(N):
        for j in range(K):
            P_los[i, j] = np.exp(-epsilon * distance_matrix_front[i, j] / 1000)
            link_type[i, j] = np.random.choice([0, 1], p=[P_los[i, j], 1 - P_los[i, j]])  # 0 for LOS, 1 for NLOS
            # if link_type[i, j] == 0:  # LOS link
            #     front_chan[i, j] = np.random.gamma(2, 1 / psi_los)  # Nakagami channel gain
            # else:  # NLOS link
            #     front_chan[i, j] = np.random.gamma(2, 1 / psi_nlos)  # Nakagami channel gain
            G[i, j] = np.random.choice(Gain, p=Gain_pro.ravel())
    # pilot assignment
    tau_p = M  # length of pilot symbol
    pilot_matrix = np.zeros([M, tau_p])
    for i in range(M):
        pilot_index = i
        pilot_matrix[i, pilot_index] = 1

    # MMSE channel estimation
    receive_pilot = np.zeros([N, varsig, tau_p], dtype=complex)
    # access_chan_estimate = np.zeros([M, N, varsig], dtype=complex)
    theta = np.zeros([M, N])

    for i in range(M):
        for j in range(N):
            theta[i, j] = tau_p * P_max * (beta[i, j] ** 2) / (tau_p * P_max * beta[i, j] + noise_access)

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
        # print(f"state: {obs} \n")
        print(f"action : {action}, reward : {reward}")
