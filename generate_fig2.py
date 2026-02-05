

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 14:00:11 2025

@author: snagchowdh
"""

import numpy as np
import matplotlib.pyplot as plt

# ----------------- PARAMETERS -----------------
N = 10
dt = 0.01
epsilon1 = 0.01     # upper triangle
epsilon2 = 0.0001   # lower triangle
total_steps = 9000000 #27000000
check_steps = 10000
last_steps = 40000
sigma_noise = 0.0
omega_mean, omega_std = 0.0, 0.0  # can be nonzero
R_threshold = 0.8
# ------------------------------------------------

rng = np.random.default_rng()
theta = rng.uniform(0, 2*np.pi, size=N)
k = rng.uniform(-1.0, 1.0, size=(N, N))
np.fill_diagonal(k, 0.0)
omega = rng.normal(omega_mean, omega_std, size=N)

# ---------- ORDER PARAMETERS ---------- 
R_arr = np.zeros(total_steps)
R2_arr = np.zeros(total_steps)

# ---------- STORE FULL k and theta ----------
k_time = np.zeros((total_steps, N, N))
theta_time = np.zeros((total_steps, N))

# ---------- PRECOMPUTE INDEX MASKS ----------
i_upper, j_upper = np.triu_indices(N, 1)
i_lower, j_lower = np.tril_indices(N, -1)

# ---------------- MAIN SIMULATION ----------------
for t in range(total_steps):
    # --- order parameters ---
    R_arr[t] = np.abs(np.mean(np.exp(1j * theta)))
    R2_arr[t] = np.abs(np.mean(np.exp(2j * theta)))

    # --- theta update ---
    theta_diff = theta[None, :] - theta[:, None]
    coupling = (k * np.sin(theta_diff)).sum(axis=1) / N
    noise = sigma_noise * np.sqrt(dt) * rng.normal(size=N)
    theta += dt * (omega + coupling) + noise
    theta %= 2*np.pi

    # --- k update ---
    k_dot = np.zeros_like(k)
    k_dot[i_upper, j_upper] = -epsilon1 * (k[i_upper, j_upper] + np.sin(theta[i_upper] - theta[j_upper] - np.pi/2))
    k_dot[i_lower, j_lower] = -epsilon2 * (k[i_lower, j_lower] + np.sin(theta[i_lower] - theta[j_lower] + np.pi/2))
    k += dt * k_dot
    np.fill_diagonal(k, 0.0)

    # --- store ---
    k_time[t] = k.copy()
    theta_time[t] = theta.copy()

# ---------------- COMPUTE AVERAGE ORDER PARAMETERS ----------------
R_last = R_arr[-check_steps:].mean()
R2_last = R2_arr[-check_steps:].mean()
print(f"Mean R over last {check_steps} steps: {R_last:.4f}")
print(f"Mean R2 over last {check_steps} steps: {R2_last:.4f}")

# Automatic classification
if R_last > R_threshold:
    state = "IN-PHASE CLUSTER"
elif R_last <= R_threshold and R2_last > R_threshold:
    state = "TWO ANTIPHASE CLUSTERS"
else:
    state = "INCOHERENT"
print("Collective state:", state)

# ---------------- TIME AXIS ----------------
t_axis = np.arange(total_steps) * dt
fontsize = 26  # Increase the font size
ticksize = 28  # Increase the tick label size
linewidth = 4  # Increase the line width
boxlinewidth = 2  # Increase the box line width

# ---------------- PLOT k_ij with subset and full data ----------------
subset_steps = 1800000  # first three subplots show only this range

fig, axes = plt.subplots(4, 1, figsize=(14, 22), sharex=False)  # extra subplot

# 1) R and R2 plot (subset)
axes[0].plot(t_axis[:subset_steps], R_arr[:subset_steps], label="R(t)", linestyle="--", linewidth=linewidth)
axes[0].plot(t_axis[:subset_steps], R2_arr[:subset_steps], label=r"$R_2(t)$", linewidth=linewidth)
axes[0].set_ylabel("Order Parameters", fontsize=fontsize)
axes[0].legend(fontsize=fontsize)
axes[0].tick_params(labelsize=ticksize)
axes[0].text(-0.1, 1.05, '(a)', ha='center', va='center', transform=axes[0].transAxes, fontsize=fontsize)

# 2) Upper-triangle k_ij (subset)
for i, j in zip(i_upper, j_upper):
    axes[1].plot(t_axis[:subset_steps], k_time[:subset_steps, i, j], linewidth=linewidth)
axes[1].set_ylabel("Upper $k_{ij}$", fontsize=fontsize)
axes[1].tick_params(labelsize=ticksize)
axes[1].text(-0.1, 1.05, '(b)', ha='center', va='center', transform=axes[1].transAxes, fontsize=fontsize)

# 3) Lower-triangle k_ij (subset)
for i, j in zip(i_lower, j_lower):
    axes[2].plot(t_axis[:subset_steps], k_time[:subset_steps, i, j], linewidth=linewidth)
axes[2].set_ylabel("Lower $k_{ij}$", fontsize=fontsize)
axes[2].tick_params(labelsize=ticksize)
axes[2].text(-0.1, 1.05, '(c)', ha='center', va='center', transform=axes[2].transAxes, fontsize=fontsize)

# 4) Lower-triangle k_ij (full)
for i, j in zip(i_lower, j_lower):
    axes[3].plot(t_axis, k_time[:, i, j], linewidth=linewidth)
axes[3].set_xlabel(r"$t$", fontsize=fontsize)
axes[3].set_ylabel("Lower $k_{ij}$ (full)", fontsize=fontsize)
axes[3].tick_params(labelsize=ticksize)
axes[3].text(-0.1, 1.05, '(d)', ha='center', va='center', transform=axes[3].transAxes, fontsize=fontsize)

plt.tight_layout()
plt.show()
