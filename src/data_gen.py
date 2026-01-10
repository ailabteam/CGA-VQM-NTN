import numpy as np
import os
from scipy.spatial.transform import Rotation as R

def generate_base_trajectory(label, n_points=50):
    t = np.linspace(0, 1, n_points)
    if label == 0: # Linear (Vệ tinh bay thẳng)
        x, y, z = t, t, t
    elif label == 1: # Spiral (UAV đổi độ cao)
        x, y, z = np.sin(5 * t), np.cos(5 * t), t
    elif label == 2: # Zigzag (Maneuvering - Đổi hướng)
        x = t
        y = np.abs(np.sin(10 * t))
        z = np.sin(5 * t)
    return np.stack([x, y, z], axis=1)

def apply_noise(traj, std=0.05):
    return traj + np.random.normal(0, std, traj.shape)

def apply_rotation(traj):
    # Tạo một góc xoay ngẫu nhiên quanh các trục
    random_rot = R.from_euler('xyz', np.random.uniform(0, 360, 3), degrees=True)
    return random_rot.apply(traj)

def create_scenario_data(scenario='clean', n_samples=30):
    X, y = [], []
    for label in range(3):
        for _ in range(n_samples):
            traj = generate_base_trajectory(label)
            if scenario == 'noisy':
                traj = apply_noise(traj, std=0.08)
            elif scenario == 'rotated':
                traj = apply_rotation(traj)
            X.append(traj)
            y.append(label)
    return np.array(X), np.array(y)
