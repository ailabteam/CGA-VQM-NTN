import numpy as np
from scipy.spatial.transform import Rotation as R

def generate_base_trajectory(label, n_points=50):
    t = np.linspace(0, 1, n_points)
    if label == 0: x, y, z = t, t, t
    elif label == 1: x, y, z = np.sin(5*t), np.cos(5*t), t
    else: x, y, z = t, np.abs(np.sin(10*t)), np.sin(5*t)
    return np.stack([x, y, z], axis=1)

def create_scenario_data(scenario='clean', n_samples=60):
    X, y = [], []
    for label in range(3):
        for _ in range(n_samples):
            traj = generate_base_trajectory(label)
            if scenario == 'noisy':
                traj = traj + np.random.normal(0, 0.15, traj.shape)
            elif scenario == 'rotated':
                # QUAN TRỌNG: Mỗi mẫu một góc xoay ngẫu nhiên khác nhau
                rot = R.from_euler('xyz', np.random.uniform(0, 360, 3), degrees=True)
                traj = rot.apply(traj)
                traj = traj + np.random.normal(0, 0.02, traj.shape)
            X.append(traj)
            y.append(label)
    X, y = np.array(X), np.array(y)
    indices = np.random.permutation(len(y))
    return X[indices], y[indices]
