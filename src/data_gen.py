import numpy as np
from scipy.spatial.transform import Rotation as R

def generate_base_trajectory(label, n_points=50):
    t = np.linspace(0, 1, n_points)
    if label == 0: # Linear
        x, y, z = t, t, t
    elif label == 1: # Spiral
        x, y, z = np.sin(5 * t), np.cos(5 * t), t
    elif label == 2: # Zigzag
        x = t
        y = np.abs(np.sin(10 * t))
        z = np.sin(5 * t)
    return np.stack([x, y, z], axis=1)

def apply_noise(traj, std=0.1): # Tăng nhiễu mặc định
    return traj + np.random.normal(0, std, traj.shape)

def apply_rotation(traj):
    # Tạo góc xoay ngẫu nhiên khác nhau cho mỗi lần gọi
    random_rot = R.from_euler('xyz', np.random.uniform(0, 360, 3), degrees=True)
    return random_rot.apply(traj)

def create_scenario_data(scenario='clean', n_samples=50):
    X, y = [], []
    for label in range(3):
        for _ in range(n_samples):
            traj = generate_base_trajectory(label)
            if scenario == 'noisy':
                traj = apply_noise(traj, std=0.15)
            elif scenario == 'rotated':
                traj = apply_rotation(traj)
                traj = apply_noise(traj, std=0.05) # Nhiễu nhẹ trong lúc xoay
            X.append(traj)
            y.append(label)
    
    X, y = np.array(X), np.array(y)
    # Shuffle dữ liệu ngay từ đầu
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    return X[indices], y[indices]
