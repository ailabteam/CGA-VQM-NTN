import numpy as np
from src.data_gen import generate_base_trajectory, apply_rotation
from src.cga_utils import CGAMapper

def check_invariance():
    mapper = CGAMapper()
    
    # 1. Tạo 1 quỹ đạo Spiral gốc
    traj_original = generate_base_trajectory(label=1, n_points=5)
    
    # 2. Xoay quỹ đạo đó
    traj_rotated = apply_rotation(traj_original)
    
    # 3. Biến đổi cả hai sang CGA
    cga_original = np.array([mapper.point_to_cga(p[0], p[1], p[2]) for p in traj_original])
    cga_rotated = np.array([mapper.point_to_cga(p[0], p[1], p[2]) for p in traj_rotated])
    
    with open("results/cga_check.txt", "w") as f:
        f.write("--- INVARIANCE CHECK: RAW 3D vs CGA 5D ---\n\n")
        f.write("RAW 3D COORDINATES (Original vs Rotated):\n")
        f.write(f"Original[0]: {traj_original[0]}\n")
        f.write(f"Rotated[0]:  {traj_rotated[0]}\n")
        raw_diff = np.linalg.norm(traj_original - traj_rotated)
        f.write(f"Total Raw Difference: {raw_diff:.4f}\n\n")
        
        f.write("CGA 5D COEFFICIENTS (Original vs Rotated):\n")
        f.write(f"CGA Original[0]: {cga_original[0]}\n")
        f.write(f"CGA Rotated[0]:  {cga_rotated[0]}\n")
        
        # Kiểm tra tính bảo toàn của r^2 (nằm trong hệ số e4 và e5)
        # r^2 = x^2 + y^2 + z^2 phải không đổi khi xoay quanh gốc tọa độ
        r2_orig = np.sum(traj_original**2, axis=1)
        r2_rot = np.sum(traj_rotated**2, axis=1)
        f.write(f"\nOriginal r^2: {r2_orig}\n")
        f.write(f"Rotated r^2:  {r2_rot}\n")
        f.write(f"r^2 Difference: {np.linalg.norm(r2_orig - r2_rot):.4e}\n")

    print("Check completed. Please see results/cga_check.txt")

if __name__ == "__main__":
    check_invariance()
