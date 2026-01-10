import clifford as cf
import numpy as np

class CGAMapper:
    def __init__(self):
        # Khởi tạo không gian CGA (4,1)
        self.layout, self.blades = cf.Cl(4, 1)
        self.e1 = self.blades['e1']
        self.e2 = self.blades['e2']
        self.e3 = self.blades['e3']
        self.e4 = self.blades['e4'] # e+
        self.e5 = self.blades['e5'] # e-
        
        # Định nghĩa e_inf và e_o theo chuẩn lý thuyết Conformal
        self.e_inf = self.e4 + self.e5
        self.e_o = 0.5 * (self.e5 - self.e4)

    def point_to_cga(self, x, y, z):
        """Ánh xạ điểm 3D (x,y,z) sang vector 5D trong CGA"""
        p_euclidean = x*self.e1 + y*self.e2 + z*self.e3
        r2 = x**2 + y**2 + z**2
        
        # Công thức: P = x + 0.5*r^2*e_inf + e_o
        p_cga = p_euclidean + 0.5 * r2 * self.e_inf + self.e_o
        
        # Trích xuất 5 hệ số từ multivector p_cga
        # Dựa trên layout của bạn: index 1=e1, 2=e2, 3=e3, 4=e4, 5=e5
        val = p_cga.value
        return np.array([val[1], val[2], val[3], val[4], val[5]])

    def batch_transform(self, X):
        """Biến đổi tập dữ liệu (N_samples, N_points, 3) -> (N_samples, N_points, 5)"""
        N, T, _ = X.shape
        X_cga = np.zeros((N, T, 5))
        for i in range(N):
            for t in range(T):
                X_cga[i, t] = self.point_to_cga(X[i, t, 0], X[i, t, 1], X[i, t, 2])
        return X_cga
