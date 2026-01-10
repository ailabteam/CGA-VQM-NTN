import clifford as cf
import numpy as np

class CGAMapper:
    def __init__(self):
        # Khởi tạo không gian CGA (4,1)
        self.layout, self.blades = cf.Cl(4, 1)
        self.e1 = self.blades['e1']
        self.e2 = self.blades['e2']
        self.e3 = self.blades['e3']
        self.e4 = self.blades['e4']
        self.e5 = self.blades['e5']
        
        self.e_inf = self.e4 + self.e5
        self.e_o = 0.5 * (self.e5 - self.e4)

    def point_to_cga(self, x, y, z):
        """Ánh xạ điểm 3D sang vector 5D trong CGA"""
        # CHỈNH SỬA TẠI ĐÂY: Ép kiểu x, y, z về float thuần túy để tránh lỗi PennyLane Tensor
        x_f, y_f, z_f = float(x), float(y), float(z)
        
        p_euclidean = x_f*self.e1 + y_f*self.e2 + z_f*self.e3
        r2 = x_f**2 + y_f**2 + z_f**2
        
        p_cga = p_euclidean + 0.5 * r2 * self.e_inf + self.e_o
        
        # p_cga.value là một mảng numpy chứa các hệ số của multivector
        # Theo kết quả check_env: index 1=e1, 2=e2, 3=e3, 4=e4, 5=e5
        val = p_cga.value
        return np.array([val[1], val[2], val[3], val[4], val[5]])

    def batch_transform(self, X):
        N, T, _ = X.shape
        X_cga = np.zeros((N, T, 5))
        for i in range(N):
            for t in range(T):
                X_cga[i, t] = self.point_to_cga(X[i, t, 0], X[i, t, 1], X[i, t, 2])
        return X_cga
