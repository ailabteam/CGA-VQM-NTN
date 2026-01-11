import clifford as cf
import numpy as np

class CGAMapper:
    def __init__(self):
        self.layout, self.blades = cf.Cl(4, 1)
        self.e1, self.e2, self.e3 = self.blades['e1'], self.blades['e2'], self.blades['e3']
        self.e4, self.e5 = self.blades['e4'], self.blades['e5']
        self.e_inf, self.e_o = (self.e4 + self.e5), 0.5*(self.e5 - self.e4)

    def point_to_cga(self, x, y, z):
        xf, yf, zf = float(x), float(y), float(z)
        p_cga = xf*self.e1 + yf*self.e2 + zf*self.e3 + 0.5*(xf**2+yf**2+zf**2)*self.e_inf + self.e_o
        val = p_cga.value
        return np.array([val[1], val[2], val[3], val[4], val[5]])
