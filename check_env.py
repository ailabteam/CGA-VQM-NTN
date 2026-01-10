import clifford as cf
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

def test():
    print("--- Checking Environment ---")
    
    # 1. Test CGA (4,1)
    # Khởi tạo không gian Minkowski cơ sở
    layout, blades = cf.Cl(4, 1)
    
    # Trong CGA, chúng ta thường định nghĩa:
    # e_plus = e4, e_minus = e5
    # n = e_inf = e_plus + e_minus
    # n_bar = e_o = 0.5 * (e_minus - e_plus)
    try:
        e4 = blades['e4']
        e5 = blades['e5']
        e_inf = e4 + e5
        e_o = 0.5 * (e5 - e4)
        print(f"CGA (4,1) initialized successfully.")
        print(f"Basis blades available: {list(blades.keys())}")
    except KeyError as e:
        print(f"Error accessing blades: {e}")

    # 2. Test Quantum with PennyLane
    dev = qml.device("default.qubit", wires=5)
    
    @qml.qnode(dev)
    def circuit(inputs):
        for i in range(5):
            qml.RY(inputs[i], wires=i)
        return [qml.expval(qml.PauliZ(i)) for i in range(5)]
    
    test_input = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    output = circuit(test_input)
    print(f"Quantum Circuit Output: {output}")

    # 3. Test PDF export
    if not os.path.exists('results'):
        os.makedirs('results')
        
    with PdfPages('results/test_plot.pdf') as pdf:
        plt.figure(figsize=(6,4))
        plt.plot(test_input, 'ro-', label='Test Data')
        plt.title("Environment Test Plot")
        plt.legend()
        pdf.savefig()
        plt.close()
    print("Test PDF saved in results/test_plot.pdf")

if __name__ == "__main__":
    test()
