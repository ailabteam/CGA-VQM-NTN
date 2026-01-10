import pennylane as qml
from pennylane import numpy as np

class CGA_VQC:
    def __init__(self, n_qubits=5, n_layers=2):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        
        # Khởi tạo tham số ngẫu nhiên cho mạch lượng tử (weights)
        self.weights = 0.01 * np.random.randn(n_layers, n_qubits, 3, requires_grad=True)
        # Bias cho output
        self.bias = np.array(0.0, requires_grad=True)

    def circuit(self, weights, features):
        # 1. Encoding: Đưa 5 hệ số CGA vào 5 Qubits
        for i in range(self.n_qubits):
            qml.RY(features[i], wires=i)
        
        # 2. Variational Layers (Ansatz)
        for l in range(self.n_layers):
            # Lớp Entanglement (Vướng víu)
            for i in range(self.n_qubits):
                qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
            
            # Lớp Xoay có tham số (Học đặc trưng)
            for i in range(self.n_qubits):
                qml.RX(weights[l, i, 0], wires=i)
                qml.RY(weights[l, i, 1], wires=i)
                qml.RZ(weights[l, i, 2], wires=i)
        
        # 3. Measurement: Đo giá trị kỳ vọng trên PauliZ
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def get_qnode(self):
        return qml.QNode(self.circuit, self.dev)

# Hàm dự đoán (Classifier wrapper)
def quantum_classifier(qnode, weights, bias, features):
    # Lấy giá trị đo từ mạch
    raw_output = qnode(weights, features)
    # Rút gọn về số lượng class (Ví dụ 3 class thì lấy 3 qubit đầu hoặc sum)
    # Ở đây chúng ta lấy tổng trọng số của các qubit đầu ra để dự đoán
    return np.stack(raw_output) + bias
