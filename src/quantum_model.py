import pennylane as qml
from pennylane import numpy as np

class CGA_VQC:
    def __init__(self, n_qubits=5, n_points=3):
        self.n_qubits = n_qubits
        self.n_points = n_points 
        # CHỈNH SỬA: Quay lại default.qubit để tránh lỗi Illegal Instruction
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        
    def circuit(self, weights, features):
        for p in range(self.n_points):
            f_p = features[p*self.n_qubits : (p+1)*self.n_qubits]
            for i in range(self.n_qubits): qml.RY(f_p[i], wires=i)
            for i in range(self.n_qubits): qml.CNOT(wires=[i, (i+1)%self.n_qubits])
            for i in range(self.n_qubits):
                qml.RX(weights[p,i,0], wires=i)
                qml.RY(weights[p,i,1], wires=i)
                qml.RZ(weights[p,i,2], wires=i)
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def get_qnode(self): return qml.QNode(self.circuit, self.dev)

def quantum_classifier(qnode, weights, bias, features):
    return np.stack(qnode(weights, features)) + bias
