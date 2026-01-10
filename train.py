import pennylane as qml
from pennylane import numpy as np
from src.data_gen import create_scenario_data
from src.cga_utils import CGAMapper
from src.quantum_model import CGA_VQC, quantum_classifier
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import time

# Cấu hình
N_SAMPLES = 40 
EPOCHS = 30
LEARNING_RATE = 0.1
SCENARIO = 'rotated' 

def cost_fn(weights, bias, qnode, features, labels, n_qubits):
    predictions = [quantum_classifier(qnode, weights, bias, f) for f in features]
    
    # Tạo targets khớp với số Qubits của mạch hiện tại
    targets = np.zeros((len(labels), n_qubits))
    for i, l in enumerate(labels):
        targets[i, l] = 1.0 # Class 0, 1, 2 tương ứng với Qubit 0, 1, 2
        
    loss = np.mean((np.array(predictions) - targets) ** 2)
    return loss

def accuracy(weights, bias, qnode, features, labels):
    predictions = [quantum_classifier(qnode, weights, bias, f) for f in features]
    # Dự đoán là index của qubit có giá trị cao nhất trong 3 qubit đầu tiên (đại diện cho 3 lớp)
    pred_labels = [np.argmax(p[:3]) for p in predictions]
    return np.mean(np.array(pred_labels) == np.array(labels))

def train_model(mode='cga'):
    X_raw, y = create_scenario_data(SCENARIO, n_samples=N_SAMPLES)
    mapper = CGAMapper()
    X_avg = np.mean(X_raw, axis=1) 
    
    if mode == 'cga':
        # Ép kiểu về float và không cần tính đạo hàm cho bước CGA
        features = np.array([mapper.point_to_cga(p[0], p[1], p[2]) for p in X_avg], requires_grad=False)
        n_qubits = 5
    else:
        features = np.array(X_avg, requires_grad=False)
        n_qubits = 3

    model = CGA_VQC(n_qubits=n_qubits, n_layers=2)
    qnode = model.get_qnode()
    weights = model.weights
    bias = model.bias
    
    opt = qml.AdamOptimizer(stepsize=LEARNING_RATE)
    loss_history = []
    acc_history = []
    
    print(f"\nTraining Mode: {mode.upper()}")
    for it in range(EPOCHS):
        # Truyền thêm n_qubits vào cost_fn qua opt.step
        weights, bias, _, _, _, _ = opt.step(cost_fn, weights, bias, qnode, features, y, n_qubits)
        
        current_loss = cost_fn(weights, bias, qnode, features, y, n_qubits)
        current_acc = accuracy(weights, bias, qnode, features, y)
        
        loss_history.append(current_loss)
        acc_history.append(current_acc)
        
        if (it + 1) % 5 == 0:
            print(f"Iter {it+1:3d} | Loss: {current_loss:.4f} | Acc: {current_acc:.4f}")
            
    return loss_history, acc_history

if __name__ == "__main__":
    start_time = time.time()
    
    loss_cga, acc_cga = train_model(mode='cga')
    loss_raw, acc_raw = train_model(mode='raw')
    
    duration = (time.time() - start_time) / 60
    
    # Xuất PDF và TXT (giữ nguyên như cũ)
    with PdfPages('results/training_results.pdf') as pdf:
        plt.figure(figsize=(10, 5))
        plt.plot(loss_cga, 'b-', label='Proposed: CGA-VQM (5 Qubits)')
        plt.plot(loss_raw, 'r--', label='Baseline: Raw-VQC (3 Qubits)')
        plt.title(f"Training Loss Comparison ({SCENARIO.capitalize()} Scenario)")
        plt.xlabel("Epochs"); plt.ylabel("Loss"); plt.legend(); pdf.savefig()
        
        plt.figure(figsize=(10, 5))
        plt.plot(acc_cga, 'b-', label='Proposed: CGA-VQM')
        plt.plot(acc_raw, 'r--', label='Baseline: Raw-VQC')
        plt.title(f"Training Accuracy Comparison ({SCENARIO.capitalize()} Scenario)")
        plt.xlabel("Epochs"); plt.ylabel("Accuracy"); plt.legend(); pdf.savefig()
        
    with open("results/final_metrics.txt", "w") as f:
        f.write(f"Scenario: {SCENARIO}\n")
        f.write(f"Final Accuracy CGA: {acc_cga[-1]:.4f}\n")
        f.write(f"Final Accuracy Raw: {acc_raw[-1]:.4f}\n")

    print(f"\nTraining completed in {duration:.2f} mins.")
