import pennylane as qml
from pennylane import numpy as np
import os
import time
from src.data_gen import create_scenario_data
from src.cga_utils import CGAMapper
from src.quantum_model import CGA_VQC, quantum_classifier

# --- Cấu hình Debug ---
SCENARIO = 'rotated'  # Chúng ta soi kịch bản khó nhất
EPOCHS = 100
LR = 0.05
N_SAMPLES = 100 # Tổng 180 mẫu

def cost_fn(w, b, qnode, f, l, nq):
    preds = [quantum_classifier(qnode, w, b, x) for x in f]
    targets = np.zeros((len(l), nq))
    for i, val in enumerate(l): targets[i, val] = 1.0
    return np.mean((np.array(preds) - targets)**2)

def get_acc(w, b, qnode, f, l):
    preds = [quantum_classifier(qnode, w, b, x) for x in f]
    # Dự đoán dựa trên qubit có giá trị cao nhất trong 3 qubit đầu
    pred_labels = np.argmax(np.array(preds)[:, :3], axis=1)
    return np.mean(pred_labels == np.array(l))

def train_debug(mode='cga'):
    print(f"\n{'='*20} DEBUGGING MODE: {mode.upper()} {'='*20}")
    
    # 1. Chuẩn bị dữ liệu
    X_raw, y_all = create_scenario_data(SCENARIO, n_samples=N_SAMPLES)
    
    # Chia tập Train/Test
    split = int(0.8 * len(y_all))
    X_train_raw, X_test_raw = X_raw[:split], X_raw[split:]
    y_train, y_test = y_all[:split], y_all[split:]
    
    mapper = CGAMapper()
    idx = [0, len(X_raw[0])//2, -1]
    nq = 5 if mode == 'cga' else 3
    
    def transform(X):
        sel = X[:, idx, :]
        if mode == 'cga':
            return np.array([np.concatenate([mapper.point_to_cga(p[0],p[1],p[2]) for p in s]) for s in sel], requires_grad=False)
        return np.array([s.flatten() for s in sel], requires_grad=False)

    f_train = transform(X_train_raw)
    f_test = transform(X_test_raw)

    # 2. Khởi tạo mô hình
    model = CGA_VQC(n_qubits=nq)
    qnode = model.get_qnode()
    w = 0.01 * np.random.randn(3, nq, 3, requires_grad=True)
    b = np.array(0.0, requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=LR)
    
    # 3. Vòng lặp huấn luyện có Logs
    print(f"{'Epoch':<8} | {'Loss':<10} | {'Train Acc':<10} | {'Test Acc':<10}")
    print("-" * 45)
    
    for epoch in range(EPOCHS):
        w, b, _, _, _, _ = opt.step(cost_fn, w, b, qnode, f_train, y_train, nq)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            loss = cost_fn(w, b, qnode, f_train, y_train, nq)
            train_acc = get_acc(w, b, qnode, f_train, y_train)
            test_acc = get_acc(w, b, qnode, f_test, y_test)
            print(f"{epoch+1:<8} | {loss:<10.4f} | {train_acc:<10.4f} | {test_acc:<10.4f}")
            
    return test_acc

if __name__ == "__main__":
    # Chạy lần lượt để so sánh
    acc_cga = train_debug(mode='cga')
    acc_raw = train_debug(mode='raw')
    
    print("\n" + "="*50)
    print(f"FINAL DEBUG RESULT for {SCENARIO.upper()}:")
    print(f"CGA Mode Test Accuracy: {acc_cga:.4f}")
    print(f"RAW Mode Test Accuracy: {acc_raw:.4f}")
    print("="*50)
