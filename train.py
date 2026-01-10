import pennylane as qml
from pennylane import numpy as np
from src.data_gen import create_scenario_data
from src.cga_utils import CGAMapper
from src.quantum_model import CGA_VQC, quantum_classifier
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import time

# Cấu hình chung
N_SAMPLES = 40  # Số mẫu mỗi lớp (tổng 120)
EPOCHS = 30
LEARNING_RATE = 0.1
SCENARIO = 'rotated' # Thử thách lớn nhất cho tọa độ thô

def cost_fn(weights, bias, qnode, features, labels):
    # labels ở dạng [0, 1, 2]. Chúng ta dùng Square Loss đơn giản
    predictions = [quantum_classifier(qnode, weights, bias, f) for f in features]
    
    # Biến đổi label thành one-hot-like cho 3 class đầu tiên
    targets = np.zeros((len(labels), 5))
    for i, l in enumerate(labels):
        targets[i, l] = 1.0 # Target là +1 tại qubit tương ứng
        
    loss = np.mean((np.array(predictions) - targets) ** 2)
    return loss

def accuracy(weights, bias, qnode, features, labels):
    predictions = [quantum_classifier(qnode, weights, bias, f) for f in features]
    # Class dự đoán là index của qubit có giá trị đo lớn nhất
    pred_labels = [np.argmax(p[:3]) for p in predictions]
    return np.mean(np.array(pred_labels) == np.array(labels))

def train_model(mode='cga'):
    # 1. Chuẩn bị dữ liệu
    X_raw, y = create_scenario_data(SCENARIO, n_samples=N_SAMPLES)
    mapper = CGAMapper()
    
    # Tiền xử lý: Lấy trung bình tọa độ của mỗi quỹ đạo để tạo 1 feature vector duy nhất
    # (Để đơn giản hóa cho mô phỏng lượng tử)
    X_avg = np.mean(X_raw, axis=1) 
    
    if mode == 'cga':
        features = np.array([mapper.point_to_cga(p[0], p[1], p[2]) for p in X_avg])
        n_qubits = 5
    else: # raw
        features = X_avg
        n_qubits = 3

    # 2. Khởi tạo Model
    model = CGA_VQC(n_qubits=n_qubits, n_layers=2)
    qnode = model.get_qnode()
    weights = model.weights
    bias = model.bias
    
    # 3. Optimization
    opt = qml.AdamOptimizer(stepsize=LEARNING_RATE)
    loss_history = []
    acc_history = []
    
    print(f"\nTraining Mode: {mode.upper()}")
    for it in range(EPOCHS):
        weights, bias, _, _, _ = opt.step(cost_fn, weights, bias, qnode, features, y)
        
        current_loss = cost_fn(weights, bias, qnode, features, y)
        current_acc = accuracy(weights, bias, qnode, features, y)
        
        loss_history.append(current_loss)
        acc_history.append(current_acc)
        
        if (it + 1) % 5 == 0:
            print(f"Iter {it+1:3d} | Loss: {current_loss:.4f} | Acc: {current_acc:.4f}")
            
    return loss_history, acc_history

if __name__ == "__main__":
    start_time = time.time()
    
    # Huấn luyện cả 2 để so sánh
    loss_cga, acc_cga = train_model(mode='cga')
    loss_raw, acc_raw = train_model(mode='raw')
    
    duration = (time.time() - start_time) / 60
    
    # Lưu kết quả vào PDF
    with PdfPages('results/training_results.pdf') as pdf:
        # Plot Loss
        plt.figure(figsize=(10, 5))
        plt.plot(loss_cga, 'b-', label='Proposed: CGA-VQM (5 Qubits)')
        plt.plot(loss_raw, 'r--', label='Baseline: Raw-VQC (3 Qubits)')
        plt.title(f"Training Loss Comparison ({SCENARIO.capitalize()} Scenario)")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        pdf.savefig()
        
        # Plot Accuracy
        plt.figure(figsize=(10, 5))
        plt.plot(acc_cga, 'b-', label='Proposed: CGA-VQM')
        plt.plot(acc_raw, 'r--', label='Baseline: Raw-VQC')
        plt.title(f"Training Accuracy Comparison ({SCENARIO.capitalize()} Scenario)")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        pdf.savefig()
        
    # Lưu thông số text
    with open("results/final_metrics.txt", "w") as f:
        f.write(f"Scenario: {SCENARIO}\n")
        f.write(f"Training Duration: {duration:.2f} minutes\n")
        f.write(f"Final Accuracy CGA: {acc_cga[-1]:.4f}\n")
        f.write(f"Final Accuracy Raw: {acc_raw[-1]:.4f}\n")

    print(f"\nTraining completed in {duration:.2f} mins. Results saved to results/")
