import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import time, os
from src.data_gen import create_scenario_data
from src.cga_utils import CGAMapper
from src.quantum_model import CGA_VQC, quantum_classifier

# --- CONFIGURATION ---
SCENARIOS = ['clean', 'noisy', 'rotated']
MODES = ['cga', 'raw']
N_TRIALS = 5  
EPOCHS = 50
LR = 0.1
N_SAMPLES = 50

def cost_fn(weights, bias, qnode, features, labels, n_qubits):
    predictions = [quantum_classifier(qnode, weights, bias, f) for f in features]
    targets = np.zeros((len(labels), n_qubits))
    for i, l in enumerate(labels): targets[i, l] = 1.0
    return np.mean((np.array(predictions) - targets) ** 2)

def get_accuracy(weights, bias, qnode, features, labels):
    predictions = [quantum_classifier(qnode, weights, bias, f) for f in features]
    pred_labels = [np.argmax(p[:3]) for p in predictions]
    return np.mean(np.array(pred_labels) == np.array(labels))

def run_single_trial(mode, scenario, seed):
    np.random.seed(seed)
    X_raw, y = create_scenario_data(scenario, n_samples=N_SAMPLES)
    mapper = CGAMapper()
    
    T = X_raw.shape[1]
    indices = [0, T//2, -1]
    X_sel = X_raw[:, indices, :]
    
    if mode == 'cga':
        features = np.array([np.concatenate([mapper.point_to_cga(p[0],p[1],p[2]) for p in s]) for s in X_sel], requires_grad=False)
        n_qubits = 5
    else:
        features = np.array([s.flatten() for s in X_sel], requires_grad=False)
        n_qubits = 3

    model = CGA_VQC(n_qubits=n_qubits, n_points=3)
    weights = 0.01 * np.random.randn(3, n_qubits, 3, requires_grad=True)
    bias = np.array(0.0, requires_grad=True)
    qnode = model.get_qnode()
    opt = qml.AdamOptimizer(stepsize=LR)
    
    for _ in range(EPOCHS):
        weights, bias, _, _, _, _ = opt.step(cost_fn, weights, bias, qnode, features, y, n_qubits)
            
    return get_accuracy(weights, bias, qnode, features, y)

if __name__ == "__main__":
    raw_results = [] # Lưu dữ liệu thô để làm thống kê
    
    print(f"Starting Professional Benchmark on {N_TRIALS} trials...")
    
    for sce in SCENARIOS:
        for mode in MODES:
            print(f"Scenario: {sce.upper()} | Mode: {mode.upper()}")
            for s in range(N_TRIALS):
                acc = run_single_trial(mode, sce, seed=s*42)
                raw_results.append({'Scenario': sce, 'Method': mode, 'Trial': s, 'Accuracy': acc})
                print(f"  > Trial {s}: {acc:.4f}")

    # 1. Chuyển đổi sang DataFrame để xử lý
    df = pd.DataFrame(raw_results)
    stats = df.groupby(['Scenario', 'Method'])['Accuracy'].agg(['mean', 'std']).reset_index()
    
    # 2. Xuất Table LaTeX
    latex_table = stats.pivot(index='Scenario', columns='Method', values=['mean', 'std'])
    print("\n--- LaTeX TABLE CODE ---")
    print(latex_table.to_latex(float_format="%.4f"))
    with open("results/table_results.tex", "w") as f:
        f.write(latex_table.to_latex(float_format="%.4f"))

    # 3. Xuất Figure PDF (Error Bars)
    with PdfPages('results/accuracy_comparison.pdf') as pdf:
        plt.figure(figsize=(10, 6))
        for mode in MODES:
            subset = stats[stats['Method'] == mode]
            plt.errorbar(subset['Scenario'], subset['mean'], yerr=subset['std'], 
                         fmt='o-', capsize=5, label=f"Method: {mode.upper()}")
        
        plt.title("Classification Accuracy: CGA-VQM vs Raw-VQC (Mean ± Std)")
        plt.ylabel("Accuracy")
        plt.ylim(0.3, 1.0)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        pdf.savefig()
        plt.close()

    # 4. Lưu CSV để backup
    stats.to_csv("results/final_stats.csv", index=False)
    print("\n[Done] All figures and tables are saved in results/")
