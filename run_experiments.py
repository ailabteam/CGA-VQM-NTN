import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from tqdm import tqdm
import time, os
from src.data_gen import create_scenario_data
from src.cga_utils import CGAMapper
from src.quantum_model import CGA_VQC, quantum_classifier

# --- CONFIG ---
SCENARIOS = ['clean', 'noisy', 'rotated']
MODES = ['cga', 'raw']
N_TRIALS = 5  
EPOCHS = 50
LR = 0.1
N_SAMPLES = 60 # Tổng 180 mẫu

def cost_fn(weights, bias, qnode, features, labels, n_qubits):
    predictions = [quantum_classifier(qnode, weights, bias, f) for f in features]
    targets = np.zeros((len(labels), n_qubits))
    for i, l in enumerate(labels): targets[i, l] = 1.0
    return np.mean((np.array(predictions) - targets) ** 2)

def get_accuracy(weights, bias, qnode, features, labels):
    predictions = [quantum_classifier(qnode, weights, bias, f) for f in features]
    pred_labels = [np.argmax(p[:3]) for p in predictions]
    return np.mean(np.array(pred_labels) == np.array(labels))

def run_single_trial(mode, scenario, seed, pbar):
    np.random.seed(seed)
    X_raw, y_all = create_scenario_data(scenario, n_samples=N_SAMPLES)
    
    # SPLIT 80/20
    split = int(0.8 * len(y_all))
    X_train_raw, X_test_raw = X_raw[:split], X_raw[split:]
    y_train, y_test = y_all[:split], y_all[split:]
    
    mapper = CGAMapper()
    T = X_raw.shape[1]
    indices = [0, T//2, -1]
    
    def transform(X):
        X_s = X[:, indices, :]
        if mode == 'cga':
            return np.array([np.concatenate([mapper.point_to_cga(p[0],p[1],p[2]) for p in s]) for s in X_s], requires_grad=False)
        return np.array([s.flatten() for s in X_s], requires_grad=False)

    feat_train = transform(X_train_raw)
    feat_test = transform(X_test_raw)
    n_q = 5 if mode == 'cga' else 3

    model = CGA_VQC(n_qubits=n_q, n_points=3)
    qnode = model.get_qnode()
    w = 0.01 * np.random.randn(3, n_q, 3, requires_grad=True)
    b = np.array(0.0, requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=LR)
    
    for _ in range(EPOCHS):
        w, b, _, _, _, _ = opt.step(cost_fn, w, b, qnode, feat_train, y_train, n_q)
        pbar.update(1)
            
    return get_accuracy(w, b, qnode, feat_test, y_test)

if __name__ == "__main__":
    if not os.path.exists('results'): os.makedirs('results')
    raw_results = []
    total_steps = len(SCENARIOS) * len(MODES) * N_TRIALS * EPOCHS
    
    with tqdm(total=total_steps, desc="Global Progress") as pbar:
        for sce in SCENARIOS:
            for mode in MODES:
                for s in range(N_TRIALS):
                    acc = run_single_trial(mode, sce, seed=s*42, pbar=pbar)
                    raw_results.append({'Scenario': sce, 'Method': mode, 'Trial': s, 'Test_Acc': acc})
                    tqdm.write(f" Done: {sce.upper()} | {mode.upper()} | Trial {s} | Test Acc: {acc:.4f}")

    # Xử lý kết quả
    df = pd.DataFrame(raw_results)
    stats = df.groupby(['Scenario', 'Method'])['Test_Acc'].agg(['mean', 'std']).reset_index()
    
    # 1. Xuất LaTeX
    latex_table = stats.pivot(index='Scenario', columns='Method', values=['mean', 'std'])
    with open("results/table_results.tex", "w") as f:
        f.write(latex_table.to_latex(float_format="%.4f"))

    # 2. Vẽ Figure
    with PdfPages('results/accuracy_comparison.pdf') as pdf:
        plt.figure(figsize=(10, 6))
        for mode in MODES:
            subset = stats[stats['Method'] == mode]
            plt.errorbar(subset['Scenario'], subset['mean'], yerr=subset['std'], fmt='o-', capsize=5, label=mode.upper())
        plt.title("Test Accuracy: CGA-VQM vs Raw-VQC (Mean +/- Std)")
        plt.ylabel("Accuracy"); plt.ylim(0.2, 1.1); plt.legend(); plt.grid(True)
        pdf.savefig()
    
    stats.to_csv("results/final_stats.csv", index=False)
    print("\n--- FINAL STATISTICS ---")
    print(stats)
