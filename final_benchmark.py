import pennylane as qml
from pennylane import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from src.data_gen import create_scenario_data
from src.cga_utils import CGAMapper
from src.quantum_model import CGA_VQC, quantum_classifier

# --- CONFIGURATION (Chốt thông số từ Debug thành công) ---
SCENARIOS = ['rotated', 'noisy', 'clean']
MODES = ['raw', 'cga']
N_TRIALS = 1  # Số lần chạy để tính Mean +/- Std
N_SAMPLES = 100
EPOCHS = 60    # Tăng lên 60 để hội tụ sâu hơn
LR = 0.05

def cost_fn(w, b, qnode, f, l, nq):
    preds = [quantum_classifier(qnode, w, b, x) for x in f]
    targets = np.zeros((len(l), nq))
    for i, val in enumerate(l): targets[i, val] = 1.0
    return np.mean((np.array(preds) - targets)**2)

def get_acc(w, b, qnode, f, l):
    preds = [quantum_classifier(qnode, w, b, x) for x in f]
    return np.mean(np.argmax(np.array(preds)[:,:3], axis=1) == np.array(l))

def run_experiment(mode, scenario, seed, pbar):
    np.random.seed(seed)
    X, y = create_scenario_data(scenario, n_samples=N_SAMPLES)
    split = int(0.8 * len(y))
    X_tr, X_te, y_tr, y_te = X[:split], X[split:], y[:split], y[split:]

    mapper = CGAMapper()
    idx = [0, len(X[0])//2, -1]
    nq = 5 if mode == 'cga' else 3

    def prep(data):
        sel = data[:, idx, :]
        if mode == 'cga':
            return np.array([np.concatenate([mapper.point_to_cga(p[0],p[1],p[2]) for p in s]) for s in sel], requires_grad=False)
        return np.array([s.flatten() for s in sel], requires_grad=False)

    f_tr, f_te = prep(X_tr), prep(X_te)
    model = CGA_VQC(n_qubits=nq)
    qnode = model.get_qnode()
    w = 0.01 * np.random.randn(3, nq, 3, requires_grad=True)
    b = np.array(0.0, requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=LR)

    for _ in range(EPOCHS):
        w, b, _, _, _, _ = opt.step(cost_fn, w, b, qnode, f_tr, y_tr, nq)
        pbar.update(1)

    return get_acc(w, b, qnode, f_te, y_te)

if __name__ == "__main__":
    if not os.path.exists('results'): os.makedirs('results')
    all_results = []
    total_steps = len(SCENARIOS) * len(MODES) * N_TRIALS * EPOCHS

    print(f"Starting Final Benchmark: 3 Scenarios, 2 Modes, {N_TRIALS} Trials.")

    with tqdm(total=total_steps, desc="Global Progress") as pbar:
        for sce in SCENARIOS:
            for m in MODES:
                for t in range(N_TRIALS):
                    acc = run_experiment(m, sce, t*100, pbar)
                    all_results.append({'Scenario': sce, 'Method': m, 'Trial': t, 'Acc': acc})
                    tqdm.write(f"Done: {sce} | {m} | Trial {t} | Acc: {acc:.4f}")

    # Xử lý thống kê
    df = pd.DataFrame(all_results)
    stats = df.groupby(['Scenario', 'Method'])['Acc'].agg(['mean', 'std']).reset_index()

    # Xuất bảng LaTeX
    latex_table = stats.pivot(index='Scenario', columns='Method', values=['mean', 'std'])
    print("\n--- FINAL LATEX TABLE ---")
    print(latex_table.to_latex(float_format="%.4f"))

    with open("results/final_table.tex", "w") as f:
        f.write(latex_table.to_latex(float_format="%.4f"))

    # Vẽ Error Bar Figure
    with PdfPages('results/final_comparison.pdf') as pdf:
        plt.figure(figsize=(10, 6))
        for m in MODES:
            subset = stats[stats['Method'] == m]
            plt.errorbar(subset['Scenario'], subset['mean'], yerr=subset['std'],
                         fmt='o-', capsize=5, label=f"Method: {m.upper()}")
        plt.title("Test Accuracy Comparison: CGA-VQM vs Raw-VQC")
        plt.ylabel("Accuracy")
        plt.ylim(0.2, 1.1)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        pdf.savefig()

    print("\n[Success] All results are saved in results/ folder.")
