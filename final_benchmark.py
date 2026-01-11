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

# --- CONFIG ---
SCENARIOS = ['rotated', 'noisy', 'clean']
MODES = ['cga', 'raw']
N_TRIALS = 3
N_SAMPLES = 100 
EPOCHS = 60
LR = 0.05

def cost_fn(w, b, qnode, f, l, nq):
    preds = [quantum_classifier(qnode, w, b, x) for x in f]
    targets = np.zeros((len(l), nq))
    for i, val in enumerate(l): targets[i, val] = 1.0
    return np.mean((np.array(preds) - targets)**2)

def get_acc_and_preds(w, b, qnode, f, l):
    preds = [quantum_classifier(qnode, w, b, x) for x in f]
    pred_labels = np.argmax(np.array(preds)[:,:3], axis=1)
    acc = np.mean(pred_labels == np.array(l))
    return acc, pred_labels

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
    
    loss_history = []
    for it in range(EPOCHS):
        w, b, _, _, _, _ = opt.step(cost_fn, w, b, qnode, f_tr, y_tr, nq)
        if it % 5 == 0:
            loss_history.append(cost_fn(w, b, qnode, f_tr, y_tr, nq))
        pbar.update(1)
        
    acc, y_pred = get_acc_and_preds(w, b, qnode, f_te, y_te)
    return acc, loss_history, y_te, y_pred

if __name__ == "__main__":
    if not os.path.exists('results'): os.makedirs('results')
    all_results = []
    histories = {m: [] for m in MODES} # Lưu loss để plot
    
    with tqdm(total=len(SCENARIOS)*len(MODES)*N_TRIALS*EPOCHS) as pbar:
        for sce in SCENARIOS:
            for m in MODES:
                trial_accs = []
                for t in range(N_TRIALS):
                    acc, loss_hist, y_true, y_pred = run_experiment(m, sce, t*42, pbar)
                    all_results.append({'Scenario': sce, 'Method': m, 'Acc': acc})
                    if sce == 'rotated' and t == 0: # Lưu history của kịch bản khó nhất
                        histories[m] = loss_hist
                    tqdm.write(f"Done: {sce} | {m} | Trial {t} | Acc: {acc:.4f}")

    # 1. Plot Loss Convergence (Quan trọng cho Paper)
    with PdfPages('results/loss_convergence.pdf') as pdf:
        plt.figure(figsize=(8, 5))
        plt.plot(range(0, EPOCHS, 5), histories['cga'], 'b-o', label='Proposed: CGA-VQM')
        plt.plot(range(0, EPOCHS, 5), histories['raw'], 'r--s', label='Baseline: Raw-VQC')
        plt.title("Learning Convergence (Rotated Scenario)")
        plt.xlabel("Epochs")
        plt.ylabel("Training Loss")
        plt.legend()
        plt.grid(True)
        pdf.savefig()

    # 2. Xuất bảng thống kê Accuracy (Mean +/- Std)
    df = pd.DataFrame(all_results)
    stats = df.groupby(['Scenario', 'Method'])['Acc'].agg(['mean', 'std']).reset_index()
    print("\n--- FINAL STATISTICS ---\n", stats)
    stats.to_csv("results/final_stats.csv", index=False)
