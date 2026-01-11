import pennylane as qml
from pennylane import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from src.data_gen import create_scenario_data
from src.cga_utils import CGAMapper
from src.quantum_model import CGA_VQC, quantum_classifier

# Config
SCENARIOS = ['clean', 'noisy', 'rotated']
MODES = ['cga', 'raw']
N_TRIALS = 3 # Để nhanh bạn có thể để 3, khi chốt bài hãy để 5
EPOCHS = 40
LR = 0.1

def cost_fn(w, b, qnode, f, l, nq):
    preds = [quantum_classifier(qnode, w, b, x) for x in f]
    targets = np.zeros((len(l), nq))
    for i, val in enumerate(l): targets[i, val] = 1.0
    return np.mean((np.array(preds) - targets)**2)

def get_acc(w, b, qnode, f, l):
    preds = [quantum_classifier(qnode, w, b, x) for x in f]
    return np.mean(np.argmax(np.array(preds)[:,:3], axis=1) == np.array(l))

def run_trial(mode, scenario, seed, pbar):
    np.random.seed(seed)
    X, y = create_scenario_data(scenario, n_samples=60)
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
    res = []
    with tqdm(total=len(SCENARIOS)*len(MODES)*N_TRIALS*EPOCHS) as pbar:
        for sce in SCENARIOS:
            for m in MODES:
                for t in range(N_TRIALS):
                    acc = run_trial(m, sce, t*10, pbar)
                    res.append({'Scenario': sce, 'Method': m, 'Acc': acc})
                    tqdm.write(f"{sce} | {m} | Acc: {acc:.4f}")
    df = pd.DataFrame(res)
    stats = df.groupby(['Scenario', 'Method'])['Acc'].agg(['mean', 'std']).reset_index()
    print("\n--- FINAL TEST STATISTICS ---")
    print(stats)
    stats.to_csv("results/final_stats.csv", index=False)
