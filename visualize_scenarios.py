import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from src.data_gen import create_scenario_data

def plot_to_pdf(filename="results/data_scenarios.pdf"):
    scenarios = ['clean', 'noisy', 'rotated']
    labels = ['Linear', 'Spiral', 'Zigzag']
    
    with PdfPages(filename) as pdf:
        for sce in scenarios:
            X, y = create_scenario_data(sce, n_samples=1) # Lấy 1 mẫu mỗi loại để vẽ
            
            fig = plt.figure(figsize=(15, 5))
            fig.suptitle(f"Scenario: {sce.upper()}", fontsize=16)
            
            for i in range(3): # Vẽ 3 class
                ax = fig.add_subplot(1, 3, i+1, projection='3d')
                idx = i # Vì mỗi class lấy 1 mẫu nên index khớp luôn
                traj = X[idx]
                ax.plot(traj[:,0], traj[:,1], traj[:,2], label=labels[i])
                ax.set_title(labels[i])
                ax.legend()
            
            pdf.savefig(fig)
            plt.close()
    print(f"Visualization saved to {filename}")

if __name__ == "__main__":
    import os
    if not os.path.exists('results'): os.makedirs('results')
    plot_to_pdf()
