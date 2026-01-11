# CGA-VQM-NTN: Conformal Geometric Algebra-based Variational Quantum Mapping for NTN Trajectory Classification

Official implementation of the paper: **"A Novel Geometric Algebra-based Variational Quantum Mapping for Non-Terrestrial Feature Representation"** submitted to the *2026 11th International Conference on Intelligent Information Technology (ICIIT 2026)*.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PennyLane](https://img.shields.io/badge/PennyLane-latest-green.svg)](https://pennylane.ai/)
[![Clifford](https://img.shields.io/badge/Clifford-CGA-orange.svg)](https://clifford.readthedocs.io/)

## 📝 Abstract
Non-Terrestrial Networks (NTN) require robust spatial feature representation for high-speed nodes like UAVs and LEO satellites. Traditional Euclidean-based learning often fails under complex 3D rotations. We propose **CGA-VQM**, a framework combining **Conformal Geometric Algebra (CGA)** and **Variational Quantum Circuits (VQC)**. By embedding trajectories into a 5D Conformal space, we extract invariant geometric signatures that significantly enhance quantum classification performance, especially in spatially dynamic environments.

## 🚀 Key Features
- **Geometric Invariance:** Leveraging CGA ($R^{4,1}$) to maintain structural integrity under 3D rotations.
- **Quantum Data Re-uploading:** Utilizing multi-layer VQCs to process temporal trajectory points.
- **Scientific Validation:** Comparative analysis across Clean, Noisy, and Rotated scenarios with Mean $\pm$ Std statistics.

## 📂 Repository Structure
```text
CGA-VQM-NTN/
├── src/
│   ├── cga_utils.py       # CGA 5D transformation logic
│   ├── data_gen.py        # NTN Trajectory generation (Clean/Noisy/Rotated)
│   └── quantum_model.py   # VQC architecture with Data Re-uploading
├── results/               # Generated figures and tables for the paper
│   ├── final_table.tex    # LaTeX source for performance table
│   ├── loss_convergence.pdf # Learning curves (CGA vs Raw)
│   ├── final_comparison.pdf # Error bar charts
│   └── data_scenarios.pdf # Visualization of NTN trajectories
├── final_benchmark.py     # Main script for full statistical evaluation
├── debug_training.py      # Script for step-by-step training analysis
└── check_env.py           # Environment and CGA-Quantum verification
```

## 🛠️ Installation & Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/ailabteam/CGA-VQM-NTN.git
   cd CGA-VQM-NTN
   ```
2. **Create Conda Environment:**
   ```bash
   conda create -n cga_quantum python=3.10 -y
   conda activate cga_quantum
   pip install numpy scipy matplotlib pandas tqdm clifford pennylane Jinja2
   ```

## 📊 Running Experiments
To reproduce the results presented in the paper:
```bash
# Run the full benchmark (3 scenarios x 2 modes x 5 trials)
python final_benchmark.py
```
*Note: The results will be automatically saved in the `results/` folder in both `.tex` and `.pdf` formats.*

## 📈 Main Results
Our framework demonstrates a significant performance gap in the **Rotated** scenario (the most challenging case):

| Scenario | Raw-VQC (Baseline) | CGA-VQM (Proposed) | Improvement |
| :--- | :---: | :---: | :---: |
| **Clean** | 1.0000 | 1.0000 | - |
| **Noisy** | 1.0000 | 1.0000 | - |
| **Rotated** | 0.5167 $\pm$ 0.07 | **0.6944 $\pm$ 0.10** | **+17.77%** |

*Findings:* While baseline models struggle with unseen spatial orientations, CGA-VQM preserves intrinsic geometric features, leading to much higher generalization stability.

## ✒️ Authors
- **Phuc Hao Do** - Danang Architecture University
- **Nang Hung Van Nguyen** (Corresponding Author) - University of Science and Technology, UD
- **Minh Tuan Pham** - University of Science and Technology, UD

## 🎓 Citation
If you find this work useful for your research, please cite:
```bibtex
@inproceedings{do2026cga,
  title={A Novel Geometric Algebra-based Variational Quantum Mapping for Non-Terrestrial Feature Representation},
  author={Do, Phuc Hao and Nguyen, Nang Hung Van and Pham, Minh Tuan},
  booktitle={2026 11th International Conference on Intelligent Information Technology (ICIIT)},
  year={2026}
}
```
