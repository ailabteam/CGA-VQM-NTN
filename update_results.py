import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

# 1. Đọc dữ liệu
df = pd.read_csv("results/final_stats.csv")

# 2. Xuất bảng LaTeX chuẩn IEEE/ACM
try:
    # Sắp xếp lại thứ tự Scenario cho đúng logic
    df['Scenario'] = pd.Categorical(df['Scenario'], categories=['clean', 'noisy', 'rotated'], ordered=True)
    stats_sorted = df.sort_values(['Scenario', 'Method'])
    
    # Tạo bảng pivot
    latex_table = stats_sorted.pivot(index='Scenario', columns='Method', values=['mean', 'std'])
    
    # Ghi file .tex
    with open("results/final_table.tex", "w") as f:
        f.write("% --- Statistical Results Table for ICIIT 2026 ---\n")
        f.write(latex_table.to_latex(float_format="%.4f", 
                                     caption="Accuracy Performance Comparison (Mean ± Std)",
                                     label="tab:results"))
    print("[Success] results/final_table.tex has been updated.")
except Exception as e:
    print(f"[Error] LaTeX export failed: {e}")

# 3. Vẽ biểu đồ Error Bar (accuracy_comparison.pdf)
try:
    with PdfPages('results/final_comparison.pdf') as pdf:
        plt.figure(figsize=(9, 6))
        
        # Vẽ từng đường cho mỗi Method
        methods = ['raw', 'cga']
        colors = ['#d62728', '#1f77b4'] # Đỏ cho Raw, Xanh cho CGA
        markers = ['s', 'o']
        
        for i, m in enumerate(methods):
            subset = df[df['Method'] == m].copy()
            subset['Scenario'] = pd.Categorical(subset['Scenario'], categories=['clean', 'noisy', 'rotated'], ordered=True)
            subset = subset.sort_values('Scenario')
            
            plt.errorbar(subset['Scenario'], subset['mean'], yerr=subset['std'], 
                         fmt=markers[i]+'-', color=colors[i], capsize=10, elinewidth=2, 
                         linewidth=2, markersize=10, label=f"Method: {m.upper()}")
        
        plt.title("Mobility Classification Performance: CGA-VQM vs Raw-VQC", fontsize=14, fontweight='bold')
        plt.ylabel("Test Accuracy Score", fontsize=12)
        plt.xlabel("Evaluation Scenarios", fontsize=12)
        plt.ylim(0.3, 1.1)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc='lower left', fontsize=11)
        
        # Thêm ghi chú về gap kết quả
        plt.annotate(f'Gap: +17.77%', xy=('rotated', 0.694), xytext=('rotated', 0.8),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                     horizontalalignment='center', fontsize=12, fontweight='bold', color='blue')
        
        plt.tight_layout()
        pdf.savefig()
        plt.close()
    print("[Success] results/final_comparison.pdf has been updated.")
except Exception as e:
    print(f"[Error] Plotting failed: {e}")
