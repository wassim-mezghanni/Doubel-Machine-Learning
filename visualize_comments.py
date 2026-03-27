import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def main():
    BASE_DIR = r"c:\Users\ge27tuv\Projects\Doubel-Machine-Learning"
    df = pd.read_csv(os.path.join(BASE_DIR, "clean_comment_metrics.csv"))

    # We will use log(1+x) since the data is highly skewed (max 100k comments)
    df['log1p_orig'] = np.log1p(df['orig_comments'])
    df['log1p_clean'] = np.log1p(df['clean_comments'])
    df['log1p_unique'] = np.log1p(df['unique_commenters'])

    # Cap at 99th percentile for plotting (prevent extreme outliers stretching plot)
    max_val_orig = df['log1p_orig'].quantile(0.999)
    max_val_clean = df['log1p_clean'].quantile(0.999)
    max_val_plot = max(max_val_orig, max_val_clean) + 1

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Orig vs Clean
    # We use a hexbin plot because scatter plot with 500k points is unintelligible
    hb1 = axes[0].hexbin(df['log1p_orig'], df['log1p_clean'], gridsize=50, cmap='Blues', mincnt=1, bins='log')
    axes[0].plot([0, max_val_plot], [0, max_val_plot], 'r--', label='y = x (No Difference)')
    axes[0].set_title("Original Comments vs. Clean Comments")
    axes[0].set_xlabel("Log(1 + Original Comment Count)")
    axes[0].set_ylabel("Log(1 + Clean Comment Count)")
    axes[0].set_xlim(-0.5, max_val_plot)
    axes[0].set_ylim(-0.5, max_val_plot)
    axes[0].legend()
    cb1 = fig.colorbar(hb1, ax=axes[0])
    cb1.set_label('Log10(Count + 1)')

    # Plot 2: Orig vs Unique
    hb2 = axes[1].hexbin(df['log1p_orig'], df['log1p_unique'], gridsize=50, cmap='Greens', mincnt=1, bins='log')
    axes[1].plot([0, max_val_plot], [0, max_val_plot], 'r--', label='y = x (No Difference)')
    axes[1].set_title("Original Comments vs. Unique Commenters")
    axes[1].set_xlabel("Log(1 + Original Comment Count)")
    axes[1].set_ylabel("Log(1 + Unique Commenters Count)")
    axes[1].set_xlim(-0.5, max_val_plot)
    axes[1].set_ylim(-0.5, max_val_plot)
    axes[1].legend()
    cb2 = fig.colorbar(hb2, ax=axes[1])
    cb2.set_label('Log10(Count + 1)')

    plt.tight_layout()
    out_file = os.path.join(BASE_DIR, "comment_metrics_comparison.png")
    plt.savefig(out_file, dpi=300)
    print(f"Plot saved to {out_file}")

if __name__ == "__main__":
    main()
