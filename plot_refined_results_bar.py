import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === Paths ===
BASE_DIR = 'results_refined'
AGG_CSV = os.path.join(BASE_DIR, 'refined_train_test_aggregated.csv')
PLOT_DIR = os.path.join(BASE_DIR, 'plots_bar')

os.makedirs(PLOT_DIR, exist_ok=True)

# === Load aggregated results ===
df = pd.read_csv(AGG_CSV)

WORKLOADS = ['Light', 'Medium', 'Heavy']
METRICS = [
    ('avg_latency_mean', 'Average Latency (ms)', 'latency'),
    ('avg_energy_mean', 'Average Energy (Wh)', 'energy'),
]


def plot_grouped_bar(sub, y_col, y_label, filename, title):
    """
    Creates a grouped bar chart:
      x-axis: task_count
      groups: different approaches
      y-axis: y_col
    """
    # Ensure sorted for consistent grouping
    sub = sub.copy()
    sub.sort_values(['task_count', 'approach'], inplace=True)

    task_counts = sorted(sub['task_count'].unique())
    approaches = sorted(sub['approach'].unique())

    # Bar positions
    x = np.arange(len(task_counts))  # task_count groups
    width = 0.8 / len(approaches)    # total width ~0.8

    plt.figure(figsize=(8, 5))

    for i, approach in enumerate(approaches):
        g = sub[sub['approach'] == approach]
        # Make sure values are aligned with task_counts
        y = [g[g['task_count'] == tc][y_col].values[0] for tc in task_counts]
        plt.bar(x + i * width, y, width, label=approach)

    plt.xlabel('Number of Tasks')
    plt.ylabel(y_label)
    plt.title(title)
    plt.xticks(x + width * (len(approaches) - 1) / 2, task_counts)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(PLOT_DIR, filename)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f'[OK] Saved: {out_path}')


def main():
    for wl in WORKLOADS:
        sub = df[df['workload_type'] == wl]
        if sub.empty:
            print(f'[WARN] No data for workload_type={wl}')
            continue

        for y_col, y_label, prefix in METRICS:
            title = f'{y_label} vs task count ({wl} workload)'
            filename = f'{prefix}_vs_tasks_{wl.lower()}_bar.png'
            plot_grouped_bar(sub, y_col, y_label, filename, title)


if __name__ == '__main__':
    main()
