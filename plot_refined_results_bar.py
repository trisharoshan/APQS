import os

import pandas as pd
import matplotlib.pyplot as plt


AGG_CSV = "results_refined/refined_train_test_aggregated.csv"
PLOT_DIR = "results_refined/plots_bar"

WORKLOADS = ["Light", "Medium", "Heavy"]


def plot_metric(df, workload, metric_mean, ylabel, title, filename):
    """Generate a grouped bar plot for one metric and workload."""

    subset = df[df["workload_type"] == workload].copy()

    if subset.empty:
        print(f"[WARNING] No data found for workload: {workload}")
        return

    pivot = subset.pivot(
        index="task_count",
        columns="approach",
        values=metric_mean
    )

    # Keep task counts in numerical order.
    pivot = pivot.sort_index()

    ax = pivot.plot(
        kind="bar",
        figsize=(10, 6)
    )

    ax.set_xlabel("Number of Tasks")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.set_xticklabels(
        [str(x) for x in pivot.index],
        rotation=0
    )

    ax.legend(
        title="Scheduler",
        loc="best"
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.4
    )

    plt.tight_layout()

    output_path = os.path.join(PLOT_DIR, filename)

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"[SAVED] {output_path}")


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    print(f"Loading: {AGG_CSV}")

    df = pd.read_csv(AGG_CSV)

    required_columns = {
        "workload_type",
        "approach",
        "task_count",
        "avg_latency_mean",
        "avg_energy_mean",
        "deadline_met_rate_mean",
        "failed_tasks_mean",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            "The aggregated CSV is missing these columns: "
            + ", ".join(sorted(missing))
        )

   
    # Average latency plots
    

    for workload in WORKLOADS:
        workload_lower = workload.lower()

        plot_metric(
            df=df,
            workload=workload,
            metric_mean="avg_latency_mean",
            ylabel="Average Latency (ms)",
            title=f"Average Latency vs Number of Tasks — {workload} Workload",
            filename=f"latency_vs_tasks_{workload_lower}_bar.png",
        )

    
    # Average energy plots
    

    for workload in WORKLOADS:
        workload_lower = workload.lower()

        plot_metric(
            df=df,
            workload=workload,
            metric_mean="avg_energy_mean",
            ylabel="Average Energy (Wh)",
            title=f"Average Energy vs Number of Tasks — {workload} Workload",
            filename=f"energy_vs_tasks_{workload_lower}_bar.png",
        )

    
    # Deadline-met-rate plots
    

    for workload in WORKLOADS:
        workload_lower = workload.lower()

        plot_metric(
            df=df,
            workload=workload,
            metric_mean="deadline_met_rate_mean",
            ylabel="Deadline Met Rate (%)",
            title=f"Deadline Met Rate vs Number of Tasks — {workload} Workload",
            filename=f"deadline_met_rate_vs_tasks_{workload_lower}_bar.png",
        )

    
    # Failed-task plots
   

    for workload in WORKLOADS:
        workload_lower = workload.lower()

        plot_metric(
            df=df,
            workload=workload,
            metric_mean="failed_tasks_mean",
            ylabel="Average Failed Tasks",
            title=f"Failed Tasks vs Number of Tasks — {workload} Workload",
            filename=f"failed_tasks_vs_tasks_{workload_lower}_bar.png",
        )

    print("\nAll plots generated successfully.")


if __name__ == "__main__":
    main()
