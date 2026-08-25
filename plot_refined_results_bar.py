import os

import pandas as pd
import matplotlib.pyplot as plt


AGG_CSV = (
    'results_refined/'
    'refined_train_test_aggregated.csv'
)

PLOT_DIR = (
    'results_refined/'
    'plots_bar'
)

WORKLOADS = [
    'Light',
    'Medium',
    'Heavy'
]


REQUIRED_COLUMNS = {
    'workload_type',
    'approach',
    'task_count',
    'avg_latency_mean',
    'avg_energy_mean',
    'deadline_met_rate_mean',
    'failed_tasks_mean',
    'executed_tasks_mean',
    'accepted_tasks_mean',
    'local_count_mean',
    'edge_count_mean',
    'fog_count_mean',
    'cloud_count_mean'
}


def plot_metric(
    df,
    workload,
    metric,
    ylabel,
    title,
    filename
):
    subset = df[
        df['workload_type'] == workload
    ].copy()

    if subset.empty:
        print(
            f'[WARNING] No data found for workload: '
            f'{workload}'
        )
        return

    pivot = subset.pivot_table(
        index='task_count',
        columns='approach',
        values=metric,
        aggfunc='mean'
    ).sort_index()

    if pivot.empty:
        print(
            f'[WARNING] No values available for '
            f'{metric} | {workload}'
        )
        return

    # Preserve a consistent scheduler order.
    scheduler_order = [
        'APQS',
        'FCFS',
        'Fuzzy Logic',
        'Genetic Algorithm',
        'Static Threshold'
    ]

    available_schedulers = [
        scheduler for scheduler in scheduler_order
        if scheduler in pivot.columns
    ]

    remaining_schedulers = [
        scheduler for scheduler in pivot.columns
        if scheduler not in available_schedulers
    ]

    pivot = pivot[
        available_schedulers + remaining_schedulers
    ]

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    pivot.plot(
        kind='bar',
        ax=ax,
        width=0.8
    )

    ax.set_xlabel(
        'Number of Tasks',
        fontsize=16
    )

    ax.set_ylabel(
        ylabel,
        fontsize=16
    )

    ax.set_title(
        title,
        fontsize=16,
        pad=45
    )

    ax.tick_params(
        axis='x',
        labelrotation=0,
        labelsize=13
    )

    ax.tick_params(
        axis='y',
        labelsize=13
    )

    ax.grid(
        axis='y',
        linestyle='--',
        alpha=0.4
    )

    # Legend above the graph, matching the attached image.
    ax.legend(
        title='Scheduler',
        loc='lower center',
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(pivot.columns),
        fontsize=11,
        title_fontsize=12,
        frameon=True,
        borderaxespad=0.0,
        columnspacing=1.2,
        handletextpad=0.5
    )

    # Reserve space above the axes for the legend.
    plt.tight_layout(
        rect=[0, 0, 1, 0.88]
    )

    output_path = os.path.join(
        PLOT_DIR,
        filename
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()

    print(
        f'[SAVED] {output_path}'
    )


def plot_apqs_execution_location_pie(
    df,
    workload
):
    subset = df[
        (df['workload_type'] == workload)
        & (df['approach'] == 'APQS')
    ].copy()

    if subset.empty:
        print(
            f'[WARNING] No APQS data found for '
            f'workload: {workload}'
        )
        return

    values = {
        'Local': subset[
            'local_count_mean'
        ].sum(),

        'Edge': subset[
            'edge_count_mean'
        ].sum(),

        'Fog': subset[
            'fog_count_mean'
        ].sum(),

        'Cloud': subset[
            'cloud_count_mean'
        ].sum()
    }

    values = {
        label: value
        for label, value in values.items()
        if value > 0
    }

    if not values:
        print(
            f'[WARNING] No execution-location '
            f'counts for APQS | {workload}'
        )
        return

    labels = list(values.keys())
    sizes = list(values.values())

    fig, ax = plt.subplots(
        figsize=(7, 7)
    )

    wedges, _, _ = ax.pie(
        sizes,
        labels=None,
        autopct=(
            lambda pct:
            f'{pct:.1f}%'
            if pct >= 3
            else ''
        ),
        startangle=90,
        textprops={
            'fontsize': 12
        }
    )

    ax.legend(
        wedges,
        labels,
        title='Execution Location',
        loc='lower center',
        bbox_to_anchor=(0.5, -0.08),
        ncol=2,
        frameon=False,
        fontsize=11,
        title_fontsize=11
    )

    ax.axis('equal')

    plt.tight_layout()

    output_path = os.path.join(
        PLOT_DIR,
        f'apqs_execution_location_'
        f'{workload.lower()}_pie.png'
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()

    print(
        f'[SAVED] {output_path}'
    )


def main():
    os.makedirs(
        PLOT_DIR,
        exist_ok=True
    )

    print(
        f'Loading: {AGG_CSV}'
    )

    if not os.path.exists(AGG_CSV):
        raise FileNotFoundError(
            f'Aggregated CSV not found: {AGG_CSV}'
        )

    df = pd.read_csv(
        AGG_CSV
    )

    missing = (
        REQUIRED_COLUMNS
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            'The aggregated CSV is missing these '
            'columns: '
            + ', '.join(sorted(missing))
        )

    metrics = [
        (
            'avg_latency_mean',
            'Average Latency of Deadline-Met Tasks (ms)',
            'Average Latency of Deadline-Met Tasks (ms) '
            'vs Number of Tasks',
            'latency'
        ),

        (
            'avg_energy_mean',
            'Average Energy of Deadline-Met Tasks (Wh)',
            'Average Energy of Deadline-Met Tasks (Wh) '
            'vs Number of Tasks',
            'energy'
        ),

        (
            'deadline_met_rate_mean',
            'Deadline Met Rate (%)',
            'Deadline Met Rate vs Number of Tasks',
            'deadline_met_rate'
        ),

        (
            'failed_tasks_mean',
            'Average Failed Tasks',
            'Failed Tasks vs Number of Tasks',
            'failed_tasks'
        ),

        (
            'executed_tasks_mean',
            'Average Executed Tasks',
            'Executed Tasks vs Number of Tasks',
            'executed_tasks'
        ),

        (
            'accepted_tasks_mean',
            'Average Accepted Tasks',
            'Accepted Tasks vs Number of Tasks',
            'accepted_tasks'
        )
    ]

    for workload in WORKLOADS:
        for (
            metric,
            ylabel,
            title,
            prefix
        ) in metrics:

            plot_metric(
                df,
                workload,
                metric,
                ylabel,
                f'{title} — {workload} Workload',
                (
                    f'{prefix}_vs_tasks_'
                    f'{workload.lower()}_bar.png'
                )
            )

        plot_apqs_execution_location_pie(
            df,
            workload
        )

    print(
        '\nAll plots generated successfully.'
    )


if __name__ == '__main__':
    main()
