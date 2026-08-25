# APQS: Intelligent Priority-Aware Task Scheduling in Heterogeneous Environments Using Deep Q-Network

This repository contains the implementation and refined experimental results for the **Adaptive Priority-Queue Scheduler (APQS)**, a Deep Q-Network (DQN)-based task-offloading policy for hierarchical IoT--Edge--Fog--Cloud environments.

APQS combines priority queues with a learned offloading policy that selects among local, edge, fog, and cloud execution tiers while balancing latency, deadline compliance, energy consumption, task completion, and resource availability.

The code corresponds to the refined **train--then--test** experiments and the associated grouped-bar and execution-location plots used in the research paper.

**Repository:** <https://github.com/trisharoshan/APQS>

---

## Repository contents

```text
APQS/
├── .gitignore
├── README.md
├── task_scheduler.py
├── refined_train_test_full.py
├── plot_refined_results_bar.py
└── results_refined/
    ├── refined_train_test_raw.csv
    ├── refined_train_test_aggregated.csv
    └── plots_bar/
        ├── latency_vs_tasks_light_bar.png
        ├── latency_vs_tasks_medium_bar.png
        ├── latency_vs_tasks_heavy_bar.png
        ├── energy_vs_tasks_light_bar.png
        ├── energy_vs_tasks_medium_bar.png
        ├── energy_vs_tasks_heavy_bar.png
        ├── deadline_met_rate_vs_tasks_light_bar.png
        ├── deadline_met_rate_vs_tasks_medium_bar.png
        ├── deadline_met_rate_vs_tasks_heavy_bar.png
        ├── failed_tasks_vs_tasks_light_bar.png
        ├── failed_tasks_vs_tasks_medium_bar.png
        ├── failed_tasks_vs_tasks_heavy_bar.png
        ├── executed_tasks_vs_tasks_light_bar.png
        ├── executed_tasks_vs_tasks_medium_bar.png
        ├── executed_tasks_vs_tasks_heavy_bar.png
        ├── accepted_tasks_vs_tasks_light_bar.png
        ├── accepted_tasks_vs_tasks_medium_bar.png
        ├── accepted_tasks_vs_tasks_heavy_bar.png
        ├── apqs_execution_location_light_pie.png
        ├── apqs_execution_location_medium_pie.png
        └── apqs_execution_location_heavy_pie.png
```

## Main files

### `task_scheduler.py`

Core simulator and scheduling algorithms:

- **APQS:** DQN-based scheduler with priority queues and feasibility-aware action selection.
- **Static Threshold:** threshold-based heuristic scheduler.
- **FCFS:** First-Come--First-Served queue-based scheduler.
- **Genetic Algorithm:** metaheuristic scheduling approach.
- **Fuzzy Logic:** fuzzy-rule-based scheduling approach.
- **Environment:** IoT devices, edge nodes, fog nodes, and a cloud tier.

### `refined_train_test_full.py`

Main train--then--test experiment script:

- Trains APQS on the Heavy workload when the trained model is unavailable and training is enabled.
- Uses the trained APQS model to evaluate Light, Medium, and Heavy workloads.
- Evaluates APQS and four baseline schedulers on Light, Medium, and Heavy workloads.
- Saves completed experiments incrementally to a checkpoint CSV.
- Skips completed experiments when resumed.
- Produces raw and aggregated result files after evaluation.

Generated files:

```text
results_refined/refined_train_test_raw.csv
results_refined/refined_train_test_aggregated.csv
results_refined/refined_train_test_checkpoint.csv
```

The checkpoint CSV is an intermediate resume file and is excluded from version control.

### `plot_refined_results_bar.py`

Reads the aggregated results and generates grouped bar plots for:

- Average latency of deadline-met tasks.
- Average energy consumption of deadline-met tasks.
- Deadline-met rate.
- Failed tasks.
- Executed tasks.
- Accepted tasks.
- APQS execution-location distributions across local, edge, fog, and cloud tiers.

Bar-chart legends are positioned above the plots in a horizontal layout to avoid overlapping the bars. Each bar represents the mean across the five repeats.

## Experimental setup

All experiments use the custom Python simulator implemented in `task_scheduler.py`.

### System model

The simulated hierarchical environment contains:

- 50 IoT devices.
- 5 edge nodes.
- 2 fog nodes.
- 1 cloud node.

Tasks originate from IoT devices and may execute locally or be offloaded to edge, fog, or cloud tiers.

### Metrics

The experiments record:

- Average latency (ms).
- P50 latency (ms).
- P99 latency (ms).
- Average energy consumption (Wh).
- Deadline-met rate (%).
- Failed tasks, representing executed tasks that miss their deadlines.
- Executed tasks, representing tasks that complete execution.
- Accepted tasks, representing executed tasks that complete within their deadlines.
- Execution counts at local, edge, fog, and cloud tiers.

Latency and energy plots in the refined analysis refer to deadline-met tasks. Therefore, these metrics should be interpreted together with the deadline-met rate and accepted-task count.

## Train--then--test protocol

The refined experiments use two phases.

### Phase 1: APQS training

APQS is trained on the Heavy workload, which contains larger CPU and data demands and tighter deadlines.

- Training episodes: `TRAIN_EPISODES = 100`.
- Tasks per training episode: `300`.
- Workload: `HeavyWorkloadConfig`.
- Random seed: `1234`.
- Exploration: epsilon-greedy action selection.
- Learning: experience replay with epsilon decay.

The trained model is saved as:

```text
results_refined/apqs_trained_refined.weights.h5
```

The weights are generated binary artifacts and are excluded from Git.

### Phase 2: Evaluation

The trained APQS model is evaluated on:

- Light workload.
- Medium workload.
- Heavy workload.

APQS is compared with:

- Static Threshold.
- FCFS.
- Genetic Algorithm.
- Fuzzy Logic.

During evaluation, APQS uses:

```python
epsilon = 0.0
```

so no exploratory learning occurs. For each workload, task count, and repeat, all approaches receive copies of the same generated task dataset.

## Workload configurations

The workload profiles are defined in `refined_train_test_full.py`.

### Light workload

- Data size: 1--5.
- CPU demand: 500--2000 MI.
- Deadline: 30--150 ms.
- Priority: 1--10.

This profile represents relatively small tasks with relaxed deadlines.

### Medium workload

- Data size: 5--15.
- CPU demand: 2000--8000 MI.
- Deadline: 15--60 ms.
- Priority: 1--10.

This profile represents intermediate computational and deadline requirements.

### Heavy workload

- Data size: 10--30.
- CPU demand: 4000--15000 MI.
- Deadline: 5--30 ms.
- Priority: 1--10.

This profile represents computationally demanding tasks with tight deadlines.

For evaluation, task arrivals use a fixed arrival gap of 20 ms, while task attributes are generated using controlled random seeds.

## Schedulers compared

### APQS

APQS combines:

- Priority-based task queues.
- DQN-based offloading decisions.
- Local, edge, fog, and cloud execution actions.
- Deadline-feasibility checks.
- A reward function involving latency, energy, deadline satisfaction, and task priority.

Feasibility-aware action selection filters actions predicted to violate the task deadline before selecting the final action.

### Static Threshold

A heuristic scheduler that uses data-size thresholds to determine the execution destination.

### FCFS

A First-Come--First-Served scheduler that uses queue information and congestion-related thresholds to determine execution placement.

### Genetic Algorithm

A metaheuristic scheduler that searches for task assignments using a latency--energy--deadline objective.

### Fuzzy Logic

A fuzzy-rule-based scheduler using task and system characteristics such as data size, queue length, and CPU availability.

## Task counts and repeats

The refined evaluation uses:

- Task counts: 200, 400, 600, 800, and 1000.
- Repeats: 5 per configuration.

The experiment count is:

```text
3 workloads × 5 task counts × 5 approaches × 5 repeats = 375 experiments
```

The raw results contain 375 experiment rows. The aggregated results contain:

```text
3 workloads × 5 task counts × 5 approaches = 75 configurations
```

Each aggregated configuration contains the mean and standard deviation across five repeats.

## Reproducibility

Synthetic task datasets are generated with controlled random seeds. For every workload, task count, and repeat:

1. A deterministic dataset seed is generated.
2. A task dataset is created from the relevant workload configuration.
3. Copies of the same dataset are passed to every scheduler.
4. Each scheduler is evaluated independently.
5. The resulting metrics are written to the raw results CSV.

This procedure ensures that the schedulers are compared using identical task instances. Python, NumPy, and TensorFlow random seeds are also configured where available.

## Checkpoint and resume

Long evaluation runs may be interrupted by the operating system. The evaluation script therefore maintains:

```text
results_refined/refined_train_test_checkpoint.csv
```

After each completed experiment, the result is written to the checkpoint. Each experiment is identified by:

```text
(workload_type, approach, task_count, repeat)
```

When the script restarts, existing checkpoint rows are loaded and completed experiment keys are skipped. For example:

```text
[SKIP] Light | APQS | tasks=600 | repeat=3
```

The checkpoint file is excluded from version control because it is an intermediate execution artifact.

## Requirements

The refined experiments were run with:

- Python 3.10.19.
- NumPy 1.26.4.
- Pandas 2.1.4.
- Matplotlib 3.6.0.
- TensorFlow 2.20.0.

Create an environment with:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the dependencies:

```bash
pip install "numpy==1.26.4" \
            "pandas==2.1.4" \
            "matplotlib==3.6.0" \
            "tensorflow==2.20.0"
```

## Reproducing the experiments

### 1. Clone the repository

```bash
git clone https://github.com/trisharoshan/APQS.git
cd APQS
```

### 2. Create the environment

```bash
python3 -m venv venv
source venv/bin/activate

pip install "numpy==1.26.4" \
            "pandas==2.1.4" \
            "matplotlib==3.6.0" \
            "tensorflow==2.20.0"
```

### 3. Train APQS

The default configuration is:

```python
TRAIN_IF_MODEL_MISSING = False
```

This prevents accidental long training runs. If the trained weights are unavailable, intentionally change it to:

```python
TRAIN_IF_MODEL_MISSING = True
```

Then run:

```bash
python refined_train_test_full.py
```

Training uses 100 episodes with 300 tasks per episode and saves:

```text
results_refined/apqs_trained_refined.weights.h5
```

After training, return `TRAIN_IF_MODEL_MISSING` to `False` to prevent unintended retraining.

### 4. Run evaluation

With trained weights available, run:

```bash
python refined_train_test_full.py
```

The script evaluates all workload and scheduler configurations. If a checkpoint exists, completed experiments are skipped automatically.

### 5. Generate plots

Run:

```bash
python plot_refined_results_bar.py
```

The script generates 18 grouped bar plots and 3 APQS execution-location pie charts in:

```text
results_refined/plots_bar/
```

The grouped bar plots include latency, energy, deadline-met rate, failed tasks, executed tasks, and accepted tasks for Light, Medium, and Heavy workloads.

## Final results

The repository includes:

- 375 raw experiment results.
- 75 aggregated configurations.
- 3 workload profiles.
- 5 task counts.
- 5 scheduling approaches.
- 5 repeats per configuration.
- Grouped bar plots for six metrics.
- APQS execution-location plots for three workloads.

Raw results:

```text
results_refined/refined_train_test_raw.csv
```

Aggregated results:

```text
results_refined/refined_train_test_aggregated.csv
```

Plots:

```text
results_refined/plots_bar/
```

## Version control

The following files should be tracked:

- `task_scheduler.py`.
- `refined_train_test_full.py`.
- `plot_refined_results_bar.py`.
- `results_refined/refined_train_test_raw.csv`.
- `results_refined/refined_train_test_aggregated.csv`.
- `results_refined/plots_bar/*.png`.

The following generated files should be excluded through `.gitignore`:

```text
results_refined/apqs_trained_refined.weights.h5
results_refined/refined_train_test_checkpoint.csv
```

The trained weights are binary generated data, and the checkpoint CSV is an intermediate resume artifact. The final raw and aggregated CSV files are retained so that reviewers can inspect the reported results without rerunning the full experiment.

## Citation

If this implementation or the APQS experimental results are used in academic work, please cite the associated research paper.

For questions, reproducibility issues, or implementation discussions, open an issue in the GitHub repository:

<https://github.com/trisharoshan/APQS>
