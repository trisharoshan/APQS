# APQS: Intelligent Priority-Aware Task Scheduling in Heterogeneous Environments Using Deep Q-Network

This repository contains the implementation and refined experimental results for the **Adaptive Priority-Queue Scheduler (APQS)**, a Deep Q-Network (DQN) based task offloading policy for hierarchical IoT–Edge–Fog–Cloud environments.

APQS combines priority queues with a learned offloading policy that selects among local, edge, fog, and cloud execution tiers while trading off latency, deadline satisfaction, and energy consumption.

The code corresponds to the refined **train–then–test** experiments and bar-plot results used in the associated research paper.

**Repository:** <https://github.com/trisharoshan/APQS>

---

## Repository contents

Current layout:

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
        └── energy_vs_tasks_heavy_bar.png
```

### `task_scheduler.py`

Core simulator and scheduling algorithms:

- **APQS**: DQN + priority queues + feasibility-aware action selection.
- **Static Threshold**: threshold-based heuristic scheduler.
- **FCFS**: First-Come–First-Served queue-based scheduler.
- **Genetic Algorithm**: metaheuristic scheduling approach.
- **Fuzzy Logic**: fuzzy-rule-based scheduling approach.
- **Environment**: IoT devices, edge nodes, fog nodes, and a cloud tier.

### `refined_train_test_full.py`

Main train–then–test experiment script. It supports checkpoint/resume for long evaluation runs:

- Phase 1 optionally trains APQS on the Heavy workload.
- Phase 2 evaluates APQS and the four baseline schedulers on Light, Medium, and Heavy workloads.
- Completed experiments are saved incrementally to a checkpoint CSV.
- Existing completed experiments are skipped when the script is resumed.

The script produces:

- `results_refined/refined_train_test_raw.csv`
- `results_refined/refined_train_test_aggregated.csv`
- `results_refined/refined_train_test_checkpoint.csv`

The checkpoint CSV is an intermediate resume file and is intentionally excluded from the Git repository.

### `plot_refined_results_bar.py`

Plotting script that reads the aggregated experimental results and generates six grouped bar plots:

- Average latency vs task count for Light, Medium, and Heavy workloads.
- Average energy vs task count for Light, Medium, and Heavy workloads.

### `results_refined/`

Contains the final experimental results and plots used for analysis:

- `refined_train_test_raw.csv`: per-experiment results.
- `refined_train_test_aggregated.csv`: means and standard deviations across the five repeats.
- `plots_bar/`: six final bar plots.

---

## Experimental setup

All experiments are conducted using a custom Python-based simulator implemented in `task_scheduler.py`.

### System model

The simulated hierarchical environment contains:

- 50 IoT devices.
- 5 edge nodes.
- 2 fog nodes.
- 1 cloud node.

Tasks originate from IoT devices and can be executed locally or offloaded to edge, fog, or cloud tiers.

### Metrics

For each experiment the following metrics are recorded:

- Average latency (ms).
- P50 latency (ms).
- P99 latency (ms).
- Average energy consumption (Wh).
- Deadline-met rate (%).
- Number of failed tasks (deadline violations).
- Offloading counts to local, edge, fog, and cloud tiers.

---

## Train–then–test protocol

The refined experiments use a two-phase protocol.

### Phase 1 — APQS training

APQS is trained only on the Heavy workload, which contains larger CPU/data demands and tighter deadlines. The training configuration is:

- Training episodes: `TRAIN_EPISODES = 100`.
- Tasks per training episode: `300`.
- Workload: `HeavyWorkloadConfig`.
- Random seed: `1234`.
- DQN exploration uses \(\epsilon\)-greedy action selection.
- Experience replay is used during training.
- \(\epsilon\) is decayed across training episodes.

After training, the APQS model weights are saved as:

- `results_refined/apqs_trained_refined.weights.h5`

The trained weights are intentionally excluded from Git because they are generated binary model data.

### Phase 2 — Evaluation

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

- \(\epsilon = 0.0\)

so no further exploratory learning occurs.

For each workload and task count, all five approaches are evaluated using copies of the same generated task dataset for each repeat.

---

## Workload configurations

The workload profiles are defined in `refined_train_test_full.py`.

### Light workload

- Data size: 1–5
- CPU demand: 500–2000 MI
- Deadline: 30–150 ms
- Priority: 1–10

This workload represents relatively small tasks with relaxed deadlines.

### Medium workload

- Data size: 5–15
- CPU demand: 2000–8000 MI
- Deadline: 15–60 ms
- Priority: 1–10

This workload represents intermediate computational and deadline requirements.

### Heavy workload

- Data size: 10–30
- CPU demand: 4000–15000 MI
- Deadline: 5–30 ms
- Priority: 1–10

This workload represents computationally demanding tasks with tight deadlines.

For evaluation, task arrival times are generated using a fixed arrival gap of 20 ms, while task attributes are generated using controlled random seeds.

---

## Schedulers compared

### APQS

APQS combines:

- Priority-based task queues.
- DQN-based offloading decisions.
- Four possible execution actions:
  - local
  - edge
  - fog
  - cloud
- Deadline feasibility checks.
- Reward-based learning involving latency, energy, deadline satisfaction, and task priority.

The implementation performs feasibility-aware action selection so that actions predicted to violate the task deadline can be filtered before the final action is selected.

### Static Threshold

Heuristic scheduler that uses data-size thresholds to determine an execution destination.

### FCFS

First-Come–First-Served scheduler that uses queue information and congestion-related thresholds to determine execution placement.

### Genetic Algorithm

Metaheuristic scheduler that searches for task assignments using a latency–energy–deadline objective.

### Fuzzy Logic

Fuzzy-rule-based scheduler using task and system characteristics such as data size, queue length, and CPU availability.

---

## Task counts and repeats

The refined evaluation uses:

- Task counts: 200, 400, 600, 800, 1000
- Repeats: 5 per configuration

There are:

- 3 workloads × 5 task counts × 5 approaches × 5 repeats = **375 experiments**

The final raw results therefore contain 375 experiment rows.

The aggregated results contain:

- 3 workloads × 5 task counts × 5 approaches = **75 configurations**

Each aggregated configuration contains the mean and standard deviation calculated across the five repeats.

---

## Dataset generation and reproducibility

Synthetic task datasets are generated using controlled random seeds.

For each workload, task count, and repeat:

1. A deterministic dataset seed is generated.
2. The task dataset is created from the corresponding workload configuration.
3. Copies of the same task dataset are provided to each scheduler.
4. Each scheduler is evaluated independently.
5. Metrics are recorded in the raw results CSV.

This procedure ensures that the five scheduling approaches are compared using the same underlying task instances for each experimental configuration.

The experiment also sets Python, NumPy, and TensorFlow random seeds where available.

---

## Checkpoint and resume behaviour

Long evaluation runs may be interrupted if the operating system terminates the Python process. The evaluation script therefore maintains:

- `results_refined/refined_train_test_checkpoint.csv`

After every completed experiment, the current results are written to this checkpoint file.

Each experiment is identified by:

- `(workload_type, approach, task_count, repeat)`

When the script starts, it loads any existing checkpoint rows and constructs a set of completed experiment keys. If an experiment has already been completed, it is skipped rather than executed again, for example:

```text
[SKIP] Light | APQS | tasks=600 | repeat=3
```

For a new experiment, the result is added to the checkpoint immediately after completion. If the operating system kills the process after hundreds of experiments, the completed experiments remain available and the script can resume from the checkpoint instead of repeating them.

The checkpoint file is excluded from version control because it is an intermediate execution artifact.

---

## Requirements and tested environment

The refined experiments were run using the following environment:

- Python: 3.10.19
- NumPy: 1.26.4
- Pandas: 2.1.4
- Matplotlib: 3.6.0
- TensorFlow: 2.20.0

The experiment environment can be recreated using:

```bash
python3 -m venv venv
source venv/bin/activate
```

Then install the required packages:

```bash
pip install "numpy==1.26.4" \
            "pandas==2.1.4" \
            "matplotlib==3.6.0" \
            "tensorflow==2.20.0"
```

---

## How to reproduce the experiments

### 1. Clone the repository

```bash
git clone https://github.com/trisharoshan/APQS.git
cd APQS
```

### 2. Create the Python environment

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

### 3. Train APQS

The default configuration is:

```python
TRAIN_IF_MODEL_MISSING = False
```

This prevents an accidental long training run.

If the trained APQS weights are not available and APQS needs to be retrained, intentionally change:

```python
TRAIN_IF_MODEL_MISSING = True
```

in `refined_train_test_full.py`, then run:

```bash
python refined_train_test_full.py
```

The training phase runs for:

- 100 episodes × 300 tasks per episode

and saves the trained weights to:

- `results_refined/apqs_trained_refined.weights.h5`

After training, `TRAIN_IF_MODEL_MISSING` can be returned to:

```python
TRAIN_IF_MODEL_MISSING = False
```

to prevent unintended retraining.

### 4. Run the evaluation

Once trained weights are available:

```bash
python refined_train_test_full.py
```

The evaluation script performs the Light, Medium, and Heavy experiments.

If an existing checkpoint is found, completed experiments are skipped automatically and the evaluation resumes from the remaining configurations.

When all experiments are complete, the script produces:

- `results_refined/refined_train_test_raw.csv`
- `results_refined/refined_train_test_aggregated.csv`

### 5. Generate the six plots

Run:

```bash
python plot_refined_results_bar.py
```

This generates:

- `results_refined/plots_bar/latency_vs_tasks_light_bar.png`
- `results_refined/plots_bar/latency_vs_tasks_medium_bar.png`
- `results_refined/plots_bar/latency_vs_tasks_heavy_bar.png`
- `results_refined/plots_bar/energy_vs_tasks_light_bar.png`
- `results_refined/plots_bar/energy_vs_tasks_medium_bar.png`
- `results_refined/plots_bar/energy_vs_tasks_heavy_bar.png`

Each bar represents the mean over five repeats. Standard deviations are retained in the aggregated CSV and can be used for error bars or additional statistical analysis.

---

## Final experimental results

The repository includes the completed refined experimental results:

- Raw experiments: 375
- Aggregated configurations: 75
- Workloads: 3
- Task counts: 5
- Schedulers: 5
- Repeats per configuration: 5

The final raw results are stored in:

- `results_refined/refined_train_test_raw.csv`

The aggregated results are stored in:

- `results_refined/refined_train_test_aggregated.csv`

The six corresponding plots are stored in:

- `results_refined/plots_bar/`

---

## Generated files and version control

The following files are tracked because they are useful for reviewing and reproducing the reported results:

- `task_scheduler.py`
- `refined_train_test_full.py`
- `plot_refined_results_bar.py`
- `results_refined/refined_train_test_raw.csv`
- `results_refined/refined_train_test_aggregated.csv`
- `results_refined/plots_bar/*.png`

The following generated files are intentionally excluded through `.gitignore`:

- `results_refined/apqs_trained_refined.weights.h5`
- `results_refined/refined_train_test_checkpoint.csv`

The trained model weights are binary generated artifacts, while the checkpoint CSV is an intermediate resume artifact. The final raw and aggregated CSV files are retained in the repository so that reviewers can inspect the experimental results directly without having to rerun the full experiment.

---

## Citation

If this implementation or the APQS experimental results are used in academic work, please cite the associated research paper.

For questions, reproducibility issues, or implementation discussions, issues can be opened in the GitHub repository:

<https://github.com/trisharoshan/APQS>
