import os
import random
from collections import Counter

import numpy as np
import pandas as pd

import task_scheduler
from task_scheduler import Task, WorkloadConfig, EdgeFogSimulator

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except Exception:
    tf = None
    keras = None
    TF_AVAILABLE = False

TASK_COUNTS = [200, 400, 600, 800, 1000]
NUM_REPEATS = 5
TRAIN_EPISODES = 100  # stronger APQS training

OUTPUT_DIR = 'results_refined'
MODEL_PATH = os.path.join(OUTPUT_DIR, 'apqs_trained_refined.weights.h5')

RAW_CSV = os.path.join(OUTPUT_DIR, 'refined_train_test_raw.csv')
AGG_CSV = os.path.join(OUTPUT_DIR, 'refined_train_test_aggregated.csv')

# Checkpoint file: results are saved after every completed experiment.
CHECKPOINT_CSV = os.path.join(
    OUTPUT_DIR,
    'refined_train_test_checkpoint.csv'
)

# Set to True only when intentionally training a new APQS model.
# For evaluation/resume runs, keep this False.
TRAIN_IF_MODEL_MISSING = False

class LightWorkloadConfig(WorkloadConfig):
    def __init__(self):
        super().__init__()
        # Smaller input sizes
        self.size_low = 1
        self.size_high = 5
        # Lower CPU demand (MI)
        self.cpu_low = 500
        self.cpu_high = 2000
        # Relaxed deadlines (ms)
        self.dl_low = 30
        self.dl_high = 150
        # Same priority range
        self.priority_low = 1
        self.priority_high = 10

class MediumWorkloadConfig(WorkloadConfig):
    def __init__(self):
        super().__init__()
        self.size_low = 5
        self.size_high = 15
        self.cpu_low = 2000
        self.cpu_high = 8000
        self.dl_low = 15
        self.dl_high = 60
        self.priority_low = 1
        self.priority_high = 10

class HeavyWorkloadConfig(WorkloadConfig):
    def __init__(self):
        super().__init__()
        self.size_low = 10
        self.size_high = 30
        self.cpu_low = 4000
        self.cpu_high = 15000
        self.dl_low = 5
        self.dl_high = 30
        self.priority_low = 1
        self.priority_high = 10


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    if TF_AVAILABLE:
        keras.utils.set_random_seed(seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass


def clone_task(task):
    return Task(
        taskid=task.taskid,
        arrivaltime=task.arrivaltime,
        datasize=task.datasize,
        cpurequirement=task.cpurequirement,
        deadline=task.deadline,
        priority=task.priority,
    )


def clone_task_list(tasks):
    return [clone_task(t) for t in tasks]


def generate_task_dataset(num_tasks, cfg, seed=0, arrival_gap_ms=20.0):
    set_all_seeds(seed)
    tasks = []
    current_time = 0.0
    for task_id in range(num_tasks):
        tasks.append(Task(
            taskid=task_id,
            arrivaltime=current_time,
            datasize=random.uniform(cfg.size_low, cfg.size_high),
            cpurequirement=random.randint(cfg.cpu_low, cfg.cpu_high),
            deadline=random.randint(cfg.dl_low, cfg.dl_high),
            priority=random.randint(cfg.priority_low, cfg.priority_high),
        ))
        current_time += arrival_gap_ms
    return tasks


def summarize_completed_tasks(completedtasks, total_input_tasks):
    if not completedtasks:
        return {
            'input_tasks': total_input_tasks,
            'executed_tasks': 0,
            'avg_latency': 0.0,
            'p50_latency': 0.0,
            'p99_latency': 0.0,
            'avg_energy': 0.0,
            'deadline_met_rate': 0.0,
            'failed_tasks': total_input_tasks,
            'local_count': 0,
            'edge_count': 0,
            'fog_count': 0,
            'cloud_count': 0,
        }

    latencies = np.array([t.actuallatency for t in completedtasks], dtype=float)
    energies = np.array([t.energyconsumed for t in completedtasks], dtype=float)
    deadline_met = sum(1 for t in completedtasks if t.meetsdeadline())
    offload_counts = Counter(t.executionlocation for t in completedtasks)

    return {
        'input_tasks': total_input_tasks,
        'executed_tasks': int(len(completedtasks)),
        'avg_latency': float(np.mean(latencies)),
        'p50_latency': float(np.percentile(latencies, 50)),
        'p99_latency': float(np.percentile(latencies, 99)),
        'avg_energy': float(np.mean(energies)),
        'deadline_met_rate': float(deadline_met / len(completedtasks) * 100.0),
        'failed_tasks': int(len(completedtasks) - deadline_met),
        'local_count': int(offload_counts.get('local', 0)),
        'edge_count': int(offload_counts.get('edge', 0)),
        'fog_count': int(offload_counts.get('fog', 0)),
        'cloud_count': int(offload_counts.get('cloud', 0)),
    }


def execute_baseline_task_fixed(task, action, source, edgenodes, fognodes, cloud, currenttime):
    if action == 0:
        node = source
        location = 'local'
    elif action == 1:
        node = min(edgenodes, key=lambda e: source.distance_to(e))
        location = 'edge'
    elif action == 2:
        node = min(fognodes, key=lambda f: source.distance_to(f))
        location = 'fog'
    else:
        node = cloud
        location = 'cloud'

    transmission = 0.0 if node.deviceid == source.deviceid else (task.datasize * 8.0) / max(node.bandwidthmbps, 1e-9)
    propagation = 0.0 if node.deviceid == source.deviceid else source.propagation_delay_to(node)
    readytime = currenttime + transmission + propagation
    waiting = max(0.0, node.busy_until - readytime)
    exectimems = task.cpurequirement / max(node.mips, 1e-9)

    starttime = readytime + waiting
    completiontime = starttime + exectimems
    latency = completiontime - currenttime
    energy = node.powerwatts * (exectimems / 1000.0) + 0.2 * ((transmission + propagation) / 1000.0)

    node.busy_until = completiontime
    task.starttime = starttime
    task.completiontime = completiontime
    task.waitingtime = waiting
    task.servicetime = exectimems
    task.actuallatency = latency
    task.energyconsumed = energy
    task.executionlocation = location
    task.computenode = node.deviceid
    return task


def train_apqs_model(train_episodes=TRAIN_EPISODES, workload_config=None, seed=1234):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cfg = workload_config or HeavyWorkloadConfig()
    set_all_seeds(seed)

    simulator = EdgeFogSimulator(num_iot=50, num_edge=5, num_fog=2, workload_config=cfg)

    for episode in range(train_episodes):
        simulator.reset_runtime()
        episode_seed = seed + episode

        train_tasks = generate_task_dataset(
            num_tasks=300,
            cfg=cfg,
            seed=episode_seed,
            arrival_gap_ms=20.0
        )

        for task in train_tasks:
            simulator.scheduler.addtask(task)

        step_counter = 0
        while not simulator.scheduler.isempty():
            task = simulator.scheduler.getnexttask()
            if task is None:
                break

            simulator.currenttime = max(simulator.currenttime, task.arrivaltime)
            source = random.choice(simulator.iotdevices)
            state = simulator.getstate(task, source)
            action = simulator.select_apqs_action(state, task, source)

            latency, energy = simulator.executetask(task, action, source)
            task.actuallatency = latency
            task.energyconsumed = energy

            reward = simulator.calculatereward(task, latency, energy)
            nextstate = simulator.getstate(task, source)
            simulator.agent.remember(state, action, reward, nextstate, True)
            simulator.completedtasks.append(task)

            if len(simulator.agent.memory) >= 32 and step_counter % 10 == 0:
                simulator.agent.replay(32)
            step_counter += 1

        # Fix: decay epsilon using the correct attribute names from DQNAgent
        if hasattr(simulator.agent, 'epsilon') and hasattr(simulator.agent, 'epsilonmin') and hasattr(simulator.agent, 'epsilondecay'):
            simulator.agent.epsilon = max(
                simulator.agent.epsilonmin,
                simulator.agent.epsilon * simulator.agent.epsilondecay
            )

        print(f'Training episode {episode + 1}/{train_episodes} complete, epsilon={getattr(simulator.agent, "epsilon", None)}')

    if hasattr(simulator.agent, 'model'):
        simulator.agent.model.save_weights(MODEL_PATH)
        print(f'Saved trained APQS weights to: {MODEL_PATH}')
    else:
        raise RuntimeError('Could not find simulator.agent.model to save weights.')


def run_apqs_test_on_dataset(tasks, workload_config, model_path=MODEL_PATH, seed=0):
    set_all_seeds(seed)
    simulator = EdgeFogSimulator(num_iot=50, num_edge=5, num_fog=2, workload_config=workload_config)
    simulator.reset_runtime()

    if hasattr(simulator.agent, 'model') and os.path.exists(model_path):
        simulator.agent.model.load_weights(model_path)
    else:
        raise FileNotFoundError(f'Trained APQS model not found at: {model_path}')

    if hasattr(simulator.agent, 'epsilon'):
        simulator.agent.epsilon = 0.0

    for task in tasks:
        simulator.scheduler.addtask(task)

    while not simulator.scheduler.isempty():
        task = simulator.scheduler.getnexttask()
        if task is None:
            break

        simulator.currenttime = max(simulator.currenttime, task.arrivaltime)
        source = random.choice(simulator.iotdevices)
        state = simulator.getstate(task, source)
        action = simulator.select_apqs_action(state, task, source)

        latency, energy = simulator.executetask(task, action, source)
        task.actuallatency = latency
        task.energyconsumed = energy
        simulator.completedtasks.append(task)

    result = summarize_completed_tasks(simulator.completedtasks, len(tasks))

    del simulator
    if TF_AVAILABLE:
        tf.keras.backend.clear_session()

    return result


def run_baseline_on_dataset(schedulertype, tasks, seed=0):
    set_all_seeds(seed)
    iotdevices, edgenodes, fognodes, cloud = task_scheduler.build_baseline_environment()

    if schedulertype == 'static':
        scheduler = task_scheduler.StaticThresholdScheduler()
    elif schedulertype == 'fcfs':
        scheduler = task_scheduler.FCFSScheduler()
    elif schedulertype == 'ga':
        scheduler = task_scheduler.GeneticAlgorithmScheduler()
    elif schedulertype == 'fuzzy':
        scheduler = task_scheduler.FuzzyLogicScheduler()
    else:
        raise ValueError(f'Unknown scheduler type: {schedulertype}')

    gasolution = scheduler.optimize(tasks) if schedulertype == 'ga' else None
    completedtasks = []
    queuelength = len(tasks)

    for i, task in enumerate(tasks):
        currenttime = task.arrivaltime
        source = random.choice(iotdevices)

        if schedulertype == 'static':
            action = scheduler.decide(task)
        elif schedulertype == 'fcfs':
            action = scheduler.decide(task, queuelength)
        elif schedulertype == 'ga':
            action = scheduler.decide(i, gasolution)
        else:
            action = scheduler.decide(task, queuelength, random.uniform(40, 90))

        completedtask = execute_baseline_task_fixed(task, action, source, edgenodes, fognodes, cloud, currenttime)
        completedtasks.append(completedtask)
        queuelength -= 1

    return summarize_completed_tasks(completedtasks, len(tasks))

def load_checkpoint():
    """Load previously completed experiment results, if available."""
    if not os.path.exists(CHECKPOINT_CSV):
        return []

    checkpoint_df = pd.read_csv(CHECKPOINT_CSV)

    if checkpoint_df.empty:
        return []

    print(
        f'[RESUME] Loaded {len(checkpoint_df)} completed '
        f'experiments from {CHECKPOINT_CSV}'
    )

    return checkpoint_df.to_dict('records')


def save_checkpoint(rows):
    """Save all completed experiment results immediately."""
    if not rows:
        return

    checkpoint_df = pd.DataFrame(rows).sort_values(
        ['workload_type', 'approach', 'task_count', 'repeat']
    )

    checkpoint_df.to_csv(CHECKPOINT_CSV, index=False)

    print(
        f'[CHECKPOINT] Saved {len(checkpoint_df)} completed '
        f'experiments to {CHECKPOINT_CSV}'
    )
    
def aggregate_results(rows):
    df = pd.DataFrame(rows)
    numeric_cols = [
        'input_tasks', 'executed_tasks', 'avg_latency', 'p50_latency', 'p99_latency',
        'avg_energy', 'deadline_met_rate', 'failed_tasks',
        'local_count', 'edge_count', 'fog_count', 'cloud_count'
    ]

    agg_rows = []
    grouped = df.groupby(['workload_type', 'approach', 'task_count'], as_index=False)
    for (workload_type, approach, task_count), g in grouped:
        row = {
            'workload_type': workload_type,
            'approach': approach,
            'task_count': task_count,
            'n_repeats': len(g)
        }
        for col in numeric_cols:
            row[f'{col}_mean'] = float(g[col].mean())
            row[f'{col}_std'] = float(g[col].std(ddof=0))
        agg_rows.append(row)

    return pd.DataFrame(agg_rows).sort_values(['workload_type', 'approach', 'task_count'])


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---------------------------------------------------------
    # Phase 1: Make sure a trained APQS model is available.
    # ---------------------------------------------------------
    if os.path.exists(MODEL_PATH):
        print(f'[MODEL] Using existing trained APQS weights: {MODEL_PATH}')

    elif TRAIN_IF_MODEL_MISSING:
        print('=== Phase 1: Train APQS on heavy workload ===')
        heavy_train_cfg = HeavyWorkloadConfig()
        train_apqs_model(
            train_episodes=TRAIN_EPISODES,
            workload_config=heavy_train_cfg,
            seed=1234
        )

    else:
        raise FileNotFoundError(
            f'Trained APQS model not found at: {MODEL_PATH}\n'
            f'Run training intentionally first, or set '
            f'TRAIN_IF_MODEL_MISSING = True.'
        )

    # ---------------------------------------------------------
    # Phase 2: Evaluation with checkpoint/resume support.
    # ---------------------------------------------------------
    print('\n=== Evaluation with checkpoint/resume ===')

    rows = load_checkpoint()

    # Set of experiments already completed.
    completed = {
        (
            r['workload_type'],
            r['approach'],
            int(r['task_count']),
            int(r['repeat'])
        )
        for r in rows
    }

    total_experiments = (
        len(TASK_COUNTS)
        * NUM_REPEATS
        * 3       # Light, Medium, Heavy
        * 5       # APQS + 4 baselines
    )

    print(
        f'[STATUS] {len(completed)}/{total_experiments} '
        f'experiments already completed.'
    )

    workload_profiles = [
        ('Light', LightWorkloadConfig()),
        ('Medium', MediumWorkloadConfig()),
        ('Heavy', HeavyWorkloadConfig()),
    ]

    for workload_name, workload_cfg in workload_profiles:
        print(f'\n--- Running workload profile: {workload_name} ---')

        for task_count in TASK_COUNTS:

            for repeat in range(NUM_REPEATS):

                repeat_number = repeat + 1

                dataset_seed = (
                    9000
                    + (100 if workload_name == 'Heavy' else 0)
                    + task_count * 10
                    + repeat
                )

                # Generate the exact same deterministic dataset
                # regardless of whether this is a fresh or resumed run.
                base_tasks = generate_task_dataset(
                    task_count,
                    cfg=workload_cfg,
                    seed=dataset_seed,
                    arrival_gap_ms=20.0
                )

                experiments = [
                    (
                        'APQS',
                        lambda ts: run_apqs_test_on_dataset(
                            ts,
                            workload_config=workload_cfg,
                            model_path=MODEL_PATH,
                            seed=dataset_seed
                        )
                    ),
                    (
                        'Static Threshold',
                        lambda ts: run_baseline_on_dataset(
                            'static',
                            ts,
                            seed=dataset_seed
                        )
                    ),
                    (
                        'FCFS',
                        lambda ts: run_baseline_on_dataset(
                            'fcfs',
                            ts,
                            seed=dataset_seed
                        )
                    ),
                    (
                        'Genetic Algorithm',
                        lambda ts: run_baseline_on_dataset(
                            'ga',
                            ts,
                            seed=dataset_seed
                        )
                    ),
                    (
                        'Fuzzy Logic',
                        lambda ts: run_baseline_on_dataset(
                            'fuzzy',
                            ts,
                            seed=dataset_seed
                        )
                    ),
                ]

                for approach, runner in experiments:

                    experiment_key = (
                        workload_name,
                        approach,
                        task_count,
                        repeat_number
                    )

                    # -------------------------------------------------
                    # Resume logic:
                    # skip experiments that were already completed.
                    # -------------------------------------------------
                    if experiment_key in completed:
                        print(
                            f'[SKIP] {workload_name} | {approach} | '
                            f'tasks={task_count} | repeat={repeat_number}'
                        )
                        continue

                    task_copy = clone_task_list(base_tasks)

                    metrics = runner(task_copy)

                    result_row = {
                        'workload_type': workload_name,
                        'approach': approach,
                        'task_count': task_count,
                        'repeat': repeat_number,
                        'dataset_seed': dataset_seed,
                        **metrics
                    }

                    rows.append(result_row)
                    completed.add(experiment_key)

                    # -------------------------------------------------
                    # CRITICAL:
                    # Save immediately after every completed experiment.
                    # -------------------------------------------------
                    save_checkpoint(rows)

                    print(
                        f'Completed: {workload_name} | {approach} | '
                        f'tasks={task_count} | repeat={repeat_number}'
                    )

    # ---------------------------------------------------------
    # Final output generation.
    # ---------------------------------------------------------
    if not rows:
        raise RuntimeError('No experiment results were collected.')

    wtypes = [r['workload_type'] for r in rows]

    print('\n[DEBUG] workload_type counts in rows:')
    print(pd.Series(wtypes).value_counts())

    raw_df = pd.DataFrame(rows).sort_values(
        ['workload_type', 'approach', 'task_count', 'repeat']
    )

    raw_df.to_csv(RAW_CSV, index=False)

    agg_df = aggregate_results(rows)
    agg_df.to_csv(AGG_CSV, index=False)

    print(f'\nSaved raw results to: {RAW_CSV}')
    print(f'Saved aggregated results to: {AGG_CSV}')
    print(f'Raw rows: {len(raw_df)}')
    print(f'Aggregated rows: {len(agg_df)}')

    if len(completed) == total_experiments:
        print('\n=== ALL EXPERIMENTS COMPLETED ===')
    else:
        print(
            f'\n[WARNING] {len(completed)}/{total_experiments} '
            f'experiments completed.'
        )


if __name__ == '__main__':
    main()
