import os
import random
from collections import Counter
import gc
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
TRAIN_EPISODES = 100
OUTPUT_DIR = 'results_refined'
MODEL_PATH = os.path.join(OUTPUT_DIR, 'apqs_trained_refined.weights.h5')
RAW_CSV = os.path.join(OUTPUT_DIR, 'refined_train_test_raw.csv')
AGG_CSV = os.path.join(OUTPUT_DIR, 'refined_train_test_aggregated.csv')
CHECKPOINT_CSV = os.path.join(OUTPUT_DIR, 'refined_train_test_checkpoint.csv')
TRAIN_IF_MODEL_MISSING = True


class LightWorkloadConfig(WorkloadConfig):
    def __init__(self):
        super().__init__()
        self.size_low, self.size_high = 1, 5
        self.cpu_low, self.cpu_high = 500, 2000
        self.dl_low, self.dl_high = 30, 150


class MediumWorkloadConfig(WorkloadConfig):
    def __init__(self):
        super().__init__()
        self.size_low, self.size_high = 5, 15
        self.cpu_low, self.cpu_high = 2000, 8000
        self.dl_low, self.dl_high = 15, 60


class HeavyWorkloadConfig(WorkloadConfig):
    def __init__(self):
        super().__init__()
        self.size_low, self.size_high = 10, 30
        self.cpu_low, self.cpu_high = 4000, 15000
        self.dl_low, self.dl_high = 5, 30


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
        priority=task.priority
    )


def clone_task_list(tasks):
    return [clone_task(task) for task in tasks]


def generate_task_dataset(num_tasks, cfg, seed=0, arrival_gap_ms=20.0):
    set_all_seeds(seed)
    tasks = []
    current_time = 0.0
    for task_id in range(num_tasks):
        tasks.append(Task(
            task_id,
            current_time,
            random.uniform(cfg.size_low, cfg.size_high),
            random.randint(cfg.cpu_low, cfg.cpu_high),
            random.randint(cfg.dl_low, cfg.dl_high),
            random.randint(cfg.priority_low, cfg.priority_high)
        ))
        current_time += arrival_gap_ms
    return tasks


def summarize_completed_tasks(completedtasks, total_input_tasks):
    executed_tasks = len(completedtasks)
    accepted_objects = [
        task for task in completedtasks
        if task.meetsdeadline()
    ]
    accepted_tasks = len(accepted_objects)
    failed_tasks = executed_tasks - accepted_tasks

    latencies = np.asarray(
        [task.actuallatency for task in accepted_objects],
        dtype=float
    )
    energies = np.asarray(
        [task.energyconsumed for task in accepted_objects],
        dtype=float
    )
    locations = Counter(
        task.executionlocation for task in completedtasks
    )

    return {
        'input_tasks': int(total_input_tasks),
        'executed_tasks': int(executed_tasks),
        'accepted_tasks': int(accepted_tasks),
        'failed_tasks': int(failed_tasks),
        'avg_latency': float(np.mean(latencies)) if len(latencies) else 0.0,
        'p50_latency': float(np.percentile(latencies, 50)) if len(latencies) else 0.0,
        'p99_latency': float(np.percentile(latencies, 99)) if len(latencies) else 0.0,
        'avg_energy': float(np.mean(energies)) if len(energies) else 0.0,
        'deadline_met_rate': float(
            accepted_tasks / executed_tasks * 100.0
            if executed_tasks else 0.0
        ),
        'local_count': int(locations.get('local', 0)),
        'edge_count': int(locations.get('edge', 0)),
        'fog_count': int(locations.get('fog', 0)),
        'cloud_count': int(locations.get('cloud', 0))
    }


def execute_baseline_task_fixed(task, action, source, edgenodes,
                                fognodes, cloud, currenttime):
    if action == 0:
        node, location = source, 'local'
    elif action == 1:
        node, location = min(edgenodes, key=lambda n: source.distance_to(n)), 'edge'
    elif action == 2:
        node, location = min(fognodes, key=lambda n: source.distance_to(n)), 'fog'
    else:
        node, location = cloud, 'cloud'

    transmission = (
        0.0 if node.deviceid == source.deviceid
        else task.datasize * 8.0 / max(node.bandwidthmbps, 1e-9)
    )
    propagation = (
        0.0 if node.deviceid == source.deviceid
        else source.propagation_delay_to(node)
    )
    ready = currenttime + transmission + propagation
    waiting = max(0.0, node.busy_until - ready)
    execution = task.cpurequirement / max(node.mips, 1e-9)
    latency = transmission + propagation + waiting + execution
    energy = (
        node.powerwatts * execution / 1000.0
        + 0.2 * (transmission + propagation) / 1000.0
    )

    task.source_device = source
    task.starttime = ready + waiting
    task.completiontime = task.starttime + execution
    task.waitingtime = waiting
    task.servicetime = execution
    task.actuallatency = latency
    task.energyconsumed = energy
    task.executionlocation = location
    task.computenode = node.deviceid
    task.executed = True
    node.busy_until = task.completiontime
    return task


def train_apqs_model(train_episodes=TRAIN_EPISODES,
                     workload_config=None, seed=1234):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cfg = workload_config or HeavyWorkloadConfig()
    set_all_seeds(seed)
    simulator = EdgeFogSimulator(
        num_iot=50, num_edge=5, num_fog=2,
        workload_config=cfg
    )

    for episode in range(train_episodes):
        simulator.reset_runtime()
        train_tasks = generate_task_dataset(
            300, cfg, seed + episode, arrival_gap_ms=20.0
        )

        for task in train_tasks:
            task.source_device = random.choice(simulator.iotdevices)
            simulator.scheduler.addtask(task)

        step = 0
        while not simulator.scheduler.isempty():
            task = simulator.scheduler.getnexttask()
            if task is None:
                break
            simulator.currenttime = max(
                simulator.currenttime, task.arrivaltime
            )
            source = task.source_device
            state = simulator.getstate(task, source)
            action = simulator.select_apqs_action(state, task, source)
            latency, energy = simulator.executetask(task, action, source)
            reward = simulator.calculatereward(task, latency, energy)
            nextstate = simulator.getstate(task, source)
            simulator.agent.remember(
                state, action, reward, nextstate, True
            )
            simulator.completedtasks.append(task)
            if len(simulator.agent.memory) >= 32 and step % 10 == 0:
                simulator.agent.replay(32)
            step += 1

        if hasattr(simulator.agent, 'epsilon'):
            simulator.agent.epsilon = max(
                simulator.agent.epsilonmin,
                simulator.agent.epsilon * simulator.agent.epsilondecay
            )
        print(
            f'Training episode {episode + 1}/{train_episodes} '
            f'complete, epsilon={simulator.agent.epsilon:.6f}'
        )

    if not TF_AVAILABLE or simulator.agent.model is None:
        raise RuntimeError('TensorFlow model is unavailable.')
    simulator.agent.model.save_weights(MODEL_PATH)
    print(f'Saved trained APQS weights to: {MODEL_PATH}')


def run_apqs_test_on_dataset(
    tasks,
    workload_config,
    model_path=MODEL_PATH,
    seed=0
):
    set_all_seeds(seed)

    simulator = EdgeFogSimulator(
        num_iot=50,
        num_edge=5,
        num_fog=2,
        workload_config=workload_config
    )

    simulator.reset_runtime()

    if simulator.agent.model is None:
        raise RuntimeError(
            'TensorFlow model is unavailable.'
        )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f'Trained APQS model not found at: {model_path}'
        )

    simulator.agent.model.load_weights(model_path)
    simulator.agent.epsilon = 0.0

    for task in tasks:
        task.source_device = random.choice(
            simulator.iotdevices
        )
        simulator.scheduler.addtask(task)

    while not simulator.scheduler.isempty():
        task = simulator.scheduler.getnexttask()

        if task is None:
            break

        simulator.currenttime = max(
            simulator.currenttime,
            task.arrivaltime
        )

        source = task.source_device
        state = simulator.getstate(task, source)
        action = simulator.select_apqs_action(
            state,
            task,
            source
        )

        simulator.executetask(
            task,
            action,
            source
        )

        simulator.completedtasks.append(task)

    result = summarize_completed_tasks(
        simulator.completedtasks,
        len(tasks)
    )

    del simulator
    del tasks

    if TF_AVAILABLE:
        tf.keras.backend.clear_session()

    gc.collect()

    return result


def run_baseline_on_dataset(schedulertype, tasks, seed=0):
    set_all_seeds(seed)
    iotdevices, edgenodes, fognodes, cloud = (
        task_scheduler.build_baseline_environment()
    )

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

    for task in tasks:
        task.source_device = random.choice(iotdevices)

    solution = scheduler.optimize(tasks) if schedulertype == 'ga' else None
    completedtasks = []
    queue_length = len(tasks)

    for index, task in enumerate(tasks):
        source = task.source_device
        if schedulertype == 'static':
            action = scheduler.decide(task)
        elif schedulertype == 'fcfs':
            action = scheduler.decide(task, queue_length)
        elif schedulertype == 'ga':
            action = scheduler.decide(index, solution)
        else:
            action = scheduler.decide(
                task, queue_length, random.uniform(40, 90)
            )

        completedtasks.append(
            execute_baseline_task_fixed(
                task, action, source, edgenodes,
                fognodes, cloud, task.arrivaltime
            )
        )
        queue_length = max(0, queue_length - 1)

    return summarize_completed_tasks(completedtasks, len(tasks))


def load_checkpoint():
    if not os.path.exists(CHECKPOINT_CSV):
        return []
    checkpoint = pd.read_csv(CHECKPOINT_CSV)
    if checkpoint.empty:
        return []
    print(
        f'[RESUME] Loaded {len(checkpoint)} completed experiments '
        f'from {CHECKPOINT_CSV}'
    )
    return checkpoint.to_dict('records')


def save_checkpoint(rows):
    if not rows:
        return
    checkpoint = pd.DataFrame(rows).sort_values(
        ['workload_type', 'approach', 'task_count', 'repeat']
    )
    checkpoint.to_csv(CHECKPOINT_CSV, index=False)
    print(f'[CHECKPOINT] Saved {len(checkpoint)} experiments')


def aggregate_results(rows):
    dataframe = pd.DataFrame(rows)
    numeric_cols = [
        'input_tasks', 'executed_tasks', 'accepted_tasks',
        'avg_latency', 'p50_latency', 'p99_latency',
        'avg_energy', 'deadline_met_rate', 'failed_tasks',
        'local_count', 'edge_count', 'fog_count', 'cloud_count'
    ]

    output = []
    grouped = dataframe.groupby(
        ['workload_type', 'approach', 'task_count']
    )
    for (workload, approach, task_count), group in grouped:
        row = {
            'workload_type': workload,
            'approach': approach,
            'task_count': task_count,
            'n_repeats': len(group)
        }
        for column in numeric_cols:
            row[f'{column}_mean'] = float(group[column].mean())
            row[f'{column}_std'] = float(group[column].std(ddof=0))
        output.append(row)

    return pd.DataFrame(output).sort_values(
        ['workload_type', 'approach', 'task_count']
    )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if os.path.exists(MODEL_PATH):
        print(f'[MODEL] Using existing weights: {MODEL_PATH}')
    elif TRAIN_IF_MODEL_MISSING:
        train_apqs_model(
            TRAIN_EPISODES,
            HeavyWorkloadConfig(),
            seed=1234
        )
    else:
        raise FileNotFoundError(
            f'Model not found: {MODEL_PATH}. '
            f'Set TRAIN_IF_MODEL_MISSING=True to train it.'
        )

    rows = load_checkpoint()
    completed = {
        (
            row['workload_type'], row['approach'],
            int(row['task_count']), int(row['repeat'])
        )
        for row in rows
    }

    profiles = [
        ('Light', LightWorkloadConfig()),
        ('Medium', MediumWorkloadConfig()),
        ('Heavy', HeavyWorkloadConfig())
    ]

    for workload_name, workload_cfg in profiles:
        for task_count in TASK_COUNTS:
            for repeat in range(NUM_REPEATS):
                repeat_number = repeat + 1
                dataset_seed = (
                    9000
                    + (100 if workload_name == 'Heavy' else 0)
                    + task_count * 10
                    + repeat
                )
                base_tasks = generate_task_dataset(
                    task_count, workload_cfg,
                    dataset_seed, arrival_gap_ms=20.0
                )

                experiments = [
                    ('APQS', lambda ts: run_apqs_test_on_dataset(
                        ts, workload_cfg, MODEL_PATH, dataset_seed
                    )),
                    ('Static Threshold', lambda ts: run_baseline_on_dataset(
                        'static', ts, dataset_seed
                    )),
                    ('FCFS', lambda ts: run_baseline_on_dataset(
                        'fcfs', ts, dataset_seed
                    )),
                    ('Genetic Algorithm', lambda ts: run_baseline_on_dataset(
                        'ga', ts, dataset_seed
                    )),
                    ('Fuzzy Logic', lambda ts: run_baseline_on_dataset(
                        'fuzzy', ts, dataset_seed
                    ))
                ]

                for approach, runner in experiments:
                    key = (
                        workload_name, approach,
                        task_count, repeat_number
                    )
                    if key in completed:
                        continue

                    metrics = runner(clone_task_list(base_tasks))
                    rows.append({
                        'workload_type': workload_name,
                        'approach': approach,
                        'task_count': task_count,
                        'repeat': repeat_number,
                        'dataset_seed': dataset_seed,
                        **metrics
                    })
                    completed.add(key)
                    save_checkpoint(rows)
                    print(
                        f'Completed {workload_name} | {approach} | '
                        f'{task_count} tasks | repeat {repeat_number}'
                    )

    if not rows:
        raise RuntimeError('No experiment results were collected.')

    raw = pd.DataFrame(rows).sort_values(
        ['workload_type', 'approach', 'task_count', 'repeat']
    )
    raw.to_csv(RAW_CSV, index=False)
    aggregate_results(rows).to_csv(AGG_CSV, index=False)

    print(f'Saved raw results to: {RAW_CSV}')
    print(f'Saved aggregated results to: {AGG_CSV}')
    print(f'Raw rows: {len(raw)}')


if __name__ == '__main__':
    main()
