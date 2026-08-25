import os
import json
import random
from dataclasses import dataclass
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs('output', exist_ok=True)
os.makedirs('results', exist_ok=True)

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except Exception:
    tf = None
    keras = None
    TF_AVAILABLE = False

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
if TF_AVAILABLE:
    tf.random.set_seed(SEED)


@dataclass
class WorkloadConfig:
    size_low: float = 10
    size_high: float = 200
    cpu_low: int = 100
    cpu_high: int = 2000
    dl_low: int = 20
    dl_high: int = 150
    priority_low: int = 1
    priority_high: int = 10


class Task:
    def __init__(self, taskid, arrivaltime, datasize, cpurequirement,
                 deadline, priority=5):
        self.taskid = taskid
        self.arrivaltime = float(arrivaltime)
        self.datasize = float(datasize)
        self.cpurequirement = float(cpurequirement)
        self.deadline = float(deadline)
        self.priority = int(priority)
        self.source_device = None

        self.starttime = None
        self.completiontime = None
        self.executionlocation = None
        self.energyconsumed = 0.0
        self.actuallatency = 0.0
        self.waitingtime = 0.0
        self.servicetime = 0.0
        self.computenode = None
        self.executed = False
        self.accepted = False
        self.rejected = False
        self.rejectionreason = None

    def meetsdeadline(self):
        return self.executed and self.actuallatency <= self.deadline


class Device:
    def __init__(self, deviceid, devicetype, cpughz, memorygb,
                 bandwidthmbps, powerwatts, location=(0, 0),
                 propagation_factor_ms_per_unit=0.005):
        self.deviceid = deviceid
        self.devicetype = devicetype
        self.cpughz = float(cpughz)
        self.memorygb = float(memorygb)
        self.bandwidthmbps = float(bandwidthmbps)
        self.powerwatts = float(powerwatts)
        self.location = location
        self.availablecpu = 100.0
        self.availablememory = float(memorygb)
        self.battery = 100.0
        self.busy_until = 0.0
        self.propagation_factor_ms_per_unit = propagation_factor_ms_per_unit

    def distance_to(self, other):
        x1, y1 = self.location
        x2, y2 = other.location
        return float(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))

    def propagation_delay_to(self, other):
        return self.distance_to(other) * self.propagation_factor_ms_per_unit

    @property
    def mips(self):
        return self.cpughz * 1000.0


class DQNAgent:
    def __init__(self, statesize=30, actionsize=4):
        self.statesize = statesize
        self.actionsize = actionsize
        self.memory = deque(maxlen=5000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilondecay = 0.995
        self.epsilonmin = 0.05
        self.learningrate = 0.001
        self.model = self.buildmodel() if TF_AVAILABLE else None

    def buildmodel(self):
        model = keras.Sequential([
            keras.layers.Input(shape=(self.statesize,)),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(self.actionsize, activation='linear')
        ])
        model.compile(
            loss='mse',
            optimizer=keras.optimizers.Adam(
                learning_rate=self.learningrate
            )
        )
        return model

    def remember(self, state, action, reward, nextstate, done):
        self.memory.append((state, action, reward, nextstate, done))

    def replay(self, batchsize=32):
        if not TF_AVAILABLE or self.model is None:
            return
        if len(self.memory) < batchsize:
            return

        minibatch = random.sample(self.memory, batchsize)
        states = np.vstack([item[0] for item in minibatch])
        nextstates = np.vstack([item[3] for item in minibatch])
        targets = self.model.predict(states, verbose=0)
        nextq = self.model.predict(nextstates, verbose=0)

        for i, (_, action, reward, _, done) in enumerate(minibatch):
            targets[i][action] = (
                reward
                if done
                else reward + self.gamma * np.max(nextq[i])
            )

        self.model.fit(states, targets, epochs=1, verbose=0)
        if self.epsilon > self.epsilonmin:
            self.epsilon = max(
                self.epsilonmin,
                self.epsilon * self.epsilondecay
            )


class PriorityScheduler:
    def __init__(self):
        self.highqueue = deque()
        self.mediumqueue = deque()
        self.lowqueue = deque()

    def addtask(self, task):
        if task.priority >= 7:
            self.highqueue.append(task)
        elif task.priority >= 4:
            self.mediumqueue.append(task)
        else:
            self.lowqueue.append(task)

    def getnexttask(self):
        if self.highqueue:
            return self.highqueue.popleft()
        if self.mediumqueue:
            return self.mediumqueue.popleft()
        if self.lowqueue:
            return self.lowqueue.popleft()
        return None

    def isempty(self):
        return not (
            self.highqueue or self.mediumqueue or self.lowqueue
        )

    def queuelength(self):
        return (
            len(self.highqueue)
            + len(self.mediumqueue)
            + len(self.lowqueue)
        )


class NoPriorityScheduler:
    def __init__(self):
        self.queue = deque()

    def addtask(self, task):
        self.queue.append(task)

    def getnexttask(self):
        return self.queue.popleft() if self.queue else None

    def isempty(self):
        return not self.queue

    def queuelength(self):
        return len(self.queue)


class EdgeFogSimulator:
    ACTION_NAMES = ['local', 'edge', 'fog', 'cloud']

    def __init__(self, num_iot=50, num_edge=5, num_fog=2,
                 workload_config=None):
        self.numiot = num_iot
        self.numedge = num_edge
        self.numfog = num_fog
        self.workload_config = workload_config or WorkloadConfig()
        self.iotdevices = self.createiotdevices()
        self.edgenodes = self.createedgenodes()
        self.fognodes = self.createfognodes()
        self.cloud = self.createcloud()
        self.agent = DQNAgent(statesize=30, actionsize=4)
        self.scheduler = PriorityScheduler()
        self.reset_runtime()

    def reset_runtime(self):
        self.completedtasks = []
        self.submittedtasks = []
        self.queuedtasks = []
        self.rejectedtasks = []
        self.currenttime = 0.0
        self.taskcounter = 0
        self.scheduler = self.scheduler.__class__()

        for device in (
            self.iotdevices
            + self.edgenodes
            + self.fognodes
            + [self.cloud]
        ):
            device.availablecpu = 100.0
            device.availablememory = device.memorygb
            device.battery = 100.0
            device.busy_until = 0.0

    def createiotdevices(self):
        return [
            Device(
                f'iot{i}', 'iot', random.uniform(0.5, 1.5),
                random.uniform(0.5, 2.0), random.uniform(5, 20),
                random.uniform(1, 5),
                (random.uniform(0, 500), random.uniform(0, 500))
            )
            for i in range(self.numiot)
        ]

    def createedgenodes(self):
        return [
            Device(
                f'edge{i}', 'edge', random.uniform(2, 4),
                random.uniform(8, 16), random.uniform(100, 300),
                random.uniform(50, 100),
                (random.uniform(100, 400), random.uniform(100, 400))
            )
            for i in range(self.numedge)
        ]

    def createfognodes(self):
        return [
            Device(
                f'fog{i}', 'fog', random.uniform(4, 8),
                random.uniform(32, 64), random.uniform(500, 1000),
                random.uniform(150, 300), (250, 250)
            )
            for i in range(self.numfog)
        ]

    def createcloud(self):
        return Device(
            'cloud0', 'cloud', 16.0, 256.0,
            5000.0, 15.0, (250, 250)
        )

    def generatetask(self):
        cfg = self.workload_config
        task = Task(
            self.taskcounter, self.currenttime,
            random.uniform(cfg.size_low, cfg.size_high),
            random.randint(cfg.cpu_low, cfg.cpu_high),
            random.randint(cfg.dl_low, cfg.dl_high),
            random.randint(cfg.priority_low, cfg.priority_high)
        )
        self.taskcounter += 1
        return task

    def candidate_nodes(self, source):
        return [
            source,
            min(self.edgenodes, key=lambda n: source.distance_to(n)),
            min(self.fognodes, key=lambda n: source.distance_to(n)),
            self.cloud
        ]

    def estimate_node_outcome(self, task, node, source):
        if node.deviceid == source.deviceid:
            transmission = 0.0
            propagation = 0.0
        else:
            transmission = (
                task.datasize * 8.0
            ) / max(node.bandwidthmbps, 1e-9)
            propagation = source.propagation_delay_to(node)

        ready = self.currenttime + transmission + propagation
        waiting = max(0.0, node.busy_until - ready)
        execution = task.cpurequirement / max(node.mips, 1e-9)
        latency = transmission + propagation + waiting + execution
        energy = node.powerwatts * execution / 1000.0
        return latency, energy

    def admit_task(self, task, source):
        task.source_device = source
        self.queuedtasks.append(task)
        self.scheduler.addtask(task)
        return True

    def getstate(self, task, source):
        def norm(value, scale):
            return float(value) / scale if scale > 0 else 0.0

        queue_len = self.scheduler.queuelength()
        state = [
            norm(task.datasize, 200.0),
            norm(task.cpurequirement, 20000.0),
            norm(task.deadline, 150.0),
            norm(task.priority, 10.0),
            norm(task.deadline, 150.0),
            norm(queue_len, 2000.0)
        ]

        for node in self.candidate_nodes(source):
            if node.deviceid == source.deviceid:
                transmission = 0.0
                propagation = 0.0
            else:
                transmission = (
                    task.datasize * 8.0
                ) / max(node.bandwidthmbps, 1e-9)
                propagation = source.propagation_delay_to(node)

            ready = self.currenttime + transmission + propagation
            waiting = max(0.0, node.busy_until - ready)
            execution = task.cpurequirement / max(node.mips, 1e-9)
            finish = transmission + propagation + waiting + execution
            slack = task.deadline - finish

            state.extend([
                norm(node.mips, 20000.0),
                norm(node.powerwatts, 500.0),
                norm(waiting, 500.0),
                norm(execution, 500.0),
                norm(finish, 1000.0),
                norm(slack, 200.0)
            ])

        return np.asarray(state, dtype=np.float32)

    def map_action_to_node(self, action, source):
        if action == 0:
            return source, 'local'
        if action == 1:
            return min(self.edgenodes, key=lambda n: source.distance_to(n)), 'edge'
        if action == 2:
            return min(self.fognodes, key=lambda n: source.distance_to(n)), 'fog'
        return self.cloud, 'cloud'

    def executetask(self, task, action, source):
        node, location = self.map_action_to_node(action, source)

        if node.deviceid == source.deviceid:
            transmission = 0.0
            propagation = 0.0
        else:
            transmission = (
                task.datasize * 8.0
            ) / max(node.bandwidthmbps, 1e-9)
            propagation = source.propagation_delay_to(node)

        ready = self.currenttime + transmission + propagation
        waiting = max(0.0, node.busy_until - ready)
        execution = task.cpurequirement / max(node.mips, 1e-9)
        latency = transmission + propagation + waiting + execution
        energy = (
            node.powerwatts * execution / 1000.0
            + 0.2 * (transmission + propagation) / 1000.0
        )

        task.waitingtime = waiting
        task.servicetime = execution
        task.executionlocation = location
        task.computenode = node.deviceid
        task.starttime = ready + waiting
        task.completiontime = task.starttime + execution
        task.actuallatency = latency
        task.energyconsumed = energy
        task.executed = True

        node.busy_until = task.completiontime
        node.availablecpu = max(
            5.0,
            100.0 * max(
                0.0,
                1.0 - min(1.0, execution / max(task.deadline, 1.0))
            )
        )
        return latency, energy

    def calculatereward(self, task, latency, energy):
        deadline = max(task.deadline, 1e-6)
        priority_weight = 1.0 + 0.2 * task.priority
        normalized_latency = latency / deadline
        waiting_penalty = task.waitingtime / deadline

        if latency <= task.deadline:
            reward = 20.0 * priority_weight
            reward += 10.0 * (task.deadline - latency) / deadline
            reward -= 3.0 * normalized_latency
            reward -= 0.2 * energy
            reward -= 2.0 * waiting_penalty
        else:
            reward = -35.0 * priority_weight
            reward -= 25.0 * (latency - task.deadline) / deadline
            reward -= 0.5 * energy
            reward -= 3.0 * waiting_penalty

        return float(np.clip(reward, -100.0, 40.0))

    def estimate_action_outcome(self, task, action, source):
        node, _ = self.map_action_to_node(action, source)
        latency, energy = self.estimate_node_outcome(task, node, source)
        return {'latency': latency, 'energy': energy}

    def get_valid_actions(self, task, source):
        valid = []
        all_actions = []

        for action in range(4):
            info = self.estimate_action_outcome(task, action, source)
            all_actions.append((action, info['latency']))
            if info['latency'] <= task.deadline:
                valid.append(action)

        if valid:
            return valid

        all_actions.sort(key=lambda item: item[1])
        return [all_actions[0][0]]

    def select_apqs_action(self, state, task, source):
        valid_actions = self.get_valid_actions(task, source)

        if random.random() <= self.agent.epsilon:
            return random.choice(valid_actions)
        if not TF_AVAILABLE or self.agent.model is None:
            return random.choice(valid_actions)

        q_values = self.agent.model.predict(
            state[np.newaxis, :], verbose=0
        )[0]
        bias = np.array([0.05, 0.02, 0.0, -0.05], dtype=np.float32)
        masked = np.full_like(q_values, -1e9)
        for action in valid_actions:
            masked[action] = q_values[action] + bias[action]
        return int(np.argmax(masked))

    def runsimulation(self, durationms=30000, arrivalrate=0.1, verbose=False):
        self.reset_runtime()
        episode = 0

        while self.currenttime < durationms:
            if random.random() < arrivalrate:
                task = self.generatetask()
                task.source_device = random.choice(self.iotdevices)
                self.submittedtasks.append(task)
                self.admit_task(task, task.source_device)

            while not self.scheduler.isempty():
                task = self.scheduler.getnexttask()
                source = task.source_device
                state = self.getstate(task, source)
                action = self.select_apqs_action(state, task, source)
                latency, energy = self.executetask(task, action, source)
                reward = self.calculatereward(task, latency, energy)
                nextstate = self.getstate(task, source)
                self.agent.remember(state, action, reward, nextstate, True)
                self.completedtasks.append(task)

            if len(self.agent.memory) >= 32 and episode % 10 == 0:
                self.agent.replay(32)

            episode += 1
            self.currenttime += 50.0

        return self.calculateresults()

    def calculateresults(self):
        executed_tasks = len(self.completedtasks)
        accepted_objects = [
            task for task in self.completedtasks
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

        return {
            'submitted_tasks': len(self.submittedtasks),
            'executed_tasks': executed_tasks,
            'accepted_tasks': accepted_tasks,
            'failed_tasks': failed_tasks,
            'rejected_tasks': 0,
            'completed_tasks': executed_tasks,
            'total_tasks': executed_tasks,
            'avg_latency': float(np.mean(latencies)) if len(latencies) else 0.0,
            'p50_latency': float(np.percentile(latencies, 50)) if len(latencies) else 0.0,
            'p99_latency': float(np.percentile(latencies, 99)) if len(latencies) else 0.0,
            'avg_energy': float(np.mean(energies)) if len(energies) else 0.0,
            'deadline_met_rate': float(
                accepted_tasks / executed_tasks * 100.0
                if executed_tasks else 0.0
            ),
            'acceptance_rate': float(
                accepted_tasks / len(self.submittedtasks) * 100.0
                if self.submittedtasks else 0.0
            )
        }


class StaticThresholdScheduler:
    def __init__(self):
        self.thresholdcloud = 120
        self.thresholdfog = 60

    def decide(self, task):
        if task.datasize >= self.thresholdcloud:
            return 3
        if task.datasize >= self.thresholdfog:
            return 2
        return 0


class FCFSScheduler:
    def decide(self, task, queuelength):
        if queuelength >= 30:
            return 3
        if queuelength >= 15:
            return 2
        return 1


class GeneticAlgorithmScheduler:
    def __init__(self, populationsize=20, generations=20, mutationrate=0.1):
        self.populationsize = populationsize
        self.generations = generations
        self.mutationrate = mutationrate

    def createindividual(self, numtasks):
        return [random.randint(0, 3) for _ in range(numtasks)]

    def estimated_latency_energy(self, task, location):
        values = [
            (1000.0, 3.0, 0.0),
            (3000.0, 75.0, task.datasize * 8.0 / 200.0),
            (6000.0, 225.0, task.datasize * 8.0 / 750.0),
            (16000.0, 15.0, task.datasize * 8.0 / 5000.0)
        ]
        mips, power, transmission = values[location]
        execution = task.cpurequirement / mips
        return transmission + execution, power * execution / 1000.0

    def evaluatefitness(self, individual, tasks):
        total = 0.0
        for index, task in enumerate(tasks):
            latency, energy = self.estimated_latency_energy(
                task, individual[index]
            )
            penalty = 0.0 if latency <= task.deadline else 1000.0
            total += 0.5 * latency + 0.2 * energy * 1000.0 + 0.3 * penalty
        return total

    def crossover(self, parent1, parent2):
        if len(parent1) <= 1:
            return parent1.copy(), parent2.copy()
        point = random.randint(1, len(parent1) - 1)
        return (
            parent1[:point] + parent2[point:],
            parent2[:point] + parent1[point:]
        )

    def mutate(self, individual):
        result = individual.copy()
        for index in range(len(result)):
            if random.random() < self.mutationrate:
                result[index] = random.randint(0, 3)
        return result

    def optimize(self, tasks):
        if not tasks:
            return []

        population = [
            self.createindividual(len(tasks))
            for _ in range(self.populationsize)
        ]
        best = population[0]
        bestfitness = float('inf')

        for _ in range(self.generations):
            fitness = [
                self.evaluatefitness(individual, tasks)
                for individual in population
            ]
            order = np.argsort(fitness)
            best_index = int(order[0])

            if fitness[best_index] < bestfitness:
                bestfitness = fitness[best_index]
                best = population[best_index].copy()

            newpopulation = [population[best_index].copy()]
            while len(newpopulation) < self.populationsize:
                parent1, parent2 = random.sample(population, 2)
                child1, child2 = self.crossover(parent1, parent2)
                newpopulation.extend([
                    self.mutate(child1),
                    self.mutate(child2)
                ])
            population = newpopulation[:self.populationsize]

        return best

    def decide(self, taskindex, solution):
        return solution[taskindex] if taskindex < len(solution) else 0


class FuzzyLogicScheduler:
    def decide(self, task, queuelength, cpuavailable):
        score = 0
        score += 3 if task.datasize > 100 else 2 if task.datasize > 50 else 1
        score += 2 if queuelength > 25 else 1 if queuelength > 15 else 0
        score += 2 if cpuavailable < 30 else 1 if cpuavailable < 60 else 0
        if score >= 5:
            return 3
        if score >= 3:
            return 2
        return 1


def build_baseline_environment():
    iotdevices = [
        Device(
            f'iot{i}', 'iot', 1.0, 1.0, 10, 3,
            (random.uniform(0, 500), random.uniform(0, 500))
        )
        for i in range(50)
    ]
    edgenodes = [
        Device(
            f'edge{i}', 'edge', 3.0, 12.0, 200, 75,
            (random.uniform(100, 400), random.uniform(100, 400))
        )
        for i in range(5)
    ]
    fognodes = [
        Device(
            f'fog{i}', 'fog', 6.0, 48.0, 750, 225, (250, 250)
        )
        for i in range(2)
    ]
    cloud = Device(
        'cloud0', 'cloud', 16.0, 256.0,
        5000.0, 15.0, (250, 250)
    )
    return iotdevices, edgenodes, fognodes, cloud


def execute_baseline_task(task, action, source, edgenodes,
                          fognodes, cloud, currenttime):
    if action == 0:
        node, location = source, 'local'
    elif action == 1:
        node, location = min(
            edgenodes, key=lambda n: source.distance_to(n)
        ), 'edge'
    elif action == 2:
        node, location = min(
            fognodes, key=lambda n: source.distance_to(n)
        ), 'fog'
    else:
        node, location = cloud, 'cloud'

    transmission = (
        0.0
        if node.deviceid == source.deviceid
        else task.datasize * 8.0 / node.bandwidthmbps
    )
    propagation = (
        0.0
        if node.deviceid == source.deviceid
        else source.propagation_delay_to(node)
    )
    waiting = max(0.0, node.busy_until - currenttime)
    execution = task.cpurequirement / node.mips
    latency = transmission + propagation + waiting + execution
    energy = (
        node.powerwatts * execution / 1000.0
        + 0.2 * (transmission + propagation) / 1000.0
    )

    task.actuallatency = latency
    task.energyconsumed = energy
    task.executionlocation = location
    task.computenode = node.deviceid
    task.executed = True
    node.busy_until = currenttime + latency
    return task


def run_baseline_simulation(schedulertype, durationms=30000,
                            arrivalrate=0.1, workload_config=None):
    cfg = workload_config or WorkloadConfig()
    iotdevices, edgenodes, fognodes, cloud = build_baseline_environment()

    if schedulertype == 'static':
        scheduler = StaticThresholdScheduler()
    elif schedulertype == 'fcfs':
        scheduler = FCFSScheduler()
    elif schedulertype == 'ga':
        scheduler = GeneticAlgorithmScheduler()
    elif schedulertype == 'fuzzy':
        scheduler = FuzzyLogicScheduler()
    else:
        raise ValueError(f'Unknown scheduler type: {schedulertype}')

    tasks = []
    currenttime = 0.0
    taskid = 0

    while currenttime < durationms:
        if random.random() < arrivalrate:
            task = Task(
                taskid, currenttime,
                random.uniform(cfg.size_low, cfg.size_high),
                random.randint(cfg.cpu_low, cfg.cpu_high),
                random.randint(cfg.dl_low, cfg.dl_high),
                random.randint(cfg.priority_low, cfg.priority_high)
            )
            task.source_device = random.choice(iotdevices)
            tasks.append(task)
            taskid += 1
        currenttime += 50.0

    solution = scheduler.optimize(tasks) if schedulertype == 'ga' else None
    completedtasks = []
    currenttime = 0.0

    for index, task in enumerate(tasks):
        source = task.source_device
        remaining = len(tasks) - index

        if schedulertype == 'static':
            action = scheduler.decide(task)
        elif schedulertype == 'fcfs':
            action = scheduler.decide(task, remaining)
        elif schedulertype == 'ga':
            action = scheduler.decide(index, solution)
        else:
            action = scheduler.decide(
                task, remaining, random.uniform(40, 90)
            )

        completedtasks.append(
            execute_baseline_task(
                task, action, source, edgenodes,
                fognodes, cloud, currenttime
            )
        )
        currenttime += 50.0

    accepted_objects = [
        task for task in completedtasks
        if task.meetsdeadline()
    ]
    accepted_tasks = len(accepted_objects)
    executed_tasks = len(completedtasks)
    failed_tasks = executed_tasks - accepted_tasks

    latencies = np.asarray(
        [task.actuallatency for task in accepted_objects],
        dtype=float
    )
    energies = np.asarray(
        [task.energyconsumed for task in accepted_objects],
        dtype=float
    )

    return {
        'submitted_tasks': len(tasks),
        'executed_tasks': executed_tasks,
        'accepted_tasks': accepted_tasks,
        'rejected_tasks': 0,
        'completed_tasks': executed_tasks,
        'total_tasks': executed_tasks,
        'avg_latency': float(np.mean(latencies)) if len(latencies) else 0.0,
        'p50_latency': float(np.percentile(latencies, 50)) if len(latencies) else 0.0,
        'p99_latency': float(np.percentile(latencies, 99)) if len(latencies) else 0.0,
        'avg_energy': float(np.mean(energies)) if len(energies) else 0.0,
        'deadline_met_rate': float(
            accepted_tasks / executed_tasks * 100.0
            if executed_tasks else 0.0
        ),
        'failed_tasks': failed_tasks,
        'acceptance_rate': float(
            accepted_tasks / len(tasks) * 100.0
            if tasks else 0.0
        )
    }


class ResultsAnalyzer:
    @staticmethod
    def compareall(resultsdict):
        rows = []
        for name, results in resultsdict.items():
            rows.append({
                'Approach': name,
                'Total Tasks': results['total_tasks'],
                'Executed Tasks': results.get('executed_tasks', 0),
                'Accepted Tasks': results.get('accepted_tasks', 0),
                'Avg Latency (ms)': round(results['avg_latency'], 4),
                'P50 Latency (ms)': round(results.get('p50_latency', 0.0), 4),
                'P99 Latency (ms)': round(results['p99_latency'], 4),
                'Avg Energy (Wh)': round(results['avg_energy'], 6),
                'Deadline Met (%)': round(results['deadline_met_rate'], 2),
                'Failed Tasks': results['failed_tasks']
            })
        return pd.DataFrame(rows)

    @staticmethod
    def plotcomparison(allresults, path='output/comparison.png'):
        approaches = list(allresults.keys())
        latencies = [allresults[a]['avg_latency'] for a in approaches]
        energies = [allresults[a]['avg_energy'] for a in approaches]
        deadline_rates = [allresults[a]['deadline_met_rate'] for a in approaches]
        colors = [
            'tab:blue', 'tab:orange', 'tab:green',
            'tab:red', 'tab:purple'
        ][:len(approaches)]

        _, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].bar(approaches, latencies, color=colors)
        axes[0].set_title('Average Latency of Deadline-Met Tasks')
        axes[1].bar(approaches, energies, color=colors)
        axes[1].set_title('Average Energy of Deadline-Met Tasks')
        axes[2].bar(approaches, deadline_rates, color=colors)
        axes[2].set_title('Deadline Met (%)')

        for axis in axes:
            axis.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()


def aggregate_results(results):
    keys = [
        'submitted_tasks', 'executed_tasks', 'accepted_tasks',
        'rejected_tasks', 'completed_tasks', 'total_tasks',
        'avg_latency', 'p50_latency', 'p99_latency',
        'avg_energy', 'deadline_met_rate', 'failed_tasks',
        'acceptance_rate'
    ]
    output = {}
    for key in keys:
        output[key] = float(
            np.mean([result.get(key, 0.0) for result in results])
        )
    output['std_latency'] = float(
        np.std([result.get('avg_latency', 0.0) for result in results])
    )
    output['std_deadline'] = float(
        np.std([
            result.get('deadline_met_rate', 0.0)
            for result in results
        ])
    )
    return output


def ablation_apqs_nopq(durationms=30000, arrivalrate=0.1, num_runs=5):
    results = []
    for _ in range(num_runs):
        simulator = EdgeFogSimulator()
        simulator.scheduler = NoPriorityScheduler()
        results.append(
            simulator.runsimulation(durationms, arrivalrate)
        )
    return aggregate_results(results)


def ablation_apqs_noenergy(durationms=30000, arrivalrate=0.1,
                           num_runs=5):
    original = EdgeFogSimulator.calculatereward

    def calculatereward_noenergy(self, task, latency, energy):
        latency_penalty = latency / max(task.deadline, 1.0)
        deadline_term = 1.0 if latency <= task.deadline else -1.0
        priority_term = task.priority / 10.0 * deadline_term
        return float(
            -0.45 * latency_penalty
            + 0.45 * deadline_term
            + 0.10 * priority_term
        )

    results = []
    try:
        EdgeFogSimulator.calculatereward = calculatereward_noenergy
        for _ in range(num_runs):
            results.append(
                EdgeFogSimulator().runsimulation(
                    durationms, arrivalrate
                )
            )
    finally:
        EdgeFogSimulator.calculatereward = original

    return aggregate_results(results)


def run_ablation_study(durationms=30000, arrivalrate=0.2, num_runs=5):
    results = {
        'APQS-NoPQ': ablation_apqs_nopq(
            durationms, arrivalrate, num_runs
        ),
        'APQS-NoEnergy': ablation_apqs_noenergy(
            durationms, arrivalrate, num_runs
        )
    }

    full = [
        EdgeFogSimulator().runsimulation(
            durationms, arrivalrate
        )
        for _ in range(num_runs)
    ]
    results['APQS (Full)'] = aggregate_results(full)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(
        os.path.join('output', f'ablation_study_{timestamp}.json'),
        'w', encoding='utf-8'
    ) as file:
        json.dump(results, file, indent=2)

    return results


def main():
    duration = 30000
    arrivalrate = 0.1
    simulator = EdgeFogSimulator()

    allresults = {
        'DQN + Priority': simulator.runsimulation(
            duration, arrivalrate
        ),
        'Static Threshold': run_baseline_simulation(
            'static', duration, arrivalrate
        ),
        'FCFS': run_baseline_simulation(
            'fcfs', duration, arrivalrate
        ),
        'Genetic Algorithm': run_baseline_simulation(
            'ga', duration, arrivalrate
        ),
        'Fuzzy Logic': run_baseline_simulation(
            'fuzzy', duration, arrivalrate
        )
    }

    analyzer = ResultsAnalyzer()
    dataframe = analyzer.compareall(allresults)
    dataframe.to_csv('output/debugged_results.csv', index=False)
    analyzer.plotcomparison(allresults, 'output/comparison.png')
    return allresults, dataframe


if __name__ == '__main__':
    main()
