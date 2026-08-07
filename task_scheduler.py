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
    def __init__(self, taskid, arrivaltime, datasize, cpurequirement, deadline, priority=5):
        self.taskid = taskid
        self.arrivaltime = arrivaltime
        self.datasize = float(datasize)
        self.cpurequirement = float(cpurequirement)
        self.deadline = float(deadline)
        self.priority = int(priority)
        self.starttime = None
        self.completiontime = None
        self.executionlocation = None
        self.energyconsumed = 0.0
        self.actuallatency = 0.0
        self.waitingtime = 0.0
        self.servicetime = 0.0
        self.computenode = None

    def meetsdeadline(self):
        return self.actuallatency <= self.deadline


class Device:
    def __init__(self, deviceid, devicetype, cpughz, memorygb, bandwidthmbps, powerwatts,
                 location=(0, 0), propagation_factor_ms_per_unit=0.005):
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
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=self.learningrate))
        return model

    def remember(self, state, action, reward, nextstate, done):
        self.memory.append((state, action, reward, nextstate, done))

    def act(self, state):
        if (not TF_AVAILABLE) or (self.model is None):
            return random.randrange(self.actionsize)
        if np.random.random() <= self.epsilon:
            return random.randrange(self.actionsize)
        state_input = state[np.newaxis, :] if state.ndim == 1 else state
        qvalues = self.model.predict(state_input, verbose=0)[0]
        return int(np.argmax(qvalues))

    def replay(self, batchsize=32):
        if (not TF_AVAILABLE) or (self.model is None) or len(self.memory) < batchsize:
            return
        minibatch = random.sample(self.memory, batchsize)
        states = np.vstack([sample[0] for sample in minibatch])
        nextstates = np.vstack([sample[3] for sample in minibatch])
        targetf = self.model.predict(states, verbose=0)
        nextq = self.model.predict(nextstates, verbose=0)
        for i, (_, action, reward, _, done) in enumerate(minibatch):
            target = reward if done else reward + self.gamma * np.max(nextq[i])
            targetf[i][action] = target
        self.model.fit(states, targetf, epochs=1, verbose=0)
        if self.epsilon > self.epsilonmin:
            self.epsilon = max(self.epsilonmin, self.epsilon * self.epsilondecay)


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
        return not (self.highqueue or self.mediumqueue or self.lowqueue)

    def queuelength(self):
        return len(self.highqueue) + len(self.mediumqueue) + len(self.lowqueue)


class NoPriorityScheduler:
    def __init__(self):
        self.queue = deque()
        self.highqueue = deque()
        self.mediumqueue = deque()
        self.lowqueue = deque()

    def addtask(self, task):
        self.queue.append(task)
        self.highqueue.append(task)

    def getnexttask(self):
        if not self.queue:
            return None
        task = self.queue.popleft()
        if self.highqueue:
            self.highqueue.popleft()
        return task

    def isempty(self):
        return len(self.queue) == 0

    def queuelength(self):
        return len(self.queue)


class EdgeFogSimulator:
    ACTION_NAMES = ['local', 'edge', 'fog', 'cloud']

    def __init__(self, num_iot=50, num_edge=5, num_fog=2, workload_config=None):
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
        self.completedtasks = []
        self.currenttime = 0.0
        self.taskcounter = 0

    def reset_runtime(self):
        self.completedtasks = []
        self.currenttime = 0.0
        self.taskcounter = 0
        self.scheduler = self.scheduler.__class__()
        for d in self.iotdevices + self.edgenodes + self.fognodes + [self.cloud]:
            d.availablecpu = 100.0
            d.availablememory = d.memorygb
            d.battery = 100.0
            d.busy_until = 0.0

    def createiotdevices(self):
        return [
            Device(
                f'iot{i}', 'iot',
                random.uniform(0.5, 1.5),
                random.uniform(0.5, 2.0),
                random.uniform(5, 20),
                random.uniform(1, 5),
                (random.uniform(0, 500), random.uniform(0, 500))
            )
            for i in range(self.numiot)
        ]

    def createedgenodes(self):
        return [
            Device(
                f'edge{i}', 'edge',
                random.uniform(2, 4),
                random.uniform(8, 16),
                random.uniform(100, 300),
                random.uniform(50, 100),
                (random.uniform(100, 400), random.uniform(100, 400))
            )
            for i in range(self.numedge)
        ]

    def createfognodes(self):
        return [
            Device(
                f'fog{i}', 'fog',
                random.uniform(4, 8),
                random.uniform(32, 64),
                random.uniform(500, 1000),
                random.uniform(150, 300),
                (250, 250)
            )
            for i in range(self.numfog)
        ]

    def createcloud(self):
        return Device('cloud0', 'cloud', 16.0, 256.0, 5000.0, 15.0, (250, 250))

    def generatetask(self):
        cfg = self.workload_config
        task = Task(
            taskid=self.taskcounter,
            arrivaltime=self.currenttime,
            datasize=random.uniform(cfg.size_low, cfg.size_high),
            cpurequirement=random.randint(cfg.cpu_low, cfg.cpu_high),
            deadline=random.randint(cfg.dl_low, cfg.dl_high),
            priority=random.randint(cfg.priority_low, cfg.priority_high)
        )
        self.taskcounter += 1
        return task

    def getstate(self, task, source):
        def norm(x, scale):
            return float(x) / float(scale) if scale > 0 else 0.0

        if hasattr(self.scheduler, 'queuelength'):
            queue_len = self.scheduler.queuelength()
        else:
            queue_len = len(getattr(self.scheduler, 'queue', []))

        candidate_nodes = [source]
        nearest_edge = min(self.edgenodes, key=lambda e: source.distance_to(e))
        nearest_fog = min(self.fognodes, key=lambda f: source.distance_to(f))
        candidate_nodes.extend([nearest_edge, nearest_fog, self.cloud])

        current_time = self.currenttime
        task_slack = task.deadline

        state = [
            norm(task.datasize, 200.0),
            norm(task.cpurequirement, 20000.0),
            norm(task.deadline, 150.0),
            norm(task.priority, 10.0),
            norm(task_slack, 150.0),
            norm(queue_len, 2000.0),
        ]

        for node in candidate_nodes:
            transmission = 0.0 if node.deviceid == source.deviceid else (task.datasize * 8.0) / max(node.bandwidthmbps, 1e-9)
            propagation = 0.0 if node.deviceid == source.deviceid else source.propagation_delay_to(node)
            ready_time = current_time + transmission + propagation
            waiting = max(0.0, node.busy_until - ready_time)
            exec_time = task.cpurequirement / max(node.mips, 1e-9)
            finish_time = transmission + propagation + waiting + exec_time
            slack_after = task.deadline - finish_time

            state.extend([
                norm(node.mips, 20000.0),
                norm(node.powerwatts, 500.0),
                norm(waiting, 500.0),
                norm(exec_time, 500.0),
                norm(finish_time, 1000.0),
                norm(slack_after, 200.0),
            ])

        return np.array(state, dtype=np.float32)

    def map_action_to_node(self, action, sourcedevice):
        if action == 0:
            return sourcedevice, 'local'
        if action == 1:
            return min(self.edgenodes, key=lambda e: sourcedevice.distance_to(e)), 'edge'
        if action == 2:
            return min(self.fognodes, key=lambda f: sourcedevice.distance_to(f)), 'fog'
        return self.cloud, 'cloud'

    def executetask(self, task, action, sourcedevice):
        computenode, location = self.map_action_to_node(action, sourcedevice)
        if computenode.deviceid == sourcedevice.deviceid:
            transmissiontime = 0.0
            propagationtime = 0.0
        else:
            transmissiontime = (task.datasize * 8.0) / max(computenode.bandwidthmbps, 1e-9)
            propagationtime = sourcedevice.propagation_delay_to(computenode)

        ready_time = self.currenttime + transmissiontime + propagationtime
        waitingtime = max(0.0, computenode.busy_until - ready_time)
        exectimems = task.cpurequirement / max(computenode.mips, 1e-9)
        latency = transmissiontime + propagationtime + waitingtime + exectimems

        exectimesec = exectimems / 1000.0
        transmissionsec = (transmissiontime + propagationtime) / 1000.0
        energy = computenode.powerwatts * exectimesec + 0.2 * transmissionsec

        task.waitingtime = waitingtime
        task.servicetime = exectimems
        task.executionlocation = location
        task.computenode = computenode.deviceid
        task.starttime = ready_time + waitingtime
        task.completiontime = task.starttime + exectimems

        computenode.busy_until = task.completiontime
        computenode.availablecpu = max(
            5.0,
            100.0 * max(0.0, 1.0 - min(1.0, exectimems / max(task.deadline, 1.0)))
        )

        return latency, energy

    def calculatereward(self, task, latency, energy):
        # Deadline- and latency-dominant reward; energy is secondary
        deadline = max(task.deadline, 1e-6)
        priority_weight = 1.0 + 0.2 * float(task.priority)
        normalized_latency = latency / deadline
        waiting_penalty = getattr(task, 'waitingtime', 0.0) / deadline

        if latency <= task.deadline:
            slack_ratio = (task.deadline - latency) / deadline
            reward = 20.0 * priority_weight
            reward += 10.0 * slack_ratio
            reward -= 3.0 * normalized_latency
            reward -= 0.2 * energy
            reward -= 2.0 * waiting_penalty
        else:
            tardiness_ratio = (latency - task.deadline) / deadline
            reward = -35.0 * priority_weight
            reward -= 25.0 * tardiness_ratio
            reward -= 0.5 * energy
            reward -= 3.0 * waiting_penalty

        return float(np.clip(reward, -100.0, 40.0))

    def estimate_action_outcome(self, task, action, source):
        if action == 0:
            node = source
        elif action == 1:
            node = min(self.edgenodes, key=lambda e: source.distance_to(e))
        elif action == 2:
            node = min(self.fognodes, key=lambda f: source.distance_to(f))
        else:
            node = self.cloud

        transmission = 0.0 if node.deviceid == source.deviceid else (task.datasize * 8.0) / max(node.bandwidthmbps, 1e-9)
        propagation = 0.0 if node.deviceid == source.deviceid else source.propagation_delay_to(node)
        ready_time = self.currenttime + transmission + propagation
        waiting = max(0.0, node.busy_until - ready_time)
        exec_time = task.cpurequirement / max(node.mips, 1e-9)
        total_latency = transmission + propagation + waiting + exec_time
        energy = node.powerwatts * (exec_time / 1000.0) + 0.2 * ((transmission + propagation) / 1000.0)
        slack_after = task.deadline - total_latency

        return {
            'node': node,
            'latency': total_latency,
            'energy': energy,
            'waiting': waiting,
            'exec_time': exec_time,
            'slack_after': slack_after,
        }

    def get_valid_actions(self, task, source):
        # Stricter feasibility: only actions predicted to meet deadline are valid.
        valid = []
        fallback = []
        for action in range(4):
            info = self.estimate_action_outcome(task, action, source)
            fallback.append((action, info['slack_after']))
            if info['slack_after'] >= 0.0:
                valid.append(action)
        if valid:
            return valid
        # If all actions miss, choose the least-bad (maximum slack_after)
        fallback.sort(key=lambda x: x[1], reverse=True)
        return [fallback[0][0]]

    def select_apqs_action(self, state, task, source):
        valid_actions = self.get_valid_actions(task, source)

        # Exploration
        if hasattr(self.agent, 'epsilon') and random.random() <= self.agent.epsilon:
            return random.choice(valid_actions)
        if (not TF_AVAILABLE) or (self.agent.model is None):
            return random.choice(valid_actions)

        q_values = self.agent.model.predict(state[np.newaxis, :], verbose=0)[0]

        # Soft bias: prefer local/edge/fog over cloud when Q-values are similar
        # action index: 0=local, 1=edge, 2=fog, 3=cloud
        latency_bias = np.array([0.05, 0.02, 0.0, -0.05], dtype=np.float32)

        masked_q = np.full_like(q_values, -1e9, dtype=np.float32)
        for a in valid_actions:
            masked_q[a] = q_values[a] + latency_bias[a]

        return int(np.argmax(masked_q))

    def runsimulation(self, durationms=30000, arrivalrate=0.1, verbose=False):
        self.reset_runtime()
        episode = 0

        while self.currenttime < durationms:
            if random.random() < arrivalrate:
                self.scheduler.addtask(self.generatetask())

            while not self.scheduler.isempty():
                task = self.scheduler.getnexttask()
                if task is None:
                    break

                source = random.choice(self.iotdevices)
                state = self.getstate(task, source)
                action = self.select_apqs_action(state, task, source)

                latency, energy = self.executetask(task, action, source)
                task.actuallatency = latency
                task.energyconsumed = energy

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
        if not self.completedtasks:
            return {
                'total_tasks': 0,
                'avg_latency': 0.0,
                'p99_latency': 0.0,
                'avg_energy': 0.0,
                'deadline_met_rate': 0.0,
                'failed_tasks': 0
            }

        latencies = np.array([t.actuallatency for t in self.completedtasks], dtype=float)
        energies = np.array([t.energyconsumed for t in self.completedtasks], dtype=float)
        deadlinemet = sum(1 for t in self.completedtasks if t.meetsdeadline())

        return {
            'total_tasks': int(len(self.completedtasks)),
            'avg_latency': float(np.mean(latencies)),
            'p50_latency': float(np.percentile(latencies, 50)),
            'p99_latency': float(np.percentile(latencies, 99)),
            'avg_energy': float(np.mean(energies)),
            'deadline_met_rate': float(deadlinemet / len(self.completedtasks) * 100.0),
            'failed_tasks': int(len(self.completedtasks) - deadlinemet),
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
    def __init__(self):
        self.queuethresholdhigh = 30
        self.queuethresholdmedium = 15

    def decide(self, task, queuelength):
        if queuelength >= self.queuethresholdhigh:
            return 3
        if queuelength >= self.queuethresholdmedium:
            return 2
        return 1


class GeneticAlgorithmScheduler:
    def __init__(self, populationsize=20, generations=20, mutationrate=0.1):
        self.populationsize = populationsize
        self.generations = generations
        self.mutationrate = mutationrate
        self.numlocations = 4

    def createindividual(self, numtasks):
        return [random.randint(0, self.numlocations - 1) for _ in range(numtasks)]

    def estimated_latency_energy(self, task, location):
        if location == 0:
            latency = task.cpurequirement / 1000.0
            energy = 3.0 * (latency / 1000.0)
        elif location == 1:
            latency = (task.datasize * 8.0 / 200.0) + (task.cpurequirement / 3000.0)
            energy = 75.0 * ((task.cpurequirement / 3000.0) / 1000.0)
        elif location == 2:
            latency = (task.datasize * 8.0 / 750.0) + (task.cpurequirement / 6000.0)
            energy = 225.0 * ((task.cpurequirement / 6000.0) / 1000.0)
        else:
            latency = (task.datasize * 8.0 / 5000.0) + (task.cpurequirement / 16000.0)
            energy = 15.0 * ((task.cpurequirement / 16000.0) / 1000.0)
        return latency, energy

    def evaluatefitness(self, individual, tasks):
        total = 0.0
        for i, task in enumerate(tasks):
            latency, energy = self.estimated_latency_energy(task, individual[i])
            deadline_penalty = 0 if latency <= task.deadline else 1000
            total += 0.5 * latency + 0.2 * energy * 1000 + 0.3 * deadline_penalty
        return total

    def crossover(self, parent1, parent2):
        if len(parent1) <= 1:
            return parent1.copy(), parent2.copy()
        cp = random.randint(1, len(parent1) - 1)
        return parent1[:cp] + parent2[cp:], parent2[:cp] + parent1[cp:]

    def mutate(self, individual):
        out = individual.copy()
        for i in range(len(out)):
            if random.random() < self.mutationrate:
                out[i] = random.randint(0, self.numlocations - 1)
        return out

    def optimize(self, tasks):
        if not tasks:
            return []

        population = [self.createindividual(len(tasks)) for _ in range(self.populationsize)]
        bestsolution = population[0]
        bestfitness = float('inf')

        for _ in range(self.generations):
            fitness = [self.evaluatefitness(ind, tasks) for ind in population]
            idx = int(np.argmin(fitness))

            if fitness[idx] < bestfitness:
                bestfitness = fitness[idx]
                bestsolution = population[idx].copy()

            order = np.argsort(fitness)
            newpop = [population[int(order[0])].copy()]
            if len(order) > 1:
                newpop.append(population[int(order[1])].copy())

            while len(newpop) < self.populationsize:
                p1, p2 = random.sample(population, 2)
                c1, c2 = self.crossover(p1, p2)
                newpop.extend([self.mutate(c1), self.mutate(c2)])

            population = newpop[:self.populationsize]

        return bestsolution

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
            f'iot{i}', 'iot',
            1.0, 1.0, 10, 3,
            (random.uniform(0, 500), random.uniform(0, 500))
        )
        for i in range(50)
    ]
    edgenodes = [
        Device(
            f'edge{i}', 'edge',
            3.0, 12.0, 200, 75,
            (random.uniform(100, 400), random.uniform(100, 400))
        )
        for i in range(5)
    ]
    fognodes = [
        Device(
            f'fog{i}', 'fog',
            6.0, 48.0, 750, 225,
            (250, 250)
        )
        for i in range(2)
    ]
    cloud = Device('cloud0', 'cloud', 16.0, 256.0, 5000.0, 15.0, (250, 250))
    return iotdevices, edgenodes, fognodes, cloud


def execute_baseline_task(task, action, source, edgenodes, fognodes, cloud, currenttime):
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
    waiting = max(0.0, node.busy_until - currenttime)
    exectimems = task.cpurequirement / max(node.mips, 1e-9)
    latency = transmission + propagation + waiting + exectimems
    energy = node.powerwatts * (exectimems / 1000.0) + 0.2 * ((transmission + propagation) / 1000.0)

    node.busy_until = currenttime + transmission + propagation + waiting + exectimems
    task.actuallatency = latency
    task.energyconsumed = energy
    task.executionlocation = location
    return task


def run_baseline_simulation(schedulertype, durationms=30000, arrivalrate=0.1, workload_config=None):
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
            tasks.append(Task(
                taskid, currenttime,
                random.uniform(cfg.size_low, cfg.size_high),
                random.randint(cfg.cpu_low, cfg.cpu_high),
                random.randint(cfg.dl_low, cfg.dl_high),
                random.randint(cfg.priority_low, cfg.priority_high)
            ))
            taskid += 1
        currenttime += 50.0

    gasolution = scheduler.optimize(tasks) if schedulertype == 'ga' else None

    completedtasks = []
    queuelength = len(tasks)
    currenttime = 0.0

    for i, task in enumerate(tasks):
        source = random.choice(iotdevices)

        if schedulertype == 'static':
            action = scheduler.decide(task)
        elif schedulertype == 'fcfs':
            action = scheduler.decide(task, queuelength)
        elif schedulertype == 'ga':
            action = scheduler.decide(i, gasolution)
        else:
            action = scheduler.decide(task, queuelength, random.uniform(40, 90))

        completedtasks.append(execute_baseline_task(task, action, source, edgenodes, fognodes, cloud, currenttime))
        queuelength -= 1
        currenttime += 50.0

    if not completedtasks:
        return {
            'total_tasks': 0,
            'avg_latency': 0.0,
            'p99_latency': 0.0,
            'avg_energy': 0.0,
            'deadline_met_rate': 0.0,
            'failed_tasks': 0
        }

    latencies = np.array([t.actuallatency for t in completedtasks], dtype=float)
    energies = np.array([t.energyconsumed for t in completedtasks], dtype=float)
    deadlinemet = sum(1 for t in completedtasks if t.meetsdeadline())

    return {
        'total_tasks': int(len(completedtasks)),
        'avg_latency': float(np.mean(latencies)),
        'p50_latency': float(np.percentile(latencies, 50)),
        'p99_latency': float(np.percentile(latencies, 99)),
        'avg_energy': float(np.mean(energies)),
        'deadline_met_rate': float(deadlinemet / len(completedtasks) * 100.0),
        'failed_tasks': int(len(completedtasks) - deadlinemet),
    }


class ResultsAnalyzer:
    @staticmethod
    def compareall(resultsdict):
        rows = []
        for name, results in resultsdict.items():
            rows.append({
                'Approach': name,
                'Total Tasks': results['total_tasks'],
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
        deadlinerates = [allresults[a]['deadline_met_rate'] for a in approaches]

        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple'][:len(approaches)]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].bar(approaches, latencies, color=colors)
        axes[0].set_title('Average Latency')
        axes[0].tick_params(axis='x', rotation=45)

        axes[1].bar(approaches, energies, color=colors)
        axes[1].set_title('Average Energy')
        axes[1].tick_params(axis='x', rotation=45)

        axes[2].bar(approaches, deadlinerates, color=colors)
        axes[2].set_title('Deadline Met %')
        axes[2].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()


def ablation_apqs_nopq(durationms=30000, arrivalrate=0.1, num_runs=5):
    all_results = []
    for _ in range(num_runs):
        simulator = EdgeFogSimulator(num_iot=50, num_edge=5, num_fog=2)
        simulator.scheduler = NoPriorityScheduler()
        all_results.append(simulator.runsimulation(durationms, arrivalrate, verbose=False))

    return {
        'total_tasks': float(np.mean([r['total_tasks'] for r in all_results])),
        'avg_latency': float(np.mean([r['avg_latency'] for r in all_results])),
        'p99_latency': float(np.mean([r['p99_latency'] for r in all_results])),
        'avg_energy': float(np.mean([r['avg_energy'] for r in all_results])),
        'deadline_met_rate': float(np.mean([r['deadline_met_rate'] for r in all_results])),
        'failed_tasks': float(np.mean([r['failed_tasks'] for r in all_results])),
        'std_latency': float(np.std([r['avg_latency'] for r in all_results])),
        'std_deadline': float(np.std([r['deadline_met_rate'] for r in all_results])),
    }


def ablation_apqs_noenergy(durationms=30000, arrivalrate=0.1, num_runs=5):
    original = EdgeFogSimulator.calculatereward

    def calculatereward_noenergy(self, task, latency, energy):
        latency_penalty = latency / max(task.deadline, 1.0)
        deadline_term = 1.0 if latency <= task.deadline else -1.0
        priority_term = (task.priority / 10.0) * (1.0 if latency <= task.deadline else -0.5)
        return float(-(0.45 * latency_penalty) + 0.45 * deadline_term + 0.10 * priority_term)

    all_results = []
    try:
        EdgeFogSimulator.calculatereward = calculatereward_noenergy
        for _ in range(num_runs):
            simulator = EdgeFogSimulator(num_iot=50, num_edge=5, num_fog=2)
            all_results.append(simulator.runsimulation(durationms, arrivalrate, verbose=False))
    finally:
        EdgeFogSimulator.calculatereward = original

    return {
        'total_tasks': float(np.mean([r['total_tasks'] for r in all_results])),
        'avg_latency': float(np.mean([r['avg_latency'] for r in all_results])),
        'p99_latency': float(np.mean([r['p99_latency'] for r in all_results])),
        'avg_energy': float(np.mean([r['avg_energy'] for r in all_results])),
        'deadline_met_rate': float(np.mean([r['deadline_met_rate'] for r in all_results])),
        'failed_tasks': float(np.mean([r['failed_tasks'] for r in all_results])),
        'std_latency': float(np.std([r['avg_latency'] for r in all_results])),
        'std_deadline': float(np.std([r['deadline_met_rate'] for r in all_results])),
    }


def run_ablation_study(durationms=30000, arrivalrate=0.2, num_runs=5):
    results = {}
    results['APQS-NoPQ'] = ablation_apqs_nopq(durationms, arrivalrate, num_runs)
    results['APQS-NoEnergy'] = ablation_apqs_noenergy(durationms, arrivalrate, num_runs)

    full = []
    for _ in range(num_runs):
        simulator = EdgeFogSimulator(num_iot=50, num_edge=5, num_fog=2)
        full.append(simulator.runsimulation(durationms, arrivalrate, verbose=False))

    results['APQS (Full)'] = {
        'total_tasks': float(np.mean([r['total_tasks'] for r in full])),
        'avg_latency': float(np.mean([r['avg_latency'] for r in full])),
        'p99_latency': float(np.mean([r['p99_latency'] for r in full])),
        'avg_energy': float(np.mean([r['avg_energy'] for r in full])),
        'deadline_met_rate': float(np.mean([r['deadline_met_rate'] for r in full])),
        'failed_tasks': float(np.mean([r['failed_tasks'] for r in full])),
        'std_latency': float(np.std([r['avg_latency'] for r in full])),
        'std_deadline': float(np.std([r['deadline_met_rate'] for r in full])),
    }

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(os.path.join('output', f'ablation_study_{ts}.json'), 'w') as f:
        json.dump(results, f, indent=2)

    return results


def main():
    duration = 30000
    arrivalrate = 0.1

    simulator = EdgeFogSimulator(num_iot=50, num_edge=5, num_fog=2)
    allresults = {
        'DQN + Priority': simulator.runsimulation(duration, arrivalrate, verbose=False),
        'Static Threshold': run_baseline_simulation('static', duration, arrivalrate),
        'FCFS': run_baseline_simulation('fcfs', duration, arrivalrate),
        'Genetic Algorithm': run_baseline_simulation('ga', duration, arrivalrate),
        'Fuzzy Logic': run_baseline_simulation('fuzzy', duration, arrivalrate),
    }

    analyzer = ResultsAnalyzer()
    df = analyzer.compareall(allresults)
    df.to_csv('output/debugged_results.csv', index=False)
    analyzer.plotcomparison(allresults, 'output/comparison.png')

    return allresults, df


if __name__ == '__main__':
    main()
