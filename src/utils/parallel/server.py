import os
import re
import time
import threading
import random
from typing import Dict

from .experiment import Experiment
from .comm import SocketServer
from .status import RunnerStatus

random.seed(int(time.time()) ^ os.getpid())
def parse_seed(seed, i, j):
    if seed == "Random":
        return random.randint(0, 2**32-1)
    elif isinstance(seed, int):
        return seed
    elif isinstance(seed, str):
        return int(eval(seed)) # Support simple expressions
    else:
        raise NotImplementedError

class Server:
    def __init__(self):
        self.socket = SocketServer(self)

        self.exps:Dict[str, Experiment] = {} # All Experiments
        self.runner_stats:Dict[str, RunnerStatus] = {} # All Runners
        self.tasks = {} # All active tasks

        self.main_thread = threading.Thread(target=self.run)
        self.main_thread.start()
    
    def add_experiment(self, e:Experiment):
        self.exps[e.identifier] = e
        e.prepare()

    def get_task(self, status):
        print(status)
        for e in self.exps.values():
            taskid, gpuinfo = e.next_task(status)
            if taskid is not None: 
                return e, taskid, gpuinfo
        return None, None, None

    def run_task(self, exp:Experiment, taskid, gpuinfo):
        success = True
        identifier = f"{exp.identifier}:{taskid[0]}-{taskid[1]}"
        seed = parse_seed(exp.tasks[taskid[0]].reqs['seed'], taskid[0], taskid[1])
        for runner in gpuinfo.keys():
            success = success and self.socket.send_task(runner, identifier, exp.tasks[taskid[0]], gpuinfo, seed)
        if success:
            exp.set_task_status(taskid[0], taskid[1], "Running")
            exp.log_task_runinfo(taskid[0], taskid[1], gpuinfo, seed)
            self.tasks[identifier] = gpuinfo
            print(f"\033[32m############  Experiment {exp.identifier} Task {taskid[0]}-{taskid[1]} Start On {gpuinfo} ############\033[0m")
            return True
        else:
            for runner in gpuinfo.keys():
                self.socket.send_command(runner, "StopTask", {"identifier": identifier})
            return False
    
    def stop_task(self, identifier):
        if identifier in self.tasks.keys():
            for runner in self.tasks[identifier].keys():
                self.socket.send_command(runner, "StopTask", {"identifier": identifier})

    def on_new_runner(self, name):
        self.runner_stats[name] = RunnerStatus(name)
    
    def on_del_runner(self, name):
        self.runner_stats.pop(name)

    def on_runner_status_update(self, name, data):
        self.runner_stats[name].unpack(data)

    def on_task_status_update(self, identifier, status):
        print("On Task Status Update", identifier, status)
        expname, i, j = re.match(r"(.*):(\d+)-(\d+)", identifier).groups()
        i, j = int(i), int(j)
        self.exps[expname].set_task_status(i, j, status)
        if status != "Running":
            self.tasks.pop(identifier)

    def run(self):
        while True:
            # Fliter finished exps
            finished_exps = []
            for n, e in self.exps.items():
                print("Experiment Finished", e.finished())
                if e.finished():
                    e.summary()
                    finished_exps.append(n)
            for n in finished_exps:
                self.exps.pop(n)

            # Try to find a new task to run
            exp, taskid, gpuinfo = self.get_task(self.runner_stats)
            if taskid:
                self.run_task(exp, taskid, gpuinfo)
            else:
                time.sleep(10)