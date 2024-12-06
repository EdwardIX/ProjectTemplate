import os
import re
import time
import threading
import random
from typing import Dict, List

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
        self.exps_active:Dict[str, Experiment] = {} # All Active Experiments

        self.runner_stats:Dict[str, RunnerStatus] = {} # All Active Runners
        self.tasks_gpuinfo:Dict[str, Dict[str, List[int]]] = {} # All Active tasks and its gpuinfo

        self.main_thread = threading.Thread(target=self.run)
        self.main_thread.start()

    def _get_task(self, status): # Get the next task to run
        print(status)
        for e in self.exps_active.values():
            taskid, gpuinfo = e.next_task(status)
            if taskid is not None: 
                return e, taskid, gpuinfo
        return None, None, None

    def _run_task(self, exp:Experiment, taskid, gpuinfo): # Run A Specific Task
        success = True
        identifier = f"{exp.identifier}:{taskid[0]}-{taskid[1]}"
        seed = parse_seed(exp.tasks[taskid[0]].reqs['seed'], taskid[0], taskid[1])
        for runner in gpuinfo.keys():
            success = success and self.socket.send_task(runner, identifier, exp.tasks[taskid[0]], gpuinfo, seed)
        if success:
            exp.set_task_status(taskid[0], taskid[1], "Running")
            exp.log_task_runinfo(taskid[0], taskid[1], gpuinfo, seed)
            self.tasks_gpuinfo[identifier] = gpuinfo
            print(f"\033[32m############  Experiment {exp.identifier} Task {taskid[0]}-{taskid[1]} Start On {gpuinfo} ############\033[0m")
            return True
        else:
            for runner in gpuinfo.keys():
                self.socket.send_command(runner, "StopTask", {"identifier": identifier})
            return False
    
    def get_task_list(self):
        ret = []
        for k,v in self.tasks_gpuinfo.items():
            exp_name, run_time, i, j = re.match(r"(.*)::([\d\.]{8}-[\d\.]{8}):(\d+)-(\d+)", k).groups()
            ret.append({"id": k, "exp_name": exp_name, "run_time": run_time, "index": f"{i}-{j}", "gpuinfo": v})
        return ret

    def get_experiment_list(self):
        return [{"id": k, "name": e.exp_name, "start_time": e.run_time, "num_tasks": len(e.tasks)} 
                for k,e in self.exps.items()]
    
    def get_task_status(self, identifier):
        exp_name = re.match(r"(.*::.*):.*", identifier).group(1)
        return self.exps_active[exp_name].get_task_status(identifier)

    def get_experiment_status(self, identifier):
        task_status = self.exps_active[identifier].get_all_task_status()
        max_repeat = max([j for i,j in task_status.keys()]) + 1
        return {
            "task_status": task_status,
            "max_repeat": max_repeat,
        }
    
    def get_runner_status(self):
        return [{"id": k, "gpumem": v.gpumem, "gpuusg": v.gpuusg, "gpupro": v.gpupro, "gpumypro": v.gpumypro} for k,v in self.runner_stats.items()]

    def add_experiment(self, e:Experiment):
        e.prepare()
        self.exps[e.identifier] = e
        self.exps_active[e.identifier] = e

    def stop_task(self, identifier):
        if identifier in self.tasks_gpuinfo.keys():
            for runner in self.tasks_gpuinfo[identifier].keys():
                self.socket.send_command(runner, "StopTask", {"identifier": identifier})

    def stop_experiment(self, identifier):
        if identifier in self.exps_active.keys():
            for taskid in self.tasks_gpuinfo.keys():
                if taskid.startswith(identifier):
                    self.stop_task(taskid)

    def on_new_runner(self, name):
        self.runner_stats[name] = RunnerStatus()
    
    def on_del_runner(self, name): # TODO: Clear All Running Tasks
        self.runner_stats.pop(name)

    def on_runner_status_update(self, name, data):
        self.runner_stats[name].unpack(data)

    def on_task_status_update(self, identifier, status):
        print("On Task Status Update", identifier, status)
        expname, i, j = re.match(r"(.*):(\d+)-(\d+)", identifier).groups()
        i, j = int(i), int(j)
        self.exps_active[expname].set_task_status(i, j, status)
        if status != "Running":
            self.tasks_gpuinfo.pop(identifier)

    def run(self):
        while True:
            # Fliter finished exps
            finished_exps = []
            for n, e in self.exps_active.items():
                print("Experiment Finished", e.finished())
                if e.finished():
                    e.summary()
                    finished_exps.append(n)
            for n in finished_exps:
                self.exps_active.pop(n)

            # Try to find a new task to run
            exp, taskid, gpuinfo = self._get_task(self.runner_stats)
            if taskid:
                self._run_task(exp, taskid, gpuinfo)
                time.sleep(3)
            else:
                time.sleep(10)