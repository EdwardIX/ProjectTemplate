import os
import re
import time
import threading
import random
from typing import Dict, List

from .experiment import Experiment
from .comm import SocketServer
from .status import RunnerStatus
from .task import parse_seed

class Server:
    """
    A task manage server enables multiple clients to run tasks in parallel.
    Components: 
        self.exps:  All the experiments (running)
        self.runner_stats:  All the active runners
        self.tasks_gpuinfo:  All the active tasks and its gpuinfo
    """
    def __init__(self):
        self.socket = SocketServer(self)

        self.exps:Dict[str, Experiment] = {} # All Experiments
        self.exps_wlock = threading.Lock()
        
        self.runner_stats:Dict[str, RunnerStatus] = {} # All Active Runners
        self.tasks_gpuinfo:Dict[str, Dict[str, List[int]]] = {} # All Active tasks and its gpuinfo

        self.main_thread = threading.Thread(target=self.run)
        self.main_thread.start()

    def _get_task(self, status): # Get the next task to run
        for e in self.exps.values():
            taskid, gpuinfo = e.next_task(status) # Query every active experiments and find the next available task
            if taskid is not None: 
                return e, taskid, gpuinfo
        return None, None, None

    def _run_task(self, exp:Experiment, taskid, gpuinfo): # Run A Specific Task
        success = True
        identifier = f"{exp.identifier}:{taskid[0]}-{taskid[1]}"
        seed = parse_seed(exp.tasks[taskid[0]].reqs['seed'], taskid[0], taskid[1])
        for runner in gpuinfo.keys():
            success = success and self.socket.send_task(runner, identifier, exp.tasks[taskid[0]], gpuinfo, seed) # Send the task to every runner
        if success:
            exp.set_task_status(taskid[0], taskid[1], "Running") # Update status
            self.tasks_gpuinfo[identifier] = gpuinfo
            print(f"\033[32m############  Experiment {exp.identifier} Task {taskid[0]}-{taskid[1]} Start On {gpuinfo} ############\033[0m")
            return True
        else:
            for runner in gpuinfo.keys():
                self.socket.send_command(runner, "StopTask", {"identifier": identifier}) # Stop that task if it fails to start
            return False
    
    def get_task_list(self): # UI: Get active tasks
        ret = []
        for k,v in self.tasks_gpuinfo.items():
            exp_name, run_time, i, j = re.match(r"(.*)::([\d\.]{8}-[\d\.]{8}):(\d+)-(\d+)", k).groups()
            ret.append({"id": k, "exp_name": exp_name, "run_time": run_time, "index": f"{i}-{j}", "gpuinfo": v})
        return ret

    def get_experiment_list(self): # UI: Get active experiments
        return [{"id": k, "finished": e.finished(), "name": e.exp_name, "start_time": e.run_time, "num_tasks": len(e.tasks)} 
                for k,e in self.exps.items()]
    
    def get_task_status(self, identifier): # UI: Get specific status of a task
        exp_name = re.match(r"(.*::.*):.*", identifier).group(1)
        return self.exps[exp_name].get_task_status(identifier)

    def get_experiment_status(self, identifier): # UI: Get status of an experiment
        task_status = self.exps[identifier].get_all_task_status()
        max_repeat = max([j for i,j in task_status.keys()]) + 1
        return {
            "task_status": task_status,
            "max_repeat": max_repeat,
        }
    
    def get_runner_status(self): # UI: Get status of all runners
        return [{
            "id": k, 
            "gpustat": v.gpustat, 
        } for k,v in self.runner_stats.items()]

    def add_experiment(self, e:Experiment): # UI: Add an experiment
        success, msg = e.prepare()
        if success:
            with self.exps_wlock:
                self.exps[e.identifier] = e
        
        return success, msg

    def stop_task(self, identifier): # UI: Stop task
        if identifier in self.tasks_gpuinfo.keys():
            for runner in self.tasks_gpuinfo[identifier].keys():
                self.socket.send_command(runner, "StopTask", {"identifier": identifier})

    def stop_experiment(self, identifier): # UI: Stop experiment
        if identifier in self.exps.keys():
            exp = self.exps[identifier]
            for (i, j) in exp.run_list:
                if exp.get_task_status(i, j) == "Running":
                    self.stop_task(f"{exp.identifier}:{i}-{j}")
                elif exp.get_task_status(i, j) == "Waiting":
                    exp.set_task_status(i, j, "Failed")
                else: # Success / Failed / Skipped ...: just do nothing
                    pass

    def del_experiment(self, identifier): # UI: Delete experiment
        if identifier in self.exps.keys():
            self.stop_experiment(identifier)
            with self.exps_wlock:
                self.exps.pop(identifier)

    def on_new_runner(self, name): # Handle a new runner, add runner status to it
        self.runner_stats[name] = RunnerStatus()
    
    def on_del_runner(self, name): # Delete a runner, stop all related tasks
        for taskid, gpuinfo in self.tasks_gpuinfo.items():
            if name in gpuinfo.keys():
                self.stop_task(taskid)
        self.runner_stats.pop(name)

    def on_runner_status_update(self, name, stat): # Update runner status
        self.runner_stats[name] = stat

    def on_task_status_update(self, identifier, status): # Update task status
        print("On Task Status Update", identifier, status)
        expname, i, j = re.match(r"(.*):(\d+)-(\d+)", identifier).groups()
        i, j = int(i), int(j)
        self.exps[expname].set_task_status(i, j, status)
        if status != "Running" and identifier in self.tasks_gpuinfo.keys():
            self.tasks_gpuinfo.pop(identifier)

    def run(self): # Main Thread
        finished_exps = set()
        while True:
            # Fliter finished exps
            for n, e in self.exps.items():
                if e.finished() and n not in finished_exps:
                    e.on_finished()
                    finished_exps.add(n)

            # Try to find a new task to run
            exp, taskid, gpuinfo = self._get_task(self.runner_stats)
            if taskid:
                self._run_task(exp, taskid, gpuinfo)
                time.sleep(3)
            else:
                time.sleep(10)