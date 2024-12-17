import time
import os
import re
import sys
import json
import shutil
import pickle
import numpy as np
import pandas as pd
import collections
from typing import List

from .task import Task, parse_seed
from .comm import SocketClient
from .status import RunnerStatus

class Experiment:
    def __init__(self, config_list, exp_name="Test", run_list="All"):
        """
        Create a Group of Task
        exp_name: the name of this experiment
        """
        self.config_list = config_list
        self.run_list = run_list
        self.exp_name = exp_name
        self.run_time = "Null"
        self.identifier = self.exp_name + "::" + self.run_time
        self.tasks:List[Task] = [] # List of tasks
        self.status = {}           # Task Running Status

    def send_to_server(self):
        client = SocketClient("Experiment")
        client.send_message(pickle.dumps(self))
        client.close()
        print("Experiment Sent to Server")

    def prepare(self):
        self.run_time = time.strftime('%y.%m.%d-%H.%M.%S')
        self.identifier = self.exp_name + "::" + self.run_time

        rootpath = os.path.join("runs", self.exp_name, self.run_time)
        os.makedirs(rootpath)
        shutil.copytree("src", os.path.join(rootpath, "src"), ignore=shutil.ignore_patterns("*.pyc", "__pycache__")) # Code Backup
        
        for config in self.config_list:
            self.tasks.append(Task(config['args'], config['reqs']))
        
        if os.path.exists(os.path.join("runs", self.exp_name, "tasks.json")):
            self.load_task()
        else:
            self.save_task()
        self.calc_run_list()
        self.save_runinfo()

    def load_task(self):
        with open(os.path.join("runs", self.exp_name, "tasks.json"), "r") as f:
            task_config = json.load(f)["TaskArgs"]
            if len(task_config) != len(self.tasks):
                print(f"\033[91m############  Server Warning: different number of Tasks detected, replace both task config and requirements ############\033[0m")
                task_reqs = json.load(f)["TaskReqs"]
                self.tasks = [(c, r) for c, r in zip(task_config, task_reqs)]
            else:
                print(f"\033[33m############  Server Warning: Load Task List from existing file. Current Task List replaced  ############\033[0m")
                for i, t in enumerate(self.tasks):
                    t.args = task_config[i]

    def save_task(self):
        with open(os.path.join("runs", self.exp_name, "tasks.json"), "w") as f:
            json.dump({
                "TaskArgs": [t.args for t in self.tasks],
                "TaskReqs": [t.reqs for t in self.tasks],
            }, f, indent=2)
    
    def save_runinfo(self):
        with open(os.path.join("runs", self.exp_name, self.run_time, "runinfo.json"), "w") as f:
            json.dump({
                "Command": "python " + " ".join(sys.argv[1:]),
                "RunList": self.run_list,
                "TaskReqs": [t.reqs for t in self.tasks],
            }, f, indent=2)

    def calc_run_list(self):
        if isinstance(self.run_list, str):
            if self.run_list == "All":
                self.run_list = [(i, j) for i, t in enumerate(self.tasks) for j in range(t.reqs["repeat"])]
            elif self.run_list == "Failed": # Find all failed tasks in the previous run
                prev_dir = max(
                    (d for d in os.listdir("runs", self.exp_name) if re.match(r'\d{2}\.\d{2}\.\d{2}-\d{2}\.\d{2}\.\d{2}', d)),
                    key=lambda d: time.strptime(d, '%y.%m.%d-%H.%M.%S'),
                    default=None
                )
                if prev_dir is not None:
                    with open(os.path.join("runs", self.exp_name, prev_dir, "runinfo.json"), "r") as f:
                        self.run_list = json.load(f)["RunList"]
                    
                    def is_failed(i, j):
                        path = os.path.join("runs", self.exp_name, prev_dir, f"{i}-{j}", f"summary.json")
                        if not os.path.exists(path): return True
                        with open(path) as f: return not json.load(f)["success"]
                    
                    self.run_list = list(filter(self.run_list, is_failed))
                else:
                    self.run_list = [(i, j) for i, t in enumerate(self.tasks) for j in range(t.reqs["repeat"])]
            else:
                raise ValueError(f"Unknown RunList Type: {self.run_list}")
        elif not isinstance(self.run_list, list):
            raise ValueError(f"Unknown RunList Type: {type(self.run_list)}")

    def add_task(self, config):
        self.tasks.append(Task(config['args'], config['reqs']))

    def get_task_status(self, i, j):
        if (i, j) not in self.run_list:
            return "Skipped"
        elif (i, j) not in self.status.keys():
            return "Waiting"
        return self.status[(i, j)]

    def get_all_task_status(self):
        ret = {}
        max_repeat = max([t.reqs['repeat'] for t in self.tasks])
        for j in range(max_repeat):
            for i, t in enumerate(self.tasks):
                if t.reqs["repeat"] >= j:
                    ret[(i, j)] = self.get_task_status(i, j)
        
        return ret

    def set_task_status(self, i, j, s):
        assert s in ("Waiting", "Running", "Success", "Failed", "Skipped"), f"Unknown task status {s}"
        self.status[(i, j)] = s
    
    def log_task_runinfo(self, i, j, gpuinfo, seed):
        os.makedirs(os.path.join("runs", self.exp_name, self.run_time, f"{i}-{j}"), exist_ok=True)
        with open(os.path.join("runs", self.exp_name, self.run_time, f"{i}-{j}", "runinfo.json"), "w") as f:
            json.dump({
                "Start Time": time.strftime("%y.%m.%d-%H:%M:%S"),
                "Seed": seed,
                "gpuinfo": gpuinfo,
            }, f, indent=2)

    def next_task(self, status):
        max_repeat = max([t.reqs['repeat'] for t in self.tasks])
        for j in range(max_repeat):
            for i, t in enumerate(self.tasks):
                if t.reqs["repeat"] >= j and self.get_task_status(i, j) == "Waiting":
                    gpuinfo = t.runnable(status)
                    if gpuinfo: 
                        return (i, j), gpuinfo
        return None, None

    def finished(self):
        max_repeat = max([t.reqs['repeat'] for t in self.tasks])
        for j in range(max_repeat):
            for i, t in enumerate(self.tasks):
                if t.reqs["repeat"] > j and self.get_task_status(i, j) not in ("Success", "Failed"):
                    return False
        return True

    def run_local(self):
        status = RunnerStatus()

        self.prepare()
        while not self.finished():
            status.update()
            taskid, gpuinfo = self.next_task({'local': status})
            if taskid is not None:
                i, j = taskid
                path = os.path.join("runs", self.exp_name, self.run_time, f"{i}-{j}")
                task = self.tasks[i].copy()

                # Run Task
                task.run(1, 0, gpuinfo['local'], path, i, j, parse_seed(task.reqs['seed'], i, j))
                print(f"\033[32m############  Experiment {self.identifier} Task {i}-{j} Start On {gpuinfo} ############\033[0m")
                task.join()

                # Update Result
                try:
                    with open(os.path.join(path, "summary.json"), "r") as f:
                        success = json.load(f)['Success']
                except:
                    success = False
                self.set_task_status(i, j, "Success" if success else "Failed")
                
            else: # Experiment Not Finished, but waiting for gpu to run
                time.sleep(10)
        
        self.summary()

    def summary(self, process_func=None):
        # Delete all __pycache__
        for dirpath, dirnames, filenames in os.walk(os.path.join("runs", self.exp_name, self.run_time, "src")):
            if '__pycache__' in dirnames:
                pycache_path = os.path.join(dirpath, '__pycache__')
                shutil.rmtree(pycache_path)
        
        # Summary
        task_data = {}
        keys = ['last', 'last_5', 'last_10', 'all']
        for entry in os.scandir(os.path.join("runs", self.exp_name, self.run_time)):
            mobj = re.match(r"^(\d+)-(\d+)$", entry.name)
            if mobj:
                i = int(mobj[1])
                j = int(mobj[2])
                task_data[i] = task_data.get(i, [])

                path = os.path.join("runs", self.exp_name, self.run_time, f"{i}-{j}")
                if not os.path.exists(os.path.join(path, "summary.json")):
                    task_data[i].append(('Running', None, None))
                else:
                    with open(os.path.join(path, "summary.json"), "r") as f:
                        summary = json.load(f)
                    if summary['Success']:
                        if process_func is not None:
                            task_data[i].append(('Success', summary['Scalars Statistics'], process_func(path)))
                        else:
                            task_data[i].append(('Success', summary['Scalars Statistics'], {}))
                    else:
                        task_data[i].append(('Failed', None, None))
        
        # 合并所有任务的数据
        dfs = {}
        for key in keys:
            rows = []
            variables = []  # all variables occured
            infos = []
            status_data = {'Success':0, 'Failed':0, 'Running':0}
            for i in range(len(task_data)):
                variable_data = collections.defaultdict(list)
                info_data = {}
                status_data = {'Success':0, 'Failed':0, 'Running':0}
                for status, scalar_data, additional_data in task_data[i]:
                    status_data[status] += 1
                    if status == 'Success':
                        for variable, value in additional_data.items():
                            if isinstance(value, str):
                                assert info_data.get(variable, value) == value, "Not all info values in the same task are the same"
                                info_data[variable] = value
                            else:
                                variable_data[variable].append(value)
                        for variable, value in scalar_data.items():
                            variable_data[variable].append(value[key])
                        
                for v in variable_data.keys():
                    if v not in variables: variables.append(v)
                for v in info_data.keys():
                    if v not in infos: infos.append(v)
                rows.append({
                    'Task': i,
                    'Repeat': len(task_data[i]),
                    **status_data,
                    **info_data,
                    **{variable + "_avg": np.nanmean(values) for variable, values in variable_data.items()},
                    **{variable + "_std": np.nanstd(values)  for variable, values in variable_data.items()},
                })

            df = pd.DataFrame(rows, columns=['Task', 'Repeat'] + \
                                            list(status_data.keys()) + \
                                            infos + \
                                            [v + "_avg" for v in variables] + [v + "_std" for v in variables])
            dfs[key] = df
            df.to_csv(os.path.join(os.path.join("runs", self.exp_name, self.run_time), f"stat_{key}.csv"), index=False)
        return dfs