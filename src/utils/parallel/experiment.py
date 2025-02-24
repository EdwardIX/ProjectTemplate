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
    def __init__(self, config_list, exp_name="Test", run_list="All", share_config="check"):
        """
        Create a Group of Task
        config_list: the list of config for each task
        exp_name: the name of this experiment
        run_list: the list of tasks to be run
        share_config: for the same exp_name, share the same config (true / false(override) / check)
        """
        self.config_list = config_list
        self.run_list = run_list
        self.exp_name = exp_name
        self.run_time = "Null"
        self.share_config = share_config.lower()
        self.identifier = self.exp_name + "::" + self.run_time
        self.tasks:List[Task] = [] # List of tasks
        self.status = {}           # Task Running Status

    def send_to_server(self):
        client = SocketClient("Experiment")
        client.send_message(pickle.dumps(self))
        success, msg = pickle.loads(client.recv_message())
        client.close()
        if success:
            if msg == "Success":
                print(f"\033[92m############  Successfully send Experiment to Server  ############\033[0m")
            else:
                print(f"\033[33m############  Server Warning: {msg} ############\033[0m")
        else:
            print(f"\033[91m############  Failed to send Experiment to Server: {msg} ############\033[0m")

    def prepare(self):
        self.run_time = time.strftime('%y.%m.%d-%H.%M.%S')
        self.identifier = self.exp_name + "::" + self.run_time

        rootpath = os.path.join("runs", self.exp_name, self.run_time)
        os.makedirs(rootpath)
        shutil.copytree("src", os.path.join(rootpath, "src"), ignore=shutil.ignore_patterns("*.pyc", "__pycache__")) # Code Backup
        
        for config in self.config_list:
            self.tasks.append(Task(config['target'], config['args'], config['reqs']))
        
        success, msg = self.handle_previous_runs()
        self.save_runinfo()
        
        return success, msg

    def handle_previous_runs(self):
        # Step 1: Calculate the directory of most recent experiment with the same name
        prev_dir = max(
            (d for d in os.listdir(os.path.join("runs", self.exp_name)) if re.match(r'\d{2}\.\d{2}\.\d{2}-\d{2}\.\d{2}\.\d{2}', d)),
            key=lambda d: time.strptime(d, '%y.%m.%d-%H.%M.%S'),
            default=None
        )
        prev_dir = os.path.join("runs", self.exp_name, prev_dir)

        if prev_dir is not None:
            # Step 2: load config / run_list from previous runs
            if self.share_config == 'check':
                return False, "The same experiment name detected, please specify share_config"
            elif self.share_config == 'false': # Use Current Config
                success, msg = self.calc_run_list(prev_dir)
            elif self.share_config == 'true':
                # Load config from previous run
                with open(os.path.join(prev_dir, "runinfo.json"), "r") as f:
                    prev_tasks = json.load(f)["Tasks"]
                    if len(prev_tasks) != len(self.tasks):
                        msg = "different number of Tasks detected, replace both task config and requirements"
                        self.tasks = [Task(**t) for t in prev_tasks]
                    else:
                        msg = "Load Task List from existing file. Current Task List replaced"
                        for i, t in enumerate(self.tasks):
                            t.target = prev_tasks[i].target
                            t.args = prev_tasks[i].args
                # Calc Run List
                success, _msg = self.calc_run_list(prev_dir)
                if not success: 
                    msg = _msg
        else:
            success, msg = self.calc_run_list()

        return success, msg
    
    def save_runinfo(self):
        with open(os.path.join("runs", self.exp_name, self.run_time, "runinfo.json"), "w") as f:
            json.dump({
                "Command": "python " + " ".join(sys.argv),
                "RunList": self.run_list,
                "Tasks": [{"target": t.target, "args": t.args, "reqs": t.reqs} for t in self.tasks],
            }, f, indent=2)

    def calc_run_list(self, prev_dir=None):
        if isinstance(self.run_list, str):
            if self.run_list == "All":
                self.run_list = [(i, j) for i, t in enumerate(self.tasks) for j in range(t.reqs["repeat"])]
            elif self.run_list == "Failed": # Find all failed tasks in the previous run
                if prev_dir is not None: # Load RunList from previous run
                    with open(os.path.join(prev_dir, "runinfo.json"), "r") as f:
                        self.run_list = json.load(f)["RunList"]
                    
                    def is_failed(i, j):
                        path = os.path.join(prev_dir, f"{i}-{j}", f"summary.json")
                        if not os.path.exists(path): return True
                        with open(path) as f: return not json.load(f)["success"]
                    
                    self.run_list = list(filter(self.run_list, is_failed))
                    if self.share_config == 'false':
                        return False, "Load RunList from previous run, this might cause error when TaskConfig changed"
                else:
                    self.run_list = [(i, j) for i, t in enumerate(self.tasks) for j in range(t.reqs["repeat"])]
            else:
                return False, f"Unknown RunList Type: {self.run_list}"
        elif not isinstance(self.run_list, list):
            return False, f"Unknown RunList Type: {type(self.run_list)}"
    
        return True, "Success"

    def add_task(self, config):
        self.tasks.append(Task(config['target'], config['args'], config['reqs']))

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

    def next_task(self, status):
        max_repeat = max([t.reqs['repeat'] for t in self.tasks])
        for j in range(max_repeat):
            for i, t in enumerate(self.tasks):
                if j <= t.reqs["repeat"] and self.get_task_status(i, j) == "Waiting":
                    gpuinfo = t.runnable(status)
                    if gpuinfo: 
                        return (i, j), gpuinfo
        return None, None

    def finished(self):
        max_repeat = max([t.reqs['repeat'] for t in self.tasks])
        for j in range(max_repeat):
            for i, t in enumerate(self.tasks):
                if j < t.reqs["repeat"] and self.get_task_status(i, j) not in ("Success", "Failed"):
                    return False
        return True

    def run_local(self, devices=None):
        status = RunnerStatus(devices)

        success, msg = self.prepare()
        if not success:
            print(f"\033[31mFailed to Prepare Experiment:{msg}\033[0m")
            return
        elif msg != "Success":
            print(f"\033[33mWarning:{msg}\033[0m")
        while not self.finished():
            status.update()
            taskid, gpuinfo = self.next_task({'local': status})
            if taskid is not None:
                i, j = taskid
                path = os.path.join("runs", self.exp_name, self.run_time, f"{i}-{j}")
                task = self.tasks[i].copy()

                # Run Task
                task.run(1, 0, gpuinfo['local'], path, i, j, parse_seed(task.reqs['seed'], i, j), True,
                         loginfo = {'start time': time.strftime("%y.%m.%d-%H:%M:%S"), 'gpuinfo': gpuinfo})
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
        
        self.on_finished()

    def summary(self, process_func=None):

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

    
    def on_finished(self):
        if os.path.exists(os.path.join("runs", self.exp_name, self.run_time, "src")):
            # Delete src folder
            shutil.rmtree(os.path.join("runs", self.exp_name, self.run_time, "src"))

        # Old: Delete __pycache__ in src
        # for dirpath, dirnames, filenames in os.walk(os.path.join("runs", self.exp_name, self.run_time, "src")):
        #     if '__pycache__' in dirnames:
        #         pycache_path = os.path.join(dirpath, '__pycache__')
        #         shutil.rmtree(pycache_path)
        
        self.summary()