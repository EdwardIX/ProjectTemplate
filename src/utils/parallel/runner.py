import re
import os
import time
import json
import threading
from typing import Dict

from .task import Task
from .comm import SocketRunner
from .status import RunnerStatus

class Runner:
    def __init__(self):
        self.status = RunnerStatus()
        self.socket = SocketRunner(self)

        self.pool:Dict[str, Task] = {}
        self.pool_lock = threading.Lock()

        self.run()

    def on_update_status(self):
        self.status.update()

    def on_run_task(self, identifier, args, reqs, gpuinfo, seed):
        task = Task(args, reqs)
        with self.pool_lock:
            self.pool[identifier] = task
        exp_name, run_time, i, j = re.match(r"(.*)::([\d\.]{8}-[\d\.]{8}):(\d+)-(\d+)", identifier).groups()
        i, j = int(i), int(j)
        nnodes = len(gpuinfo)
        node_rank = list(gpuinfo.keys()).index(self.socket.hostname)
        task.run(nnodes, \
                 node_rank, \
                 gpuinfo[self.socket.hostname], \
                 path = os.path.join("runs", exp_name, run_time, f"{i}-{j}"), \
                 taskid = i, \
                 repeatid = j, \
                 seed = seed)
        try:
            for gid in gpuinfo[self.socket.hostname]:
                self.status.start_task(gid, task.reqs)
        except Exception as e:
            print(gpuinfo[self.socket.hostname], task.gpuids)
            print(e)

    def on_stop_task(self, identifier):
        if identifier in self.pool.keys(): # The Task is still active
            self.pool[identifier].terminate()
    
    def on_stop_all_task(self):
        with self.pool_lock:
            for n, t in self.pool.items():
                t.terminate()

    def on_task_finished(self, identifier):
        task = self.pool[identifier]
        exp_name, run_time, i, j = re.match(r"(.*)::([\d\.]{8}-[\d\.]{8}):(\d+)-(\d+)", identifier).groups()
        i, j = int(i), int(j)
        try:
            with open(os.path.join("runs", exp_name, run_time, f"{i}-{j}", "summary.json"), "r") as f:
                success = json.load(f)['Success']
        except:
            success = False
        self.socket.send_task_status(identifier, "Success" if success else "Failed")
        for gid in task.gpuids:
            self.status.end_task(gid, task.reqs)

    def run(self):
        try:
            while self.socket.connected:
                inactive_tasks = []
                for n, t in self.pool.items():
                    if not t.alive():
                        inactive_tasks.append(n)
                        self.on_task_finished(n)
                
                with self.pool_lock:
                    for n in inactive_tasks:
                        self.pool.pop(n)

                time.sleep(10)
        
        except Exception as e:
            print(e)
        finally:
            self.on_stop_all_task()