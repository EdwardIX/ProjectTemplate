# 1. Support Multitask Running (Gpu allocation)
# 2. Support Task Register (Give function(dill), Give args(dict), function receives gpu&path&i&j&config)
# 3. Support Task Logging (Log params for a function call)

import os
import re
import sys
import time
import json
import dill
import copy
import base64
import socket
import random
import argparse
import traceback
import tracemalloc
import collections
import pandas as pd
import subprocess as sp
import numpy as np
from typing import List, Dict, Callable, Any
dill.settings['recurse'] = True

def find_free_port():
    # 创建一个临时套接字
    temp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 将套接字绑定到本地主机的一个随机端口
    temp_socket.bind(('localhost', 0))
    # 获取绑定后的端口号
    _, port = temp_socket.getsockname()
    # 关闭套接字
    temp_socket.close()
    
    return port

class HookedOutput():
    def __init__(self, filename, original) -> None:
        self.file = open(filename, 'w')
        self.original = original

    def __getattr__(self, name):
        return getattr(self.original, name)

    def write(self, data):
        self.original.write(data)
        self.file.write(data)

    def flush(self):
        self.original.flush()
        self.file.flush()

class SeedGenerator():
    def __init__(self, method=None):
        if method == "Random":
            self.method = "Random"
            random.seed(int(time.time()) ^ os.getpid())
        elif method == "Uniform":
            self.method = "Uniform"
        elif isinstance(method, int):
            self.method = "List"
            self.seed = [method]
        elif isinstance(method, (list, tuple)):
            self.method = "List"
        elif isinstance(method, str):
            try:
                self.seed = list(map(int, method.split(',')))
                self.method = "List"
            except:
                raise ValueError(f"Invaild seed method {method}")
        else:
            raise ValueError(f"Invalid seed method {method}")
    
    def __call__(self, i, j):
        if self.method == "Random":
            return random.randint(0, 2**32-1)
        elif self.method == "List":
            return self.seed[i % len(self.seed)]
        elif self.method == "Uniform":
            return j
    
    def to_json(self):
        return self.seed if self.method == "List" else self.method
    
    @classmethod
    def from_json(cls, json):
        return cls(json)

class StrJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (dict, list, tuple)):
            return [self.default(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)

class Task:
    """
    Register a Task
    To initialize: give an dict containing "target", "args" and "repeat"
    Target should accept (gpu, path, taskid, repeatid) four args
    Args is strongly recommended to be json serializable (string, numbers, dict, list, ...)
    """

    def __init__(self, server, target_id, args_id, numgpu=0, repeat=1):
        self.server:Server = server
        self.target_id = target_id
        self.args_id = args_id
        self.numgpu = numgpu
        self.repeat = repeat

        self.progress = None
    
    @property
    def target(self):
        return self.server.targets[self.target_id]
    
    @property
    def args(self):
        return self.server.args[self.args_id]

    def copy(self):
        return Task(self.server, self.target_id, self.args_id, self.repeat)

    def dict(self):
        return {"Target":self.target.__name__, "Args": self.args, "Numgpu": self.numgpu, "Repeat": self.repeat, "Target ID": self.target_id, "Args ID": self.args_id}

    def save(self, filepath, additional={}):
        with open(filepath, "w") as f:
            json.dump({**additional, **self.dict()}, f, indent=4, cls=StrJSONEncoder)
    
    def run_python(self, gpu, path, taskid, repeatid, seed, parallel):
        assert len(gpu) == 1
        if parallel:
            self.process = sp.Popen(['python', __file__, \
                                    '--target', base64.b64encode(dill.dumps(self.target)), \
                                    '--args', base64.b64encode(dill.dumps(self.args)), \
                                    '--gpu', str(gpu[0]), \
                                    '--path', path, \
                                    '--taskid', str(taskid), \
                                    '--repeatid', str(repeatid), \
                                    '--seed', str(seed)],
                                    cwd=os.getcwd())
        else:
            self.process = None
            subprocess(dill.dumps(self.target), dill.dumps(self.args), gpu[0], path, taskid, repeatid, seed, -1)
    
    def run_accelerate(self, gpu, path, taskid, repeatid, seed, parallel):
        self.process = sp.Popen(['accelerate', 'launch', \
                                '--num_processes', str(len(gpu)),
                                '--gpu_ids', ','.join(map(str, gpu)),
                                '--main_process_port', str(find_free_port()), \
                                '--num_machines', '1', \
                                '--mixed_precision', 'no', \
                                '--dynamo_backend', 'no', \
                                __file__, '--use_accelerate', \
                                '--target', base64.b64encode(dill.dumps(self.target)), \
                                '--args', base64.b64encode(dill.dumps(self.args)), \
                                '--gpu', '-1', \
                                '--path', path, \
                                '--taskid', str(taskid), \
                                '--repeatid', str(repeatid), \
                                '--seed', str(seed)],
                                cwd=os.getcwd())
        if not parallel:
            self.process.wait()
    
    def run(self, gpu, path, taskid, repeatid, seed, parallel):
        if len(gpu) > 1:
            self.run_accelerate(gpu, path, taskid, repeatid, seed, parallel)
        else:
            self.run_python(gpu, path, taskid, repeatid, seed, parallel)
    
    def alive(self):
        return (self.process.poll() == None) if self.process else False
    
    def join(self):
        if self.process is not None:
            self.process.wait()

class Server: # Gpu program Run Server
    def __init__(self, path=None, exp_name="Test", device="Auto", repeat=1, numgpu=1, hostname=None, seed="Random", parallel=True, capacity=1):
        """
        Create a Server for multitask running
        Params:
        path: if specified, the server will work in this dir, and load the task existed in this path (if any)
        exp_name: the name of this experiment
        device: the devices available. if device=None/Auto, this server will automatically find available devices using gpustat
        repeat: the default repeat times for each task. but can be overwritted by task args
        numgpu: the number of gpus one task needed. Set to 0 to automatically use all gpu available.
        hostname: specific name for this host, if not specified, hostname will be got from gpustat
        seed: the mode for seed for each task. 
        parallel: if the programs should run in parallel
        capacity: maximum processes running on a single device
        """

        self.exp_name = exp_name
        self.path = path if path is not None else os.path.join('runs', time.strftime('%y.%m.%d-%H.%M.%S-') + self.exp_name)
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        self.device = device
        if isinstance(self.device, str):
            self.device = "Auto" if self.device == "Auto" else list(map(int, self.device.split(",")))
        elif isinstance(self.device, (list, tuple)):
            self.device = list(map(int, self.device))
        elif isinstance(self.device, int):
            self.device = [self.device]
        else:
            raise ValueError(f"Unknown device format: {self.device}")
        self.repeat = repeat
        self.numgpu = numgpu
        self.hostname = hostname if hostname is not None else sp.check_output(['hostname']).decode().strip()
        self.seedgen = SeedGenerator(seed)
        self.parallel = parallel
        self.capacity = capacity

        self.dependency = False
        self.default_target = None
        self.tasks:List[Task] = []     # List of tasks
        self.targets = []              # List of runnable targets
        self.args = []                 # List of args (when args are not serializeable)
        self.pool:Dict[int, List[Task]] = {} # Current Running Process

    def allocate_gpu(self, num=1):
        def available(gpu):
            self.pool[gpu] = list(filter((lambda task:task.alive()), self.pool.get(gpu, [])))
            return len(self.pool[gpu]) < self.capacity
        
        available_gpus = []
        if self.device == "Auto":  # Auto Gpu Allocation Based on gpustat
            gpustat = json.loads(sp.check_output(['gpustat', '--json']).decode())["gpus"]
            for gpu in random.sample(gpustat, len(gpustat)):
                if available(gpu['index']) and len(gpu['processes']) == len(self.pool[gpu['index']]): # Ensuring no other process are running 
                    available_gpus.append(gpu['index'])
        else:                      # Traditional Gpu Allocation (run even gpu is busy)
            for gpu in random.sample(self.device, len(self.device)):
                if available(gpu):
                    available_gpus.append(gpu)

        if num == 0 and len(available_gpus): # use all gpus available
            return available_gpus
        elif len(available_gpus) >= num:
            available_gpus.sort(key=(lambda g:len(self.pool[g]))) # balance the usage of devices
            return available_gpus[:num]
        else:
            return None
        
    def dict(self):
        return {
            "Experiment Name": self.exp_name,
            "Devices": self.device,
            "Path": self.path,
            "Command Args": sys.argv[1:],
            "NumTask": len(self.tasks),
            "Repeat": self.repeat,
            "Numgpu": self.numgpu,
            "Dependency": self.dependency,
            "Seed": self.seedgen.to_json(),
            "Tasks": list(map((lambda t:t.dict()), self.tasks))
        }
    
    def save_bin(self):
        with open(os.path.join(self.path, "tasks.bin"), "wb") as f:
            dill.dump((self.targets, self.args), f)

    def save_task(self):
        with open(os.path.join(self.path, "tasks.json"), "w") as f:
            json.dump(self.dict(), f, indent=4, cls=StrJSONEncoder)
    
    def load_bin(self):
        with open(os.path.join(self.path, "tasks.bin"), "rb") as f:
            self.targets, self.args = dill.load(f)

    def load_task(self):
        with open(os.path.join(self.path, "tasks.json"), "r") as f:
            config = json.load(f)
        
        self.tasks=[]
        for task in config["Tasks"]:
            self.tasks.append(Task(self, task['Target ID'], task['Args ID'], task['Numgpu'], task['Repeat']))

    def set_default_target(self, target):
        self.targets.append(target)
        self.default_target = len(self.targets) - 1

    def add_task(self, target=None, args=None, numgpu=None, repeat=None):
        target = target if target is not None else self.default_target
        if target is None: raise ValueError("Task Target Cannot be None")
        if not isinstance(target, int): # if the task is new
            self.targets.append(target)
            target = len(self.targets) - 1
        
        args = args if args is not None else {}
        self.args.append(args)
        args = len(self.args) - 1

        numgpu = numgpu if numgpu is not None else self.numgpu
        repeat = repeat if repeat is not None else self.repeat

        self.tasks.append(Task(self, target_id=target, args_id=args, numgpu=numgpu, repeat=repeat))
    
    def add_task_args(self, taskargs:Dict, scan_params:Dict[str, List[Any]]=None, all_combination=True):
        """
        A function to add a single / group of task(s).
        if scan_params is not None, it will modify taskargs using copy.deepcopy() and add them to tasklist

        Parameters: 
        taskargs: the args for target function
        scan_params: a dict containing the name+value of parameters to be scaned. 
            name: to modify taskargs['a']['b'][1]['c'], pass in "a/b/1/c"
            value: a list like object
        all_combination: if True, will scan all combinations of the params
        """
        if scan_params is None:
            self.add_task(args=taskargs)
            return
        
        def _modify_params(taskargs, name, value):
            names = name.split('/')
            for i, n in enumerate(names):
                if isinstance(taskargs, (list, tuple)):
                    n = int(n)

                if i != len(names) - 1:
                    taskargs = taskargs[n]
                else:
                    taskargs[n] = value
                    
        num = list(len(v) for v in scan_params.values())
        if all_combination:
            from .arrayutil import meshgrid
            for idx in meshgrid(*[np.arange(n) for n in num]):
                new_taskarg = copy.deepcopy(taskargs)
                for i, (k, v) in enumerate(scan_params.items()):
                    _modify_params(new_taskarg, k, v[idx[i]])
                
                self.add_task(args=new_taskarg)
        else:
            assert len(set(num)) == 1, "All params should have the same number of choices"

            for i in range(num[0]):
                new_taskarg = copy.deepcopy(taskargs)
                for k, v in scan_params.items():
                    _modify_params(new_taskarg, k, v[i])
                
                self.add_task(args=new_taskarg)

    def set_dependency(self, dependency):
        """
        Set The dependency for each task. List is available.
        dependency = None&False:  No dependency
        dependency = True: dependency on previous task
        dependency = int: for specific repeat of previous task. 
        dependency = str: for dependency on certain path
        """
        self.dependency = dependency

    def get_training_state(self): # Get the times each task runned
        claimed_tasks = []
        finished_tasks = []
        for entry in os.scandir(self.path):
            mobj = re.match(r"^(\d+)-(\d+)$", entry.name)
            if mobj:
                claimed_tasks.append((int(mobj[1]), int(mobj[2])))
                if os.path.exists(os.path.join(entry.path, "summary.json")):
                    finished_tasks.append((int(mobj[1]), int(mobj[2])))
        
        next_repeat = collections.defaultdict(int)
        for i,j in sorted(claimed_tasks):
            if next_repeat[i] == j:
                next_repeat[i] += 1
        return next_repeat, claimed_tasks, finished_tasks
    
    def get_path_training_state(self, path):
        if not os.path.exists(path):
            return 'Waiting'
        if os.path.exists(os.path.join(path, 'summary.json')):
            with open(os.path.join(path, "summary.json"), "r") as f:
                summary = json.load(f)
                if summary['Success']:
                    return 'Success'
                else:
                    return 'Failed'
        else:
            return 'Running'

    def get_next_task(self):
        """
        Select the task with minimum repeat times & minimum id.
        The infomation is based on whether the corresponding dir is created.
        If self.dependency is set, it will apply according to the dependency setting.
        """
        if not isinstance(self.dependency, list):                 # Extend Dependency info after loading task
            self.dependency = [self.dependency] * len(self.tasks)
        
        taskid, minrepeat = None, np.inf
        next_repeat, claimed_tasks, finished_tasks = self.get_training_state()
        for i, task in enumerate(self.tasks):
            if next_repeat[i] >= task.repeat or minrepeat <= next_repeat[i]:
                continue

            if self.dependency[i] in (None, False):
                depend_check = True 
            elif isinstance(self.dependency[i], int):
                depend_check = i == 0 or (i-1, self.dependency[i]) in finished_tasks
            elif self.dependency[i] == True:
                depend_check = i == 0 or (i-1, next_repeat[i] % self.tasks[i-1].repeat) in finished_tasks
            elif isinstance(self.dependency[i], str):
                depend_check = self.get_path_training_state(self.dependency[i]) in ('Success', 'Failed')
            else:
                raise NotImplementedError(f"Unsupported dependency type {type(self.dependency[i])}")
            
            if depend_check:
                minrepeat = next_repeat[i]
                taskid = i
        
        return taskid, minrepeat

    def finished(self):
        next_repeat, *_ = self.get_training_state()
        for i, task in enumerate(self.tasks):
            if next_repeat[i] < task.repeat:
                return False
        return True

    def run(self, parallel=None):
        with open(os.path.join(self.path, 'command.txt'), 'a') as f:
            f.write(time.strftime('%y.%m.%d-%H.%M.%S-') + self.hostname + ':' + ' '.join(sys.argv) + '\n')
            
        parallel = parallel if parallel is not None else self.parallel
        if not os.path.exists(os.path.join(self.path, "tasks.json")):
            self.save_bin()
            self.save_task()
        else:
            print(f"\033[33m############  Server Warning: Load Task List from existing file. Current Task List replaced  ############\033[0m")
            self.load_bin()
            self.load_task()
        
        warn_no_gpu = True
        warn_dependency = True
        while not self.finished():                                # always try to find the next task
            while True:
                i, j = self.get_next_task()                       # first get a task to run
                if i is None:
                    if warn_dependency == True:
                        warn_dependency = False
                        print("\033[33m############  Server: Dependent task not finished yet  ############\033[0m")
                    time.sleep(10)
                    continue
                numgpu = self.tasks[i].numgpu                     # get numgpu the task required
                gpu = self.allocate_gpu(numgpu)
                if gpu is None:
                    if warn_no_gpu == True:
                        warn_no_gpu = False
                        print("\033[33m############  Server: Cannot find available GPU Now  ############\033[0m")
                    time.sleep(10 + random.randint(1, 5))         # sleep if failed to get gpu
                else:
                    warn_no_gpu = False
                    break                                         # gpu found

            path = os.path.join(self.path, f"{i}-{j}") + os.path.sep
            try:
                os.mkdir(path)                                # try to occupy the path
            except FileExistsError:
                time.sleep(3 + random.randint(1, 5))          # failed to occupy, try again later
                continue

            task:Task = self.tasks[i].copy()
            for g in gpu:
                if g not in self.pool.keys(): self.pool[g] = []
                self.pool[g].append(task)
            seed = self.seedgen(i, j)
            task.save(os.path.join(path, "task.json"), additional={"Hostname": self.hostname, "GPU": gpu, "TaskID":i, "RepeatID": j, "Seed": seed})
            print(f"\033[32m############  Task {i}-{j} Start On GPU {gpu} ############\033[0m")
            task.run(gpu, path, taskid=i, repeatid=j, seed=seed, parallel=parallel)
        
        for key, tasks in self.pool.items():
            for task in tasks:
                task.join()
        
        print(f"\033[32m############  All Tasks Finished ############\033[0m")
        self.summary()
    
    def summary(self, process_func:Callable[[str], Dict]=None):
        task_data = {}

        for entry in os.scandir(self.path):
            mobj = re.match(r"^(\d+)-(\d+)$", entry.name)
            if mobj:
                i = int(mobj[1])
                j = int(mobj[2])
                task_data[i] = task_data.get(i, [])

                path = os.path.join(self.path, f"{i}-{j}")
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
        # key = 'Last Non-NaN Value'
        key = 'Last 10 Non-NaN Mean'
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
        df.to_csv(os.path.join(self.path, "stat.csv"), index=False)
        return df

def subprocess(target, args, gpu, path, taskid, repeatid, seed, localrank):
    """This method will be the target of subprocess and started as a new process."""
    # 1: Set Gpu run environment:
    import torch
    if localrank == -1:
        torch.cuda.set_device("cuda:"+str(gpu))
        torch.set_default_dtype(torch.float32)
    else: # the Gpu environment is set by other libraries
        gpu = torch.cuda.current_device()

    # 2: Set Random Seed
    import numpy as np
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

    # 3: setup logging: Hook Stdout & Stderr + setup logger (only for local main process)
    import os
    import sys
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(proj_root) # it is quite ugly here... maybe consider put this file to the root of project
    from src.utils.logger import logger
    if localrank in (-1, 0):
        import os
        import sys
        sys.stdout = HookedOutput(os.path.join(path, "log.txt"), sys.stdout)
        sys.stderr = HookedOutput(os.path.join(path, "err.txt"), sys.stderr)
        logger.initialize(path, taskid, repeatid)

    # 4: Load, Setup stat
    func = dill.loads(target)
    args = dill.loads(args)
    exception = None
    start_time = {"Wall": time.time(), "User": time.process_time()}
    tracemalloc.start()

    # 5: Run
    try:
        func(gpu=gpu, path=path, taskid=taskid, repeatid=repeatid, **args)
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        exception = e

    # 6: Save Summary (only for local main process)
    if localrank in (-1, 0):
        _, mem_peak = tracemalloc.get_traced_memory()
        mem_snapshot = tracemalloc.take_snapshot()
        tracemalloc.stop()
        statistics = logger.summary(path = os.path.join(path, "stat.csv"))
        with open(os.path.join(path, "summary.json"), "w") as f:
            json.dump({
                "Success": True if exception is None else False,
                "Wall Time": time.time() - start_time["Wall"],
                "User Time": time.process_time() - start_time["User"],
                "Peak Memory Usage": mem_peak,
                "Top 10 Memory Used": list(map(str, mem_snapshot.statistics('lineno')[:10])),
                "Scalars Statistics": statistics,
            }, f, indent=4)

if __name__ == "__main__":
    """
    The interface for calling script with sp.Popen
    """

    # Accept args
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_accelerate', action='store_true')
    parser.add_argument('--target', type=str)
    parser.add_argument('--args', type=str)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--path', type=str)
    parser.add_argument('--taskid', type=int)
    parser.add_argument('--repeatid', type=int)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()

    if args.use_accelerate:
        import accelerate
        subprocess(base64.b64decode(args.target), \
                   base64.b64decode(args.args), \
                   args.gpu, \
                   args.path, \
                   args.taskid, \
                   args.repeatid, \
                   args.seed, \
                   accelerate.PartialState().local_process_index)
    else:
        subprocess(base64.b64decode(args.target), \
                   base64.b64decode(args.args), \
                   args.gpu, \
                   args.path, \
                   args.taskid, \
                   args.repeatid, \
                   args.seed, \
                   -1)