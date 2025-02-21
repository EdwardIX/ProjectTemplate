import importlib.util
import os
import pickle
import base64
import subprocess as sp
import time
import argparse
import traceback
import tracemalloc
import json
import signal
import socket
import random

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(('10.255.255.255', 1))
    return s.getsockname()[0]

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

def get_func_by_path(path, func_name):
    spec = importlib.util.spec_from_file_location("temp_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    func = getattr(module, func_name)
    return func

def get_func_by_module_name(module_name, func_name):
    module = importlib.import_module(module_name)
    func = getattr(module, func_name)
    return func

def parse_seed(seed, i, j):
    random.seed(int(time.time()) ^ os.getpid())
    if seed == "Random":
        return random.randint(0, 2**32-1)
    elif isinstance(seed, int):
        return seed
    elif isinstance(seed, str):
        return int(eval(seed)) # Support simple expressions
    else:
        raise NotImplementedError

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

class Task:
    def __init__(self, target, args, reqs):
        self.target = target
        self.args = args
        self.reqs = reqs

        # Runtime Information
        self.progress = None
        self.gpuids = None
    
    def copy(self):
        return Task(self.target, self.args, self.reqs)

    def runnable(self, status):
        def get_avail_gpus(s): # Return all available gpus on single node, Ranked as Priority
            gpu_prio_list = []
            for i in range(len(s.gpumem)):
                if s.available(i) \
                   and s.gpumem[i] >= self.reqs['gpumem'] \
                   and 100-s.gpuusg[i] >= self.reqs['gpuusg'] \
                   and s.gpumem[i] + s.gpumymem[i] - s.taskmem[i] >= self.reqs['gpumem'] \
                   and (s.taskpro[i] < self.reqs['gpupro'] or self.reqs['gpupro'] == -1) \
                   and (s.gpupro[i] - s.gpumypro[i] == 0 or self.reqs['occupy']):
                    gpu_prio_list.append((s.gpupro[i], s.gpuusg[i], i))
            gpu_prio_list.sort()
            return [g[-1] for g in gpu_prio_list]

        gpuinfo = {}
        if self.reqs['mulnode']:
            tot = 0
            for n, s in status.items():
                avail_gpus = get_avail_gpus(s)
                if avail_gpus:
                    tot += len(avail_gpus)
                    gpuinfo[n] = avail_gpus
            
            if tot >= self.reqs['numgpu']:
                prio = sorted([(len(v), k) for k,v in gpuinfo.items()])
                tot = 0
                for _, n in prio:
                    gpuinfo[n] = gpuinfo[n][:self.reqs['numgpu']-tot]
                    if not len(gpuinfo[n]): 
                        gpuinfo.pop(n)
                    tot += len(gpuinfo[n])
                return gpuinfo
            else:
                return None
        else:
            for n, s in status.items():
                avail_gpus = get_avail_gpus(s)
                if len(avail_gpus) >= self.reqs['numgpu']:
                    gpuinfo[n] = avail_gpus[:self.reqs['numgpu']]
                    return gpuinfo
    
            return None

    def run_python(self, gpu, path, taskid, repeatid, seed, localsrc):
        current_env = {
            "CUDA_VISIBLE_DEVICES": str(gpu),
        }

        self.process = sp.Popen(['python', __file__,
                                 '--mode', 'python',
                                 '--args', base64.b64encode(pickle.dumps(self.args)),
                                 '--path', path,
                                 '--taskid', str(taskid),
                                 '--repeatid', str(repeatid),
                                 '--seed', str(seed),
                                 '--localsrc', str(localsrc)],
                                cwd=os.getcwd(),
                                env = {**os.environ, **current_env})

    def run_torchrun_multinode(self, nnodes, node_rank, gpu_ids, path, taskid, repeatid, seed, localsrc):
        current_env = {
            "CUDA_VISIBLE_DEVICES": ','.join(map(str, gpu_ids)),
        }

        if node_rank == 0:
            ip = get_local_ip()
            port = find_free_port()
            with open(os.path.join(path, "host_ip_port.txt") ,"w") as f:
                print(f"ip:{ip}\nport:{port}",file=f)
        else:
            while not os.path.exists(os.path.join(path, "host_ip_port.txt")):
                print("Waiting for Main Process to Start ...")
                time.sleep(1)
            with open(os.path.join(path, "host_ip_port.txt"), "r") as f:
                for l in f.readlines():
                    if l.startswith("ip:"): ip = l[3:].strip()
                    if l.startswith("port:"): port = int(l[5:].strip())
        
        self.process = sp.Popen(['torchrun', 
                                 f'--nproc_per_node={len(gpu_ids)}', 
                                 f'--nnodes={nnodes}',
                                 f'--node_rank={node_rank}',
                                 f'--master_addr={ip}',
                                 f'--master_port={port}',
                                 __file__,
                                 '--mode', 'torchrun_multinode',
                                 '--args', base64.b64encode(pickle.dumps(self.args)),
                                 '--path', path,
                                 '--taskid', str(taskid),
                                 '--repeatid', str(repeatid),
                                 '--seed', str(seed),
                                 '--localsrc', str(localsrc)],
                                cwd = os.getcwd(),
                                env = {**os.environ, **current_env})

    def run_torchrun_singlenode(self, gpu_ids, path, taskid, repeatid, seed, localsrc):
        current_env = {
            "CUDA_VISIBLE_DEVICES": ','.join(map(str, gpu_ids)),
        }
        
        self.process = sp.Popen(['torchrun',
                                 '--rdzv-backend=c10d',
                                 f'--rdzv-endpoint=localhost:{find_free_port()}', # Handle Multiple task on single machine
                                 '--nnodes=1',
                                 f'--nproc-per-node={len(gpu_ids)}',
                                 __file__,
                                 '--mode', 'torchrun_singlenode',
                                 '--args', base64.b64encode(pickle.dumps(self.args)),
                                 '--path', path,
                                 '--taskid', str(taskid),
                                 '--repeatid', str(repeatid),
                                 '--seed', str(seed),
                                 '--localsrc', str(localsrc)],
                                cwd = os.getcwd(),
                                env = {**os.environ, **current_env})

    def run(self, nnodes, node_rank, gpuids, path, taskid, repeatid, seed, localsrc, loginfo={}):
        self.gpuids = gpuids
        # TODO: Start a new process by directly start the target python script
        # Log Task Info
        os.makedirs(path, exist_ok=True)
        if node_rank == 0:
            with open(os.path.join(path, 'task.json'), 'w') as f:
                json.dump({
                    **loginfo,
                    'path': path,
                    'taskid': taskid,
                    'repeatid': repeatid,
                    'seed': seed,
                    'args': self.args,
                    'reqs': self.reqs
                }, f, indent=4)
                
        # Run Task
        if nnodes > 1:
            self.run_torchrun_multinode(nnodes, node_rank, gpuids, path, taskid, repeatid, seed, localsrc)
        elif len(gpuids) > 1:
            self.run_torchrun_singlenode(gpuids, path, taskid, repeatid, seed, localsrc)
        else:
            self.run_python(gpuids[0], path, taskid, repeatid, seed, localsrc)

    def alive(self):
        return (self.process and self.process.poll() == None)

    def terminate(self):
        if self.process:
            self.process.terminate()
        self.process=None
        self.gpuids=None

    def join(self):
        if self.process:
            retcode = self.process.wait()
        else:
            retcode = None
        self.process=None
        self.gpuids=None
        return retcode

def subprocess(args, info):
    """This method will be the target of subprocess and started as a new process."""
    # 1: Set Gpu run environment:
    import torch
    torch.set_default_dtype(torch.float32) # TODO: Use more complex precision setting (eg Mixed...). This should not be set

    # 2: Set Random Seed (TODO: Assign Same Seed for every proc?)
    import numpy as np
    if info['seed'] is not None:
        torch.manual_seed(info['seed'])
        torch.cuda.manual_seed(info['seed'])
        np.random.seed(info['seed'])

    # 3: setup logging: Hook Stdout & Stderr + setup logger + write task info(only for local main process)
    import os
    import sys
    def get_parent_dir(path, n): return os.path.dirname(get_parent_dir(path, n-1)) if n else path
    sys.path.append(get_parent_dir(info['path'], (4 if info['localsrc'] else 1))) # Source code is in rundir/../src, set the root to src
    from src.utils.logger import logger
    if info['rank'] == 0:
        import os
        import sys
        sys.stdout = HookedOutput(os.path.join(info['path'], "log.txt"), sys.stdout)
        sys.stderr = HookedOutput(os.path.join(info['path'], "err.txt"), sys.stderr)
        logger.initialize(info['path'], info['taskid'], info['repeatid'])

    # 4: Set up Signal Handle function & Summary function
    ppid = os.getppid()
    def summary(signum, frame):
        if info['rank'] == 0 and os.getppid() == ppid: # Ensures only the main process enters this function
            if signum is not None: 
                print(f"\033[91m######### Signal {signum} received! #########\033[0m", file=sys.stderr)
                print("Stack trace for signal:", file=sys.stderr)
                traceback.print_stack(frame, file=sys.stderr)
            _, mem_peak = tracemalloc.get_traced_memory()
            mem_snapshot = tracemalloc.take_snapshot()
            tracemalloc.stop()
            statistics = logger.summary(path = os.path.join(info['path'], "stat.csv"))
            with open(os.path.join(info['path'], "summary.json"), "w") as f:
                json.dump({
                    "Success": True if exception is None and signum is None else False,
                    "Exception": f"Signal {signum} received" if signum is not None else (f"{type(exception)}:{exception}" if exception is not None else None),
                    "End Time": time.strftime('%y.%m.%d-%H.%M.%S'),
                    "Wall Time": time.time() - start_time["Wall"],
                    "User Time": time.process_time() - start_time["User"],
                    "Peak Memory Usage": mem_peak,
                    "Top 10 Memory Used": list(map(str, mem_snapshot.statistics('lineno')[:10])),
                    "Scalars Statistics": statistics,
                }, f, indent=4)
        exit(0)

    if info['rank'] == 0:
        signal.signal(signal.SIGTERM, summary)
        signal.signal(signal.SIGINT, summary) # Ctrl-C

    # 5: Try to load function, Setup stat
    exception = None
    try:
        func = get_func_by_module_name(os.path.join(get_parent_dir(info['path'], 4), 'main.py'), 'target')
    except Exception as e:
        exception = e
    if exception:
        try:
            func = get_func_by_path(os.path.join(get_parent_dir(info['path'], 4), 'main.py'), 'target')
            exception = None
        except Exception as e:
            exception = e

    args = pickle.loads(args)
    start_time = {"Wall": time.time(), "User": time.process_time()}
    tracemalloc.start()

    # 6: Run
    try:
        func(cfg=args, info=info)
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        exception = e
    summary(None, None)

if __name__ == "__main__":
    """
    The interface for calling script with sp.Popen
    """

    # Accept args
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str)
    parser.add_argument('--args', type=str)
    parser.add_argument('--path', type=str)
    parser.add_argument('--taskid', type=int)
    parser.add_argument('--repeatid', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--localsrc', type=str)
    args = parser.parse_args()

    if args.mode == 'python':
        info = {
            'path': args.path,
            'taskid': args.taskid,
            'repeatid': args.repeatid,
            'seed': args.seed,
            'device': "cuda:0",
            'localrank': 0,
            'localworldsize': 1, 
            'rank': 0,
            'worldsize': 1,
            'localsrc': args.localsrc == 'True',
        }
    elif args.mode in ('torchrun_singlenode', 'torchrun_multinode'):
        info = {
            'path': args.path,
            'taskid': args.taskid,
            'repeatid': args.repeatid,
            'seed': args.seed,
            'device': f"cuda:{os.environ['LOCAL_RANK']}",
            'localrank': int(os.environ['LOCAL_RANK']),
            'localworldsize': int(os.environ['LOCAL_WORLD_SIZE']),
            'rank': int(os.environ['RANK']),
            'worldsize': int(os.environ['WORLD_SIZE']),
            'localsrc': args.localsrc == 'True',
        }

    subprocess(base64.b64decode(args.args), info)