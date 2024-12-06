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
    def __init__(self, args, reqs):
        self.args = args
        self.reqs = reqs

        # Runtime Information
        self.progress = None
        self.gpuids = None
    
    def runnable(self, status):
        def get_avail_gpus(s): # Return all available gpus on single node, Ranked as Priority
            gpu_prio_list = []
            for i in range(len(s.gpumem)):
                if s.gpumem[i] >= self.reqs['gpumem'] and 100-s.gpuusg[i] >= self.reqs['gpuusg'] and s.gpupro[i] < self.reqs['gpupro'] \
                   and s.gpumem[i] + s.gpumymem[i] - s.taskmem[i] >= self.reqs['gpumem'] \
                   and s.gpupro[i] - s.gpumypro[i] + s.taskpro[i] < self.reqs['gpupro'] \
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

    def run_python(self, gpu, path, taskid, repeatid, seed):
        current_env = {
            "CUDA_VISIBLE_DEVICE": str(gpu),
        }

        self.process = sp.Popen(['python', __file__,
                                 '--mode', 'python',
                                 '--args', base64.b64encode(pickle.dumps(self.args)),
                                 '--path', path,
                                 '--taskid', str(taskid),
                                 '--repeatid', str(repeatid),
                                 '--seed', str(seed)],
                                cwd=os.getcwd(),
                                env = {**os.environ, **current_env})

    def run_torchrun_multinode(self, nnodes, node_rank, gpu_ids, path, taskid, repeatid, seed):
        current_env = {
            "CUDA_VISIBLE_DEVICE": ','.join(map(str, gpu_ids)),
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
                                 '--seed', str(seed)],
                                cwd = os.getcwd(),
                                env = {**os.environ, **current_env})

    def run_torchrun_singlenode(self, gpu_ids, path, taskid, repeatid, seed):
        current_env = {
            "CUDA_VISIBLE_DEVICE": ','.join(map(str, gpu_ids)),
        }
        
        self.process = sp.Popen(['torchrun',
                                 '--standalone',
                                 '--nnodes=1',
                                 f'--nproc-per-node={len(gpu_ids)}',
                                 __file__,
                                 '--mode', 'torchrun_singlenode',
                                 '--args', base64.b64encode(pickle.dumps(self.args)),
                                 '--path', path,
                                 '--taskid', str(taskid),
                                 '--repeatid', str(repeatid),
                                 '--seed', str(seed)],
                                cwd = os.getcwd(),
                                env = {**os.environ, **current_env})

    def run(self, nnodes, node_rank, gpuids, path, taskid, repeatid, seed):
        self.gpuids = gpuids
        os.makedirs(path, exist_ok=True)
        if nnodes > 1:
            self.run_torchrun_multinode(nnodes, node_rank, gpuids, path, taskid, repeatid, seed)
        elif len(gpuids) > 1:
            self.run_torchrun_singlenode(gpuids, path, taskid, repeatid, seed)
        else:
            self.run_python(gpuids[0], path, taskid, repeatid, seed)
    
    def alive(self):
        return (self.process.poll() == None)

    def terminate(self):
        self.process.terminate()
        self.process=None
        self.gpuids=None

    def join(self):
        retcode = self.process.wait()
        self.process=None
        self.gpuids=None
        return retcode

def subprocess(args, worldinfo, path, taskid, repeatid, seed):
    """This method will be the target of subprocess and started as a new process."""
    # 1: Set Gpu run environment:
    import torch
    torch.cuda.set_device(worldinfo['device'])
    torch.set_default_dtype(torch.float32)

    # 2: Set Random Seed
    import numpy as np
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

    # 3: setup logging: Hook Stdout & Stderr + setup logger (only for local main process)
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(path), "src")) # Source code is in rundir/../src
    from utils.logger import logger
    if worldinfo['rank'] == 0:
        import os
        import sys
        sys.stdout = HookedOutput(os.path.join(path, "log.txt"), sys.stdout)
        sys.stderr = HookedOutput(os.path.join(path, "err.txt"), sys.stderr)
        logger.initialize(path, taskid, repeatid)

    # 4: Set up Signal Handle function & Summary function
    def summary(signum, *args):
        if worldinfo['rank'] == 0:
            if signum is not None: print(f"Signal {signum} received!")
            _, mem_peak = tracemalloc.get_traced_memory()
            mem_snapshot = tracemalloc.take_snapshot()
            tracemalloc.stop()
            statistics = logger.summary(path = os.path.join(path, "stat.csv"))
            print(statistics)
            with open(os.path.join(path, "summary.json"), "w") as f:
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

    signal.signal(signal.SIGTERM, summary)
    signal.signal(signal.SIGINT, summary)

    # 5: Load, Setup stat (the target is in target.py target)
    spec = importlib.util.spec_from_file_location("target", os.path.join(os.path.dirname(path), "src", "target.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    func = module.target
    args = pickle.loads(args)
    exception = None
    start_time = {"Wall": time.time(), "User": time.process_time()}
    tracemalloc.start()

    # 6: Run
    try:
        func(config=args, worldinfo=worldinfo, path=path, taskid=taskid, repeatid=repeatid, seed=seed)
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        exception = e
    summary(None)

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
    args = parser.parse_args()

    if args.mode == 'python':
        worldinfo = {
            'device': "cuda:" + os.environ['CUDA_VISIBLE_DEVICE'],
            'localrank': 0,
            'localworldsize': 1, 
            'rank': 0,
            'worldsize': 1,
        }
    elif args.mode in ('torchrun_singlenode', 'torchrun_multinode'):
        worldinfo = {
            'device': "cuda:" + os.environ['CUDA_VISIBLE_DEVICE'].split(',')[int(os.environ['LOCAL_RANK'])],
            'localrank': int(os.environ['LOCAL_RANK']),
            'localworldsize': int(os.environ['LOCAL_WORLD_SIZE']),
            'rank': int(os.environ['RANK']),
            'worldsize': int(os.environ['WORLD_SIZE']),
        }

    subprocess(base64.b64decode(args.args), \
               worldinfo, \
               args.path, \
               args.taskid, \
               args.repeatid, \
               args.seed)