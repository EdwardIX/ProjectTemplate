import importlib.util
import sys
import os
import pickle
import base64
import subprocess as sp

def subprocess(args, gpu, path, taskid, repeatid, seed, localrank):
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
    sys.path.append(os.path.join(path, "src"))
    from src.utils.logger import logger
    if localrank in (-1, 0):
        import os
        import sys
        sys.stdout = HookedOutput(os.path.join(path, "log.txt"), sys.stdout)
        sys.stderr = HookedOutput(os.path.join(path, "err.txt"), sys.stderr)
        logger.initialize(path, taskid, repeatid)

    # 4: Load, Setup stat (the target is in main.py target)
    spec = importlib.util.spec_from_file_location("main", os.path.join(path, "src", "main.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    func = module.target
    args = pickle.loads(args)
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

class Task:
    def __init__(self, args, reqs):
        self.args = args
        self.reqs = reqs

        self.progress = None
    
    def save(self, filepath):
        with open(filepath, "w") as f:
            json.dump(self.args)
    
    def run_python(self, gpu, path, taskid, repeatid, seed):
        assert len(gpu) == 1
        self.process = sp.Popen(['python', __file__, \
                                '--args', base64.b64encode(pickle.dumps(self.args)), \
                                '--gpu', str(gpu[0]), \
                                '--path', path, \
                                '--taskid', str(taskid), \
                                '--repeatid', str(repeatid), \
                                '--seed', str(seed)],
                                cwd=os.getcwd())

    def run_torchrun(self, gpu, path, taskid, repeatid, seed):
        raise NotImplementedError

    def run(self, gpu, path, taskid, repeatid, seed):
        os.makedirs(path)
        if len(gpu) > 1:
            self.run_torchrun(gpu, path, taskid, repeatid, seed)
        else:
            self.run_python(gpu, path, taskid, repeatid, seed)
    
    def alive(self):
        return (self.process.poll() == None)

if __name__ == "__main__":
    """
    The interface for calling script with sp.Popen
    """

    # Accept args
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_accelerate', action='store_true')
    parser.add_argument('--args', type=str)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--path', type=str)
    parser.add_argument('--taskid', type=int)
    parser.add_argument('--repeatid', type=int)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()

    if args.use_accelerate:
        import accelerate
        subprocess(base64.b64decode(args.args), \
                   args.gpu, \
                   args.path, \
                   args.taskid, \
                   args.repeatid, \
                   args.seed, \
                   accelerate.PartialState().local_process_index)
    else:
        subprocess(base64.b64decode(args.args), \
                   args.gpu, \
                   args.path, \
                   args.taskid, \
                   args.repeatid, \
                   args.seed, \
                   -1)