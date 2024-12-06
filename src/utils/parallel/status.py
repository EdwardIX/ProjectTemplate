import pickle
import json
import subprocess as sp

class RunnerStatus:
    def __init__(self):
        self.username = sp.check_output(['whoami']).decode().strip()

        # Informations from gpustat
        self.gpumem = []
        self.gpuusg = []
        self.gpupro = []
        self.gpumymem = []
        self.gpumypro = []

        # Informations from task
        self.taskmem = []
        self.taskpro = []

    def update(self):
        gpustat = json.loads(sp.check_output(['gpustat', '--json']).decode())
        gpus = gpustat['gpus']
        self.gpumem = [g['memory.total']-g['memory.used'] for g in gpus]
        self.gpuusg = [g['utilization.gpu'] for g in gpus]
        self.gpupro = [len(g['processes']) for g in gpus]
        self.gpumymem = [sum([p['gpu_memory_usage'] for p in g['processes'] if p['username'] == self.username]) for g in gpus]
        self.gpumypro = [len([p for p in g['processes'] if p['username'] == self.username]) for g in gpus]
        if len(self.taskmem) < len(gpus):
            self.taskmem.extend([0] * (len(gpus) - len(self.taskmem)))
        if len(self.taskpro) < len(gpus):
            self.taskpro.extend([0] * (len(gpus) - len(self.taskpro)))

    def start_task(self, gpuid, reqs):
        self.taskmem[gpuid] += reqs['gpumem']
        self.taskpro[gpuid] += 1
    
    def end_task(self, gpuid, reqs):
        self.taskmem[gpuid] -= reqs['gpumem']
        self.taskpro[gpuid] -= 1

    def pack(self):
        return pickle.dumps((
            self.gpumem,
            self.gpuusg,
            self.gpupro,
            self.gpumymem,
            self.gpumypro,
            self.taskmem,
            self.taskpro,
        ))
    
    def unpack(self, data):
        (
            self.gpumem,
            self.gpuusg,
            self.gpupro,
            self.gpumymem,
            self.gpumypro,
            self.taskmem,
            self.taskpro
        ) = pickle.loads(data)
    
    def __str__(self):
        return f"<RunnerStatus hostname={sp.check_output(['hostname']).decode()} gpumem={self.gpumem} gpuusg={self.gpuusg} gpupro={self.gpupro} gpumymem={self.gpumymem} gpumypro={self.gpumypro} taskmem={self.taskmem} taskpro={self.taskpro}>"
    def __repr__(self):
        return str(self)