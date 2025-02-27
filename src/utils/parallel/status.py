import json
import subprocess as sp
import os

class RunnerStatus:
    def __init__(self, devices=None):
        self.username = sp.check_output(['whoami']).decode().strip()

        if isinstance(devices, str):
            devices = set(map(int, devices.split(',')))

        cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        if cuda_devices is not None: 
            cuda_devices = set(map(int, cuda_devices.split(',')))
            self.devices = cuda_devices & set(devices) if devices is not None else cuda_devices
        else:
            self.devices = devices

        # Informations from gpustat
        self.gpumem = []
        self.gpuusg = []
        self.gpupro = []
        self.gpumymem = []
        self.gpumypro = []

        # Informations from task
        self.taskmem = []
        self.taskpro = []

        # Raw output of gpustat
        self.gpustat = None

    def available(self, idx):
        if idx >= len(self.gpumem):
            return False
        if self.devices is None:
            return True
        return idx in self.devices

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
        self.gpustat = sp.check_output(['gpustat']).decode()

    def start_task(self, gpuid, reqs):
        self.taskmem[gpuid] += reqs['gpumem']
        self.taskpro[gpuid] += 1
    
    def end_task(self, gpuid, reqs):
        self.taskmem[gpuid] -= reqs['gpumem']
        self.taskpro[gpuid] -= 1
    
    def __str__(self):
        return f"<RunnerStatus hostname={sp.check_output(['hostname']).decode()} gpumem={self.gpumem} gpuusg={self.gpuusg} gpupro={self.gpupro} gpumymem={self.gpumymem} gpumypro={self.gpumypro} taskmem={self.taskmem} taskpro={self.taskpro}>"
    
    def __repr__(self):
        return str(self)