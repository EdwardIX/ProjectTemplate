import pickle
import json
import subprocess as sp

class RunnerStatus:
    def __init__(self, hostname):
        # self.hostname = hostname
        self.username = sp.check_output(['whoami']).decode()
        self.gpumem = []
        self.gpuusg = []
        self.gpupro = []


    def update(self):
        gpustat = json.loads(sp.check_output(['gpustat', '--json']).decode())
        gpus = gpustat['gpus']
        self.gpumem = [g['memory.total']-g['memory.used'] for g in gpus]
        self.gpuusg = [g['utilization.gpu'] for g in gpus]
        self.gpupro = [len(g['processes']) for g in gpus]
        self.gpumypro = [len([p for p in g['processes'] if p['username'] == self.username]) for g in gpus]

    def pack(self):
        package = (
            self.gpumem,
            self.gpuusg,
            self.gpupro,
            self.gpumypro,
        )
        return pickle.dumps(package)
    
    def unpack(self, data):
        (
            self.gpumem,
            self.gpuusg,
            self.gpupro,
            self.gpumypro,
        ) = pickle.loads(data)
    
    def __str__(self):
        return f"<RunnerStatus hostname={sp.check_output(['hostname']).decode()} gpumem={self.gpumem} gpuusg={self.gpuusg} gpupro={self.gpupro}>"
    def __repr__(self):
        return str(self)