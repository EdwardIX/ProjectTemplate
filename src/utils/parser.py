import os
import copy
import itertools
from importlib.util import spec_from_file_location, module_from_spec

def load_config_from_py(filepath):
    # 确保文件存在
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"No such file: '{filepath}'")

    # 动态加载模块
    spec = spec_from_file_location("module.name", filepath)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    # 获取CONFIG字典
    if hasattr(module, 'CONFIG') and isinstance(module.CONFIG, dict):
        return module.CONFIG
    else:
        raise AttributeError("The specified file does not contain a dictionary named 'CONFIG'.")

class ListParser:
    def __init__(self, ref):
        self.ref = ref
    
    def __call__(self, s):
        out = s.strip().split(',')
        if len(self.ref) != len(out):
            for i in range(1, len(self.ref)):
                assert type(self.ref[i]) == type(self.ref[0]), f"Different type encountered in reference list"
            T = type(self.ref[0])
            out = list(map(T, out))
        else:
            for i in range(len(out)):
                out[i] = type(self.ref[i])(out[i])
        
        return out

class BoolParser:
    def __init__(self):
        pass
    
    def __call__(self, s):
        if s == 'True' or s == 'true':
            return True
        elif s == 'False' or s == 'false':
            return False
        else:
            raise ValueError(f"Unrecognized boolean type {s}")

class ConfigParser:
    """
    ConfigParser: A class to parse config file and generate list of config base on cmdargs
    config: a config file (.py) defines a constant dict CONFIG, with the following keys:
        'args'  (required): the base args. ** CAN BE OVERWRITTEN BY CMDARGS **
        'reqs'  (optional): the resources to run this program (dict with following keys)
            ** CAN BE OVERWRITTEN BY CMDARGS & SCAN, PRIORITY: SCAN > CMDARGS > DEFAULT**
            seed: seed for task (default None)
            numgpu: number of parallel gpus (default 1)
            gpumem: minimum memory requirement for each gpu (default 0)
            gpuusg: minimum gpu usage required to run this program (default 0)
            occupy: use GPUs that others are currently using (default False)
            repeat: number of repeat needed (default 1)
            mulnode: allow multi node training (default False)
        'cmdargs' (optional): a dict with (name of cmdargs)-(linked param path in args)
            e.g.: '--batchsize': 'Training/Batchsize'
        'scan'    (optional): a dict with (param path in args)-(a list of values to be scaned)
            all the combinations of the values will be enumerated 
            Note: the list can be replace by a dict to be scaned together
    """
    def __init__(self, config):
        if isinstance(config, str):
            config = load_config_from_py(os.path.join('config', config if config.endswith('.py') else config + '.py'))
        
        assert isinstance(config, dict), f"Not supported config type: {type(config)}"

        if 'args' not in config.keys():
            raise ValueError('the provided config file does not contains args argument')
        
        self.params = {
            "args": config['args'],
            "reqs": {
                'seed': "Random",
                'numgpu': 1,
                'gpumem': 0,
                'gpuusg': 0,
                'repeat': 1,
                'occupy': False,
                'mulnode': False,
                **config.get('reqs', {}), # Overwrite default settings
            }
        }
        self.cmdargs = config.get('cmdargs', {})
        self.scan = config.get('scan')

        self.config_list = []
    
    def _get_params(self, params, name):
        names = name.split('/')
        args = params['args']
        for i, n in enumerate(names):
            if isinstance(args, list):
                n = int(n)

            if i != len(names) - 1:
                args = args[n]
            else:
                return args[n]
    
    def _modify_params(self, params, name, modify_value):
        if name.startswith(':'): # Modify Running Params
            reqs = params['reqs']
            reqs[name[1:]] = modify_value
        else:
            names = name.split('/')
            args = params['args']
            for i, n in enumerate(names):
                if isinstance(args, list):
                    n = int(n)

                if i != len(names) - 1:
                    args = args[n]
                else: 
                    args[n] = modify_value

    def add_parser_args(self, parser):
        # for runreq arguments
        parser.add_argument('--seed', type=str, default=None)
        parser.add_argument('--numgpu', type=int, default=None)
        parser.add_argument('--gpumem', type=int, default=None)
        parser.add_argument('--gpuusg', type=int, default=None)
        parser.add_argument('--repeat', type=int, default=None)
        parser.add_argument('--occupy', type=BoolParser(), default=None)
        parser.add_argument('--mulnode', type=BoolParser(), default=None)

        # for cmdargs arguments
        for k, v in self.cmdargs.items():
            k = (k if k.startswith('--') else '--' + k)
            p = self._get_params(self.params, v)
            if isinstance(p, list):
                parser.add_argument(k, type=ListParser(p), default=None)
            elif isinstance(p, bool):
                parser.add_argument(k, type=BoolParser(), default=None)
            else:
                parser.add_argument(k, type=type(p), default=None)

    def calc_config(self, args):
        if len(self.config_list):
            return self.config_list
    
        # parse cmdarg arguments
        args = vars(args)
        for k, v in self.cmdargs.items():
            p = args.get(k.removeprefix('--'))
            if p is not None:
                self._modify_params(self.params, v, p)

        # parse reqs arguments
        for k in self.params['reqs'].keys(): # Overwrite reqs setting by cmdargs
            p = args.get(k)
            if p is not None:
                self.params['reqs'][k] = p
        
        # Calculate config list
        self.config_list = []
        if self.scan is None:
            self.config_list.append(self.params)
        else:
            lengths = []
            for val in self.scan.values(): # Step 1: Calculate scan length for each group
                if isinstance(val, list):
                    lengths.append(range(len(val)))
                elif isinstance(val, dict):
                    val = list(map((lambda l:len(l)), val.values()))
                    assert len(set(val)) == 1, f"Multiple lengths ({val}) encountered in same scan group"
                    lengths.append(range(val[0]))
                else:
                    raise RuntimeError("Error in parsing scan params")
            
            for idx in itertools.product(*lengths): # Step 2: enumerate every setting
                new_params = copy.deepcopy(self.params)
                for i, (key, val) in enumerate(self.scan.items()):
                    if isinstance(val, list):
                        self._modify_params(new_params, key, val[idx[i]])
                    else:
                        for k, v in val.items():
                            self._modify_params(new_params, k, v[idx[i]])
                
                self.config_list.append(new_params)
        
        return self.config_list