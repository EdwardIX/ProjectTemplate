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
        if len(ref) != len(out):
            for i in range(1, len(ref)):
                assert type(ref[i]) == type(ref[0]), f"Different type encountered in reference list"
            T = type(ref[0])
            out = list(map(T, out))
        else:
            for i in range(len(out)):
                out[i] = type(ref[i])(out[i])
        
        return out

class ConfigParser:
    """
    ConfigParser: A class to parse config file and generate list of config base on cmdargs
    config: a config file (.py) defines a constant dict CONFIG, with the following keys:
        'config'  (required): the base config. ** CAN BE OVERWRITTEN BY CMDARGS **
        'run'     (optional): the resources to run this program (dict with following keys)
            ** CAN BE OVERWRITTEN BY CMDARGS & SCAN, PRIORITY: SCAN > CMDARGS > DEFAULT**
            numgpu: number of parallel gpus (default 1)
            gpumem: minimum memory requirement for each gpu (default 0)
            gpuusg: minimum gpu usage required to run this program (default 0)
            repeat: number of repeat needed (default 1)
        'cmdargs' (optional): a dict with (name of cmdargs)-(linked param path in config)
            e.g.: '--batchsize': 'Training/Batchsize'
        'scan'    (optional): a dict with (param path in config)-(a list of values to be scaned)
            all the combinations of the values will be enumerated 
            Note: the list can be replace by a dict to be scaned together
    """
    def __init__(self, config):
        if isinstance(config, str):
            config = load_config_from_py(os.path.join('config', config if config.endswith('.py') else config + '.py'))
        
        assert isinstance(config, dict), f"Not supported config type: {type(config)}"

        if 'config' not in config.keys():
            raise ValueError('the provided config file does not contains config argument')
        
        self.params = {
            "config": config['config'],
            "run": {
                'numgpu': 1,
                'gpumem': 0,
                'gpuusg': 0,
                'repeat': 1,
                **config.get('run', {}), # Overwrite default settings
            }
        }
        self.cmdargs = config.get('cmdargs', {})
        self.scan = config.get('scan')

        self.config_list = []
    
    def _get_params(self, params, name):
        names = name.split('/')
        config = params['config']
        for i, n in enumerate(names):
            if isinstance(config, list):
                n = int(n)

            if i != len(names) - 1:
                config = config[n]
            else:
                return config[n]
    
    def _modify_params(self, params, name, modify_value):
        if name.startswith(':'): # Modify Running Params
            config = params['run']
            config[name[1:]] = modify_value
        else:
            names = name.split('/')
            config = params['config']
            for i, n in enumerate(names):
                if isinstance(config, list):
                    n = int(n)

                if i != len(names) - 1:
                    config = config[n]
                else: 
                    config[n] = modify_value

    def add_parser_args(self, parser):
        parser.add_argument('--numgpu', type=int, default=None)
        parser.add_argument('--gpumem', type=int, default=None)
        parser.add_argument('--gpuusg', type=int, default=None)
        parser.add_argument('--repeat', type=int, default=None)

        for k, v in self.cmdargs.items():
            k = (k if k.startswith('--') else '--' + k)
            p = self._get_params(self.params, v)
            if isinstance(p, list):
                parser.add_argument(k, type=ListParser(p), default=None)
            elif isinstance(p, bool):
                parser.add_argument(k, type=str, default=None)
            else:
                parser.add_argument(k, type=type(p), default=None)

    def calc_config(self, args):
        if len(self.config_list):
            return self.config_list
    
        args = vars(args)
        for k, v in self.cmdargs.items():
            p = args.get(k.removeprefix('--'))
            if p is not None:
                self._modify_params(self.params, v, p)

        for k in self.params['run'].keys(): # Overwrite run setting by cmdargs
            p = args.get(k)
            if p is not None:
                self.params['run'][k] = p
        
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