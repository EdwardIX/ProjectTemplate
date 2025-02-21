import os
import re
import copy
import itertools
import json
import yaml
from dataclasses import dataclass, is_dataclass, fields, MISSING
from typing import Optional, Union, Type, TypeVar, get_type_hints
from importlib.util import spec_from_file_location, module_from_spec

T = TypeVar("T")
def parse_structured_config(cls: Type[T], config: dict) -> T:
    """
    Converts a nested dictionary to a dataclass object.

    Args:
        cls: The dataclass type to instantiate.
        config: The nested dictionary to convert.

    Returns:
        An instance of the dataclass `cls` with the data populated.
    """
    init_args = {}
    type_hints = get_type_hints(cls)  # Resolve actual types from string annotations

    # Ensure all keys in config are valid fields of cls
    valid_fields = {field.name for field in fields(cls)}
    invalid_keys = set(config.keys()) - valid_fields
    if invalid_keys:
        raise ValueError(f"Invalid keys in config: {invalid_keys}")

    for field in fields(cls):
        field_name = field.name
        field_type = type_hints[field_name]  # Use resolved type hints
        if field_name in config:
            value = config[field_name]
            if is_dataclass(field_type):  # Handle nested dataclasses
                init_args[field_name] = parse_structured_config(field_type, value)
            else:
                init_args[field_name] = value
        else:
            # Use default values if the field is missing
            if field.default is not MISSING:
                init_args[field_name] = field.default
            elif field.default_factory is not MISSING:  # For default_factory
                init_args[field_name] = field.default_factory()
            elif is_dataclass(field_type): # Init the dataclass with default params
                try:
                    init_args[field_name] = field_type() # If failed (required params not specified), will raise an Exception Here
                except Exception as e:
                    raise ValueError(f"Required field {field_name} failed to initialize: {e}")
            else:
                raise ValueError(f"Missing required field: {field_name} in {cls}")
    
    return cls(**init_args)

class Configurable:
    @dataclass
    class Config:
        pass

    cfg: Config

    def __init__(self, cfg: Optional[Union[dict, Type[T]]] = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        cfg = cfg or {}  # Default to an empty dictionary
        config_cls = getattr(self, "Config", None)
        if config_cls is None or not is_dataclass(config_cls):
            raise AttributeError("Configurable classes must define a Config dataclass.")
        
        if isinstance(cfg, config_cls):
            self.cfg = cfg
        elif isinstance(cfg, dict):
            self.cfg = parse_structured_config(config_cls, cfg)
        else:
            raise TypeError("Config must be a dictionary or an instance of the Config dataclass.")

def load_config_from_file(filepath):
    # 确保文件存在
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"No such file: '{filepath}'")

    if filepath.endswith(".yaml"):
        with open(filepath, 'r') as f:
            return yaml.safe_load(f)
    elif filepath.endswith(".json"):
        with open(filepath, 'r') as f:
            return json.load(f)

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
    def __init__(self, dtype=None, ref=None):
        self.dtype = dtype
        self.ref = ref
    
    def __call__(self, s):
        out = s.strip().split(',')
        if self.dtype is not None:
            for i in range(len(out)):
                out[i] = self.dtype(out[i])
        elif self.ref is not None:
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
    ConfigParser: A class to parse config file and generate list of config base on file & cmdargs
    parser: an ArgumentParser with one required argument: config, and other arguments needed
    config: a config file (.py) defines a constant dict CONFIG, with the following keys:
        'args'  (required): the base args. 
            ** CAN BE OVERWRITTEN BY CMDARGS & SCAN, PRIORITY: SCAN > CMDARGS > DEFAULT**
            Note 1: Use '${}' to link to another parameter in args (e.g. 'Scheduler.epoch': '${Training.epoch}')
        'target'(optional): the target function to run (default: [main, target])
        'reqs'  (optional): the resources to run this program (dict with following keys)
            ** CAN BE OVERWRITTEN BY CMDARGS & SCAN, PRIORITY: SCAN > CMDARGS > DEFAULT**
            seed: seed for task (default None)
            numgpu: number of parallel gpus (default 1)
            gpumem: minimum memory requirement for each gpu (default 0)
            gpuusg: minimum gpu usage required to run this program (default 0)
            gpupro: maximum gpu programs running on each gpu, ignores others' program. Set to -1 to ignore. (default 1)
            occupy: use GPUs that others are currently using (default False)
            repeat: number of repeat needed (default 1)
            mulnode: allow multi node training (default False)
        'cmdargs' (optional): a dict with (name of cmdargs)-(linked param path in args)
            e.g.: '--batchsize': 'Training.Batchsize'
        'scan'    (optional): a dict with (param path in args)-(a list of values to be scaned)
            all the combinations of the values will be enumerated 
            Note 1: the list can be replace by a dict to be scaned together
            Note 2: reqs parameter can be modified by adding ':' character before the key (e.g. ':seed': [1,2,3])
            Note 3: to modify a parameter, use . to split. (e.g.: 'Training.Epochs': [1,2,3])
    """
    def __init__(self, parser):
        self.parser = parser
        config = parser.parse_known_args()[0].config

        if isinstance(config, str):
            suffix_list = ["", ".yaml", ".json", ".py"] # Try to load config file with different suffix

            for suffix in suffix_list:
                config_path = os.path.join("config", config + suffix)
                if os.path.isfile(config_path):
                    break
            else:
                raise FileNotFoundError(f"No such file: '{config}'")
            
            config = load_config_from_file(config_path)
        
        assert isinstance(config, dict), f"Not supported config type: {type(config)}"

        if 'args' not in config.keys():
            raise ValueError('the provided config file does not contains args argument')
        
        self.params = { # The default setting for params
            "target": config['target'] if 'target' in config else ('main', 'target'),
            "args": config['args'],
            "reqs": {
                'seed': "Random",
                'numgpu': 1,
                'gpumem': 0,
                'gpuusg': 0,
                'gpupro': 1,
                'repeat': 1,
                'occupy': False,
                'mulnode': False,
                **config.get('reqs', {}), # Overwrite default settings
            }
        }
        self.cmdargs = config.get('cmdargs', {}) # The mapping from cmdargs to params
        self.scan = config.get('scan') # parameter scan list

        self.config_list = []
    
    def parse_args(self):
        self.add_parser_args(self.parser)
        cmdargs = self.parser.parse_args()
        return cmdargs
    
    def _get_params(self, params, name):
        try:
            if name.startswith(':'): # Getting Running Params
                return params['reqs'][name[1:]]
            else:
                names = name.split('.')
                args = params['args']
                for i, n in enumerate(names):
                    if isinstance(args, list):
                        n = int(n)

                    if i != len(names) - 1:
                        args = args[n]
                    else:
                        return args[n]
        except Exception as e:
            raise ValueError(f"Get {name} in config file error: {str(e)}") from e
    
    def _modify_params(self, params, name, modify_value):
        if name.startswith(':'): # Modify Running Params
            reqs = params['reqs']
            reqs[name[1:]] = modify_value
        else:
            names = name.split('.')
            args = params['args']
            for i, n in enumerate(names):
                if isinstance(args, list):
                    n = int(n)

                if i != len(names) - 1:
                    args = args[n]
                else: 
                    args[n] = modify_value
    
    def interpolate_params(self, params, ref_params):
        for k, v in params.items():
            if isinstance(v, dict):
                self.interpolate_params(v, ref_params)
            elif isinstance(v, str):
                if re.search(r"\${(.*?)}", v):
                    v = re.sub(r"\${(.*?)}", lambda x: str(self._get_params(ref_params, x.group(1))), v)
                    try:
                        params[k] = eval(v)
                    except Exception as e:
                        raise ValueError(f"Failed to interpolate param {k} (parsed as {v}) in config file: {str(e)}") from e

    def add_parser_args(self, parser):
        # for runreq arguments
        parser.add_argument('--seed', type=str, default=None)
        parser.add_argument('--numgpu', type=int, default=None)
        parser.add_argument('--gpumem', type=int, default=None)
        parser.add_argument('--gpuusg', type=int, default=None)
        parser.add_argument('--gpupro', type=int, default=None)
        parser.add_argument('--repeat', type=int, default=None)
        parser.add_argument('--occupy', type=BoolParser(), default=None)
        parser.add_argument('--mulnode', type=BoolParser(), default=None)

        # for cmdargs arguments
        for k, v in self.cmdargs.items():
            k = (k if k.startswith('--') else '--' + k)
            p = self._get_params(self.params, v)
            if isinstance(p, list):
                parser.add_argument(k, type=ListParser(ref=p), default=None)
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
        
        # Interpolate Parameters
        for config in self.config_list:
            self.interpolate_params(config, config)
        return self.config_list