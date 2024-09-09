import os
import copy
import argparse

from src.utils.parallel import Server
from src.utils.parser import ConfigParser

def target(config, **kwargs):
    config['Device'] = "cuda:" + str(kwargs['gpu'])
    config['Savepath'] = kwargs['path']
    
    from src.model.mlp import MLP
    import torch

    print(config)

    model = MLP()
    model.to(config['Device'])
    while False:
        x = torch.zeros(10, 10).to(config['Device'])
        y = model(x)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='bpiml')
    parser.add_argument('--config', type=str, required=True)

    config = parser.parse_known_args()[0].config
    config_parser = ConfigParser(config)

    parser.add_argument('--name', type=str, default="debug")
    parser.add_argument('--device', type=str, default="Auto")
    parser.add_argument('--seed', type=str, default="Random")
    parser.add_argument('--path', type=str, default=None)
    parser.add_argument('--capacity', type=int, default=1)
    parser.add_argument('--depend', type=str, default=False)

    config_parser.add_parser_args(parser)
    cmdargs = parser.parse_args()
    print(cmdargs)

    # serv = Server(exp_name=cmdargs.name, path=cmdargs.path, device=cmdargs.device, numgpu=cmdargs.numgpu, repeat=cmdargs.repeat, seed=cmdargs.seed, capacity=cmdargs.capacity)
    # serv.set_default_target(target)
    # serv.set_dependency(cmdargs.depend)

    for config in config_parser.calc_config(cmdargs):
        print(config)
        # serv.add_task_args({"config":config})
    # serv.run(parallel=False)