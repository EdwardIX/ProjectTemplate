import os
import copy
import argparse

from src.utils.parallel import Server, Experiment, Runner
from src.utils.parser import ConfigParser

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Project Template')
    parser.add_argument('--config', type=str, required=True)

    config = parser.parse_known_args()[0].config
    config_parser = ConfigParser(config)

    config_parser.add_parser_args(parser)
    cmdargs = parser.parse_args()
    print(cmdargs)

    for config in config_parser.calc_config(cmdargs):
        print(config)

    exp = Experiment(config_parser.config_list)
    serv = Server()
    serv.add_experiment(exp)