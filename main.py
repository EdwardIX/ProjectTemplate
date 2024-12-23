import argparse

from src.utils.parallel import Experiment
from src.utils.config import ConfigParser

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Project Template')
    parser.add_argument('--config', type=str, required=True)

    config_parser = ConfigParser(parser)
    cmdargs = config_parser.parse_args()
    print(cmdargs)

    for config in config_parser.calc_config(cmdargs):
        print(config)

    exp = Experiment(config_parser.config_list)
    # exp.send_to_server()
    exp.run_local()