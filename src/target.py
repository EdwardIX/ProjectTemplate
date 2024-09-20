def target(config, **kwargs):
    config['Device'] = kwargs['worldinfo']['device']
    config['Savepath'] = kwargs['path']
    
    from model.mlp import MLP
    import torch
    from utils.logger import logger

    model = MLP()
    model.to(config['Device'])

    logger.info("config:", config)
    print("kwargs:", kwargs)
    logger.add_scalars({"seed": 1}, 0)