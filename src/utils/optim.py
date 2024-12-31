import dataclasses
from typing import Optional

import torch
from torch.optim import lr_scheduler

from ..utils.logger import logger

@dataclasses.dataclass
class OptimizerConfig:
    name: str
    args: dict
    params: Optional[dict] = dataclasses.field(default_factory=dict)

def getattr_recursive(m, attr):
    for name in attr.split("."):
        m = getattr(m, name)
    return m

def get_parameters(model, name):
    module = getattr_recursive(model, name)
    if isinstance(module, torch.nn.Module):
        return module.parameters()
    elif isinstance(module, torch.nn.Parameter):
        return module
    return []

def parse_optimizer(config, model):
    if config.params is not None:
        params = [
            {"params": get_parameters(model, name), "name": name, **args}
            for name, args in config.params.items()
        ]
        logger.info(f"Specify optimizer params: {config.params}")
    else:
        params = model.parameters()
    if config.name in ["FusedAdam"]:
        import apex

        optim = getattr(apex.optimizers, config.name)(params, **config.args)
    elif config.name in ["Adam8bit", "AdamW8bit"]:
        import bitsandbytes as bnb

        optim = bnb.optim.Adam8bit(params, **config.args)
    else:
        optim = getattr(torch.optim, config.name)(params, **config.args)
    return optim

def get_scheduler(name):
    if hasattr(lr_scheduler, name):
        return getattr(lr_scheduler, name)
    else:
        raise NotImplementedError

def parse_scheduler(config, optimizer, steps_per_epoch=None):
    if config is None: return None
    assert config.interval in ["epoch", "step"]
    if config.name == "SequentialLR":
        return lr_scheduler.SequentialLR(
                optimizer,
                [
                    parse_scheduler(conf, optimizer, steps_per_epoch)
                    for conf in config.schedulers
                ],
                milestones=config.args['milestones'],
            )
    elif config.name == "ChainedScheduler":
        return lr_scheduler.ChainedScheduler(
                [
                    parse_scheduler(conf, optimizer, steps_per_epoch)
                    for conf in config.schedulers
                ]
            )
    else:
        if config.name == "OneCycleLR":
            return lr_scheduler.OneCycleLR(optimizer, **config.args, steps_per_epoch=steps_per_epoch)
        return get_scheduler(config.name)(optimizer, **config.args)
