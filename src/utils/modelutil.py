from collections import OrderedDict, defaultdict
import torch

def unwrap_ddp_state_dict(model: torch.nn.Module):
    """
    提取模型的 state_dict，如果模型或其直接子模块被 DDP 包裹，
    则返回未被 DDP 包裹的 state_dict，保持与未封装模型一致的格式。

    Parameters:
    model : torch.nn.Module
        输入的模型，其本身或者直接子模块可能被 DDP 包裹。

    Returns:
    state_dict : OrderedDict
        未被 DDP 包裹的模型的 state_dict，与未封装模型一致的格式。
    """

    # 如果模型本身是 DDP，提取其内部模块
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module.state_dict()
    else:
        state_dict = OrderedDict()
        for name, module in model.named_children():
            # 如果子模块是 DDP，提取其内部模块的 state_dict
            if isinstance(module, torch.nn.parallel.DistributedDataParallel):
                unwrapped_state_dict = module.module.state_dict()
            else:
                # 子模块未被 DDP 包裹，直接添加其 state_dict
                unwrapped_state_dict = module.state_dict()
            
            for key, value in unwrapped_state_dict.items():
                    full_key = f"{name}.{key}"
                    state_dict[full_key] = value

    return state_dict

def load_ddp_compatible_state_dict(model: torch.nn.Module, state_dict: OrderedDict, partial: bool = False):
    """
    根据提供的 state_dict 将参数加载到模型中，即使模型或其子模块被 DDP 包裹。

    Parameters:
    model : torch.nn.Module
        目标模型，其本身或直接子模块可能被 DDP 包裹。
    state_dict : OrderedDict
        已保存的模型参数。
    """
    # 如果模型本身是 DDP，获取内部模块
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module

    # 按子模块前缀对 state_dict 进行分类
    sub_state_dicts = defaultdict(OrderedDict)
    for key, value in state_dict.items():
        prefix, sub_key = key.split(".", 1)  # 获取顶层子模块的前缀
        sub_state_dicts[prefix][sub_key] = value

    # 加载子模块的 state_dict
    for name, module in model.named_children():
        if isinstance(module, torch.nn.parallel.DistributedDataParallel):
            module = module.module

        if name in sub_state_dicts:
            module.load_state_dict(sub_state_dicts[name])
        else: # Some module is not in the state_dict, but it has a state dict
            if not partial and len(module.state_dict()) != 0:
                raise ValueError(f"State dict does not contain the module: {name}")

    return model