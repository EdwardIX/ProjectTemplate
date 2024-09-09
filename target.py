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