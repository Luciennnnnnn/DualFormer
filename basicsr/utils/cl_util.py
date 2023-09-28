import torch

def zerolike_params_dict(model):
    """
    Create a list of (name, parameter), where parameter is initalized to zero.
    The list has as many parameters as model, with the same size.
    :param model: a pytorch model
    """

    return [
        (k, torch.zeros_like(p).to(p.device))
        for k, p in model.named_parameters()
    ]
