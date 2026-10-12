from torch import nn

class Linear(nn.Module):
    weight: nn.Parameter
    # Typed like torch.nn.Linear.bias: it is only None when the layer is built with bias=False.
    bias: nn.Parameter
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True) -> None: ...
