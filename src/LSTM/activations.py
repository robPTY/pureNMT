import torch
from torch import Tensor

class Sigmoid:
    def forward(self, x: Tensor) -> Tensor:
        return 1.0 / (1.0 + torch.exp(-x))
    
class Tanh:
    def forward(self, x: Tensor) -> Tensor:
        return (torch.exp(x) - torch.exp(-x))/ (torch.exp(x) + torch.exp(-x))

class h:
    def forward(self, x: Tensor) -> Tensor:
        return (2.0 / (1.0 + torch.exp(-x))) - 1.0

class g:
    def forward(self, x: Tensor) -> Tensor:
        return (4.0 / (1.0 + torch.exp(-x))) - 2.0