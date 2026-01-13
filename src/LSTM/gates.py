import torch 
from torch import Tensor
from activations import Sigmoid, Tanh

class ForgetGate:
    def forward(self, Wf: Tensor, Xt: Tensor, bf: Tensor) -> Tensor:
        s = Sigmoid()
        return s.forward(Xt @ Wf + bf)

class InputGate:
    def forward(self, Wi: Tensor, Xt: Tensor, bi: Tensor) -> Tensor:
        s = Sigmoid() 
        return s.forward(Xt @ Wi + bi)

class CandidateGate:
    def forward(self, Wc: Tensor, Xt: Tensor, bc: Tensor) -> Tensor:
        t = Tanh()
        return t.forward(Xt @ Wc + bc)

class OutputGate:
    def forward(self, Wo: Tensor, Xt: Tensor, bo: Tensor) -> Tensor:
        s = Sigmoid() 
        return s.forward(Xt @ Wo + bo)