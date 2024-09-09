from torch.nn import Module, Parameter
from tools.gen_shell import generate
import torch


def make_rand():
    return torch.rand(1, 3, 224, 224)


class AddW(Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.w = Parameter(data=make_rand())

    def forward(self, y):
        return self.w + y


y = make_rand()

generate("Add_W", AddW(), [y], "./AddW")


class Add(Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.w = Parameter(data=make_rand())

    def forward(self, x, y):
        return x + y


x = make_rand()
y = make_rand()

generate("Add", Add(), [x, y], "./Add")
