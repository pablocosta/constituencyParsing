from modules.dropout import sharedDropout

import torch


class mlp(torch.nn.Module):

    def __init__(self, nIn, nOut, dropout=0):
        super(mlp, self).__init__()

        self.nIn        = nIn
        self.nOut       = nOut
        self.linear     = torch.nn.Linear(nIn, nOut)
        self.activation = torch.nn.LeakyReLU(negative_slope=0.1)
        self.dropout    = sharedDropout(p=dropout)

        self.resetParameters()

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f"n_in={self.n_in}, n_out={self.n_out}"
        if self.dropout.p > 0:
            s += f", dropout={self.dropout.p}"
        s += ')'

        return s

    def resetParameters(self):
        torch.nn.init.orthogonal_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)

        return x