import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LearnableAct(th.nn.Module):
    def __init__(self, inputdim, gridsize=1, addbias=True, base_fn=None):
        super(LearnableAct, self).__init__()
        self.gridsize = gridsize
        self.addbias = addbias
        self.inputdim = inputdim

        # The normalization has been chosen so that if given inputs where each coordinate is of unit variance,
        # then each coordinates of the output is of unit variance
        # independently of the various sizes
        self.fouriercoeffs = th.nn.Parameter(th.randn(2, inputdim, gridsize) /
                                             (np.sqrt(inputdim) * np.sqrt(self.gridsize)))

        self.sin_coeffs = th.nn.Parameter(th.randn(inputdim, gridsize) / (np.sqrt(inputdim) * np.sqrt(self.gridsize)))
        self.cos_coeffs = th.nn.Parameter(th.randn(inputdim, gridsize) / (np.sqrt(inputdim) * np.sqrt(self.gridsize)))
        self.base_fn = base_fn

        self.learn_coeffs = nn.Parameter(th.rand(inputdim) / 100)
        self.act_coeffs = nn.Parameter(th.rand(inputdim) / 100)

        self.learn_coeffs2 = nn.Parameter(th.rand(inputdim) / 100)
        # limit = 1 / (np.sqrt(inputdim) * np.sqrt(self.gridsize))

        nn.init.xavier_normal_(self.sin_coeffs, )
        nn.init.xavier_normal_(self.cos_coeffs, )

        self.norm = nn.LayerNorm(inputdim)

    # x.shape ( ... , indim )
    # out.shape ( ..., outdim)
    def forward(self, x):
        k = th.arange(1, self.gridsize + 1, device=x.device)
        temp = torch.einsum('...d,j->...dj', x, k)

        c = th.cos(temp)
        s = th.sin(temp)
        y = th.einsum('...dj,dj->...d', c, self.cos_coeffs)

        y += th.einsum('...dj,dj->...d', s, self.sin_coeffs)

        # print(y.std())

        y = torch.einsum('...d,d->...d', y, self.learn_coeffs)

        if self.base_fn is not None:
            y += torch.einsum('...d,d->...d', self.base_fn(x), self.act_coeffs)
        # y = torch.einsum('...d,d->...d', y, self.learn_coeffs2)
        y = F.gelu(y)
        return y


if __name__ == '__main__':
    b, d = 1, 10
    noise = torch.rand(b, d)
    model = LearnableAct(d)

    model(noise)
