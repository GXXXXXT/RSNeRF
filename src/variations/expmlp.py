import torch
import torch.nn as nn
from .KAN import get_kan_model


class Decoder(nn.Module):
    def __init__(self,
                 args,
                 width=128,
                 lum_in_dim=1,
                 lum_dim=2,
                 color_dim=6,
                 kan_basis_type='rbf',
                 **kwargs):
        """
        """
        super().__init__()
        KAN = get_kan_model(kan_basis_type)
        self.lum_linears = nn.Sequential(
            KAN(
                layers_hidden=[lum_in_dim] +
                              [width // 2] +
                              [6]
            ),
            nn.Sigmoid()
        )

        self.exp_output = nn.Sequential(
            KAN(
                layers_hidden=[lum_dim + color_dim] +
                              [width // 2] +
                              [6]
            ),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        color_dim = self.lum_linears(inputs['rgb_mean'])
        input_encoding = torch.cat([color_dim, inputs['lum']], dim=-1)
        exposure = self.exp_output(input_encoding)
        return exposure
