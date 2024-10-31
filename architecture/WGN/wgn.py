import torch.nn as nn

from architecture.WGN.deconv import DeConvNet
from architecture.WGN.mlp import MLP
from architecture.WGN.pignn import PIGNN

class WGN(PIGNN):
    def __init__(self,
                 edge_in_dim: int,
                 node_in_dim: int,
                 global_in_dim: int,
                 edge_hidden_dim: int = 32,
                 node_hidden_dim: int = 32,
                 global_hidden_dim: int = 32,
                 num_nodes: int =  10,
                 output_dim: int = 16384,
                 n_pign_layers: int = 3,
                 residual: bool = True,
                 input_norm: bool = True,
                 pign_mlp_params: dict = None,
                 mlp_params: dict = None,
                 deconv_params: dict = None,
                 output_size: tuple = (128, 128)):
        super(WGN, self).__init__(edge_in_dim, node_in_dim, global_in_dim, edge_hidden_dim, node_hidden_dim,
                                        global_hidden_dim, output_dim, n_pign_layers, residual, input_norm,
                                        pign_mlp_params)


        self.num_nodes = num_nodes

        self.mlp = MLP(input_dim= self.num_nodes*node_hidden_dim,
                       output_dim= 64,
                       num_neurons=mlp_params["num_neurons"],
                       hidden_act='ReLU')

        self.deconv = DeConvNet(1,
                                layer_channels = deconv_params["layer_channels"],
                                output_size=output_size)

        # self.mlp = MLP(input_dim=self.num_nodes*node_hidden_dim, output_dim=64,num_neurons=[128, 128, 64], hidden_act='ReLU')
        # self.deconv = DeConvNet(1, [64, 128, 256, 1], output_size=output_size)

    # Override
    def forward(self, data, nf, ef, gf):

        unf, uef, ug = self._forward_graph(data, nf, ef, gf)

        output_pignn = unf.reshape(-1, 1, self.num_nodes*unf.size(1))
        output_mlp = self.mlp(output_pignn).reshape(-1, 1, 8, 8)
        output = self.deconv(output_mlp)

        return output