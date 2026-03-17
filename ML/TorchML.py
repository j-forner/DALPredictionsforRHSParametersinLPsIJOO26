import torch
from torch import nn
import math
from torch.nn import BatchNorm1d as BN
from math import sqrt
import einops
# build linear model

class LayerNorm(nn.Module):
    def __init__(self, dimension, layer_norm_eps = 1e-5):
        super().__init__()
        self.w = nn.Parameter(torch.ones(dimension))
        self.b = nn.Parameter(torch.zeros(dimension))
        self.layer_norm_eps = layer_norm_eps
    
    def forward(self, inp):
        # inp: [batch,  dimension]
        residual = inp - einops.reduce(inp, "batch dimension -> batch 1", "mean")
        # Calculate the variance, square root it. Add in an epsilon to prevent divide by zero.
        scale = (einops.reduce(residual.pow(2), "batch dimension -> batch 1", "mean") + self.layer_norm_eps).sqrt()
        normalized = residual / scale
        normalized = normalized * self.w + self.b

        return normalized

class LinearRegressionforSyn(nn.Module):
    
    '''
    num_feature: dimension of the feature vector
    num_var: dimension of the variable 
    '''
    def __init__(self, num_feature, num_var, dim, bias = True , identity_initialization = False):
        super().__init__()
        self.linear = nn.Linear(num_feature, dim, bias = bias )
        self.dim = dim
        self.num_var = num_var
        nn.init.xavier_uniform_(self.linear.weight)
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, feature):
        out = self.linear(feature)
        # out = torch.clamp(out, min = 0. )
        # out = out.view(-1, self.dim, self.num_var)
        return out

class MLP(nn.Module):
    """
    num_feature: Dimension of the input feature vector.
    num_var: Dimension of the output variable.
    hidden_sizes: A list of integers where each integer specifies the number of neurons in the corresponding hidden layer.
    The first item represents the size of the first hidden layer, and so on.
    squeeze: If True, the output will be squeezed to a scalar value.
    withrelu: If True, ReLU activation is applied after each hidden layer.
    """
    def __init__(self, num_feature,  num_var,  hidden_sizes, squeeze= False, withrelu= True ):
        super().__init__()
        layers = []
        self.squeeze = squeeze

        # Input layer
        layers.append(nn.Linear(num_feature,  hidden_sizes[0]))
        if withrelu:
            layers.append(nn.SiLU())
        # layers.append(LayerNorm(hidden_sizes[0]))

        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            if withrelu:
                layers.append(nn.GELU())
            # layers.append(LayerNorm(hidden_sizes[i + 1]))

        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1],  num_var))

        # Create Sequential model
        self.net = nn.Sequential(*layers)

    def forward(self, x):

        out = self.net(x)
        if self.squeeze:
            out = out.squeeze(-1) 
        return out