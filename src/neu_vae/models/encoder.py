"""
Encoder stuff.
"""

from torch.nn import Module
from torch.nn import Linear


class LinearEncoder(Module):
    """
    A simple neural encoder.

    Parameters
    ----------
    input_dim : `int`
        The size of the flattened input.
    hidden_dim : `int`
        The number of hiden units in the single hidden layer.
    z_dim : `int`
        The dimensionality of the latent space.
    act_func : `nn.Functional.function`
        The activation function of the single hidden layer.
    """
    def __init__(self, input_dim, hidden_dim, z_dim, act_func):
        """
        The constructor.
        """
        super(LinearEncoder, self).__init__()

        self.z_dim = z_dim
        self.linear = Linear(input_dim, hidden_dim)
        self.mean = Linear(hidden_dim, z_dim)
        self.logvar = Linear(hidden_dim, z_dim)
        self.act_func = act_func

    def forward(self, input_data):
        """
        Do a forward pass.
        """
        hidden = self.act_func(self.linear(input_data))  # (batch_size, hidden_dim)
        z_mean = self.mean(hidden)  # (batch_size, z_dim)
        z_logvar = self.logvar(hidden)  # (batch_size, z_dim)

        return z_mean, z_logvar
