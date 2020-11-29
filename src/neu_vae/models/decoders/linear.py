"""
Encoder stuff.
"""

from torch.nn import Module
from torch.nn import Linear


class LinearDecoder(Module):
    """
    A simple neural decoder.

    Parameters
    ----------
    z_dim : `int`
        The dimensionality of the latent space.
    hidden_dim : `int`
        The number of hiden units in the single hidden layer.
    output_dim : `int`
        The size of the flattened output.
    act_func : `nn.Functional.function`
        The activation function of the single hidden layer.
    pred_func : ``torch.function`
        The activation function of the output layer.
    """
    def __init__(self, z_dim, hidden_dim, output_dim, act_func, pred_func, **kwargs):
        """
        The constructor.
        """
        super(LinearDecoder, self).__init__()

        self.z_dim = z_dim
        self.linear = Linear(z_dim, hidden_dim)
        self.output = Linear(hidden_dim, output_dim)
        self.act_func = act_func
        self.pred_func = pred_func

    def forward(self, input_data):
        """
        Do a forward pass.

        Parameters
        ----------
        input_data : `torch.Tensor`
            The reparametrized data.

        Returns
        -------
        prediction : `torch.Tensor`
            The predicted data.
        """
        hidden = self.act_func(self.linear(input_data))  # (batch_size, hidden_dim)

        return self.pred_func(self.output(hidden))  # (batch_size, output_dim)
