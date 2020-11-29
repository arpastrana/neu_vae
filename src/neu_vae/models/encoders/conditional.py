"""
Encoder stuff.
"""

from torch.nn import Linear

from neu_vae.models.encoders import LinearEncoder


class ConditionalLinearEncoder(LinearEncoder):
    """
    A simple conditional encoder.

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
    n_classes : `int`
        The number of classes to condition on.
    """
    def __init__(self, input_dim, hidden_dim, z_dim, act_func, n_classes, **kwargs):
        """
        The constructor.
        """
        super(ConditionalLinearEncoder, self).__init__(input_dim + n_classes,
                                                       hidden_dim,
                                                       z_dim,
                                                       act_func)
        self.n_classes = n_classes
