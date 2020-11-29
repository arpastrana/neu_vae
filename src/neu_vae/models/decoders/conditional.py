"""
Encoder stuff.
"""

from torch.nn import Linear

from neu_vae.models.decoders import LinearDecoder


class ConditionalLinearDecoder(LinearDecoder):
    """
    A simple conditional decoder.

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
    n_classes : `int`
        The number of classes to condition on.
    """
    def __init__(self, z_dim, hidden_dim, output_dim, act_func, pred_func, n_classes, **kwargs):
        """
        The constructor.
        """
        super(ConditionalLinearDecoder, self).__init__(z_dim,
                                                       hidden_dim,
                                                       output_dim,
                                                       act_func,
                                                       pred_func)

        self.linear = Linear(z_dim + n_classes, hidden_dim)
        self.n_classes = n_classes
