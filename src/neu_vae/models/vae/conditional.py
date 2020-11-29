"""
Vanilla VAE.
"""

from torch import zeros
from torch import cat

from neu_vae.models.vae import BetaVAE


class ConditionalBetaVAE(BetaVAE):
    """
    A VAE (almost) as per Higgins et al. (2016) with one-hot class labels.

    Parameters
    ----------
    encoder : `models.Encoder`
        An initialized encoder.
    decoder : `models.Decoder`
        An initializer decoder.
    recon_loss_func : `torch.nn.functional.function`
        The function to compute the reconstruction loss.
        For a Bernoulli decoder, the binary cross entropy is recommended.
        For a Gaussian decoder, the squared error is suggested.
        TODO: Is it the mean squared error or just the squared error?
    beta : `float`
        The beta hyperparemeter that scales the KL divergence term in the loss.
    """
    def __init__(self, encoder, decoder, recon_loss_func, beta, **kwargs):
        """
        The constructor.
        """
        super(ConditionalBetaVAE, self).__init__(encoder,
                                                 decoder,
                                                 recon_loss_func,
                                                 beta)

        assert encoder.n_classes == decoder.n_classes
        self.name = "Conditional Beta VAE"
        self.n_classes = encoder.n_classes


    def forward(self, input_data, input_labels, *args, **kwargs):
        """
        Do a forward step.

        Parameters
        ----------
        input_data : `torch.Tensor`
            The input data.
        input_labels : `torch.Tensor`
            The one-hot labels of the data.

        Returns
        -------
        reconstructed_data : `torch.Tensor`
            The reconstructed data.
        z_mean : `torch.Tensor`
            The mean of the Gaussian latent space.
        z_logvar : `torch.Tensor`
            The log-variance of the Gaussian latent space.

        """
        # one-hot encoding
        input_labels = self.one_hot(input_labels)

        # concatenate data and labels
        input_data = cat((input_data, input_labels), dim=1)

        # encode
        z_mean, z_logvar = self.encode(input_data)

        # reparametrize
        z_data = self.reparametrize(z_mean, z_logvar)

        # concatenate latent vector and labels
        z_data = cat((z_data, input_labels), dim=1)  # TODO: is dim=1 okay?

        # decode
        reconstruction = self.decode(z_data)

        return reconstruction, z_mean, z_logvar

    def one_hot(self, labels):
        """
        Encodes an integer numeric label as a one-hot tensor.

        Parameters
        ----------
        label : `torch.Tensor`
            The labels of the data. Integer values from 0 to number of classes.

        Returns
        -------
        one_hot : `torch.Tensor`
            The one-hot tensor.

        Notes
        -----
        For example, if the total number of classes is 5 and the input label 4,
        the one hot vector would be [0, 0, 0, 1, 0].
        """
        assert labels.shape[1] == 1
        assert max(labels).item() < self.n_classes

        one_hot = zeros(labels.size(0), self.n_classes, device=labels.device)
        one_hot.scatter_(1, labels.data, 1)

        return one_hot
