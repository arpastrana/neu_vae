"""
Vanilla VAE.
"""

from torch import exp
from torch import randn_like
from torch import sum

from neu_vae.models import VanillaVAE


class BetaVAE(VanillaVAE):
    """
    A beta-VAE inspired by Higgins et al (2016).

    Parameters
    ----------
    beta : `float`
        The beta hyperparemeter that scales the KL divergence term in the loss.
    encoder : `models.Encoder`
        An initialized encoder.
    decoder : `models.Decoder`
        An initializer decoder.
    recon_loss_func : `torch.nn.functional.function`
        The function to compute the reconstruction loss.
        For a Bernoulli decoder, the binary cross entropy is recommended.
        For a Gaussian decoder, the squared error is suggested.
    """
    def __init__(self, beta, encoder, decoder, recon_loss_func):
        """
        The constructor.
        """
        super(BetaVAE, self).__init__(encoder, decoder, recon_loss_func)

        self.name = "Beta VAE"
        self.beta = beta

    def loss(self, input_data, recon_data, mean, logvar):
        """
        Calculate the training loss.

        Parameters
        ----------
        input_data : `torch.Tensor`
            The input data.
        recon_data : `torch.Tensor`
            The reconstructed data.
        mean : `torch.Tensor`
            The mean of the Gaussian latent space.
        logvar : `torch.Tensor`
            The log-variance of the Gaussian latent space.

        Returns
        -------
        loss_dict : `dict`
            A dictionary with computed losses.

        Notes
        -----
        The losses dictionary has keys ["loss", "recon_loss", "kld_loss"]
        """
        # must be a partial for BCE, because reduction="sum"
        r_loss = self.recon_loss_func(recon_data, input_data)

        # analytical KLD between two Gaussian distributions
        # TODO: What if beta is normalized? (beta = beta * M / N)
        # where M = batch size and N = number of batches
        kld_loss = self.beta * self.kl_divergence_gaussian(mean, logvar)

        # sum of the two previous losses
        loss = r_loss + kld_loss

        return {"loss": loss, "recon_loss": r_loss, "kld_loss": kld_loss}
