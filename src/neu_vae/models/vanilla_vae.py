"""
Vanilla VAE.
"""

from torch import exp
from torch import randn_like
from torch import sum

from neu_vae.models import BaseVAE


class VanillaVAE(BaseVAE):
    """
    A VAE (almost) as described by Kingma and Welling (2014).

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
    """
    def __init__(self, encoder, decoder, recon_loss_func):
        """
        The constructor.
        """
        super(VanillaVAE, self).__init__()

        assert encoder.z_dim == decoder.z_dim

        self.name = "Vanilla VAE"
        self.z_dim = encoder.z_dim
        self.encoder = encoder
        self.decoder = decoder
        self.recon_loss_func = recon_loss_func

    def forward(self, input_data):
        """
        Do a forward step.

        Parameters
        ----------
        input_data : `torch.Tensor`
            The input data.

        Returns
        -------
        reconstructed_data : `torch.Tensor`
            The reconstructed data.
        z_mean : `torch.Tensor`
            The mean of the Gaussian latent space.
        z_logvar : `torch.Tensor`
            The log-variance of the Gaussian latent space.

        """
        # encode
        z_mean, z_logvar = self.encode(input_data)

        # reparametrize
        z_data = self.reparametrize(z_mean, z_logvar)

        # decode
        reconstruction = self.decode(z_data)

        return reconstruction, z_mean, z_logvar

    def encode(self, input_data):
        """
        Encodes the input as mean and log-variance tensors.

        Parameters
        ----------
        input_data : `torch.Tensor`
            The input data.

        Returns
        -------
        mean : `torch.Tensor`
            The mean of the Gaussian latent space.
        logvar : `torch.Tensor`
            The log-variance of the Gaussian latent space.
        """
        mean, logvar = self.encoder(input_data)

        return mean, logvar

    def decode(self, z_data):
        """
        Decode a sample from the latent space back into data space.

        Parameters
        ----------
        z_data : `torch.Tensor`
            The sampled data from the Gaussian latent space.

        Returns
        -------
        reconstructed_data : `torch.Tensor`
            The reconstructed data.
        """
        return self.decoder(z_data)

    def generate(self, input_data):
        """
        Given some input data return its corresponding reconstruction.

        Parameters
        ----------
        input_data : `torch.Tensor`
            The input data.

        Returns
        -------
        reconstructed_data : `torch.Tensor`
            The reconstructed data
        """
        return self.forward(input_data)[0]  # is [0] necessary?

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
        kld_loss = self.kl_divergence_gaussian(mean, logvar)

        # sum of the two previous losses
        loss = r_loss + kld_loss

        return {"loss": loss, "recon_loss": r_loss, "kld_loss": kld_loss}

    @staticmethod
    def kl_divergence_gaussian(mean, logvar):
        """
        Analytically computes the negative KL Divergence of two Gaussians.

        Parameters
        ----------
        mean : `torch.Tensor`
            The mean of the Gaussian latent space.
        logvar : `torch.Tensor`
            The log-variance of the Gaussian latent space.

        Returns
        -------
        kl_divergence : `float`
            The value of the KL divergence.

        Notes
        -----
        One of the two distributions is assumed to be a unit Gaussian N~(0, I).
        """
        return - 0.5 * sum(1.0 + logvar - mean.pow(2) - logvar.exp())

    @staticmethod
    def reparametrize(mean, logvar):
        """
        Reparametrization trick to sample N(mean, var) from N(0, I).

        Parameters
        ----------
        mean : `torch.Tensor`
            The mean of the Gaussian latent space.
        logvar : `torch.Tensor`
            The log-variance of the Gaussian latent space.

        Returns
        -------
        z_data : `torch.Tensor`
            The sampled data from the Gaussian latent space.
        """
        std = exp(0.5 * logvar)
        eps = randn_like(std)
        return eps * std + mean
