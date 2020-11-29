"""
An VAE factory.
"""

from neu_vae.models import ModelFactory

from neu_vae.models.vae import VanillaVAE
from neu_vae.models.vae import BetaVAE
from neu_vae.models.vae import ConditionalBetaVAE


class VAEFactory(ModelFactory):
    """
    A factory to streamline the creation VAEs.
    """
    supported = {}

    @classmethod
    def register(cls, name, model_obj):
        """
        Registers a supported VAE model by name.

        Parameters
        ----------
        name : `str`
            The name key to store the VAE with.
        model : `neu_vae.models.BaseVAE`
            A VAE class.
        """
        cls.supported[name] = model_obj

    @classmethod
    def create(cls, name):
        """
        Creates a VAE model by name.

        Parameters
        ----------
        name : `str`
            The name of the VAE model to create.

        Returns
        -------
        model : `neu_vae.models.BaseVAE`
            A VAE class.
        """
        return cls.supported[name]


# TODO: This can be done more elegantly.
VAEFactory.register("VanillaVAE", VanillaVAE)
VAEFactory.register("BetaVAE", BetaVAE)
VAEFactory.register("ConditionalBetaVAE", ConditionalBetaVAE)
