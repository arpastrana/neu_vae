"""
An encoder factory.
"""

from neu_vae.models import ModelFactory

from neu_vae.models.encoders import LinearEncoder
from neu_vae.models.encoders import ConditionalLinearEncoder


class EncoderFactory(ModelFactory):
    """
    A factory to streamline the creation of encoders.
    """
    supported = {}

    @classmethod
    def register(cls, name, encoder):
        """
        Registers a supported encoder by name.

        Parameters
        ----------
        name : `str`
            The name key to store the encoder with.
        encoder : `neu_vae.models.Encoder`
            An encoder object.
        """
        cls.supported[name] = encoder

    @classmethod
    def create(cls, name):
        """
        Creates a encoder by name.

        Parameters
        ----------
        name : `str`
            The name of the encoder to create.

        Returns
        -------
        encoder : `neu_vae.models.Encoder`
            An encoder class.
        """
        return cls.supported[name]


# TODO: This can be done more elegantly.
EncoderFactory.register("LinearEncoder", LinearEncoder)
EncoderFactory.register("ConditionalLinearEncoder", ConditionalLinearEncoder)
