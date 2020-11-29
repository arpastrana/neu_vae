"""
A decoder factory.
"""

from neu_vae.models import ModelFactory

from neu_vae.models.decoders import LinearDecoder
from neu_vae.models.decoders import ConditionalLinearDecoder


class DecoderFactory(ModelFactory):
    """
    A factory to streamline the creation of encoders.
    """
    supported = {}

    @classmethod
    def register(cls, name, decoder):
        """
        Registers a supported decoder by name.

        Parameters
        ----------
        name : `str`
            The name key to store the decoder with.
        decoder : `neu_vae.models.Decoder`
            A decoder object
        """
        cls.supported[name] = decoder

    @classmethod
    def create(cls, name):
        """
        Creates a decoder by name.

        Parameters
        ----------
        name : `str`
            The name of the decoder to create.

        Returns
        -------
        decoder : `neu_vae.models.Decoder`
            A decoder class.
        """
        return cls.supported[name]


# TODO: This can be done more elegantly.
DecoderFactory.register("LinearDecoder", LinearDecoder)
DecoderFactory.register("ConditionalLinearDecoder", ConditionalLinearDecoder)
