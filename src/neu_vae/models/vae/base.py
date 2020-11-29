"""
An abtract base VAE.
"""

from abc import abstractmethod

from torch.nn import Module


class BaseVAE(Module):
    """
    The basis for VAEs.
    """
    def __init__(self):
        super(BaseVAE, self).__init__()

    def encode(self):
        raise NotImplementedError

    def decode(self):
        raise NotImplementedError

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def loss(self):
        pass
