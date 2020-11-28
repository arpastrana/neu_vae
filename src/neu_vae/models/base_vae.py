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

    def encode(self, input_data):
        raise NotImplementedError

    def decode(self, input_data):
        raise NotImplementedError

    @abstractmethod
    def forward(self, input_data):
        pass

    @abstractmethod
    def loss(self, *inputs, **kwargs):
        pass
