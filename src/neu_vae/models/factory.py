"""
An abstract factory.
"""
from abc import ABC
from abc import abstractclassmethod


class ModelFactory(ABC):
    """
    A factory to streamline the creation model objects.
    """
    @abstractclassmethod
    def register(cls, name, model_obj):
        pass

    @abstractclassmethod
    def create(cls, name):
        pass
