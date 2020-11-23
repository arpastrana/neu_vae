"""
Let the good times roll.
"""
from datetime import datetime
from time import time

from functools import partial

from torchsummary import summary

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

from torch import device
from torch import cuda
from torch import manual_seed

from torch.nn import functional
from torch import optim

from neu_vae.models import LinearEncoder
from neu_vae.models import LinearDecoder
from neu_vae.models import VanillaVAE

from train_model import train
from test_model import test


def run(config):
    """
    Run training and testing.
    """
    # Set random seeds
    manual_seed(config["seed"])
    cuda.manual_seed(config["seed"])

    # override device
    dev = device("cuda" if cuda.is_available() else "cpu")
    config["device"] = dev

    # load datasets
    transformations = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.MNIST(config["data_path"],
                                   train=True,
                                   download=True,
                                   transform=transformations)

    test_dataset = datasets.MNIST(config["data_path"],
                                  train=False,
                                  download=True,
                                  transform=transformations)

    # define batchers
    train_batcher = DataLoader(train_dataset,
                               batch_size=config["batch_size"],
                               shuffle=True)

    test_batcher = DataLoader(test_dataset,
                              batch_size=config["batch_size"])

    config["train_batcher"] = train_batcher
    config["test_batcher"] = test_batcher

    # create encoder
    enc_act_func = getattr(functional, config["encoder_act_func"])

    encoder = LinearEncoder(config["input_dim"],
                            config["encoder_hidden_dim"],
                            config["z_dim"],
                            act_func=enc_act_func)

    # create decoder
    dec_act_func = getattr(functional, config["decoder_act_func"])
    dec_pred_func = getattr(functional, config["decoder_pred_func"])

    decoder = LinearDecoder(config["z_dim"],
                            config["decoder_hidden_dim"],
                            config["input_dim"],
                            act_func=dec_act_func,
                            pred_func=dec_pred_func)

    # assemble VAE
    reconstruction_loss = partial(getattr(functional, config["rec_loss"]),
                                  reduction="sum")

    model = VanillaVAE(encoder,
                       decoder,
                       reconstruction_loss)

    # send model to device and store it
    model = model.to(dev)
    config["model"] = model

    # print model summary
    summary(model, (1, config["input_dim"]))

    # create the optimizer
    optimizer = getattr(optim, config["optimizer"])
    config["optimizer"] = optimizer(model.parameters(), lr=config["lr"])

    # train and test
    print("Training...")
    print("----------------------------------------------------------------")

    # current date and time
    start = time()
    print(f"Start datetime: {datetime.now()}")
    print("----------------------------------------------------------------")

    for e in range(config["epochs"]):

        train_loss = train(config)
        test_loss = test(config)

        train_loss /= len(config["train_batcher"].dataset)
        test_loss /= len(config["test_batcher"].dataset)

        print(f'Epoch {e}, Train Loss: {train_loss:.2f}, Test Loss: {test_loss:.2f}')

    # current time
    print("----------------------------------------------------------------")
    print(f"End datetime: {datetime.now()}")
    print(f"Elapsed time: {round((time() - start) / 60.0, 2)} minutes")
    print("----------------------------------------------------------------")


if __name__ == "__main__":

    import yaml


    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    run(config)

    print("Done!")
