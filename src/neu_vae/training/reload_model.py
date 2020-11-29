"""
Let the good times roll.
"""

import torch

from functools import partial

from torchsummary import summary

from torch import device
from torch import load
from torch import cuda
from torch import manual_seed

from torch.nn import functional

from neu_vae.models import EncoderFactory
from neu_vae.models import DecoderFactory
from neu_vae.models import VAEFactory


def reload_model(config):
    """
    Run training and testing.
    """
    # Set random seeds
    manual_seed(config["seed"])
    cuda.manual_seed(config["seed"])

    # override device
    use_cuda = cuda.is_available()
    dev = device("cuda" if use_cuda else "cpu")
    config["device"] = dev

     # create encoder
    n_classes = config["n_classes"]
    enc_kwargs = {"input_dim": config["input_dim"],
                  "hidden_dim": config["encoder_hidden_dim"],
                  "z_dim": config["z_dim"],
                  "act_func": getattr(torch, config["encoder_act_func"]),
                  "n_classes": n_classes}

    encoder = EncoderFactory.create(config["encoder_name"])
    encoder = encoder(**enc_kwargs)

    # create decoder
    dec_kwargs = {"z_dim": config["z_dim"],
                  "hidden_dim": config["decoder_hidden_dim"],
                  "output_dim": config["input_dim"],
                  "act_func": getattr(torch, config["decoder_act_func"]),
                  "pred_func": getattr(torch, config["decoder_pred_func"]),
                  "n_classes": n_classes}

    decoder = DecoderFactory.create(config["decoder_name"])
    decoder = decoder(**dec_kwargs)

    # assemble VAE
    reconstruction_loss = partial(getattr(functional, config["rec_loss"]),
                                  reduction="sum")

    vae_kwargs = {"encoder": encoder,
                  "decoder": decoder,
                  "recon_loss_func": reconstruction_loss,
                  "beta": config["beta"]}

    # selecte VAE model
    vae = VAEFactory.create(config["vae_name"])
    model = vae(**vae_kwargs)

    # load checkpoint
    checkpoint = load(config["checkpoint_path"])

    # load state dict
    model.load_state_dict(checkpoint["model_state_dict"])

    # send model to device and store it
    model = model.to(dev)

    # print model summary
    # summary(model, (1, config["input_dim"]))

    # print out
    print("----------------------------------------------------------------")
    print(f"Model Loaded")
    print("----------------------------------------------------------------")

    return model


if __name__ == "__main__":
    pass
