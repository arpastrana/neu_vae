"""
Let the good times roll.
"""

from functools import partial

from torchsummary import summary

from torch import device
from torch import load
from torch import cuda
from torch import manual_seed

from torch.nn import functional

from neu_vae.models import LinearEncoder
from neu_vae.models import LinearDecoder
from neu_vae.models import VanillaVAE


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

    # load checkpoint
    checkpoint = load(config["checkpoint_path"])

    # load state dict
    model.load_state_dict(checkpoint["model_state_dict"])

    # send model to device and store it
    model = model.to(dev)

    # print model summary
    summary(model, (1, config["input_dim"]))

    # print out
    print("----------------------------------------------------------------")
    print(f"Model Loaded")
    print("----------------------------------------------------------------")

    return model


if __name__ == "__main__":
    pass
