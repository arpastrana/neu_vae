"""
Let the good times roll.
"""
import os
import torch
import wandb

from datetime import datetime
from time import time

from functools import partial

from torchsummary import summary

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

from torch import device
from torch import save
from torch import cuda
from torch import manual_seed
from torch import cat

from torch.nn import functional
from torch import optim

from neu_vae.models import LinearEncoder
from neu_vae.models import LinearDecoder
from neu_vae.models import VanillaVAE

from neu_vae.training import train
from neu_vae.training import test

from neu_vae.visualization import image_grid
from neu_vae.visualization import show_image_grid


def run(config):
    """
    Run training and testing.
    """
    # wandb
    if config["wandb"]:
        wandb.init(config=config,
                   project=config["project"],
                   group=config["group"])

    # Set random seeds
    manual_seed(config["seed"])
    cuda.manual_seed(config["seed"])

    # override device
    use_cuda = cuda.is_available()
    dev = device("cuda" if use_cuda else "cpu")
    config["device"] = dev

    d_kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # load datasets
    transformations = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.MNIST(config["data_path"],
                                   train=True,
                                   download=True,
                                   transform=transformations,
                                   **d_kwargs)

    test_dataset = datasets.MNIST(config["data_path"],
                                  train=False,
                                  download=True,
                                  transform=transformations,
                                  **d_kwargs)

    # define batchers
    train_batcher = DataLoader(train_dataset,
                               batch_size=config["batch_size"],
                               shuffle=True)

    test_batcher = DataLoader(test_dataset,
                              batch_size=config["batch_size"],
                              shuffle=False)

    config["train_batcher"] = train_batcher
    config["test_batcher"] = test_batcher

    # create encoder
    enc_act_func = getattr(torch, config["encoder_act_func"])

    encoder = LinearEncoder(config["input_dim"],
                            config["encoder_hidden_dim"],
                            config["z_dim"],
                            act_func=enc_act_func)

    # create decoder
    dec_act_func = getattr(torch, config["decoder_act_func"])
    dec_pred_func = getattr(torch, config["decoder_pred_func"])

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

    # wandb
    if config["wandb"]:
        wandb.watch(model, log="all")

    # create the optimizer
    optimizer = getattr(optim, config["optimizer"])
    optimizer = optimizer(model.parameters(), lr=config["lr"])
    config["optimizer"] = optimizer

    # train and test
    print("Training...")
    print("----------------------------------------------------------------")

    # current date and time
    start = time()
    print(f"Start datetime: {datetime.now()}")
    print("----------------------------------------------------------------")

    # log control image
    test_losses = test(config)
    _, _, _, x, x_hat = test_losses
    num_img = config["test_num_img"]
    x_cat = cat((x[:num_img], x_hat[:num_img]), dim=0)
    grid = image_grid(x_cat, nrow=num_img)

    if config["wandb"]:
        wandb.log({"Test Example": wandb.Image(grid, caption="Epoch: 0")})


    for e in range(config["epochs"] + 1):

        train_losses = train(config)
        train_losses = [loss / len(train_batcher.dataset) for loss in train_losses]
        train_loss, train_recon_loss, train_kld_loss = train_losses

        # average
        test_losses = test(config)
        test_losses = [loss / len(test_batcher.dataset) for loss in test_losses]
        test_loss, test_recon_loss, test_kld_loss, x, x_hat = test_losses

        # print stuff
        print(f'Epoch {e}, Train Loss: {train_loss:.2f}, Test Loss: {test_loss:.2f}')

        # log images
        num_img = config["test_num_img"]
        x_cat = cat((x[:num_img], x_hat[:num_img]), dim=0)
        grid = image_grid(x_cat, nrow=num_img)
        # show_image_grid(x_cat, nrow=5)  # BUG: only works in test()?

        if not config["wandb"]:
            continue

        # wandb - merge into one logging operation
        wandb.log({"Train Loss - Total": train_loss,
                   "Train Loss - Reconstruction": train_recon_loss,
                   "Train Loss - KL Divergence": train_kld_loss,
                   "Test Loss - Total": test_loss,
                   "Test Loss - Reconstruction": test_recon_loss,
                   "Test Loss - KL Divergence": test_kld_loss,
                   "Test Example": wandb.Image(grid, caption=f"Epoch: {e}")})

    # save model with torch to wandb run dir (uploads after training is complete)
    # TODO: Save intermediary checkpoints instead
    if config["wandb"] and config["save_model"]:
        save({"epoch": e,
              "model_state_dict": model.state_dict(),
              "optimizer_state_dict": optimizer.state_dict(),
              "z_dim": config["z_dim"],
              "train_loss": train_loss,
              "test_loss": test_loss},
             os.path.join(wandb.run.dir, "model_state.pt"))

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
