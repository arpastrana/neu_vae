"""
Test a VAE.
"""
from torch import no_grad


def test(kwargs):
    """
    Test a VAE.
    """
    # parse arguments
    model = kwargs["model"]
    test_batcher = kwargs["test_batcher"]
    device = kwargs["device"]
    optimizer = kwargs["optimizer"]

    # set the model to evaluation mode
    model.eval()

    # loss of the epoch
    test_loss = 0.0
    test_recon_loss = 0.0
    test_kld_loss = 0.0

    # disregard gradients
    with no_grad():

        for x, _ in test_batcher:

            # reshape the data into [batch_size, 784]
            x = x.view(-1, kwargs["input_dim"])
            x = x.to(device)

            # forward pass
            x_hat, z_mean, z_logvar = model(x)

            # loss
            loss_dict = model.loss(x, x_hat, z_mean, z_logvar)

            loss = loss_dict["loss"]
            recon_loss = loss_dict["recon_loss"]
            kld_loss = loss_dict["kld_loss"]

            # log losses
            test_loss += loss.item()
            test_recon_loss += recon_loss.item()
            test_kld_loss += kld_loss.item()

            # wandb logging

    return test_loss
