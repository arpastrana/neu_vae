"""
Train a VAE.
"""


def train(kwargs):
    """
    Train the VAE.
    """

    # parse arguments
    model = kwargs["model"]
    train_batcher = kwargs["train_batcher"]
    device = kwargs["device"]
    optimizer = kwargs["optimizer"]

    # set the train mode
    model.train()

    # loss of the epoch
    train_loss = 0.0
    train_recon_loss = 0.0
    train_kld_loss = 0.0

    for x, _ in train_batcher:

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

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # log losses
        train_loss += loss.item()
        train_recon_loss += recon_loss.item()
        train_kld_loss += kld_loss.item()

    return train_loss, train_recon_loss, train_kld_loss
