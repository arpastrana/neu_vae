"""
Visualize the inner workings of a VAE.
"""
import torch

from torch import no_grad
from torch import randn
from torch import cat
from torch import arange
from torch import tensor
from torch import FloatTensor
from torch import LongTensor
from torch import exp

from neu_vae.visualization import image_grid
from neu_vae.visualization import show_image_grid

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# Displaying routine

def plot_images(out, n_cols=10, count=False, figsize=(8, 8)):
    """
    Plot a lot of images a la Canziani (not really).
    """

    out_pic = out.data.cpu().view(-1, 28, 28)
    n_rows = int(out_pic.size()[0] / n_cols)
    fig = plt.figure(figsize=figsize)

    gs = GridSpec(n_rows,
                  n_cols,
                  width_ratios=[1]*n_cols,
                  wspace=0.0,
                  hspace=0.0,
                  top=0.95,
                  bottom=0.05,
                  left=0.17,
                  right=0.845)

    index = 0
    for i in range(n_rows):
        for j in range(n_cols):
            ax = plt.subplot(gs[i, j])
            ax.imshow(out_pic[index], cmap="gray")
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.axis("off")
            # ax.set_title(f"image: {index}")
            index += 1

    plt.show()


def latent_traversal(model, x_data, y_data, use_y=False, n_images=1, bound=4):
    """
    Walk across every dimension of the latent space of a trained model.
    """
    model.eval()

    with no_grad():

        x_hat, mean, logvar = model(x, y)

        std_dev = exp(0.5 * logvar)
        std_dev_max = torch.max(std_dev)
        z_data = model.reparametrize(mean, logvar)
        z_dims = model.z_dim

        traversal = arange(-bound, bound + 1, step=1)

        for i in range(n_images):

            z_sample = z_data[i, :]

            if use_y:
                y_sample = y_data[i, :].view(-1, 1)
                y_sample = model.one_hot(y_sample)

            # create a grid of modified z samples
            z_stream = []

            for idx, new_val in enumerate(traversal):

                for j in range(z_dims):

                    # if j not in [5, 8, 9]:
                    #     continue

                    z = z_sample.clone().detach().view(1, -1)
                    z[:, j] = new_val

                    if use_y:
                        z = cat((z, y_sample), dim=1)

                    z_stream.append(z)


            z_cat = cat(z_stream, dim=0)

            print("z cat shape", z_cat.shape)
            # print("z stack shape", z_cat.shape)

            x_hat = model.decode(z_cat)


            print(x_hat.size())

            show_image_grid(x_hat, nrow=z_dims, padding=4)
            # show_image_grid(x_hat, nrow=3, padding=4)


def latent_traversal_manual(model, x_data, y_data, use_y=False, n_images=1, bound=4):
    """
    Walk across every dimension of the latent space of a trained model.
    """
    model.eval()

    with no_grad():

        z_dims = model.z_dim
        x_hat, mean, logvar = model(x, y)
        std_dev = exp(0.5 * logvar)

        traversal = arange(-bound, bound + 1, step=1)

        for i in range(n_images):

            if use_y:
                y_sample = y_data[i, :].view(-1, 1)
                y_sample = model.one_hot(y_sample)

            std_sample = std_dev[i, :]
            mean_sample = mean[i, :]

            # create a grid of modified z samples
            z_stream = []

            for new_val in traversal:

                for j in range(z_dims):

                    z = new_val * std_sample + mean_sample
                    z = z.view(1, -1)
                    z[:, j] = new_val

                    if use_y:
                        z = cat((z, y_sample), dim=1)

                    z_stream.append(z)

            z_cat = cat(z_stream, dim=0)

            x_hat = model.decode(z_cat)

            show_image_grid(x_hat, nrow=z_dims, padding=4)

if __name__ == "__main__":

    import yaml

    from torchvision import datasets
    from torchvision import transforms
    from torch.utils.data import DataLoader

    from torch import device
    from torch import cuda
    from torch import manual_seed

    from neu_vae.training import reload_model


    with open("../training/config.yaml") as f:
        config = yaml.safe_load(f)

    # Set random seeds
    if config["seed"]:
        manual_seed(config["seed"])
        cuda.manual_seed(config["seed"])

    # override device
    use_cuda = cuda.is_available()
    dev = device("cuda" if use_cuda else "cpu")
    config["device"] = dev

    # load test dataset
    transformations = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(config["data_path"],
                                  train=False,
                                  download=True,
                                  transform=transformations)

    idx = test_dataset.targets==config["single_label"]
    test_dataset.targets = test_dataset.targets[idx]
    test_dataset.data = test_dataset.data[idx]

    # define test batcher
    d_kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    test_batcher = DataLoader(test_dataset,
                              batch_size=config["batch_size"],
                              shuffle=True,
                              **d_kwargs)

    # reload model
    model = reload_model(config)
    model.eval()

    # define some stuff
    num_images = 1
    bound=5
    for i in range(num_images):

        x, y = next(iter(test_batcher))
        x = x.view(-1, 784).to("cpu")
        y = y.view(-1, 1).to("cpu")

        use_y = True

        # NOTE: Maybe try only with a single digit? Like digit 1 or 4?

        latent_traversal(model, x, y, use_y, n_images=1, bound=bound)
        # latent_traversal_manual(model, x, y, use_y, n_images=1, bound=bound)

    print("Done!")
