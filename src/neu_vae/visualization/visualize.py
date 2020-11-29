"""
Visualize the inner workings of a VAE.
"""
import time

import numpy as np

import matplotlib.pyplot as plt

from math import pi
from math import sin
from math import cos

from torch import no_grad
from torch import randn
from torch import cat
from torch import FloatTensor
from torch import LongTensor

from torchvision.utils import make_grid


def show_latent_grid(model, device, bound=1.5, num_samples=20):
    """
    Make a grid of images using torchvision.
    """
    # load a network that was trained with a 2d latent space
    assert model.z_dim == 2, "Only 2d latent space currently supported"

    with no_grad():
        # no training mode
        model.eval()

        # create a sample grid in 2d latent space
        z_x = np.linspace(-bound, bound, num_samples)
        z_y = np.linspace(-bound, bound, num_samples)

        z = FloatTensor(len(z_y), len(z_x), 2)

        for i, lx in enumerate(z_x):
            for j, ly in enumerate(z_y):
                z[j, i, 0] = lx
                z[j, i, 1] = ly

        z = z.view(-1, 2)  # flatten grid into a batch
        z = z.to(device)

        # reconstruct images from the latent vectors
        x_hat = model.decode(z)

        show_image_grid(x_hat, nrow=num_samples)


def save_circular_walk(model, fpath, device, start, end, num_samples=20, figsize=(8, 8)):
    """
    Make a grid of images using torchvision.
    """
    # load a network that was trained with a 2d latent space
    assert model.z_dim == 2, "Only 2d latent space currently supported"

    with no_grad():
        # no training mode
        model.eval()

        # assertions
        assert len(start) == 2, "End point must have two entries"
        assert len(end) == 2, "End point must have two entries"

        # create radius
        radius = (np.array(start) + np.array(end)) / 2.0
        rx = radius[0]
        ry = radius[1]

        # create theta vector
        theta = np.linspace(0.0, 2.0 * pi, num_samples)

        for i in range(len(theta) - 1):
            t = theta[i]
            # calculate new x and y
            x = rx + cos(t)
            y = ry + sin(t)

            z = FloatTensor(1, 2)
            z[:, 0] = x
            z[:, 1] = y

            z = z.view(-1, 2)  # flatten grid into a batch
            z = z.to(device)

            # reconstruct images from the latent vectors
            x_hat = model.decode(z)

            path = fpath + f"{i}.png"

            save_image_grid(x_hat, path, nrow=1)

            print(f"saved {i}")


def numpy_image(image):
    """
    Creates a numpy array image from a torch tensor.
    """
    # image = clamp_image(image)
    np_img = image.numpy()

    return np.transpose(np_img, (1, 2, 0))  # (1, 2, 0) shifts the axes


def show_numpy_image(image):
    """
    Shows a numpy image using matplotlib's imshow.
    """
    return plt.imshow(numpy_image(image))


def clamp_image(image, bounds=(0, 1)):
    """
    Clamps the values of an image to be within bounds.
    """
    a, b = bounds

    return image.clamp(a, b)


def image_grid(images, nrow=2, padding=0, imgsize=(1, 28, 28), figsize=(8, 8)):
    """
    Make a grid of images using torchvision.
    """
    # send images to cpu
    images = images.to("cpu")

    # for now, hard coded for the mnist dataset (1, 28, 28)
    images = images.view(-1, *imgsize)  # batch size, channels, height, width

    # gridding
    return make_grid(images.data, nrow=nrow, padding=padding)


def process_image_grid(images, nrow, padding, imgsize, figsize):
    """
    Make a grid of images using torchvision.
    """
    grid = image_grid(images, nrow, padding, imgsize)

    # make subplots -- NOTE: is this needed? better plt.figure?
    plt.subplots(figsize=figsize)

    # convert to numpy and do imshow
    show_numpy_image(grid)

    # clean up axes and whitespace
    plt.axis('off')
    plt.tight_layout()


def show_image_grid(images, nrow=2, padding=0, imgsize=(1, 28, 28), figsize=(8, 8)):
    """
    Make a grid of images using torchvision.
    """
    process_image_grid(images, nrow, padding, imgsize, figsize)

    plt.show()


def save_image_grid(images, path, nrow=2, padding=0, imgsize=(1, 28, 28), figsize=(8, 8)):
    """
    Make a grid of images using torchvision.
    """
    process_image_grid(images, nrow, padding, imgsize, figsize)

    plt.savefig(path, bbox_inches='tight', pad_inches=0)


def plot_conditional_image(model, label, device):
    """
    Plots a single image from a trained conditional VAE.
    """
    # create a random latent vector
    z = randn(1, model.z_dim).to(device)

    y = LongTensor([label]).view((-1, 1))
    y = model.one_hot(y).to(device, dtype=z.dtype)
    z = cat((z, y), dim=1)

    reconstructed_img = model.decode(z)
    img = reconstructed_img.view(28, 28).data

    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.title(label)

    plt.show()


if __name__ == "__main__":

    import yaml

    from neu_vae.training import reload_model


    with open("../training/config.yaml") as f:
        config = yaml.safe_load(f)


    path = "/Users/arpj/Desktop/vae_interp/"

    model = reload_model(config)
    # show_latent_grid(model, "cpu", bound=1.5, num_samples=20)
    # save_circular_walk(model, path, "cpu", (-1.5, -1.5), (1.5, 1.5), num_samples=10)
    for i in range(10):
        plot_conditional_image(model, i, "cpu")

    print("Done!")
