"""
Visualize the inner workings of a VAE.
"""

import numpy as np

import matplotlib.pyplot as plt

from torch import no_grad
from torch import randn
from torch import FloatTensor

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
        x_hat = model.decode(z).cpu()
        x_hat = x_hat.view(-1, 1, 28, 28)

        plt.subplots(figsize=(8, 8))
        show_image(make_grid(x_hat.data, nrow=num_samples, padding=0))

        plt.axis('off')
        plt.tight_layout()
        plt.show()


def show_image(img):
    """
    """
    # img = to_img(img)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


def to_img(x):
    """
    """
    x = x.clamp(0, 1)

    return x


if __name__ == "__main__":

    import yaml

    from neu_vae.training import reload_model


    with open("../training/config.yaml") as f:
        config = yaml.safe_load(f)

    model = reload_model(config)
    show_latent_grid(model, "cpu", bound=1.5, num_samples=20)

    print("Done!")
