import os
import imageio
from pygifsicle import optimize


DIR = "/Users/arpj/desktop/vae_interp/"
FPS = 16
OUT_NAME = "vae_mnist_circular_interp_{}_fps.gif".format(FPS)
SUFFIX = ".png"

LOOP = 1
OPTIMIZE = False
COLORS = 56  # NOTE: 56 colors hurts animation quality on MNIST!

filenames = os.listdir(DIR)
OUT = os.path.join(DIR, OUT_NAME)

images = []
filenames = [filename for filename in filenames if filename.endswith(SUFFIX)]
# sorting by numbers, assuming filenames are number.png
filenames = sorted(filenames, key=lambda f: int(os.path.splitext(f)[0]))

for filename in filenames:
    file_path = os.path.join(DIR, filename)
    images.append(imageio.imread(file_path))

print('baking...')
# imageio.mimsave(OUT, images, fps=FPS)  # mp4
imageio.mimsave(OUT, images, loop=LOOP, fps=FPS)  # gif
print('baked!')

if OPTIMIZE:  # only for gif
    print('optimizing gif...')
    optimize(OUT, colors=COLORS)

print('done!')