from setuptools import setup
from setuptools import find_packages


setup(
    name='neu_vae',
    version='0.1.0',
    description='An experiment with VAEs for NEU 560 at Princeton University',
    author='Rafael Pastrana',
    license='MIT',
    # packages=find_packages(),
    packages=["neu_vae"],
    package_dir={"": "src"}
)
