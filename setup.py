from setuptools import setup, find_packages

setup(
    name="kronos",
    version="0.1",
    author="Kevin",
    description="Temporal metric learning.",
    python_requires=">=3.6",
    packages=find_packages(exclude=("configs", "tests")),
    license="MIT",
    install_requires=[
        "torch",
        "torchvision",
        "tqdm",
        "numpy",
        "scipy",
        "opencv-python",
        "tensorboard",
        "albumentations",
        "dtw",
        "yacs",
        "termcolor",
        "imageio",
        "imageio-ffmpeg",
        "ipdb",
    ],
)
