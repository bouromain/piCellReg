from setuptools import setup

setup(
    name="piCellReg",
    version="0.1",
    description="Multiday Registration toolbox",
    author="Bourboulou Romain",
    author_email="bouromain@gmail.com",
    packages=["piCellReg"],
    install_requires=[
        "numpy",
        "scipy",
        "bottleneck",
        "scikit-image",
        "tqdm",
        "matplotlib",
        "PyQt5",
    ],
)
