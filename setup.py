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
        "bottleneck",
        "suite2p",
        "scanimage-tiff-reader",
        "h5py",
        "mat73",
        "glob",
    ],
)