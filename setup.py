from setuptools import setup, find_packages

VERSION = "0.1"
DESCRIPTION = "Bayesian inference for SDE models using ABC-SMC"

# Setting up
setup(
    name="datacondabc",
    version=VERSION,
    author="Petar Jovanovski",
    author_email="<petarj@chalmers.se>",
    description=DESCRIPTION,
    packages=find_packages(),
    install_requires=["numpy", "scipy", "numba", "pot", "torch", "pytorch-lightning"],
    keywords=["python", "abc", "sde"],
    classifiers=[
        "Intended Audience :: Researchers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
)
