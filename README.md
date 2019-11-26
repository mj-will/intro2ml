# [intro2ml](https://mj-will.github.io/intro2ml/)
An assortment of scripts to help you get started in machine learning.

Some of scripts are based on the [examples](https://github.com/keras-team/keras/tree/master/examples) provided for Keras. They have been transferred to Jupyter notebooks and then edited and added to.

These notebooks are also intended to used at the meetings of the [Physics and Astronomy Machine Learning Journal Club at the University of Glasgow](https://phas-ml.github.io/)


## Accessing the notebooks

### Using Google Colab

These notebooks are intended to be used with [Google Colab](https://colab.research.google.com/) since this eliminates the need to install python packages locally and allows for training with GPUs.

They can be run using the Google Colab icons, for example:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mj-will/intro2ml/blob/master/notebooks/classification-MLP.ipynb)

To then enable the GPU go to:  **Runtime -> Change runtime type -> Hardware accelerator -> GPU** 

### Running locally with Conda

These scripts can also be ran locally by cloning the repository and using [conda](https://docs.conda.io/en/latest/miniconda.html) and the provided file `environment.yml` to create an environment to run a jupyter-notebook from.

Conda can be installed from [here](https://docs.conda.io/en/latest/miniconda.html). A conda environment can then be created:

```
conda env create -f environment.yml
```

The notebooks can then be accessed by activating the conda environment and typing in the directory containing the notebooks:

```
jupyter-notebook
```

See [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) for a guide on using conda environments.
