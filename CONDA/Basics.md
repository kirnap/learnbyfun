# Introduction to Conda
Conda is a pip-virtualenv like package manager containing its own pip, python and libraries etc.

## How to install conda
1. You need to download the conda installer (either miniconda -lightweight- or anaconda -containing data-science packages-) from [miniconda link](https://conda.io/miniconda.html)
2. chmod +x your-installer
3. Specify your .local dir
4. Add to your path

## How to create environment
---
```bash
conda create -n your-env-name (python=your-pyhton-version optional)
source activate your-env-name # to activate your env
source deactivate # to deactivate your env
conda env list # To list environment names
conda list # to see what is installed in your conda env
conda remove -n your-env-name # to remove your env
```
Please search conda your-package-name from search banner in [official conda page](https://anaconda.org/)
