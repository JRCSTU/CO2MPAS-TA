#!/usr/bin/env bash
#
# NOTE: conda has now MKL support for `numpy`:
#   https://stackoverflow.com/a/37224954/548792
#
conda install -c conda-forge \
    numpy==1.14.3  \
    scikit-learn==0.19.1 \
    scipy==1.1.0 \
    xgboost==0.72 \
    pandas==0.23.0 \
    matplotlib

pip install -r exe.pip --no-cache-dir
