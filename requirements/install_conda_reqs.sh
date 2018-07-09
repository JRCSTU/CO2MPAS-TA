#!/usr/bin/env bash
#
# NOTE: conda has now MKL support for `numpy`:
#   https://stackoverflow.com/a/37224954/548792
#
mydir="${0%/*}"

conda install \
    numpy==1.14.3  \
    numpy-base==1.14.3  \
    scikit-learn==0.19.1 \
    scipy==1.1.0 \
    pandas==0.23.0 \
    matplotlib

conda install -c conda-forge \
    xgboost==0.72

pip install -r "$mydir/exe.pip" --no-cache-dir
