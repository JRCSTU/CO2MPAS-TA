#!/usr/bin/env bash
#
# NOTE: conda has now MKL support for `numpy`:
#   https://stackoverflow.com/a/37224954/548792
#
mydir="${0%/*}"

## Note that `numpy` from feed-stock
#  is NOT MKL variant!
#
conda install \
    numpy==1.11.3  \
    numpy-base==1.11.3  \
    scikit-learn==0.18.1 \
    scipy==0.19.0 \
    pandas==0.20.1 \
    matplotlib

conda install -c conda-forge \
    xgboost==0.72.1 \
    regex  # Needed if no GCC in linux bc no pre-compiled exists in PyPi.

pip install -r "$mydir/exe.pip" --no-cache-dir
