#!/usr/bin/env bash
#
## Install dependencies that are hard to find in conda-environments
#  like those with native C++ extensions and in other channels.
#
## NOTE: Used for reproducible docker because
#  conda has MKL support for `numpy`:
#   https://stackoverflow.com/a/37224954/548792
conda install \
    numpy==1.14.3  \
    numpy-base==1.14.3  \
    scikit-learn==0.19.1 \
    scipy==1.1.0 \
    pandas==0.23.0
## Add both channels since `defaults has newer certify/ssl
#
conda install -c defaults -c conda-forge \
    xgboost==0.72.1 \
    regex  # Needed if no GCC in linux bc no pre-compiled exists in PyPi.
