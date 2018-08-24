#!/usr/bin/env bash
#
# Prequisite:
#   Any Python (even Py-2.x) with `setuptools` installed.
#
# SYNTAX:
#       build.sh [<fallback-version>  [<docker-compose-cli-arg-1>]...]
#
# Example:
#       ./build.sh  1.2.3b2  --no-cache
#
## BUT....better use Makefile, to avoid rebuilding wheel(!):
#       make docker
#
mydir="$(dirname "$0")"
cd "$mydir"

co2mpas_version() {
    local pname="$1" userver="$2"

    POLY_VERSION="$(git describe "--match=$pname-v*")"

    ## Maybe we are on an r-tag?
    #
    if [ -z "$POLY_VERSION" ]; then
        POLY_VERSION="$(python ../src/co2mpas/__init__.py version)"
    fi

    ## Fallback to user-provided version
    #
    if [ -z "$POLY_VERSION" ]; then
        POLY_VERSION="$userver"
    fi

    POLY_VERSION="${POLY_VERSION/+/-}"
}

## Master project defining version assumed `co2sim`.
co2mpas_version co2sim $1

export CO2MPAS_TAG="vinci1it2000/co2mpas:v$POLY_VERSION"
echo "Building co2mpas-wheel $POLY_VERSION..."

## Remove any wheels & build-dir(!) from the past.
rm -rf ../dist/* ../build/*
## Generate wheel in my dir, to send small docker-context as possible.
#  Wheel will derive version from Git or failback to user-version (if any).
#  Also provide `polyversion` lib without installing it.
PYTHONPATH=docker/pvlib.run  co2sim_VERSION="$1"  python ../setup.py bdist_wheel


echo "Building co2mpas-image $CO2MPAS_TAG..."
## Pass <docker-compose-cli-args>.
shift
docker-compose build "$@"
