#!/usr/bin/env bash

cd "$(dirname "$0")"

co2mpas_VERSION=$(python ../co2mpas/_version.py --version)
export co2mpas_VERSION=${co2mpas_VERSION/+/-}
export CO2MPAS_TAG_DEBIAN=vinci1it2000/co2mpas-debian:v$co2mpas_VERSION
export CO2MPAS_TAG_EXE=vinci1it2000/co2mpas-exe:v$co2mpas_VERSION

echo "Building images for CO2MPAS version: $co2mpas_VERSION..."
docker-compose build "$@"
