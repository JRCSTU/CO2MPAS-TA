#!/usr/bin/env bash
#
## Note: must run from an r-commit,
#  for version to have been engraved.
#
ver=$(python ../co2mpas/__init__.py version)
export CO2MPAS_TAG=vinci1it2000/co2mpas:v${ver}
echo "Building image $CO2MPAS_TAG..."
docker-compose build
