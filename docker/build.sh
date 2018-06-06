#!/usr/bin/env bash
#
## Note: must run from an r-commit,
#  for version to have been engraved.
#

##
curbranch="$(git branch | grep '*')"
if [ "$curbranch" != 'latest' -a $# -ne 1 ]; then
    echo "Ensure you are dock-building with *polyversions* engraved
    and either do 'checkout latest', or invoke it again with an extra arg
    (assuming you know what you're doing)." > /dev/stderr
    exit 1
fi
ver=$(python ../co2mpas/__init__.py version)
export CO2MPAS_TAG=vinci1it2000/co2mpas:v${ver}
echo "Building image $CO2MPAS_TAG..."
docker-compose build
