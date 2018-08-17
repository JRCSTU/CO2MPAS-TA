#!/bin/bash
#
## Build wheels for all sub-projects.
#
#  Options:
#  ========
#  -n      dry-run

#  INFO: Release checklistm oved to:
#   https://github.com/JRCSTU/co2mpas/wiki/Developer-Guidelines#release-checklist
#
shopt -s extglob  # for wildcards to surely work

action="python setup.py bdist_wheel"

my_dir=`dirname "$0"`
cd $my_dir/..
projects="pCO2* ."

if [ "$1" = '-n' ]; then
    action="echo $action"
fi

p_ok=''
for p in $projects; do
    echo "Entering '$(realpath "$p")'..."
    cd $p
    rm -rf build/* dist/*
    $action && p_ok="$p_ok$(printf "\n  $p")"
    cd -
done
echo -e "\nExecuted '$action' for: $p_ok" >/dev/stderr

## Check CO2SIM WHEEL archive.
#
wildcard='./pCO2SIM/dist/co2mpas-*.whl'
matches="$(echo $wildcard)"
if [ "$wildcard" != "$matches" ]; then
    whl_list="$(unzip -l $wildcard)"
    ( echo "$whl_list" | grep -q co2mpas_template; ) || echo "FAIL: No TEMPLATE-file in WHEEL($(ldcard)!)!"
    ( echo "$whl_list" | grep -q co2mpas_demo; ) || echo "FAIL: No DEMO in WHEEL($(ldcard)!)!"
    ( echo "$whl_list" | grep -q simVehicle.ipynb; ) || echo "FAIL: No IPYNBS in WHEEL($(ldcard)!)!"
    ( echo "$whl_list" | grep -q .co2mpas_cache; ) && echo "FAIL!!!! CACHE IN DEMOS!!!"
else
        echo "No WHEEL ($wildcard) generated!!"
        exit 1
fi
