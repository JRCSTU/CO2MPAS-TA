#!/bin/bash
#
# Release checklist
# =================
# INFO: Moved to: https://github.com/JRCSTU/co2mpas/wiki/Developer-Guidelines#release-checklist
#

my_dir=`dirname "$0"`
cd $my_dir/..

if [ $# -gt 1 ]; then
    ## Generate Site:
    rm -r ./doc/_build/
    #cmd /C co2mpas modelgraph -O doc/_build/html/_static/ co2mpas.model.model co2mpas.model.physical.wheels.wheels
    cmd /C python setup.py build_sphinx

    ## Package docs
    #
    gitver="`git describe --tags`"
    gitver="${gitver:1}"
    zipfolder="co2mpas-doc-$gitver"
    docdir="build/doc/$zipfolder"
    mkdir -p build/doc

    ## Pack docs
    #
    cp -lr doc/_build/html "$docdir"
    pushd build/doc
    #zip -r9 "../../dist/$zipfolder.zip" "$zipfolder"
    7z a -r "../../dist/$zipfolder.7z" "$zipfolder"
    popd

fi

rm -rf build/* dist/*
python setup.py build bdist_wheel


## Check SDIST archive.
#
wildcard='./dist/co2mpas-*.zip'
matches="$(echo $wildcard)"
if [ "$wildcard" != "$matches" ]; then
    src_list="$(unzip -l ./dist/co2mpas-*.zip)"
    ( echo "$src_list" | grep -q co2mpas_template; ) || echo "FAIL: No TEMPLATE-file in SDIST($wildcard)!"
    ( echo "$src_list" | grep -q co2mpas_demo; ) || echo "FAIL: No DEMO in SDIST($wildcard)!"
    ( echo "$src_list" | grep -q simVehicle.ipynb; ) || echo "FAIL: No IPYNBS in SDIST($wildcard)!"
else
        echo "No SDIST($wildcard) generated, fine."
fi

## Check WHEEL archive.
#
wildcard='./dist/co2mpas-*.whl'
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
