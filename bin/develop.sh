#!/bin/bash
#
## Pip-install in "develop" mode all sub-projects.
#
#  Options:
#  ========
#  -n      dry-run
#
shopt -s extglob  # for wildcards to surely work

action="pip install -e ."

my_dir=`dirname "$0"`
cd $my_dir/..
projects="pCO2SIM pCO2DICE pCO2GUI ."  # order important

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
