#!/usr/bin/env bash

cd "$(dirname "$0")"
rm -rf build dist
pyinstaller --clean -y -F ../../co2mpas/cli/__init__.py -n co2mpas --additional-hooks-dir=hooks