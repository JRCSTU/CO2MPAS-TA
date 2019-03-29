#!/usr/bin/env bash
cd "$(dirname "$0")"
rm -rf test
mkdir test
cd test
co2mpas=../dist/co2mpas
$co2mpas -h
$co2mpas template
$co2mpas conf
$co2mpas demo
$co2mpas syncing -h
$co2mpas syncing template
$co2mpas syncing sync ../files/datasync.xlsx sync.xlsx
$co2mpas run inputs
