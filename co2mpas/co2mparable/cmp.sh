#!/bin/bash
#
## Example commands:
#
#  - See all function invocations in order::
#       xzcat <co2mparable-XXX.txt.xz> | grep -v PRINT | cut -d, -f2 |awk '!a[$0]++'
#
#  - Show all diff-lines (from 1st file)::
#       \ls -tr *xz |head -n2 | xargs ./cmp.sh
#
#  - Show all diff args (~310 args differ, ~300 same, out of ~360(!))::
#       ls -t /tmp/co2mp*.xz | head -n2 | xargs ./cmp.sh | cut -d, -f3 |sort --unique
#
#  - Show all diff args:
#    (same as above with `cut -f1`, ~1400 funcs out of ~2160)
#

# diff \
#     --speed-large-files \
#     <(xzcat "$1" | grep -v PRINT | awk '!a[$0]++' ) \
#     <(xzcat "$2" | grep -v PRINT | awk '!a[$0]++' )

comm <(xzcat "$1" | grep -v PRINT | sort -u )  <(xzcat "$2" | grep -v PRINT | sort -u ) -23
