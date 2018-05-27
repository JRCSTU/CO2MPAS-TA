#!/bin/bash
#
## Example commands:
#
#  - Count functions::
#       xzcat <co2mparable-XXX.txt.xz> | cut -d, -f1 |sort --unique |wc -l
#
#  - Show all diff-lines (from 1st file)::
#       \ls -tr *xz |head -n2 | xargs ./cmp.sh
#
#  - Show all diff args (~310 args differ, ~300 same, out of ~360(!))::
#       ls -t /tmp/co2mp*.xz | head -n2 | xargs ./cmp.sh | \
#           grep -v PRINT |cut -d, -f2 |sort --unique
#
#  - Show all diff args:
#    (same as above with `cut -f1`, ~1400 funcs out of ~2160)
#

#diff <(xzcat "$1" | awk '!a[$0]++' ) <(xzcat "$2" | awk '!a[$0]++' ) --speed-large-files
comm <(xzcat "$1" | sort -u ) <(xzcat "$2" | sort -u ) -23
