#!/bin/bash
#
# \ls -tr *xz |head -n2 | xargs ./cmp.sh
diff <(xzcat $1 | awk '!a[$0]++' ) <(xzcat $2 | awk '!a[$0]++' )
