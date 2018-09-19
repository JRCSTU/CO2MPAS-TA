#!/bin/bash
#
# Release checklist
# =================
# 1. Bump-ver & Update Date+Title in ./CHANGES.rst.
# 2. (if FINAL)REMOVE `pip install --pre` from README!!!
# 3. Run TCs.
# 4. commit & TAG & push
# 5. Gen DOCS & MODEL in `cmd.exe` & check OK (i.e. diagrams??)
#    and build `wheel,` `sdist` , `doc` archives:
#       ./bin/package.sh
# 6. Upload to PyPi:
#    - DELETE any BETAS (but the last one?)!!
#       - twine upload -su <gpg-user> dist/* # Ignore warn about doc-package.
#
# +++MANUAL+++
# 7. Prepare site at http://co2mpas.io/
#   - copy ALLINONES
#   - copy `allinone/CO2MPAS/packages` dir created during:
#            pip install co2mpas --download %home%\packages
#    - Expand docs, link STABLE ad LATEST
# 8. Prepare email (and test)
#    - Use email-body to draft a new "Release" in github (https://github.com/JRCSTU/co2mpas/releases).
#

my_dir=`dirname "$0"`
cd $my_dir/..

## Generate Site:
rm -r ./doc/_build/
cmd /C python setup.py build_sphinx

