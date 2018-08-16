#!/usr/bin/env bash
echo 'DERECATION: script `requirements/install_requirements.sh` renamed as `install_coda_reqs.sh`!
  Please update the name in your scripts.' > /dev/stderr

"$(realpath "$(dirname "$0")")/install_conda_reqs.sh"