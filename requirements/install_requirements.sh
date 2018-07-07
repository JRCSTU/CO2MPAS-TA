#!/usr/bin/env bash
echo 'DERECATION: script `requirements/install_requirements.sh` renamed as `conda_requirements.sh`!
  Please update the name in your scripts.' > /dev/stderr

"$(realpath "$(dirname "$0")")/conda_requirements.sh"