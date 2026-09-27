#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo "Usage: bash scripts/train.sh configs/drugs_mixed.yaml" >&2
  exit 2
fi
exec "${PYTHON:-python}" -m mollift train --config "$1"
