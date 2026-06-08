#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${1:-.env}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_PATH="$REPO_ROOT/$ENV_FILE"

if [[ ! -f "$ENV_PATH" ]]; then
  echo "Environment file not found: $ENV_PATH" >&2
  exit 1
fi

while IFS= read -r line || [[ -n "$line" ]]; do
  line="${line%$'\r'}"
  line="${line#"${line%%[![:space:]]*}"}"
  line="${line%"${line##*[![:space:]]}"}"

  if [[ -z "$line" || "${line:0:1}" == "#" ]]; then
    continue
  fi

  if [[ "$line" == export[[:space:]]* ]]; then
    line="${line#export }"
  fi

  if [[ "$line" != *=* ]]; then
    continue
  fi

  name="${line%%=*}"
  value="${line#*=}"
  name="${name%"${name##*[![:space:]]}"}"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"

  if [[ ! "$name" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
    continue
  fi

  if [[ "${#value}" -ge 2 ]]; then
    first="${value:0:1}"
    last="${value: -1}"
    if [[ "$first" == "$last" && ( "$first" == "'" || "$first" == '"' ) ]]; then
      value="${value:1:${#value}-2}"
    fi
  fi

  export "$name=$value"
done < "$ENV_PATH"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is missing in $ENV_PATH. Add HF_TOKEN=hf_your_token before starting MAX." >&2
  exit 1
fi

export MAX_SERVE_API_TYPES="${MAX_SERVE_API_TYPES:-[\"responses\"]}"

BASE_URL="${MAX_IMAGE_BASE_URL:-${MAX_BASE_URL:-http://localhost:8010}}"
MODEL="${MAX_IMAGE_MODEL:-${NOTES_IMAGE_MODEL:-black-forest-labs/FLUX.2-dev}}"
DEVICES="${MAX_IMAGE_DEVICES:-gpu:0}"
PORT="$(python3 - "$BASE_URL" <<'PY'
from urllib.parse import urlparse
import sys
u = urlparse(sys.argv[1])
print(u.port or 8010)
PY
)"

if ! command -v max >/dev/null 2>&1; then
  cat >&2 <<EOF
The 'max' CLI is not installed or not available on PATH.

Install Modular MAX in this WSL/Linux environment, then rerun:
  ./scripts/start_max_images.sh

Recommended by Modular docs:
  pixi init max-images -c https://conda.modular.com/max-nightly/ -c conda-forge
  cd max-images
  pixi add modular
  pixi shell

Then return to this repo:
  cd /mnt/d/Edva/eddva_ai_service
  ./scripts/start_max_images.sh
EOF
  exit 127
fi

echo "Starting MAX image server"
echo "  model: $MODEL"
echo "  port:  $PORT"
echo "  url:   http://localhost:$PORT/v1/responses"

max serve --model "$MODEL" --devices "$DEVICES" --port "$PORT"
