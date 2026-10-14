#!/usr/bin/env bash
# End-to-end pipeline: data -> train -> calibrate -> benchmark -> judge -> tests -> readme -> publish.
# Secrets are loaded from .env by uv and read from the environment; nothing here prints them.
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -f .env ]]; then
    echo "note: .env not found; judge and publish steps will skip" >&2
    touch_env=$(mktemp)
    ENV_FILE="$touch_env"
else
    ENV_FILE=.env
fi

step() {
    echo
    echo "==> $1"
    shift
    uv run --env-file "$ENV_FILE" python -m "laya_gate.$1" "${@:2}"
}

uv sync --extra dev
step "1/8 data"       data
step "2/8 train"      train
step "3/8 calibrate"  calibrate
step "4/8 benchmark"  benchmark
step "5/8 judge"      judge
step "5b/8 probe"     probe   # fluent-vs-fluent quality probe (skips without BASE_URL/API_KEY)
step "6/8 tests"      testing
step "7/8 readme"     readme
step "8/8 publish"    publish
