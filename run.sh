#!/usr/bin/env bash
# End-to-end pipeline: data -> train -> calibrate -> benchmark -> judge -> tests -> readme -> publish.
# v2 (5c-5f): fluent-vs-fluent training pairs from Kimi-k3, then the two-question v2 gate.
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
step "5c/8 fluent"    fluent  # v2 training pairs (cached; skips without BASE_URL/API_KEY)
if [[ -f cache/fluent/train.jsonl ]]; then
    step "5d/8 v2 train"     v2 train
    step "5e/8 v2 calibrate" v2 calibrate
    step "5f/8 v2 evaluate"  v2 evaluate
    step "5f/8 v1 evaluate"  v2 evaluate --checkpoint checkpoints/best_model --section v1_eval
else
    echo "note: no fluent training pairs; skipping v2" >&2
fi
step "6/8 tests"      testing
step "7/8 readme"     readme
if [[ -f checkpoints/v2/model.safetensors ]]; then
    step "8/8 publish"    publish --checkpoint v2
else
    step "8/8 publish"    publish
fi
