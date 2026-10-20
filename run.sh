#!/usr/bin/env bash
# End-to-end pipeline: data -> train -> calibrate -> benchmark -> judge -> tests -> readme -> publish.
# v2 (5c-5f): fluent-vs-fluent training pairs from Kimi-k3, then the two-question v2 gate.
# v3 (5g-5i): v2's questions plus single-response quality and natural-Swedish questions;
# then the CPU benchmark (5j), the machine-translation test (5k), the translationese benchmark
# (5l), the DaLAJ/SweFAQ/Europarl benchmarks (5m) and the context-length check (5n).
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

uv sync --extra dev --extra pilot
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
    step "5g/8 v3 train"     v3 train
    step "5h/8 v3 calibrate" v3 calibrate
    step "5i/8 v3 evaluate"  v3 evaluate
    step "5j/8 cpu"          cpu --checkpoint checkpoints/v3
    step "5k/8 mt test"      mtqa translate
    step "5k/8 mt test"      mtqa evaluate
    step "5l/8 translationese benchmark" liu_eval gate
    step "5l/8 translationese benchmark" liu_eval judge   # skips without BASE_URL/API_KEY
    step "5m/8 external benchmarks" external_eval prepare    # DaLAJ, SweFAQ, Europarl + OPUS-MT
    step "5m/8 external benchmarks" external_eval translate  # Kimi-k3 MT; skips without BASE_URL/API_KEY
    step "5m/8 external benchmarks" external_eval gate
    step "5m/8 external benchmarks" external_eval judge      # skips without BASE_URL/API_KEY
    step "5n/8 context length"      external_eval length
    step "5o/8 v4 ablation"         v4 prepare   # Kimi-k3 MT of Europarl; skips Europarl without BASE_URL/API_KEY
    for arm in control europarl native both; do
        step "5o/8 v4 ablation: $arm" v4 run --arm "$arm"
    done
    step "5o/8 v4 ablation: seed"   v4 run --arm control --seed 20261015
    step "5o/8 v4 ablation: grammar"  v4 run --arm grammar
    step "5o/8 v4 ablation: adequacy" v4 run --arm adequacy   # v4: builds the adequacy negatives
    step "5o/8 minimal pairs"       external_eval prepare-minimal-pairs   # BLiMPPI-sv, MultiBLiMP-sv
    step "5o/8 v4.1"                v4 run --arm v41          # v4.1: registers + hard wrong sources
    step "5o/8 v4 rescore"          v4 rescore --tags v3 control adequacy v41
    step "5o/8 n-gram baseline"     ngram build
    step "5o/8 v4 baselines"        v4 baselines
    step "5o/8 v4 ablation"         v4 compare
    step "5o/8 v4 sanity tests"     v4 sanity
    step "5p/8 EuroEval pilot"      pilot generate --batch-size 8 --delete-after   # needs --extra pilot for 4-bit
    step "5p/8 EuroEval pilot"      pilot score
    step "5p/8 EuroEval pilot"      pilot analyze
else
    echo "note: no fluent training pairs; skipping v2 and v3" >&2
fi
step "6/8 tests"      testing
step "7/8 readme"     readme
if [[ -f checkpoints/v4-adequacy/model.safetensors ]]; then
    step "8/8 publish"    publish --checkpoint v4 --public
    step "8/8 publish"    publish_data --upload
elif [[ -f checkpoints/v3/model.safetensors ]]; then
    step "8/8 publish"    publish --checkpoint v3
elif [[ -f checkpoints/v2/model.safetensors ]]; then
    step "8/8 publish"    publish --checkpoint v2
else
    step "8/8 publish"    publish
fi
