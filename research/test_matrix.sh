#!/bin/zsh
# Phase 4 matrix: the engine test page (20 strokes, text per stroke) on five browser profiles. Output: research/data/matrix.jsonl
cd "$(dirname "$0")/.." || exit 1
out=research/data/matrix.jsonl; : > $out
run() { echo "== $*" >&2; node research/measure_engine.mjs --strokes 20 --seconds 3 --text 1 "$@" 2>/dev/null | tail -1 | tee -a $out; }
run --browser chromium --device "Desktop Chrome"
run --browser chromium --device "Desktop Chrome" --nogpu
run --browser webkit --device "Desktop Safari"
run --browser webkit --device "iPhone 11"
run --browser chromium --device "Pixel 5"
echo "matrix done: $out"
