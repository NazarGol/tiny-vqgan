#!/bin/zsh
# Copy the Kaggle evaluation images/metrics into web/research/results/ (committed, used by the report).
set -e
HERE=$(cd "$(dirname "$0")" && pwd); R="$HERE/results"; mkdir -p "$R"
[ -d "$HERE/out/decoder/tiny" ] && { cp "$HERE/out/decoder/tiny/eval_sheet.png" "$R/decoder_eval_sheet.png"; cp "$HERE/out/decoder/tiny/eval.json" "$R/decoder_eval.json"; for f in "$HERE"/out/decoder/tiny/pair_*.png; do cp "$f" "$R/decoder_$(basename $f)"; done; cp "$HERE/out/decoder/tiny/train.log" "$R/decoder_train.log" 2>/dev/null || true; }
[ -d "$HERE/out/scorer/scorer" ] && { cp "$HERE/out/scorer/scorer/eval.json" "$R/scorer_eval.json"; cp "$HERE/out/scorer/scorer/train.log" "$R/scorer_train.log" 2>/dev/null || true; }
[ -d "$HERE/out/text/text" ] && { cp "$HERE/out/text/text/eval.json" "$R/text_eval.json"; cp "$HERE/out/text/text/train.log" "$R/text_train.log" 2>/dev/null || true; }
[ -d "$HERE/out/onepass/onepass" ] && { cp "$HERE/out/onepass/onepass/eval.json" "$R/onepass_eval.json"; cp "$HERE/out/onepass/onepass/train.log" "$R/onepass_train.log" 2>/dev/null || true; }
ls -la "$R"
