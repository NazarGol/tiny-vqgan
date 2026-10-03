#!/bin/zsh
# Copy trained weights from research/out/* into web/models/tiny (served as models/tiny/ by the app and tests), regenerate the
# PyTorch references and re-run the browser equivalence tests. Usage: web/research/install_weights.sh [decoder|scorer|text|all]
set -e
HERE=$(cd "$(dirname "$0")" && pwd); WEB=$(cd "$HERE/.." && pwd); ROOT=$(cd "$WEB/.." && pwd); PY="$ROOT/.venv-export/bin/python"
what=${1:-all}; T="$HERE/tiny/test"; M="$WEB/models/tiny"
[ -L "$M" ] && rm "$M"; mkdir -p "$M"
if [[ $what == decoder || $what == all ]] && [ -f "$HERE/out/decoder/tiny/ckpt.pt" ]; then
  for v in A B; do "$PY" "$HERE/tiny/dump_ref.py" "$HERE/out/decoder/tiny/ckpt.pt" $v "$T" | tail -1; cp "$T/tiny_decoder_$v".{bin,json} "$M/"; done
  (cd "$WEB" && node research/test_tinydec.mjs --variant A --browser all 2>/dev/null | cut -c1-200)
fi
if [[ $what == scorer || $what == all ]] && [ -f "$HERE/out/scorer/scorer/ckpt.pt" ]; then
  for v in S M; do "$PY" "$HERE/tiny/dump_ref_scorer.py" "$HERE/out/scorer/scorer/ckpt.pt" $v "$T" | tail -1; cp "$T/tiny_scorer_$v".{bin,json} "$M/"; done
  (cd "$WEB" && node research/test_tinydec.mjs --page test_tinyscorer.html --variant S --browser all 2>/dev/null | cut -c1-200)
fi
if [[ $what == text || $what == all ]] && [ -f "$HERE/out/text/text/ckpt.pt" ]; then
  for v in S M; do "$PY" "$HERE/tiny/dump_ref_text.py" "$HERE/out/text/text/ckpt.pt" $v "$T" | tail -1; cp "$T/tiny_text_$v".{bin,json} "$M/"; done
  (cd "$WEB" && node research/test_tinydec.mjs --page test_tinytext.html --variant S --browser chromium 2>/dev/null | cut -c1-200)
fi
"$HERE/tiny_manifest.sh"
ls -la "$M"
