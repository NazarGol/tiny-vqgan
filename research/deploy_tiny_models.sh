#!/bin/zsh
# Publish ONLY web/models/tiny/* to the gh-pages branch (the live app code is deployed by the product agent, never from here).
# The app's fetchCached falls back from Hugging Face to GitHub Pages, so the light engine's files only need to exist on Pages.
set -e
HERE=$(cd "$(dirname "$0")" && pwd); WEB=$(cd "$HERE/.." && pwd); ROOT=$(cd "$WEB/.." && pwd)
ORIGIN="$(git -C "$ROOT" remote get-url origin)"; OUT="$ROOT/.gh-pages-tiny"
if [ ! -d "$OUT/.git" ]; then rm -rf "$OUT"; git clone -q --depth 1 --branch gh-pages "$ORIGIN" "$OUT"; fi
cd "$OUT" && git fetch -q --depth 1 origin gh-pages && git reset -q --hard origin/gh-pages
mkdir -p models/tiny
for f in "$WEB"/models/tiny/*.bin "$WEB"/models/tiny/*.json; do [ -f "$f" ] && cp "$f" "models/tiny/$(basename "$f")"; done
git add models/tiny
if git diff --cached --quiet; then echo "tiny models unchanged on gh-pages"; exit 0; fi
git -c user.name="deploy" -c user.email="deploy@local" commit -q -m "models/tiny: light engine weights ($(git -C "$ROOT" rev-parse --short HEAD) $(date -u +%FT%TZ))"
for try in 1 2 3; do git -c http.postBuffer=1048576000 push -q origin gh-pages:gh-pages && break; echo "push failed (try $try), retrying"; sleep 5; done
echo "deployed: $(ls models/tiny | tr '\n' ' ')"
