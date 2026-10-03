#!/bin/zsh
# Write web/models/tiny/manifest.json: { "<file>": "<first 8 hex of md5>" } for every .bin/.json in web/models/tiny.
# The engine appends ?v=<hash> to each file URL, so a new model reaches every device automatically (Cache Storage keys change)
# and stale copies are evicted. Run after install_weights.sh; the deploy scripts run it again (idempotent).
set -e
HERE=$(cd "$(dirname "$0")" && pwd); M=$(cd "$HERE/../models/tiny" && pwd); cd "$M"
h() { if command -v md5 >/dev/null; then md5 -q "$1"; else md5sum "$1" | cut -d' ' -f1; fi; }
{ echo "{"; first=1; for f in $(ls *.bin *.json | grep -v '^manifest.json$' | sort); do [ $first = 1 ] || echo ","; first=0; printf '  "%s": "%s"' "$f" "$(h "$f" | cut -c1-8)"; done; echo; echo "}"; } > manifest.json
echo "manifest: $(grep -c '"' manifest.json) files → $M/manifest.json"
