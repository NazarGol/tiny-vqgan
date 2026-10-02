#!/bin/zsh
# Copy the engine files from a tiny-vqgan checkout into this app (vendoring). Usage: web/research/sync_from_tiny_vqgan.sh ../tiny-vqgan
set -e
SRC=${1:?path to a tiny-vqgan checkout}; HERE=$(cd "$(dirname "$0")" && pwd); WEB=$(cd "$HERE/.." && pwd)
rm -rf "$WEB/engine"; cp -R "$SRC/engine" "$WEB/engine"
for f in glnn.js tinydec.js tinyscorer.js clipvision.js; do cp "$SRC/lib/$f" "$WEB/lib/$f"; done
mkdir -p "$WEB/lib/engine"; cp "$SRC/lib/engine/light.js" "$WEB/lib/engine/light.js"
rsync -a --delete --exclude out --exclude data --exclude 'tiny/test/*.bin' "$SRC/research/" "$WEB/research/"
rev=$(git -C "$SRC" rev-parse --short HEAD); sed -i '' "s/^Pinned: tiny-vqgan .*/Pinned: tiny-vqgan \`main\` @ $rev ($(date +%F))./" "$WEB/engine/VENDOR.md"
echo "vendored tiny-vqgan @ $rev"
