#!/bin/zsh
# Rebase this branch onto the latest origin/<base> (default web-spikes); room.js conflicts are resolved by taking theirs and re-applying patch_room.py.
set -e
BASE=${1:-web-spikes}; ROOT=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
git fetch -q origin
if ! git rebase "origin/$BASE"; then
  while git status --short | grep -q "^UU\|^AA"; do
    for f in $(git diff --name-only --diff-filter=U); do
      if [ "$f" = "web/app/room.js" ]; then git checkout --theirs -- "$f" 2>/dev/null || git show "origin/$BASE:web/app/room.js" > "$f"; "$ROOT/.venv-export/bin/python" web/research/patch_room.py web/app/room.js; git add "$f"
      else echo "unexpected conflict in $f"; exit 1; fi
    done
    GIT_EDITOR=true git rebase --continue || true
  done
fi
git log --oneline -3 | cat
