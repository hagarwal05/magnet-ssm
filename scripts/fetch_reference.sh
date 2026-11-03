#!/usr/bin/env bash
# Sparse-clone the upstream MagNet Challenge 2 repo into third_party/, pinned to
# the commit the reproduction targets were checked against. Only the code,
# demo data and weights are checked out (no slide decks).
set -euo pipefail
cd "$(dirname "$0")/.."

REPO_URL="https://github.com/minjiechen/magnetchallenge-2.git"
REF="${REF:-ab4913ca63ebda1f8eab429c64d3e93f96204713}"
DEST="third_party/magnetchallenge-2"

if [ ! -d "$DEST/.git" ]; then
  git clone --filter=blob:none --sparse --no-checkout "$REPO_URL" "$DEST"
fi

git -C "$DEST" sparse-checkout set --no-cone \
  '/README.md' \
  '/Test/*' '!/Test/*.pdf' \
  '/Pretest/Readme.md' '/Pretest/PreEvaluationExample.ipynb' '/Pretest/*.m' '/Pretest/Pretest_Models/*' \
  '/tutorials/tools/*' \
  '/tutorials/tutorial-1/*' '!/tutorials/tutorial-1/*.pdf' \
  '/tutorials/tutorial-2/data/*' \
  '/tutorials/tutorial-3/*' '!/tutorials/tutorial-3/*.pdf'
git -C "$DEST" checkout --quiet "$REF"
git -C "$DEST" log -1 --format='reference repo at %H (%cd)'
