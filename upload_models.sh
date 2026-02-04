save_config.py#!/bin/bash
set -euo pipefail
repo_id=$1
start=1
end=20
step=1

CKPT_ROOT="/scratch/xiulyang/pos_bench/models/$repo_id"

hf repo create "$repo_id"

echo "📦 Uploading top-level files → revision: main"
hf upload "$repo_id" $CKPT_ROOT/ \
  --repo-type model \
  --revision main \
  --exclude "checkpoint-*"

mapfile -t CKPTS < <(find "$CKPT_ROOT" -maxdepth 1 -mindepth 1 -type d -name 'checkpoint-*' -printf '%f\n' | sort -V)

if ((${#CKPTS[@]} == 0)); then
  echo "ℹ️  No checkpoints found under: $CKPT_ROOT"
  exit 0
fi

for base in "${CKPTS[@]}"; do
  src="$CKPT_ROOT/$base"
  revision="$base"   # e.g., checkpoint-500, checkpoint-750, ...
  echo "📦 Uploading $src → revision: $revision"
  hf upload "$repo_id" "$src" \
    --repo-type model \
    --revision "$revision" \
    --commit-message "Add $base as revision $revision"
done


echo "✅all checkpoints have been uploaded to revision。"
