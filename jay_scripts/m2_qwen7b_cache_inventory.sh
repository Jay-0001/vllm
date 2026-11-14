#!/usr/bin/env bash
# Collect text-only artifacts from the existing Qwen-7B AOT cache on Delta.

set -Eeuo pipefail

repo=/u/jsaravananindira/vllm
source_root="$repo/specialization_experiments/cache/qwen7b-512x4-23650318"
output_root="$repo/jay_outputs/m2_qwen7b_last_chance/inventory"

cd "$repo"
mkdir -p "$output_root"
{
  date -u +%FT%TZ
  git rev-parse HEAD
  printf 'source_run=qwen7b-512x4-23650318\n'
  printf 'source_root=%s\n' "$source_root"
} > "$output_root/environment.txt"
old_environment="$repo/specialization_experiments/metadata/qwen7b-512x4-23650318/environment.txt"
if [[ -f "$old_environment" ]]; then
  cp "$old_environment" "$output_root/source_environment.txt"
fi

for arm in dynamic spec2048; do
  cache_dir="$source_root/$arm/torch_compile_cache"
  if [[ ! -d "$cache_dir" ]]; then
    printf 'Missing existing cache: %s\n' "$cache_dir" >&2
    exit 1
  fi
  (
    cd "$cache_dir"
    find . -type f \( -name '*.py' -o -name '*.json' -o -name '*.txt' \) \
      -print | LC_ALL=C sort > "$output_root/$arm-manifest.txt"
    find . -type f \( -name '*.py' -o -name '*.json' -o -name '*.txt' \) \
      -print0 | tar --null -T - -czf "$output_root/$arm-sources.tgz"
  )
  sha256sum "$output_root/$arm-sources.tgz" \
    > "$output_root/$arm-sources.sha256"
  printf '%s: %s text files\n' "$arm" \
    "$(wc -l < "$output_root/$arm-manifest.txt")"
done

printf 'Saved inventory in %s\n' "$output_root"
