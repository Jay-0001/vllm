#!/usr/bin/env bash
set -euo pipefail

repo=/u/jsaravananindira/vllm
run_id=shape-dist-20261108T165305Z-job23666234
out="$repo/jay_outputs"

cd "$repo/specialization_experiments"

# Results, four run logs, timing files, cases.tsv, and environment details.
tar -czf "$out/shape_distribn_records.tgz" \
  "results/$run_id" \
  "logs/$run_id" \
  "metadata/$run_id"

# Locate the final AOT model for each arm within this job's cache.
models=()
for arm in dynamic spec64; do
  cache_arm="cache/$run_id/$arm"
  test -d "$cache_arm"

  mapfile -d '' -t matches < <(
    find "$cache_arm" -type f -path '*/rank_0_0/model' -print0
  )
  if (( ${#matches[@]} != 1 )); then
    printf 'Expected one AOT model for %s; found %d\n' \
      "$arm" "${#matches[@]}" >&2
    exit 1
  fi
  models+=("${matches[0]}")
done

printf 'Archiving AOT models:\n%s\n' "${models[@]}"
tar -czf "$out/shape_distribn_aot_models.tgz" "${models[@]}"

# Check the contents and transfer sizes before copying.
tar -tzf "$out/shape_distribn_records.tgz"
tar -tzf "$out/shape_distribn_aot_models.tgz"
ls -lh "$out/shape_distribn_records.tgz" "$out/shape_distribn_aot_models.tgz"