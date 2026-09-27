#!/usr/bin/env bash
# Build the final submission zip in the layout the challenge asks for:
#   <team>_submission.zip
#   ├── output/{matching_results.tsv, candidate_pairs.tsv}
#   ├── code/business_entity_resolution/{src, scripts, configs, utils, tests, notebooks, README.md, ...}
#   └── Documentation_template.md
# Usage: scripts/make_package.sh <team_name> <output_dir_with_both_tsvs>
set -euo pipefail
cd "$(dirname "$0")/.."
TEAM="${1:?team name}"
OUT="${2:?output dir with matching_results.tsv and candidate_pairs.tsv}"
STAGE="$(mktemp -d)/${TEAM}_submission"
mkdir -p "$STAGE/output" "$STAGE/code/business_entity_resolution"
cp "$OUT/matching_results.tsv" "$OUT/candidate_pairs.tsv" "$STAGE/output/"
git ls-files -z | tar --null -T - -cf - | tar -xf - -C "$STAGE/code/business_entity_resolution"
# final models and splits are needed to reproduce without retraining
cp artifacts/model/model_m10_*.txt artifacts/model/meta_m10.json "$STAGE/code/business_entity_resolution/artifacts/model/" 2>/dev/null || true
cp docs/Documentation.md "$STAGE/Documentation_template.md"
( cd "$(dirname "$STAGE")" && rm -f "$OLDPWD/${TEAM}_submission.zip" && zip -qr "$OLDPWD/${TEAM}_submission.zip" "$(basename "$STAGE")" )
echo "wrote ${TEAM}_submission.zip"
unzip -l "${TEAM}_submission.zip" | tail -1
