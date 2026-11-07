#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Regenerate asset_sha256.txt from the GitHub release API.
#
#   perfmon/scripts/lib/refresh_asset_sha256.sh > perfmon/scripts/lib/asset_sha256.txt
#
# Writes to stdout so the result can be diffed before it replaces anything --
# a digest changing for an asset that already had one is either a re-uploaded
# release or something worth asking about, and neither should be applied by a
# script without a human seeing it.
#
# curl, not the gh CLI: one GET per tag does not justify that dependency, and
# gh is not installed in the perfmon image. GITHUB_TOKEN is honoured for the
# rate limit only -- these are public releases.

set -euo pipefail

REPO="${PERFMON_RELEASE_REPO:-ROCm/aotriton}"
TAGS=("$@")
if [ "${#TAGS[@]}" -eq 0 ]; then
  # The tags perfmon has release build scripts for (head is built from the
  # checkout and downloads nothing). Kept in sync by hand; a tag with no script
  # cannot be built, so recording its digests would be noise.
  # TODO(perfmon): 0.14.2b is listed, but its digests are not in
  # asset_sha256.txt yet: this was written without network access. Run this
  # script, review the diff, and replace the file.
  TAGS=(0.9.2b 0.10b 0.11b 0.11.2b 0.12.1b 0.13b 0.14.2b)
fi

auth=()
[ -n "${GITHUB_TOKEN:-}" ] && auth=(-H "Authorization: Bearer ${GITHUB_TOKEN}")

cat <<'HDR'
# SHA-256 digests of the AOTriton release assets perfmon downloads.
#
# Format is sha256sum(1)'s, so a line can be checked directly:
#     grep ' <asset>$' asset_sha256.txt | sha256sum -c -
#
# Source: GitHub's release API reports a `digest` field per asset
# (`sha256:<hex>`), which is what the release page shows. Regenerate with
#     perfmon/scripts/lib/refresh_asset_sha256.sh > perfmon/scripts/lib/asset_sha256.txt
# Not fetched at build time on purpose: a digest fetched over the same
# connection as the file it vouches for checks that the download completed, not
# that it is the file this repo was tested against. Committing it makes the
# expected bytes reviewable, and a change to them show up in a diff.
#
# Every asset of every tag is listed, not just the ones the build scripts ask
# for today: the extra lines cost nothing, and a tag whose asset name changes
# should fail on the name rather than by falling through to "digest unknown".
#
# An asset with no entry here is a WARNING, not an error -- a new tag can be
# built before anyone records its digests. Add them here to make it enforced.
HDR

for tag in "${TAGS[@]}"; do
  echo "[refresh] ${tag}" >&2
  curl -sfL "${auth[@]}" \
    "https://api.github.com/repos/${REPO}/releases/tags/${tag}" \
  | python3 -c '
import json, sys
tag = sys.argv[1]
r = json.load(sys.stdin)
rows = [(a["digest"][7:], a["name"]) for a in r["assets"]
        if (a.get("digest") or "").startswith("sha256:")]
print(f"\n# --- {tag} ({len(rows)} assets) ---")
for sha, name in sorted(rows, key=lambda x: x[1]):
    print(f"{sha}  {name}")
' "${tag}"
done
