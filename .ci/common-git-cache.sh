#!/bin/bash
#
# Reusable git caching primitive backed by a docker volume.
#
#   sync_mirror <mirror_volume> <origin> [base_image]
#       Maintain a bare mirror of <origin> inside <mirror_volume> (at
#       /mirror), via `git init --bare` + `fetch` (idempotent, heals a
#       missing/partial mirror in place). Safe to call every run; never
#       deletes anything.
#
# Workflow (used for both aotriton and triton): GitHub -> local mirror volume
# -> the build/wheel container clones the exact commit from the LOCAL mirror
# (offline, fast). Consumers can fetch any reachable commit SHA from the
# mirror, so a commit that is not a branch/tag tip still resolves without a
# network round-trip or a destructive re-clone.

# Maintain a bare mirror of a git repo in a docker volume.
function sync_mirror() {
  local mirror_volume="$1"
  local origin="$2"
  local base_docker_image="${3:-aotriton:base}"

  # A local file:// origin must be visible inside the container: bind-mount the
  # path read-only at a fixed location and rewrite the URL the container uses.
  local origin_mount=()
  local origin_in_container="${origin}"
  if [[ "${origin}" == file://* ]]; then
    local origin_path
    origin_path=$(realpath "${origin#file://}")
    if [[ ! -d "${origin_path}" ]]; then
      echo "Error: file:// origin path does not exist: ${origin_path}" >&2
      return 1
    fi
    origin_mount=(-v "${origin_path}:/mirror-origin:ro")
    origin_in_container="file:///mirror-origin"
  fi

  docker volume create --name "${mirror_volume}" >/dev/null
  docker run --network=host -i --rm \
    -v "${mirror_volume}:/mirror" \
    "${origin_mount[@]}" \
    "${base_docker_image}" \
    bash -s "${origin_in_container}" << 'EOF'
set -ex
origin="$1"
git config --global --add safe.directory '*'

# One unconditional repair path, no branching on whether /mirror already
# looks valid: `git init --bare` is idempotent (a no-op scaffold-check on an
# already-valid repo, a plain init on empty, a non-destructive fill-in of
# missing structure otherwise -- it never touches existing objects/refs) and
# `fetch` heals anything missing/partial via git's content-addressed store.
# Never delete: there is no case where wiping the volume first helps.
git init --bare /mirror
git -C /mirror config remote.origin.fetch '+refs/*:refs/*'
git -C /mirror remote set-url origin "${origin}" \
  || git -C /mirror remote add origin "${origin}"
git -C /mirror fetch --prune origin

# Let consumers fetch any reachable commit SHA (not just branch/tag tips).
git -C /mirror config uploadpack.allowReachableSHA1InWant true
EOF
}
