#!/bin/bash
#
# Reusable git caching primitive backed by a docker volume.
#
#   sync_mirror <mirror_volume> <origin> [base_image] [pat_environ]
#       Maintain a bare mirror of <origin> inside <mirror_volume> (at
#       /mirror), via `git init --bare` + `fetch` (idempotent, heals a
#       missing/partial mirror in place). Safe to call every run; never
#       deletes anything.
#
#       pat_environ, if non-empty, names an environment variable (in THIS
#       shell) holding a GitHub PAT used to authenticate the fetch against a
#       private origin. Only the variable's NAME crosses into traced/logged
#       commands; its value is forwarded via `docker run -e <name>` (no
#       `=value`, so docker pulls it from this shell's environment) and only
#       ever touched inside the container under `set +x`.
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
  local pat_environ="${4:-}"

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

  # Forward the PAT by NAME only (no `=value`): docker resolves the value
  # from this shell's environment, so the token never appears as a literal
  # in the docker run argv (safe even under a stray `set -x`).
  local pat_env_arg=()
  if [[ -n "${pat_environ}" ]]; then
    if [[ -z "${!pat_environ:-}" ]]; then
      echo "Error: pat_environ '${pat_environ}' is set but that environment variable is empty/unset." >&2
      return 1
    fi
    pat_env_arg=(-e "${pat_environ}")
  fi

  docker volume create --name "${mirror_volume}" >/dev/null
  docker run --network=host -i --rm \
    -v "${mirror_volume}:/mirror" \
    "${origin_mount[@]}" \
    "${pat_env_arg[@]}" \
    "${base_docker_image}" \
    bash -s "${origin_in_container}" "${pat_environ}" << 'EOF'
set -ex
origin="$1"
pat_environ="$2"
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

if [[ -n "${pat_environ}" ]]; then
  # Never let `set -x` echo the token value: resolve and consume it with
  # tracing off, and scope the credential helper to this ephemeral
  # container's local git config (never persisted into the /mirror volume).
  set +x
  pat_value="${!pat_environ}"
  git -C /mirror config --local credential.helper \
    "!f() { echo username=x-access-token; echo password=${pat_value}; }; f"
  set -x
fi
git -C /mirror fetch --prune origin

# Let consumers fetch any reachable commit SHA (not just branch/tag tips).
git -C /mirror config uploadpack.allowReachableSHA1InWant true
EOF
}
