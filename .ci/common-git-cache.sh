#!/bin/bash
#
# Reusable git caching primitive backed by a docker volume.
#
#   sync_mirror <mirror_volume> <origin> [base_image] [pat_environ] <rev>...
#       Make each <rev> available in the bare mirror inside <mirror_volume>
#       (at /mirror), fetching from <origin> only what is not already there. Prints one resolved 40-hex commit SHA per <rev>, in
#       order, on stdout (and nothing else; progress goes to stderr). If any
#       <rev> cannot be supplied, reports it on stderr and returns non-zero;
#       the caller only has to stop.
#
#       <rev> is a full SHA, a tag or branch name, or a full ref (refs/...).
#       A name is a moving ref, so it is always resolved against <origin> (an
#       `ls-remote`: a ref listing, no objects), never against a local copy
#       that may be stale -- a transient network failure is an error, not a
#       silent fallback to yesterday's tip.
#
#       pat_environ, if non-empty, names an environment variable (in THIS
#       shell) holding a GitHub PAT used to authenticate against a private
#       origin. Only the variable's NAME crosses into traced/logged commands;
#       its value is forwarded via `docker run -e <name>` (no `=value`, so
#       docker pulls it from this shell's environment) and only ever touched
#       inside the container under `set +x`.
#
# Workflow (used for aotriton, triton, llvm and flydsl alike, one volume per
# project): GitHub -> the project's local mirror volume -> the build container
# shallow-fetches the exact commit from the LOCAL mirror (file:///mirror,
# offline, fast).
#
# The one goal of this cache is the least possible network traffic for git
# objects. Everything below follows from that:
#
#   * ONE volume, ONE object store, for every origin of a project. A fork
#     shares nearly all of its history with its upstream; with the objects in
#     one store, a fetch from the fork downloads only what the fork adds.
#   * No object is ever deleted: no gc, no prune. Every commit ever fetched
#     is pinned as refs/kept/<sha>, and a pin is never removed. Refs are what git offers the
#     server as "have"s during negotiation, so pinning is what lets the NEXT
#     fetch, from ANY origin, skip everything already here. (Objects kept but
#     unreferenced would still be on disk, yet invisible to negotiation, and
#     be downloaded again.)
#   * Only the requested commits are fetched, never an origin's whole ref
#     namespace: no branch/PR churn is downloaded that no build asked for.
#     A commit that is already here (and connected) costs no network at all.
#   * A requested branch or tag name is also kept as a ref of that same name
#     (refs/heads/<b>, refs/tags/<t>), pointing where the LATEST fetch found
#     it -- from whichever origin that was: a fork's `main` overwrites
#     upstream's. Losing the old value costs nothing, the commit it named is
#     still pinned under refs/kept/.
#   * The fetch is by URL, with no remote configured, so there is no ref
#     namespace to prune and no fetch refspec pulling in unrequested refs.
function sync_mirror() {
  local mirror_volume="$1"
  local origin="$2"
  local base_docker_image="${3:-aotriton:base}"
  local pat_environ="${4:-}"
  shift 4
  if [[ "$#" -eq 0 ]]; then
    echo "Error: sync_mirror: no revision requested from ${origin}." >&2
    return 1
  fi

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
  #
  # `${!pat_environ}` only checks that a variable by this name is SET in this
  # shell, not that it's exported -- `docker run -e NAME` (no `=value`) pulls
  # from the calling process's environ, which a plain (non-exported) shell
  # variable never reaches. Export it ourselves so a caller that forgot to
  # export still works, instead of silently forwarding an empty value.
  local pat_env_arg=()
  if [[ -n "${pat_environ}" ]]; then
    if [[ -z "${!pat_environ:-}" ]]; then
      echo "Error: pat_environ '${pat_environ}' is set but that environment variable is empty/unset." >&2
      return 1
    fi
    export "${pat_environ}"
    pat_env_arg=(-e "${pat_environ}")
  fi

  docker volume create --name "${mirror_volume}" >/dev/null
  if ! docker run --network=host -i --rm \
    -v "${mirror_volume}:/mirror" \
    "${origin_mount[@]}" \
    "${pat_env_arg[@]}" \
    "${base_docker_image}" \
    bash -s "${origin_in_container}" "${pat_environ}" "$@" << 'EOF'
set -ex
origin="$1"
pat_environ="$2"
shift 2
export GIT_TERMINAL_PROMPT=0
git config --global --add safe.directory '*'

# Every build of this project shares this one repo. Serialize the writers
# (fetches, pins, config) so two builds syncing at once cannot trip over each
# other's ref or config locks; readers (the build containers' fetch from file:///mirror) need
# no lock, git publishes packs and refs atomically.
exec 9>/mirror/aotriton-mirror.lock
flock 9

# One unconditional repair path, no branching on whether /mirror already
# looks valid: `git init --bare` is idempotent (a no-op scaffold-check on an
# already-valid repo, a plain init on empty, a non-destructive fill-in of
# missing structure otherwise -- it never touches existing objects/refs).
# Never delete: there is no case where wiping the volume first helps.
git init --bare /mirror >&2
# Never delete an object, not even an unreachable one: no auto-gc on fetch
# (gc.auto), no auto-maintenance (maintenance.auto), and should anybody run
# `git gc` by hand anyway, it must not prune (gc.pruneExpire).
git -C /mirror config gc.auto 0
git -C /mirror config maintenance.auto false
git -C /mirror config gc.pruneExpire never
# Let consumers fetch any commit by SHA, whatever does or does not point at it.
git -C /mirror config uploadpack.allowAnySHA1InWant true

# Credentials, if any, via `-c` (a per-invocation override) rather than any
# config file, so the token never lands in the reused mirror volume. The empty
# `credential.helper=` first clears any helper that some older script left in
# /mirror/config: git runs configured helpers BEFORE a `-c` one, and a stale
# helper answering first is how a valid PAT ends up rejected.
auth=(-c credential.helper=)
if [[ -n "${pat_environ}" ]]; then
  # Never let `set -x` echo the token value: resolve and consume it with
  # tracing off. The credential.helper VALUE becomes part of the `git`
  # process's own argv (visible via `ps`/`docker top` while it runs) -- so it
  # must never contain the raw token. Export the token under a fixed name and
  # have the helper reference that name: only the (harmless) variable name
  # crosses into argv, the secret itself stays in the environment.
  set +x
  export AOTRITON_GIT_PAT="${!pat_environ}"
  set -x
  auth+=(-c 'credential.helper=!f() { echo username=x-access-token; echo "password=$AOTRITON_GIT_PAT"; }; f')
fi

# "Here" means the commit AND everything it reaches are in the store: walk
# from it, stopping at anything already reachable from a ref. A pinned commit
# is instant; a commit whose history is missing a piece is not "here".
have_commit() {
  git -C /mirror rev-list --quiet --objects "$1" --not --all 2>/dev/null
}

resolved=()
for rev in "$@"; do
  # ref/oid: the named ref and what it points at (an annotated tag's own
  # object, not its commit); empty for a bare SHA, which names no ref.
  ref=""
  oid=""
  if [[ "${rev}" =~ ^[0-9a-fA-F]{40}$ ]]; then
    sha="${rev,,}"
  else
    # A name: ask the origin what it points at NOW. Exact refs only, in git's
    # own rev-parse precedence (full ref, then tag, then branch). A tag
    # resolves through its peeled `^{}` entry, i.e. the commit, if annotated.
    if [[ "${rev}" == refs/* ]]; then
      candidates=("${rev}")
    else
      candidates=("refs/tags/${rev}" "refs/heads/${rev}")
    fi
    patterns=()
    for c in "${candidates[@]}"; do patterns+=("${c}" "${c}^{}"); done
    listing="$(git "${auth[@]}" ls-remote "${origin}" "${patterns[@]}")"
    for c in "${candidates[@]}"; do
      read -r oid sha <<< "$(awk -v r="${c}" '$2 == r "^{}" { p = $1 } $2 == r { d = $1 } END { if (d != "") print d, (p != "" ? p : d) }' <<< "${listing}")"
      if [[ -n "${oid}" ]]; then
        ref="${c}"
        break
      fi
    done
    if [[ -z "${ref}" ]]; then
      echo "Error: '${rev}' is not a branch, tag or ref at ${origin}." >&2
      exit 1
    fi
    echo "Resolved ${rev} -> ${sha} at ${origin}" >&2
  fi

  # Fetch only what is missing, by exact object id -- never by name, so a ref
  # moving at the origin since the ls-remote above cannot slip in a different
  # commit than the one resolved. Every ref in the store is offered as a
  # "have", so only objects not already here -- from this origin or any
  # other -- come over the wire. --no-tags: no auto-followed tags, only what
  # was asked for. --no-write-fetch-head: FETCH_HEAD is shared scratch nobody
  # reads.
  refspecs=()
  have_commit "${sha}" || refspecs+=("+${sha}:refs/kept/${sha}")
  if [[ -n "${ref}" ]] && ! git -C /mirror cat-file -e "${oid}" 2>/dev/null; then
    refspecs+=("+${oid}:${ref}")
  fi
  if [[ "${#refspecs[@]}" -gt 0 ]]; then
    git "${auth[@]}" -C /mirror fetch --no-tags --no-write-fetch-head \
      "${origin}" "${refspecs[@]}"
  fi
  # Pin it even when it was already here (e.g. reachable only through a ref
  # that is not ours): the pin is what keeps it offered as a "have".
  git -C /mirror update-ref "refs/kept/${sha}" "${sha}"
  # The name follows the latest fetch; its previous commit stays pinned.
  if [[ -n "${ref}" ]]; then
    git -C /mirror update-ref "${ref}" "${oid}"
  fi
  if ! have_commit "${sha}"; then
    echo "Error: ${sha} is still not complete in the mirror after fetching from ${origin}." >&2
    exit 1
  fi
  resolved+=("$(git -C /mirror rev-parse --verify "${sha}^{commit}")")
done
unset AOTRITON_GIT_PAT
printf '%s\n' "${resolved[@]}"
EOF
  then
    echo "Error: could not fetch ${*} from ${origin} into ${mirror_volume}." >&2
    return 1
  fi
}
