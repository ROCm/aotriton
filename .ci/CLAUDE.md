# Project Instructions for `.ci/`

## Never delete local git caches/mirrors

Scripts in this directory maintain local git caches (e.g. `common-git-cache.sh`'s
`sync_mirror`, backing the Triton/AOTriton mirror volumes used by
`build_triton_wheels.sh`, `releasesuite-git-head.sh`, etc.). When a cache
appears missing, empty, or not-yet-valid, do **not** `rm -rf`/wipe it to
"reclone fresh" — this defeats the entire purpose of caching (avoiding
network round-trips) and is unsafe under any kind of concurrent access.

Instead, use an idempotent, non-destructive repair: `git init --bare <dir>`
(a no-op on an already-valid repo, a plain init on empty, a non-destructive
scaffold-fill-in otherwise — it never touches existing objects/refs, and
tolerates unrelated stray files in the directory) followed by `git fetch`,
which heals anything missing or partial via git's content-addressed object
store. There is no inspection step and no case where wiping first helps —
see `common-git-cache.sh`'s `sync_mirror` for the reference implementation.
