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

## Choosing the backend for level-3 unit tests

When asked to run level-3 unit tests (`.ci/run-test.sh <pass#> 3 <backend>`)
without an explicit backend, do not run every backend. Pick the backend from
what is being tested, and state that justification before starting:

| What is being tested | Backend to run |
|---|---|
| Tuning-database changes | `split` |
| A release (candidate) | `v3` |
| An AITER bump | `aiter` |
| Triton kernel changes (forward or split backward) | `split` |
| Triton fused backward kernel changes | `fused` |
| FlyDSL kernel changes | `flyc` |

If the case is unclear or spans several rows, ask the user before running.
Level 3 takes hours per backend, so a backend that cannot be affected is
wasted GPU time.
