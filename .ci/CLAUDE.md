# Project Instructions for `.ci/`

## The git cache exists to minimize network traffic

`common-git-cache.sh`'s `sync_mirror` maintains one bare git mirror per project,
each in its own docker volume (`triton-mirror`, `llvm-mirror`, `flydsl-mirror`,
`aotriton-mirror`), shared by every origin of that project. Its sole purpose is
the absolute minimum network traffic for downloading git objects. No
"optimization" or "correctness improvement" may work against that goal. In
particular:

1. **One volume, one object store, for all origins of a project.** Never
   introduce a per-origin mirror volume: a fork cloned after its upstream
   would download all the shared history again.
2. **Never remove anything.** No `git gc`, no pruning, no ref deletion, no
   `fetch --prune`. If repo A, then B, then A again is fetched, nothing of A
   may be downloaded twice.
3. **Keep every object ever fetched, and supply the SHA the current build
   asks for.** Every fetched commit stays pinned under `refs/kept/<sha>`:
   local refs are what git offers as "have"s during negotiation, so an
   unreferenced object would be downloaded again even though it is on disk.
   A requested branch/tag name is kept as a ref of the same name and may be
   overwritten by the latest fetch (from any origin) -- that is fine, the
   commit it used to name is still pinned.
   Fetch only the commits a build requests, never an origin's whole ref
   namespace, and fetch nothing at all when the commit is already present.

## Never delete local git caches/mirrors

When a cache appears missing, empty, or not-yet-valid, do **not** `rm -rf`/wipe
it to "reclone fresh" — this defeats the entire purpose of caching (avoiding
network round-trips) and is unsafe under any kind of concurrent access.

Instead, use an idempotent, non-destructive repair: `git init --bare <dir>`
(a no-op on an already-valid repo, a plain init on empty, a non-destructive
scaffold-fill-in otherwise — it never touches existing objects/refs, and
tolerates unrelated stray files in the directory) followed by `git fetch`,
which heals anything missing or partial via git's content-addressed object
store. There is no inspection step and no case where wiping first helps —
see `common-git-cache.sh`'s `sync_mirror` for the reference implementation.
