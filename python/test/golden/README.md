# Golden codegen snapshot harness

`golden.py` captures the C++ that the ATI generator emits for the **flash**
family and lets you prove that a change is **generation-output-neutral**
byte-for-byte. It is a developer tool, not a committed test fixture.

> **Why is there no checked-in golden?** A frozen snapshot goes stale the moment
> the generator's output legitimately changes, and reviewing a 500 KB blob diff
> is useless. Instead you generate a **local baseline** before your change and
> diff against it after. The baseline lands in `snapshot/`, which is gitignored.

## Workflow

Run from the **repo root**. Generation is deterministic (driven entirely by the
checked-in tuning database, `--noimage_mode`, no HSACO compilation), so identical
inputs yield identical bytes.

```bash
# 1. On a clean tree (before your change), record the baseline:
python python/test/golden/golden.py --update

# 2. Make your generator/IR change.

# 3. Confirm the emitted C++ is unchanged:
python python/test/golden/golden.py --check
```

- `--check` exits **0** and prints `OK: N generated files match...` when the
  output is identical.
- It exits **1** and lists every `MISSING` / `ADDED` / `CHANGED` file when the
  output differs. A non-empty diff means your change is **not** output-neutral —
  either that is a bug, or it is intended and you should re-run `--update` to move
  the baseline.
- `--check` exits **2** if no baseline exists yet (run `--update` first).

## What the snapshot contains

`snapshot/manifest.json` — a `{relative path: sha256}` map over every generated
`.h`/`.cc` file. The human-readable-signature comment block is normalized out
before hashing (it is a non-load-bearing C++ comment whose grouping differs
between the legacy and ATI IRs; see the `_normalize` docstring in `golden.py`).

`snapshot/files/` — verbatim copies of a few primary shim/op files
(`shim.attn_fwd.*`, `iface.op_attn_fwd.*`,
`shim.debug_simulate_encoded_softmax.*`) for eyeball-friendly diffs.

## Options

| flag / env | default | meaning |
|---|---|---|
| `--arch ARCH` | `gfx942_mod0` | target GPU passed to `--target_gpus` |
| `--keep_dir DIR` | *(temp dir)* | generate into `DIR` and keep the full tree on disk for file-level diffing instead of a throwaway temp dir |
| `AOTRITON_GOLDEN_DB` | `/tmp/ati_golden_fused_db` | directory holding the prebuilt fused `tuning_database.sqlite3` (+ optional `op_database.sqlite3`). If absent, the harness falls back to composing the decomposed DB from the schema tarballs. |

## Notes for automation / AI agents

- Always invoke with the repo's Python interpreter and from the repo root; the
  harness resolves paths relative to its own location, but the generator
  subprocess runs with `cwd=<repo root>`.
- The tool shells out to `python -m aotriton.generate` and
  `aotriton.database_compose`; the `aotriton` package must be importable
  (install it, or run inside the build venv).
- Treat a clean `--check` (exit 0) as the pass condition for "no codegen
  regression". Do not commit anything under `snapshot/`.
