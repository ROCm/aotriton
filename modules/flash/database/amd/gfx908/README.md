# MI100 forward inference profile

`attn_fwd.sqlite3.tar.xz` contains 24 measured forward entries in the native
`FLASH$attn_fwd` schema. CMake includes this shard in the normal tuning database.
`attn_fwd.json` records the selected performance/compiler options, both head
configurations' timings, compiler versions, and the source log's SHA-256.
Candidate indices in that JSON identify the original tuning build only; runtime
dispatch uses the stored options, not those indices. `validation.json` captures
the subsequent normal-build GPU validation, including all 66 cases' timings,
output/LSE errors, selected tiles/indices, and SHA-256 hashes of the source logs.

The measurements used an MI100 with PyTorch `2.14.0+rocm10.1.0`, pinned Triton
`3.8.0+gitaa3cf1a1`, batch 1, and query/KV head counts 32/32 and 32/8. Each row
selects the common finalist with the lowest geometric mean latency relative to
the default across those two head configurations. Finalists were timed in three
rounds after all 3,888 candidate output and log-sum-exp checks passed against
the reference calculations.

Coverage is deliberately exact:

| Mode | Actual head dimensions | Query/key lengths |
| --- | --- | --- |
| Causal prefill | 64, 128 | 512/512, 2048/2048 |
| Noncausal decode | 64, 128 | 1/1024, 1/8192 |
| Causal irregular | 56, 120 | 33/33 |
| Noncausal irregular | 56, 120 | 1/1009 |

All rows use FP16 or BF16, equal Q/K and V head dimensions, no dropout, and no
bias. The tiny causal cases explicitly retain the default. The family lookup
policy in `modules/flash/aot/sancheck.py` limits automatic selection to these
fixed-length shapes and the tested batch/head configurations. Unknown sequence
lengths, unmeasured combinations, different actual head dimensions, other
batch/head counts, and variable-length inputs select the generic image at
index 0. Other architectures retain their existing LUT completeness checks.

Normal automatic dispatch was validated on MI100 using the same PyTorch and
compiler versions. All 48 measured cases and 18 uncovered cases passed output
and log-sum-exp checks. Dispatch traces confirmed every expected tile and the
default image at index 0 for all 18 uncovered cases, including FP32, adjacent
sequence lengths, different batch/head counts, and unequal head dimensions.
The broader forward/backward and variable-length suites passed 3,907 tests with
split backward and 3,895 with fused backward (7,802 total).

Fresh baseline and profiled builds, timed in separate processes with three
rounds per case, produced:

| Cases | Count | Speedup range | Geometric mean |
| --- | --- | --- | --- |
| Aligned causal prefill | 16 | 1.56–4.40× | 2.52× |
| Aligned noncausal decode | 16 | 1.21–2.04× | 1.55× |
| Irregular dimensions/lengths | 16 | 0.975–1.547× | 1.13× |
| Uncovered shapes using defaults | 18 | 0.996–1.003× | 1.000× |

The tiny causal irregular cases retain the identical default images; their
small timing differences reflect run variation. The timing measurements use a
1 GiB cache flush between samples. These results establish gains for the tested
standalone forward calls; they do not measure end-to-end LLM speedups or
PyTorch's built-in SDPA integration. Profiles match only the exact coverage
listed above, so an adjacent decoding length continues to use the default.

A normal gfx908 build retains every generic image and adds 16 distinct selected
images, for 1,783 images total. Run the CPU database/lookup regression with:

```bash
cd /workspaces/aotriton
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /opt/aotriton-venv/bin/python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/aotriton-profile-cpu-tests python/test/test_gfx908_profile.py
```

The regression composes the native shard, compares every selected configuration
with the measurements, and executes the emitted C++ lookup on the CPU. It also
checks fallback and manual tuning selection without importing GPU libraries.
