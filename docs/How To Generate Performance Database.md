# Overview

AOTriton perfmon measures how fast AOTriton's public operators run, for every
release and every GPU architecture, and turns the measurements into the
performance database (perfdb) committed under `modules/<family>/perfdb/`. This
guide walks an operator through the whole process with the command line tools:

1. Build the **subjects** (AOTriton builds with a measurement harness) on a build node.
2. Relay them to the GPU hosts.
3. **Dispatch** measurement tasks and run **nodes** on the GPU hosts.
4. **Export** the measurements from PostgreSQL into perfdb, one architecture per commit.
5. **Analyze** the result: release-over-release regressions and backend-selection faults.

Perfmon is an application of the distributed execution framework, the same one
the tuner uses. It shares the workdir, the PostgreSQL server, `workers.db`, and
the remote-build mechanism with the tuner, so an operator who has set up the
tuner has most of the work done. Engine details are in
[Distributed Execution Framework](Distributed%20Execution%20Framework.md); the
file format, the merge rules and the export semantics are in
[Human Readable Performance Database Format](Human%20Readable%20Performance%20Database%20Format.md).
This guide does not repeat either.

# Concepts

| Term | Meaning |
|------|---------|
| **Subject** | An AOTriton build with a native harness (the runner) linked against it. Named `aotriton-<tag>+rocm<ver>`. |
| **Release subject** | A subject of a shipped release tag. The committed perfdb holds release subjects only, and a release's perfdb is generated after the release has shipped. |
| **`head`** | The committed tip of your branch, cloned by `prepwkdir` into `<wkdir>/aotriton.src` (uncommitted work is not in it). Subject `aotriton-head+rocm<ver>`. Measured only for live comparison (`aotriton.perfmon.live`); never committed, and `export` refuses it. |
| **MAJOR subject** | One subject per release serving all arches of `AOTRITON_MAJOR_ARCHES` (`python/gpu_targets.py`), built once with all of the release's images. Other arches get a subject of their own. |
| **Core library** | `libperfmon_core+rocm<ver>.so`, built once per ROCm version and linked by every subject of that ROCm. |
| **Entry set** | Which problems to measure. `prime`: bf16/fp16, eight head dimensions, causal on/off, dropout 0/0.1, a diagonal sequence-length ladder 16..16384. `coverage`: `prime` plus fp32, bias, more head dimensions and L-shaped sequence lengths. `llm`: real model cases from `modules/flash/perfmon/workloads.csv`. |
| **Select and observe** | For every (entry, operator) the harness measures the *unforced* operator (the backend the tuning database selects) and every backend the subject lists, *forced*. A forced backend that beats the unforced operator exposes a tuning-database fault. |

Result statuses of one measurement:

| Status | Meaning | In perfdb files |
|--------|---------|-----------------|
| `OK` | Six statistics, in microseconds | the statistics line |
| `NotOK` | The runner rejected the call | `x` |
| `crash` | The runner died or timed out | `x` |
| `excluded` | Unsupported by the subject or arch, or does not fit the GPU | `-` |

Sizes derive from the problem and the sequence lengths alone, so an entry is
the same call on every GPU. Each measurement takes `clamp(ceil(10 s / call
time), 20, 400)` iterations; the count is stored in the file.

What ends up in the repository: only generated files under
`modules/<family>/perfdb/` (data, `README.md`, `<arch>.md`, `.metadata/*.csv`).
They are canonical and written by `export`; never edit them by hand.

# Prerequisites

## System Requirements

* **Dev Node**: a host that can reach every GPU host and the build node over
  SSH. It holds the checkout and the workdir, and runs the commands in this guide.
* **Server Node**: runs PostgreSQL; reachable from all GPU hosts (or through the
  SSH tunnel for firewalled hosts). Can be the dev node.
* **Build Node**: a host with the ROCm toolchain of every ROCm version you
  measure. Can be the dev node. This is the same remote build node as the tuner's.
* **GPU Hosts**: AMD GPUs, reachable over SSH from the dev node, with the
  checkout and the ROCm Python environment that `setup_session` prepares.

Tools: `ssh`, `rsync`, `sqlite3` on the dev node; `psql` or `docker` for
`initdb`; Python >= 3.10 with `psycopg` and the checkout's `aotriton` package
importable (`pip install -e .`, as in [How To Run Tests](How%20To%20Run%20Tests.md)).

All commands are run from the repository root. The scripts take the workdir from
`--workdir` or from `AOTRITON_WORKDIR`, so you can export it once:

```bash
export AOTRITON_WORKDIR=<wkdir>
```

Verify any option in a script's header or with `--help` before relying on it:

```bash
.perfmon/bin/build_subjects --help
python -m aotriton.perfmon.dispatch --help
```

## Create the Working Directory

If you already have a tuner workdir, reuse it: perfmon tasks are a separate
category in the same database. Otherwise:

```bash
.def/bin/initproj <wkdir>        # needs a TTY; writes config.rc and workers.db
```

`config.rc` holds the infrastructure settings only (PostgreSQL, service host).
See [How To Generate Tuning Database](How%20To%20Generate%20Tuning%20Database.md#create-working-directory)
for its fields. Never put a database name into a connection setting: the
default database is used everywhere.

## Register Hosts and Settings

Perfmon reads its settings from `<wkdir>/workers.db`. Edit them in the WebUI
(**Workers** tab for hosts, **Perfmon** page for the release list), or
directly with `sqlite3`. The WebUI is launched as for the tuner:

```bash
python .def/bin/dashboard.py <wkdir> [--port <port>]
```

| Setting | Where | Meaning |
|---------|-------|---------|
| Perfmon host | Workers tab: add the host with its arch, tick the **Perfmon** role | Hosts that receive subjects and run nodes |
| `buildnode::hostname` | `config` table | SSH name of the build node |
| `buildnode::workdir_override` | `config` table | The workdir on the build node (else `default_workdir`) |
| `perfmon::rocm` | `config` table, JSON list | ROCm versions to build; default `["10.1.0"]` |
| `perfmon::tags` | `config` table, JSON list | Release tags to build, e.g. `["0.13b","0.14.2b"]` |
| `<host>::gpu_selection` | `config` table | GPUs a node uses: `-1` for all, or `0,1,2` |
| `thermal_gate::<host>` | `config` table | `1`: the node waits until the GPU is cool enough (see Troubleshooting) |
| `firewalled::<host>` | `config` table | `1`: the host reaches PostgreSQL through the SSH tunnel that `srvctl` runs |

```bash
sqlite3 <wkdir>/workers.db <<'EOF'
INSERT OR REPLACE INTO config (key, value) VALUES ('buildnode::hostname', '<buildnode>');
INSERT OR REPLACE INTO config (key, value) VALUES ('buildnode::workdir_override', '<remote wkdir>');
INSERT OR REPLACE INTO config (key, value) VALUES ('perfmon::rocm', '["10.1.0"]');
INSERT OR REPLACE INTO config (key, value) VALUES ('perfmon::tags', '["0.13b","0.14.2b"]');
INSERT OR REPLACE INTO config (key, value) VALUES ('<host>::gpu_selection', '0,1');
INSERT OR REPLACE INTO config (key, value) VALUES ('thermal_gate::<host>', '1');
EOF
```

The host list is shared with the tuner; the role decides which application
uses a host. A host can have both roles, but do not run a tuner and a perfmon
node on the same GPU at once.

## Start PostgreSQL and Apply the Schemas

```bash
.def/bin/srvctl <wkdir> start     # PostgreSQL, plus the tunnel manager for firewalled hosts
.def/bin/initdb <wkdir>
```

`srvctl` takes `start|stop|restart|status`. Tunnels to firewalled hosts need
non-interactive SSH from the dev node to each of them (start an `ssh-agent` and
`ssh-add` your key if it has a passphrase); see the "Firewalled hosts" section
of the [Distributed Execution Framework](Distributed%20Execution%20Framework.md).

`initdb` is idempotent: run it again after pulling a newer checkout.

> **Never pass `--recreate` to a database that holds data.** It drops the
> schemas, including the tuner's results and every perfmon measurement. The
> database is normally shared with the tuner.

## Prepare the Source and Sync to the Build Node

Perfmon builds on the remote build node exactly as the tuner does
(`prepwkdir`, `<wkdir>/aotriton.src`, `sync_workdir.sh --buildnode`). If you
already run the tuner, the flow is familiar and a shared build node needs no
new setup.

```bash
.def/bin/prepwkdir <wkdir>
.def/libexec/sync_workdir.sh <wkdir> <buildnode> --buildnode
```

`prepwkdir` clones or updates `<wkdir>/aotriton.src` from your branch. Release
subjects build from release tarballs and do not depend on it, but the build
scripts on the build node come from that copy, so run both commands after
every change to the checkout that should reach the build.

# Build and Deploy Subjects

## Build on the Build Node

```bash
.perfmon/bin/build_subjects --workdir <wkdir> --host <buildnode> \
        [--rocm <ver>]... [--arch MAJOR|ALL|<arch>...] [--without_amdsmi] \
        [<tag>...] [-- <build_subject.sh options>]
```

| Option | Meaning |
|--------|---------|
| `--host <h>` | Build over SSH on `<h>`, from its copy of the workdir. Without it the build runs here and needs the toolchain in the environment. |
| `--rocm <ver>` | Repeatable. Default: `perfmon::rocm`, else `10.1.0`. |
| `--arch` | `MAJOR` (default): one subject per tag for the major arches. `ALL`: `MAJOR` plus a subject for every other supported arch. `<arch>`: a subject for that arch only (a major arch means `MAJOR`). |
| `<tag>...` | Release tags. Default: `perfmon::tags`. |
| `--without_amdsmi` | Build the core without amd-smi. Such a core refuses the thermal gate. |
| `--prefix <dir>` | Default `<wkdir>/installed/perfmon` (on the build node with `--host`). |
| `-- ...` | Options for `perfmon/build_subject.sh`, e.g. `--cache_dir <dir>` to cache release tarballs. |

It builds the core once per ROCm (`perfmon/build_core.sh`), then the subjects
(`perfmon/build_subject.sh`). Each tag builds in its own subshell, so one
failing tag does not stop the others. Example:

```bash
.perfmon/bin/build_subjects --workdir <wkdir> --host <buildnode> --rocm 10.1.0 0.13b 0.14.2b
```

Do not build `head` for a campaign meant for the committed perfdb (see
[Measuring head](#measuring-head)).

## Relay and Deploy

```bash
.def/bin/fetchbuild <wkdir> --category perfmon    # build node -> <wkdir>/installed/perfmon on the dev node
.def/bin/deploy <wkdir> --category perfmon        # dev node -> Perfmon-role hosts
```

`deploy` sends each host the core, the MAJOR subjects and the subjects of its
own arch. A subject that is missing under `installed/perfmon/` is an error at
dispatch, not on a worker.

## Check What Was Built

Every subject records its origin in `source.txt`:

```bash
cat <wkdir>/installed/perfmon/MAJOR/aotriton-0.14.2b+rocm10.1.0/source.txt
```

For a release it should name the release tag and report no changed files. The
same information is stored in each report's `platform.source`.

# Prepare the GPU Hosts

```bash
.perfmon/bin/setup_session --workdir <wkdir> --host <host> [--rocm <ver>] [--arch <arch>]
```

Over SSH it checks that the checkout and the scratch directory (by default the
parent of the workdir) are visible, that `/dev/kfd` exists and that `rocminfo`
lists an arch, creates the ROCm Python environment if it is missing, and
prints what a node needs. It is idempotent: run it again whenever it reports
something missing. Fix everything it reports before starting a node.

# Dispatch a Campaign

A campaign is subjects x entry sets x arches. Always run `--dry_run` first and
read the counts:

```bash
.perfmon/bin/dispatch --module flash --arch <arch>... --entry_set prime \
        [--subject aotriton-<tag>+rocm<ver> ...] --dry_run
.perfmon/bin/dispatch --module flash --arch <arch>... --entry_set prime --yes
```

| Option | Meaning |
|--------|---------|
| `--arch <arch>...` | The pools to dispatch to (synonym of `--pool`). |
| `--entry_set prime\|coverage\|llm` | Which entries. |
| `--subject ...` | Default: every subject installed for the arch (`installed/perfmon/<MAJOR or arch>/`). |
| `--phase fwd fwd_bwd` | `llm` only: which phases. |
| `--hdim 64 128`, `--causal 0`, ... | Axis options replace the entry set's values for that axis (`--entry_set prime --help` lists them). `llm` takes none. |
| `--skip_completed` | Add only tasks that have no done task yet. |
| `--dry_run`, `-y/--yes` | Print without inserting; skip the confirmation prompt. |

A typical campaign for a release on one arch:

```bash
for set in prime llm coverage; do
  .perfmon/bin/dispatch --module flash --arch <arch> --entry_set $set \
          --subject aotriton-<tag>+rocm<ver> --yes
done
```

Entries the arch cannot run are printed and recorded as `excluded` instead of
being dispatched. Tasks are generated subject by subject, because switching
subjects restarts the native runner.

## GPU Hygiene

**A GPU shared with any other process gives worthless timings.** Before
starting a node, check that nothing else uses the GPUs of the host:

```bash
ssh <host> 'rocm-smi --showuse --showmemuse'    # or: amd-smi monitor
```

Select only idle GPUs: set `<host>::gpu_selection` in `workers.db`, or pass
`--multi_gpu <id>...` to `node` (it overrides the stored selection). Do not
start a tuner worker, a test run or an interactive job on the same GPU while
a perfmon node runs, and do not mix GPU models between tags you intend to
compare (see [Analyze](#analyze)).

## Start Nodes

```bash
.perfmon/bin/node --workdir <wkdir> --host <host> [--rocm <ver>] [--arch <arch>] \
        [--multi_gpu <id>...] [--nreader <n>] [--ncpu <n>] [--session_name <n>] start
.perfmon/bin/node --workdir <wkdir> --host <host> status
.perfmon/bin/node --workdir <wkdir> --host <host> stop        # add --graceful to let running tasks finish
```

The command applies the host's settings from `workers.db`: GPU selection,
thermal gate, subpool, and the PostgreSQL endpoint for firewalled hosts. If
the hosts run the worker container instead, use `wkctl`:

```bash
.def/bin/wkctl <wkdir> start --category perfmon [--host <host>]...
```

Nodes never write core dumps by design (`RLIMIT_CORE` is 0); a crash is
recorded as a `crash` result.

# Monitor

| What | How |
|------|-----|
| Task counts per subject (pending/running/done/failed/cancelled) | WebUI **Perfmon** page, or `pgdef.pq.progress.queue_progress` and `perfmon.pq.measurements.subject_progress` from a script |
| Measurement counts per selection, with shortfalls | `.perfmon/bin/export -l [--module flash] [--arch <arch>] [--subject ...]` |
| Node logs | On the host, `<rundir>/logs/[<session_name>/]`; the node's `<rundir>` is `${XDG_RUNTIME_DIR:-/tmp}/aotriton-perfmon` unless `AOTRITON_RUNDIR` is set |

`export -l` writes nothing. It lists, per (module, arch, variant, subject), the
`measured`, `OK`, `NotOK`, `crash` and `excluded` counts, the newest date,
whether the data was already exported, and an `incomplete` column.

Telling the outcomes apart:

| Outcome | Meaning | Action |
|---------|---------|--------|
| `failed` task | A real failure of the task, not of a measurement | Read the node log before re-dispatching |
| `crash` result | The runner died or timed out on that point | Expected occasionally; many in one region mean a real bug or a too-long entry |
| `NotOK` result | The runner rejected the call | Usually an unsupported combination of that backend |
| `excluded` | Unsupported or does not fit the GPU | Nothing; recorded for completeness |

Failures and crashes of AITER-family backends on some entries are **expected
results**, not harness errors. They are counted in the summaries and are not
flagged as faults.

> **A task is marked `done` before all its result rows are written.** After the
> last task is done, wait until the CPU queue has drained and `export -l`
> counts stop changing before you export.

# Export, Verify and Commit

Export one arch at a time, and commit after each:

```bash
.perfmon/bin/export --module flash --arch <arch> [--subject aotriton-<tag>+rocm<ver> ...] --verify
python -m aotriton.perfmon.perfdb render modules/flash/perfdb --check
git diff --stat
git add modules/flash/perfdb/<explicit paths>
git commit -m "perfdb: flash <arch>, <tag list>"
```

| Option | Meaning |
|--------|---------|
| `--variant V` | Only this variant (default: all) |
| `--subject ...` | Only these subjects; `head` is refused |
| `--entry_set prime llm` | Entry sets whose arch-rejected entries are written as excluded |
| `--output <dir>` | Default `modules/<module>/perfdb` |
| `--verify` | Read the files back and check every exported line |
| `--include_incomplete` | Export tasks that look incomplete. **Never use it for committed data.** |

Rules to know (details in the
[format document](Human%20Readable%20Performance%20Database%20Format.md)):

* The newest measurement of each point replaces the file's line; lines absent
  from PostgreSQL are kept, so a partial campaign never erases data.
* Incomplete tasks are left out and `export` exits with status 3 after writing
  the rest. Wait, then run it again.
* Two ROCm versions of one tag cannot share `<tag>/`; the export refuses them.
  A file whose timing method or L2-flush setting differs from the new
  measurements is skipped with a warning: do not mix measurement methods in one tag.
* `render --check` must be clean after an export. If it is not, run
  `python -m aotriton.perfmon.perfdb render modules/flash/perfdb` and review the diff.

**One arch per commit.** `README.md` and `.metadata/*.csv` change with every
export. Exporting and committing one arch at a time attributes those shared
files to the arch that changed them. Stage explicit paths, never `git add -A`.

# Analyze

## Compare Two Releases

```bash
python -m aotriton.perfmon.perfdb compare <tag A> <tag B> [--arch <arch>] [--variant V] \
        [--top 10] [--noise 0.03] [--include_forced] [--csv_file <file>|-]
```

`A` is the older tag, `B` the newer. The summary lists, per arch, the number
of lines, the geometric mean, how many lines are slower or faster beyond the
noise, and the `--top` largest regressions with their coordinates (dtype,
hdim, sequence lengths, causal, dropout, ...), so each row maps to a cell of
the tuning database. By default only unforced operators are compared (what
users get); `--include_forced` adds each forced backend on its own row;
`--csv_file` writes every compared line.

| Situation | `--noise` |
|-----------|-----------|
| Both tags measured on the same GPU | `0.03` (default) |
| Tags measured on different GPUs (see `platform.csv`) | about `0.06` |

Files whose timing method or L2-flush setting differ are refused and make the
exit status 1.

## Check a Subject Live

`aotriton.perfmon.live` reads subjects straight from PostgreSQL and compares
them with committed tags or with other subjects:

```bash
python -m aotriton.perfmon.live --workdir <wkdir> --module flash --arch <arch>... \
        --against <tag>... [--against_subject <subject>...] [--subject <subject>] \
        [--check regression forced_beats_unforced] [--noise 0.03] [--top 10] \
        [--summary_only] [--format csv|markdown] [--output_file <file>]
```

| Check | Meaning | What it says about the tuning database |
|-------|---------|----------------------------------------|
| `forced_beats_unforced` | A forced backend is faster than the unforced operator beyond noise (`ratio` above 1) | The tuning database picked a worse backend or variant for that cell: a **backend-selection fault**. Report the cell, the backend and the margin. Works on a finished release campaign with `--subject`. |
| `regression` | A point is slower than the same point in a reference beyond noise (`ratio` = reference time / measured time, below 1) | A **candidate** faulty cell. Confirm on the same GPU with repeats before filing. |

A backend that fails or crashes at a point is not a flag; it is counted in the
summary. `--forced_check_baselines` runs the forced-beats-unforced check on the
references too.

# Measuring head

`head` answers "did my branch make something slower?" before a release. It is
for live comparison only and is never exported.

```bash
git commit ...                                           # head is the committed tip; a WIP commit is fine
.def/bin/prepwkdir <wkdir>
git -C <wkdir>/aotriton.src rev-parse HEAD               # must be the commit you meant
.def/libexec/sync_workdir.sh <wkdir> <buildnode> --buildnode
.perfmon/bin/build_subjects --workdir <wkdir> --host <buildnode> --rocm <ver> head
.def/bin/fetchbuild <wkdir> --category perfmon
.def/bin/deploy <wkdir> --category perfmon
cat <wkdir>/installed/perfmon/<pool>/aotriton-head+rocm<ver>/source.txt   # clean, and names that commit
.perfmon/bin/dispatch --module flash --arch <arch> --entry_set prime \
        --subject aotriton-head+rocm<ver> --yes
# start nodes and monitor as for a release, then:
python -m aotriton.perfmon.live --workdir <wkdir> --module flash --arch <arch> \
        --against <tag> --check regression forced_beats_unforced
```

Rebuilding head after new commits means repeating the sequence; to measure
the new build from scratch, clean the old head tasks first (next section).

# Restart or Clean a Campaign

* **Missing tasks only**: re-dispatch with `--skip_completed`. Newer
  measurements supersede older ones at export (the latest wins).
* **Stop first**: `node ... stop --graceful` on every host (or `wkctl ... stop`).
* **Delete or reset perfmon tasks** through the `pgdef` pq API. There is no
  CLI for it, and raw SQL must never be written outside a `pq` package. Select
  by category and, if wanted, by pool, status or payload; an empty selector is
  refused.

```python
# clean_head.py: run with the workdir's configuration loaded
import psycopg
from pathlib import Path

from aotriton.pgdef.pq import tasks
from aotriton.pgdef.pq.conn import params_from_env
from aotriton.pgdef.workdir import load_config

load_config(Path('<wkdir>'))
conn = psycopg.connect(**params_from_env(), autocommit=True)
selector = tasks.TaskSelector(pool='<arch>',
                              category=('perfmon', None),
                              payload_contains={'subject': 'aotriton-head+rocm<ver>'})
print(tasks.count(conn, selector), 'tasks selected')
tasks.delete(conn, selector)      # or: tasks.reset(conn, selector, purge_reports=True)
```

Never select "every task": the database may hold the tuner's work. Never
truncate or drop tables.

**Changing the measurement method means a clean restart.** If the core changes
(timing, iteration count, L2 flush), rebuild the core and subjects, delete the
affected campaign's tasks and re-dispatch. Do not mix methods in one tag.

# Troubleshooting

| Symptom | Cause and fix |
|---------|---------------|
| Very long entries are recorded as `crash` | Each runner reply has a 600 s timeout (120 s for the first launch); a long entry can exceed it. Treat as a result, not a harness error; only large-seqlen entries should be affected. |
| Tasks stay `running` and nodes are idle | With `thermal_gate::<host>` = 1 the node waits for the GPU to cool. Check the GPU temperature and airflow, or set the row to 0 if the timings need not be thermally stable. A core built `--without_amdsmi` refuses the gate. |
| `dispatch` reports a missing subject | Run `fetchbuild` and `deploy` again; check `installed/perfmon/`. |
| Disk fills on the build node or hosts | Release tarballs and builds accumulate. Use `--cache_dir`, and remove old subjects from `installed/perfmon/` once deployed. |
| Node cannot reach PostgreSQL | Is the host marked `firewalled::<host>`? Does `srvctl <wkdir> status` show the tunnels? SSH from the dev node must be non-interactive. |
| `export` exits with status 3 | Incomplete tasks were left out. Wait for the queue to drain, check `export -l`, run again. |
| `pytest -m pg` drops schemas | Those tests drop and recreate the pgdef, perfmon and tune schemas. Run them only on a disposable PostgreSQL (`AOTRITON_PG_TEST_DISPOSABLE=1` and the `PG*` variables), never on the campaign database. |
| A crashed runner left no core dump | By design: nodes set `RLIMIT_CORE` to 0. The crash is in the results; use the node log. |
