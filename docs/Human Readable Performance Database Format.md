# Human Readable Performance Database Format

The performance database records how fast AOTriton's public APIs run, for every
release and every GPU architecture that was measured. It lives in plain text
under `modules/<family>/perfdb/`, next to the tuning database, so that the
answer to "did AOTriton get faster or slower since release X?" is permanent,
reviewable as ordinary `git diff`s, and readable on GitHub without any tool.

The format follows [Human Readable Tuning Database Format](Human%20Readable%20Tuning%20Database%20Format.md)
(HRDB): implicit ordinals, `#!` sections, a shebang, a canonical writer. Read
that document first for the common conventions; this one only restates what
perfdb relies on. The writer, reader, renderer and comparison tool are one
module, `aotriton.perfmon.perfdb`. The tool that fills the database from
PostgreSQL is `aotriton.perfmon.export`.

All examples are small and made up; real files have the same shape.

## Design Principles

### Diffable and deterministic

The same data always produces the same bytes. Lines are sorted, variant blocks
are sorted, and numbers are rounded to 4 significant digits by one function.
Rendering the markdown is deterministic too, so CI can check that the committed
markdown matches the data (`render --check`).

### Avoid massive updates

Performance data accumulates for years, so a change must only touch what it is
about:

- **One directory per release tag.** Measuring a new release adds a directory
  and never rewrites another release's files. Redoing a release is `rm -r` of
  its directory and a new export.
- **Shared, append-only problem and entry ordinals.** One `.metadata/` for the
  family, shared by every tag, arch and variant. Testing a new head dimension,
  a new sequence length or a new LLM workload appends rows. Existing files do
  not change, and one entry ordinal names the same call in every release, so
  two releases are compared by joining on ordinals.
- **Every planned axis is a column from day one.** GQA, varlen and storage flip
  are columns of `problems.csv` even while they hold their default. Adding a
  column later would rewrite every row.
- **One line per measurement.** Re-measuring an entry with one backend changes
  one line of one file.

### Organised for reading by release

The question people ask is "how did release X do on arch Y", so the page for it
is `<tag>/<arch>.md`, and its data is `<tag>/data/<arch>/`. Variants are blocks
inside a data file and sections of the page, not a directory level.

### Tested entries are their own list

perfdb tests what callers ask for, which need not match what is compiled:
`hdim=384` is tested as 384 even where the call runs the 512 kernel with
padding. The entry list is therefore not the tuning database's functionals.
Tying them together would hide exactly the cases where padding costs
performance, and would renumber the results whenever the build's functional
space changes.

### Backends stored by name, in file names

Releases number their backends differently and rename them. Each backend's
measurements are a file named after its canonical name, so there is no backend
ordinal to keep stable. perfmon selects and observes: it measures the unforced
operator, and the operator forced to each backend. It never records which
backend the unforced operator picked.

### Explicit where it is cheap

Each data file states its iface and backend in its `#! metadata` section as
well as in its name, and `entries.csv` states every size instead of
recomputing it.

### HRDB features not adopted

- **Dense tensors and run-length encoding.** The tested entries are an
  irregular set (a diagonal, an L of sequence lengths, LLM workloads off the
  ladder), so a dense tensor would be mostly blank. Each line carries six
  statistics, so the cost of naming its entry is small.
- **Base references and `.dbpatch` overlays.** Performance data is never shared
  between architectures, and a re-measurement is already a one-line diff.

## Directory Layout

```
modules/<family>/perfdb/
  README.md                       generated: index of tags, release-over-release table per arch
  .metadata/                      shared by all tags
    problems.csv                  what is asked, without sizes
    entries.csv                   a problem and all its sizes
  <tag>/                          e.g. 0.14.2b
    README.md                     generated: the release's summary against the previous tag
    <arch>.md                     generated: prime tables, LLM cases, forced backends; a section per variant
    data/<arch>/
      <iface>.perfdb              the operator, unforced (what users get)
      <iface>.<backend>.perfdb    the operator forced to <backend>
      platform.csv                the machines that measured this arch
```

`<tag>` is the release tag: the subject `aotriton-0.14.2b+rocm10.1.0` is
exported to `0.14.2b/`. A subject names the build under test, the tag and the
ROCm it was built against. Two subjects of one tag but different ROCm versions
cannot share a directory and the export refuses them. `head` is never
exported: it would rewrite every number of its directory each time, and is
compared live from PostgreSQL instead. Release order is the order of the
numbers in the tag, so `0.9.2b < 0.10b < 0.11.2b`.

`<arch>` is the architecture identifier, `<iface>` the operator
(`op_attn_fwd`, `op_attn_bwd`). `<backend>` is a canonical backend name,
`[a-z0-9_]+`, so the first dot of a file name always separates iface and
backend. `<iface>.perfdb` needs no invented word for "unforced", and a
directory listing sorts each operator next to its backends. A backend that
never ran on an arch has no file there. A release that cannot force backends
has only `<iface>.perfdb`.

A **variant** is a group of machines of one arch whose performance is
expected to be the same; it is named by the operator of the fleet. Tasks
dispatched without a variant are variant `any`. Variants are never merged and
never derived from hardware details.

## File Types

| File | Content |
|---|---|
| `.perfdb` | the measurements of one iface and backend on one arch of one tag |
| `problems.csv`, `entries.csv` | the entry list, plain CSV with a header |
| `platform.csv` | the machines of one arch, plain CSV with a header |
| `.md` | generated; do not edit |

`.perfdb` files are not CSV: they are written in sections, like tensor files in
HRDB, so they do not use the `.csv` extension that tools would open as a
spreadsheet.

## Common Conventions

### Implicit Ordinals

As in HRDB, a row of a `.metadata/` table is referred to by its line number,
counting the header as line 0: the first data row is ordinal 1. Both tables are
append-only: a row, once referenced, keeps its ordinal.

The writer finds rows by content. Exporting an entry that is already listed
reuses its ordinal; an entry that is not is appended. The order in which the
family generates its entries never matters. A row listed twice is an error, as
is a header that differs from the family's columns: a new column needs a
migration, not an append.

### Comments, Sections, Shebang

The shebang, `#` comments and `#!` sections are HRDB's (and are read by the
tuning database's `aotriton.database.interpreter`). The shebang of a data file
is

```
#!/usr/bin/env -S python -m aotriton.perfmon.perfdb
```

so `./op_attn_fwd.perfdb` prints the file with every entry spelled out, see
[Tooling](#tooling).

### Numbers

Statistics are in microseconds. A number has 4 significant digits, no exponent
and no trailing zeros: `12.3456` is `12.35`, `123456` is `123500`, `0.081`
stays `0.081`, `12.0` is `12`. The reader accepts any spelling; the writer
emits only this one.

## Entry List

### problems.csv

What a caller asks the API to compute, apart from the sizes. One row per
distinct problem. The header is the family's `PROBLEM_COLUMNS`; for flash:

```
# .metadata/problems.csv
T_io,hdim_qk,hdim_v,causal,window_left,window_right,dropout_p,bias_type,bias_shape,gqa_ratio,varlen,storage_flip
bf16,64,64,0,-1,-1,0,0,-,1,0,0
bf16,384,384,0,-1,-1,0,0,-,1,0,0        <- tested as 384, whatever kernel runs it
fp16,64,64,0,-1,-1,0.1,1,b1qk,1,0,0     <- a padding-mask bias broadcast over heads
bf16,128,128,1,-1,-1,0,0,-,4,0,0        <- a GQA problem
```

Columns use ATI notation where an operator argument exists (`T_io`) and the
API's own names otherwise. `gqa_ratio` is `n_heads_q / n_heads_kv`; absolute
head counts are sizes, not part of the problem. `bias_shape` says which
dimensions of the bias are real: `bhqk`, `b1qk` (broadcast over heads, as
PyTorch passes a key-padding mask), `11qk`, or `-` without bias.

Cells are spelled canonically so that matching by content does not depend on
how a value was typed: integers and booleans as `0`/`1`, a float with an
integral value as an integer (`0.0` is `0`), any other float by its shortest
representation (`0.1`). A cell that would need CSV quoting is refused.

The table is per family: forward and backward of every release measure the
same problems, so one ordinal names the same problem in every file.

### entries.csv

One row per tested point, every size filled:

```
# .metadata/entries.csv
problem,seqlen_q,seqlen_k,batch,n_heads_q
1,16,16,4096,32                          <- grid entry: sizes from the size rule
1,16384,16384,8,32
3,512,512,384,16                         <- an LLM case: the workload's own sizes
```

`problem` is an ordinal into `problems.csv`; it must exist. `n_heads_kv` follows
from the problem's `gqa_ratio`. A grid entry and an LLM case with the same
problem and seqlens but different sizes are two entries. The entry set an entry
belongs to (prime, coverage, LLM) is not stored: sets overlap, and the renderer
re-derives membership from the family.

## Data Files

```
#!/usr/bin/env -S python -m aotriton.perfmon.perfdb
#! metadata
iface,op_attn_bwd
backend,triton_split
timing_method,hipgraph_ev100
iterations,100
l2_flush,1
unit,us
#! results any
entry,median,mean,stddev,min,p05,p95
1,12.34,12.4,0.081,12.21,12.25,12.61
2,x
3,25.61,25.7,0.12,25.43,25.48,25.94
188,-
#! results pool_a
entry,median,mean,stddev,min,p05,p95
...
```

### `#! metadata`

`key,value` lines, in this order: `iface`, `backend`, `timing_method`,
`iterations`, `l2_flush`, `unit`. `backend` is `-` in the unforced
`<iface>.perfdb`. A reader checks that `iface` and `backend` agree with the
file name. The last four keys say how the times were taken. Three of them are the
**measurement keys**, `timing_method`, `l2_flush` and `unit`: a file never mixes them (the export refuses to merge a
measurement taken another way into a file), and two files whose measurement
keys differ are never compared.

`iterations` is how many timed iterations the lines of the file used: one
number when they all used the same count, else the range `MIN-MAX` (for
example `20-400`). The count is adaptive: the harness times one warm call and
takes `clamp(ceil(10 s / call time), 20, 400)` iterations, so entries whose
call takes 25 ms or less have 400 and long entries (500 ms and up) as few as 20.
A report stored without a count was measured before the count became adaptive,
with 100 iterations, and counts as 100. The key only describes the file: it is not
a measurement key, and the export never refuses a measurement because its count differs.

### `#! results <variant>`

One block per variant, sorted by variant name. Re-exporting one variant changes
only its block, and re-exporting one arch only that arch's files. A block with
no lines is not written. The first line of a block is the header
`entry,median,mean,stddev,min,p05,p95`.

One line per entry, sorted by entry. `entry` is an ordinal into `entries.csv`.
There is no backend column (the file is one backend) and no size column. The
six statistics are in the fixed order above; `median` is the one that is
rendered and compared. A reader accepts lines in any order and refuses an entry
listed twice in a block.

An iface measures only the entries that apply to it: an LLM case that is
measured forward only has no lines in the backward files.

### Markers

A marker replaces the six values:

| Marker | Meaning |
|---|---|
| `x` | measured, no data: the backend failed or crashed (details stay in PostgreSQL) |
| `-` | excluded: the entry is not supported by this subject or arch, or does not fit the GPU |
| no line | never measured |

## platform.csv

One row per distinct machine configuration that contributed to the arch's data,
sorted by its cells:

```
variant,gpu_device_id,vram_gib,power_cap_w,power_cap_default_w,gpus_per_node,thermal_gate,cpu,ram_speed,os,kernel,amdgpu,rocm,harness,measured
any,0x75a0,288,1400,1400,8,off,AMD EPYC 9575F 64-Core Processor,,Debian GNU/Linux 13,6.12.43,6.19.14,10.1.0,abc1234,2026-11-01..2026-11-03
```

`harness` is the source commit of the measuring core. `measured` is the first
and last date of the measurements that carried this configuration
(`YYYY-MM-DD..YYYY-MM-DD`, or one date). Host names are never published. An
empty cell is a fact the report did not carry. The worker reads the power caps
(the GPU's amdgpu hwmon `power1_cap` and `power1_cap_default`, found by the
GPU's PCI address) and the `amdgpu` module version (`/sys/module/amdgpu/version`)
from the node's sysfs, so campaigns measured before that was added have them
empty, and so does an in-tree driver without a module version. `ram_speed` is
always empty for now: only the DMI tables have it, and a container without root
does not see them. Machines of one variant that
report different device ids or different default power caps are listed on
separate rows, and the export warns, since the variant is then not expected to
perform the same.

## Merging

The export merges the newest measurement of every (variant, subject, entry,
iface, backend) into the tree. Per data file and variant:

- A line present in the database replaces the file's line for the same entry.
- Lines that are not in the database are kept, so exporting a partial campaign
  never erases data.
- Entries the arch cannot run are re-derived from the family and written as
  excluded (`-`) to every file of the covered variants, unless the file already
  has a line for them.

Status mapping: `OK` becomes the statistics (an `OK` row without them is `x`),
`NotOK` and `crash` become `x`, `excluded` becomes `-`. Re-exporting unchanged
data writes nothing.

## Rendering

`render <perfdb dir>` regenerates every generated markdown file from the data;
with `--check` it writes nothing and fails if a committed file differs. TFLOPS
are computed only when rendering, from the median time and the entry; the
repository stores times.

- **`<tag>/<arch>.md`**, the page people look for. One section per variant,
  sorted by name, with a platform header from `platform.csv` of at most five
  lines. Then, per variant:
  - the **prime tables**, per iface, of the unforced operator: rows are head
    dimensions, columns sequence lengths, one table per combination of causal
    and dropout. A cell is the TFLOPS of the main dtype with the difference of
    the alternative dtype in parentheses, e.g. `0.512 (-0.1463)`; where the
    main one has no number but the alternative has, `✗ (fp16 0.1432)`.
  - the **LLM cases**, one table per iface, a row per case in the order of the
    family's workloads list, labelled by its model, with the TFLOPS of the
    unforced operator and of each forced backend. Cases whose adapters cannot
    run yet are not listed.
  - the prime tables of every forced backend, each in a `<details>` block. A
    backend that is excluded in every cell of the variant (flyc on an arch that
    does not support it) has no block and no LLM column; a line names it instead.

  `—` marks excluded, `✗` marks no data, a blank cell was not measured. A
  generated note explains that the unforced operator can be slower than its
  fastest backend: the operator database's choice is measured as it is, not
  corrected.
- **`<tag>/README.md`**: per arch and variant, the geometric mean of the
  median-time ratio (old over new, above 1 is faster) of the unforced operator
  against the previous tag that has the same arch, per iface and for the LLM
  cases, then a per-case table of the LLM cases (old time, new time, ratio).
- **`README.md`**: the index of tags, and per arch a release-over-release table
  of the same geometric means.

## Tooling

```
python -m aotriton.perfmon.perfdb show <file.perfdb>
python -m aotriton.perfmon.perfdb render <perfdb dir> [--check]
python -m aotriton.perfmon.perfdb compare <tag A> <tag B> [--arch A] [--variant V]
                                  [--perfdb DIR] [--top N] [--noise F]
                                  [--include_forced] [--csv_file F|-]
python -m aotriton.perfmon.export --workdir <workdir> --module flash --arch gfx942
                                  [--variant V] [--subject S ...] [--output DIR]
                                  [--entry_set prime llm] [--verify] [--include_incomplete]
python -m aotriton.perfmon.export --workdir <workdir> -l [--module M] [--arch A]
                                  [--variant V] [--subject S ...]
```

- `show` prints the file as CSV with the entries spelled out: variant, entry,
  the problem columns, the sizes and the statistics. Naming a `.perfdb` file
  alone does the same, which is what the shebang runs.
- `compare` compares any two tags, not only adjacent ones. It joins files of
  the same name, then blocks of the same variant, then lines of the same entry;
  entries that are a marker or missing on either side are skipped. By default
  only the unforced operators are compared (what users get); `--include_forced`
  adds the forced backends, each reported on its own row. The summary comes
  first: per arch and file, the number of lines, the geometric mean of A over B,
  how many lines are slower or faster beyond the noise (`--noise`, default 3%)
  and how many of those are LLM cases; then the largest regressions
  beyond noise (`--top`) with the entry's coordinates (dtype, hdim, causal,
  dropout, heads, batch, seqlens, ...), and every LLM case that is slower beyond
  noise. `--csv_file` also writes every compared line with its coordinates as
  CSV (`-`: to stdout, after the summary). A file pair whose measurement keys
  differ is refused, reported, and makes the exit status 1.
- `export` reads `perfmon.latest_measurement` for the selection, prints how
  many points were measured (with the `OK`, `NotOK` and `crash` counts), how
  many are excluded, and how many rows a newer one superseded,
  merges, writes `platform.csv`, re-renders the markdown, and with `--verify`
  reads the files back and checks every exported line. `--entry_set` names the
  sets whose arch-rejected entries are written as excluded (default prime and
  llm). `--output` defaults to `modules/<module>/perfdb`.
- A DEF task is marked `done` before all its result rows are written (by
  design), so `done` alone is not trusted. For each task that owns rows of the
  selection, `export` compares the (iface, backend) pairs recorded done in the
  task's appcookie with the report rows present. A task that is not `done`, or
  has recorded pairs without a report row, is incomplete: its rows are left out
  with a warning naming the missing pairs, and the exit status is 3 after the
  rest was exported. `--include_incomplete` exports them anyway. A done task
  with no report row at all is warned about as "no results yet". Wait for the
  result writes to finish (re-run `-l` until nothing is incomplete) before
  exporting a campaign.
- `export -l` writes nothing. It lists the (module, arch, variant, subject)
  combinations that exist in PostgreSQL for the options that were not given,
  with the number of points that were timed (`measured`, the sum of `OK`,
  `NotOK` and `crash`; excluded points were never run and are counted in
  `excluded` only), the newest date, and whether `<tag>/data/<arch>/` already has a block for the
  variant, and an `incomplete` column with the shortfall of the selection's
  tasks (`-` when complete, else `<n> tasks, <m> results missing`; the details
  go to stderr):

  ```
  variant  subject                      measured    OK  NotOK  crash  excluded  newest      exported  incomplete
  any      aotriton-0.13b+rocm10.1.0        2816  2790     26      0      1450  2026-11-20  yes       -
  any      aotriton-0.14.2b+rocm10.1.0      2816  2801     15      0      2878  2026-11-21  no        2 tasks, 7 results missing
  ```

### Live comparison of `head`

`head` is never committed, so it is compared live, from PostgreSQL, by
`python -m aotriton.perfmon.live`:

```
python -m aotriton.perfmon.live --workdir <workdir> --module flash --arch gfx942 gfx950
                                --against 0.14.2b 0.13b [--against_subject S ...]
                                [--subject aotriton-head+rocm...] [--variant V]
                                [--check regression forced_beats_unforced]
                                [--noise 0.03] [--forced_check_baselines] [--top 10]
                                [--format csv|markdown] [--output_file F] [--summary_only]
                                [--perfdb DIR]
```

It lists the flagged entries of the fault hunt: tuning-database cells tuned at
tiny batch or heads lose at real sizes, and the flagged problem coordinates map
to those cells. Two checks, both with the noise rule of `compare`
(`--noise`, default 3% for the same GPU; use `--noise 0.06` across GPUs, and
the tool warns when the committed `platform.csv` has another GPU device id than
the live rows):

`--check` runs only the named checks (default: both). `--subject` may name any
subject of the database, so `--check forced_beats_unforced --subject <release
subject>` runs the second check on a finished campaign without a head subject.

- `regression`: a point of head slower than the same (variant, iface, backend,
  entry) in a committed tag (`--against`) or another subject in the database
  (`--against_subject`) by more than the noise.
- `forced_beats_unforced`: within head (and, with `--forced_check_baselines`,
  within each reference), a forced backend faster than the unforced operator by
  more than the noise.

A summary goes to stderr before the list: per check, reference, arch, iface and
backend, the number of flags and the largest margin; then the `--top` flags of
each check with the largest margin and their coordinates. `--summary_only` omits
the list.

Only points where both sides have a median are compared: a backend that fails
or crashes (AITER-family backends often do) is not a flag; the summary on
stderr counts the points without reference data. Files with a different timing
method or L2 flush are refused. Without `--subject`, the only head subject in
the database is used. The output is CSV (or markdown) with the columns
`check, arch, variant, iface, backend, reference, <problem columns>,
<size columns>, reference_us, measured_us, ratio`; `ratio` is
`reference_us / measured_us` (below 1: head is slower; above 1: the forced
backend is faster), worst first within each check.
