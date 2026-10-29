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
file name. The last four keys say how the times were taken, and are the
**measurement keys**: a file never mixes them (the export refuses to merge a
measurement taken another way into a file), and two files whose measurement
keys differ are never compared.

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
any,0x75a0,288,,,8,off,AMD EPYC 9575F 64-Core Processor,,Debian GNU/Linux 13,6.12.43,,10.1.0,abc1234,2026-11-01..2026-11-03
```

`harness` is the source commit of the measuring core. `measured` is the first
and last date of the measurements that carried this configuration
(`YYYY-MM-DD..YYYY-MM-DD`, or one date). Host names are never published. An
empty cell is a fact the report did not carry. Machines of one variant that
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
  - the prime tables of every forced backend, each in a `<details>` block.

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
                                  [--perfdb DIR] [--top N] [--summary_only]
python -m aotriton.perfmon.export --workdir <workdir> --module flash --arch gfx942
                                  [--variant V] [--subject S ...] [--output DIR]
                                  [--entry_set prime llm] [--verify]
python -m aotriton.perfmon.export --workdir <workdir> -l [--module M] [--arch A]
                                  [--variant V] [--subject S ...]
```

- `show` prints the file as CSV with the entries spelled out: variant, entry,
  the problem columns, the sizes and the statistics. Naming a `.perfdb` file
  alone does the same, which is what the shebang runs.
- `compare` compares any two tags, not only adjacent ones. It joins files of
  the same name, then blocks of the same variant, then lines of the same entry;
  entries that are a marker or missing on either side are skipped. It prints a
  ratio (A over B) per line, the geometric mean per file, the overall geometric
  mean and the entries with the largest regressions. A file pair whose
  measurement keys differ is refused, reported, and makes the exit status 1.
- `export` reads `perfmon.latest_measurement` for the selection, prints how
  many rows a newer one superseded and how many were `NotOK` or `crash`,
  merges, writes `platform.csv`, re-renders the markdown, and with `--verify`
  reads the files back and checks every exported line. `--entry_set` names the
  sets whose arch-rejected entries are written as excluded (default prime and
  llm). `--output` defaults to `modules/<module>/perfdb`.
- `export -l` writes nothing. It lists the (module, arch, variant, subject)
  combinations that exist in PostgreSQL for the options that were not given,
  with the number of measurements and how many are `OK`, `NotOK` and `crash`,
  the newest date, and whether `<tag>/data/<arch>/` already has a block for the
  variant:

  ```
  variant  subject                      measured    OK  NotOK  crash  newest      exported
  any      aotriton-0.13b+rocm10.1.0        2816  2790     26      0  2026-11-20  yes
  any      aotriton-0.14.2b+rocm10.1.0      2816  2801     15      0  2026-11-21  no
  ```
