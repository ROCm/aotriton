# Human Readable Tuning Database Format

The tuning database records, for every kernel and operator, which compiled
variant (or which backend) to use for a given set of inputs on a given GPU. It
lives in plain text under `modules/<family>/database/`, so that changes can be
reviewed as ordinary `git diff`s and contributors can submit incremental
updates.

This document describes the on-disk format. All examples are taken from
`modules/flash/database/`.

## Design Principles

### Avoid massive updates

A change to the database, or to the code that consumes it, should only touch
the entries it is about. A diff that rewrites thousands of unrelated lines
cannot be reviewed, and it conflicts with every other contribution in flight.
Each of the following mechanisms keeps updates local:

- **Append-only IDs.** Rows of `.metadata/` tables are never renumbered or
  removed. New functionals, psels, copts and hsaco options are appended, so
  existing tensor cells keep pointing at the same rows.
- **Functional IDs are not godel numbers.** The code generator numbers
  functionals by enumerating the functional space, so extending an axis
  (e.g. adding hdim=384) renumbers it. The database assigns its own IDs and
  appends the new functionals after the existing ones instead.
- **Backends are stored by name, not by index.** `op_backend.csv` maps each
  backend name, as declared by `@ati.backend`, to the small integer stored in
  the tensors. The index of a backend in the ATI DSL, and the C++ enum value
  derived from it, never appear in the database. Reordering or inserting
  backends in the operator declaration therefore leaves every operator tensor
  unchanged; only the mapping from name to backend in the code generator
  follows the new declaration.
- **Per-interface tables.** Each kernel and operator has its own
  `functional.csv`, `psels.csv`, `hsaco_options.csv` and `op_backend.csv`,
  so retuning one kernel never shifts the IDs of another.
- **Patches and symlinks.** A partially tuned architecture symlinks its base
  architecture and stores only its own entries in `.patch<N>` files. The base
  is not copied, and retuning the base does not touch the patches.
- **Line-oriented tensors.** Each tensor row is one line, and run-length
  encoding never crosses lines, so changing one cell changes one line.

### Diffable and deterministic

The database is plain text, and the writer's output is canonical: blocks are
sorted by functional ID, and run-length encoding is applied exactly where it
is shorter. Regenerating unchanged data produces identical files, so a diff
only ever shows real changes.

### Explicit where it is cheap

Large tables use implicit row IDs to stay compact. `op_backend.csv` has only
a handful of rows, so it lists its integers explicitly: a reader can decode
an operator tensor without counting lines.

## Directory Layout

```
modules/<family>/database/
  .metadata/
    copts.csv                         compiler options, shared by all triton kernels
    <iface>/
      functional.csv                  functionals of this kernel/operator
      psels.csv                       performance selections        (kernels)
      hsaco_options.csv               (psels, copts) pairs          (kernels)
      op_backend.csv                  backend name -> ID            (operators)
  amd/
    <arch>/
      <iface>.csv                     tuning tensors for <arch>, mod0
      <iface>.csv.patch<N>            incremental changes to <iface>.csv
      mod<N>/
        <iface>.csv                   tuning tensors for <arch>, mod<N>
```

`<iface>` is the interface name of a kernel (`attn_fwd`, `bwd_kernel_dq`, ...)
or an operator (`op_attn_fwd`, `op_attn_bwd`). `<arch>` is the architecture
identifier, e.g. `gfx942`. The files directly under `amd/<arch>/` hold the
entries of GPU `<arch>_mod0`, and the files under `amd/<arch>/mod<N>/` hold
those of `<arch>_mod<N>`.

`.metadata/` holds the tables shared by all architectures. Everything under
`amd/` is per architecture.

## Common Conventions

### Implicit IDs

Every table under `.metadata/` is a CSV file with a header line. Rows are
referred to by their **implicit ID**: the line number, counting the header as
line 0. The first data row therefore has ID 1, which matches the line number
shown by most editors minus one.

```
waves_per_eu,num_warps,num_stages      <- line 0, header
2,4,1                                  <- ID 1
1,4,1                                  <- ID 2
```

IDs are append-only. A row, once referenced, keeps its ID: new entries are
appended at the end of the table, and existing rows are never reordered or
removed. Adding a new value (e.g. a new head dimension) appends new
functionals after the existing ones instead of renumbering them.

`op_backend.csv` is the one exception, see [Operator Backends](#operator-backends).

### Comments and Sections

In tensor files (`amd/<arch>/<iface>.csv` and their patches):

- A line starting with `#!` starts a named metadata section. The section name
  is the rest of the line, e.g. `#! tuning keys`.
- Any other line starting with `#` is a comment and is ignored.
- Blank lines are ignored.

## Metadata Tables

### functional.csv

A functional is one combination of the inputs that select a compiled kernel,
excluding the tuning keys (the sequence lengths). Each kernel and operator has
its own `functional.csv`, with exactly its own input columns:

```
# .metadata/attn_fwd/functional.csv
T_io,CAUSAL_TYPE,BLOCK_DMODEL,ENABLE_DROPOUT,PADDED_HEAD,BIAS_TYPE,USE_ALIBI,INT8,INT8_KV,USE_P_SCALE
bf16,0,16,0,0,0,0,0,0,0
bf16,0,16,0,0,1,0,0,0,0
...

# .metadata/bwd_kernel_dq/functional.csv
T_io,CAUSAL_TYPE,BLOCK_DMODEL,ENABLE_DROPOUT,PADDED_HEAD,BIAS_TYPE
bf16,0,16,0,0,0
...
```

Columns use ATI notation:

- A tensor dtype axis is named after its ATI type variable, e.g. `T_io` from
  `@ati.type_var('T_io', ...)`. It is not named after a tensor argument
  (`Q`), because several tensors share the type variable.
- Dtypes are spelled the way ATI declares them, without the pointer and
  alignment: `*fp16:16` is stored as `fp16`, likewise `bf16`, `fp32`.
- Other columns are named after the kernel argument (`CAUSAL_TYPE`,
  `BLOCK_DMODEL`, ...). Boolean inputs are `0`/`1`.

Functional IDs are local to the database. They are deliberately not the godel
numbers the code generator assigns, so the database stays stable when the
functional space is extended.

### copts.csv

Compiler options, shared by all triton kernels:

```
# .metadata/copts.csv
waves_per_eu,num_warps,num_stages
2,4,1
1,4,1
...
```

### psels.csv

Performance selections (the tuned `tl.constexpr` values) of one kernel:

```
# .metadata/attn_fwd/psels.csv
PERSISTENT_TYPE,GRID_CU_MULTIP,BLOCK_M,BLOCK_N,PRE_LOAD_V
0,2,32,16,1
2,2,64,32,1
...
```

### hsaco_options.csv

Each row is one compiled variant of the kernel: a pair of IDs into the
kernel's `psels.csv` and the shared `copts.csv`. Tensors of triton kernels
refer to this table, so a single ID identifies everything needed to build the
HSACO. Other compiler backends can add their own option tables later.

```
# .metadata/attn_fwd/hsaco_options.csv
psels,copts
1,1
2,2
...
```

### Operator Backends

`op_backend.csv` maps each backend of an operator, by the name its
`@ati.backend` declaration gives it, to the ID used in this database. Unlike
other tables, it is small and lists its IDs explicitly:

```
# .metadata/op_attn_bwd/op_backend.csv
name,id
triton_split,1
triton_fuse,2
aiter,3
flyc,4
```

These IDs belong to the tuning database only. They do not have to equal the
values of the C++ backend enum, and may diverge from them, e.g. when backends
are reordered or added in the operator declaration. Code reading the database
maps backends by name. ID 0 is reserved for "no backend" (see
[Precedence](#precedence)).

## Tensor Files

`amd/<arch>/<iface>.csv` stores, for every functional tuned on `<arch>`, a
tensor indexed by the tuning keys.

```
#! tuning keys
Max_seqlen_q,16,32,64,128,256,512,1024,2048,4096,8192
Max_seqlen_k,16,32,64,128,256,512,1024,2048,4096,8192
#! tuning tensor
functional,tuning keys->hsaco_options
1,
149,27,191[3],170,150,170[3]
27,6,191[2],170,150[2],170[2],150
...
2,
...
```

### `#! tuning keys`

One line per tuning key: its name, then the bucket values of its ladder in
ascending order. Key names keep the spelling of the kernel argument
(`Max_seqlen_q` in `attn_fwd`, `max_seqlen_q` in the backward kernels).

### `#! tuning tensor`

The first line is a header of the form `functional,tuning keys-><target>`,
naming what the tensor cells contain:

| Header | Used by | Cell content |
|---|---|---|
| `functional,tuning keys->hsaco_options` | kernels | ID in `.metadata/<iface>/hsaco_options.csv` |
| `functional,tuning keys->precedence` | operators | backend precedence, see [Precedence](#precedence) |

After the header, the file lists one block per functional:

- A line `<functional ID>,` (note the trailing comma) starts the block.
- It is followed by the tensor in row-major order, one line per combination of
  all tuning keys but the last, one cell per bucket of the last key. With the
  two keys above, a block has 10 lines (`Max_seqlen_q` = 16 ... 8192) of 10
  cells each (`Max_seqlen_k` = 16 ... 8192).

The size of each block is inferred from `#! tuning keys`. A blank cell means
the entry is missing.

### Run-Length Encoding

Within a line, `n` consecutive copies of a value `v` may be written `v[n]`:

```
104,104,104,104,104,104,104,104,104,104    is written as    104[10]
```

- Runs never cross lines, so a line still covers exactly one tensor row.
- Blank cells are values too: `[5]` is five blank cells.
- Writers use `v[n]` only where it is shorter than the expanded form, so
  `1,1` stays as is and `1,1,1` becomes `1[3]`. Readers accept either form.

### Precedence

Operator tensors store, per cell, the backends to use in order of preference,
packed into a hexadecimal integer. The packing is described by its own
section:

```
#! precedence
bits_per_slot,4
#! tuning tensor
functional,tuning keys->precedence
1,
12[3],42[7]
...
```

- Each slot holds an ID from `.metadata/<iface>/op_backend.csv`, in
  `bits_per_slot` bits.
- Slots are little endian: the most preferred backend is in the lowest bits.
- ID 0 means "no backend" and ends the list.
- Values are written in lowercase hexadecimal, without a `0x` prefix.

With 4 bits per slot, each hex digit is one slot, read right to left. Example
for `op_attn_bwd`:

| Cell | Slots, most preferred first | Backends |
|---|---|---|
| `12` | 2, 1 | `triton_fuse`, then `triton_split` |
| `42` | 2, 4 | `triton_fuse`, then `flyc` |
| `123` | 3, 2, 1 | `aiter`, then `triton_fuse`, then `triton_split` |

The format records a full precedence of any length. Databases migrated from
sqlite carry at most two slots (the former `best1st`/`best2nd` columns); a
single slot means there was no runner-up.

## Incremental Changes

### Patches

`<iface>.csv.patch<N>`, next to `<iface>.csv`, changes it without touching
the base file:

- A patch uses the same tensor format as its base. Its `#! tuning keys` and
  other metadata sections must equal the base's.
- Each functional listed in the patch overrides the base's cells for that
  functional. A blank cell in the patch keeps the base value, so a patch can
  change a single cell.
- A functional not present in the base is added.
- Patches are applied in increasing order of `N`, so later patches stack on
  earlier ones.

A patch is an overlay rather than a textual diff, so it keeps applying when
its base changes, e.g. when the base architecture is retuned.

### Sharing a Base Through Symlinks

An architecture that is tuned only partially uses another architecture's
tensors as its base, through relative symlinks, and stores its own entries as
patches. gfx1150 and gfx1151 only have their own tuning for hdim=64 with
dropout=0.0, and use gfx1100 for everything else:

```
amd/gfx1150/attn_fwd.csv          -> ../gfx1100/attn_fwd.csv
amd/gfx1150/attn_fwd.csv.patch0      9 functionals tuned on gfx1150
amd/gfx1150/op_attn_bwd.csv       -> ../gfx1100/op_attn_bwd.csv
...
```

The effective gfx1150 database is the gfx1100 one with gfx1150's own entries
on top. Retuning gfx1100 updates gfx1150 automatically, except for the cells
gfx1150 overrides.

### Mods

A GPU mod with its own entries (e.g. `gfx942_mod1`) is stored under
`amd/<arch>/mod<N>/`. It can use the same mechanism: symlink
`amd/<arch>/mod<N>/<iface>.csv` to `../<iface>.csv` and put the differences in
`amd/<arch>/mod<N>/<iface>.csv.patch0`.

## Worked Example

Which variant of `attn_fwd` does gfx1100 use for functional 13 at
`Max_seqlen_q=16`, `Max_seqlen_k=16`?

1. `amd/gfx1100/attn_fwd.csv`, block `13,`, first line, first cell: `67`.
2. `.metadata/attn_fwd/functional.csv`, ID 13:
   `bf16,0,64,0,0,0,0,0,0,0`, i.e. bf16, no causal, hdim 64,
   no dropout, no padded head, no bias.
3. `.metadata/attn_fwd/hsaco_options.csv`, ID 67: `13,11`, i.e. psels 13,
   copts 11.
4. `.metadata/attn_fwd/psels.csv`, ID 13: `PERSISTENT_TYPE=0`,
   `GRID_CU_MULTIP=2`, `BLOCK_M=16`, `BLOCK_N=16`, `PRE_LOAD_V=1`.
5. `.metadata/copts.csv`, ID 11: `waves_per_eu=1`, `num_warps=4`,
   `num_stages=2`.

For gfx1150, `amd/gfx1150/attn_fwd.csv` resolves to the same gfx1100 file, but
`attn_fwd.csv.patch0` lists functional 13 too, and its first cell (`104`)
overrides the gfx1100 value.

## Tooling

`python/database_to_csv.py` converts sqlite tuning databases (as produced by
the tuning infrastructure, see [How To Generate Tuning Database.md](How%20To%20Generate%20Tuning%20Database.md))
into this format:

```bash
python -m python.database_to_csv \
    --kernel_databases <dir>/<arch>/*.sqlite3 \
    --op_database <dir>/op_database.sqlite3 \
    --output modules/flash/database \
    --patch_on gfx1150=gfx1100 gfx1151=gfx1100 \
    --verify
```

- Existing `.metadata/` tables are loaded first, so existing IDs are kept and
  new rows are appended.
- `--patch_on <arch>=<base>` writes `<arch>`'s entries as `.patch0` and
  symlinks the base architecture's tensor files.
- `--verify` reads the result back, with patches applied, and checks that it
  reproduces every row of the sqlite input.
