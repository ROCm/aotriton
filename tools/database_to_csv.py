#!/usr/bin/env python
# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

'''
Convert the sqlite tuning database shards into the human readable (git
diff-able) CSV layout. A one-time migration tool, kept for reference; the
format itself is specified in docs/Human Readable Tuning Database Format.md.

Usage, from the repository root:

    tools/database_to_csv.py --kernel_databases <dir>/<arch>/*.sqlite3 \
        --op_database <dir>/op_database.sqlite3 \
        --output modules/flash/database \
        --patch_on gfx1150=gfx1100 gfx1151=gfx1100 --sparse_patch gfx1151 \
        --verify

Layout:

    <output>/.metadata/copts.csv                       (shared by all triton kernels)
    <output>/.metadata/<iface>/functional.csv
    <output>/.metadata/<iface>/psels.csv               (kernels)
    <output>/.metadata/<iface>/hsaco_options.csv       (kernels, psels,copts ordinal pairs)
    <output>/.metadata/<iface>/op_backend.csv          (operators, name,bord)
    <output>/<vendor>/<arch>/[mod<N>/]<iface>.txtdb      (tuning tensors)
    <output>/<vendor>/<arch>/[mod<N>/]<iface>.<NNNN>-<name>.dbpatch

Ordinals: rows of the .metadata/ tables are referred to by integer ordinals.
An ordinal is implicit: line 0 of every CSV is the header, so the first data
row has ordinal 1. The exception is op_backend.csv, which is small and lists
its ordinals, the bords, explicitly.

functional.csv uses ATI notation, which differs from sqlite: the tensor dtype
column is named after the ATI type variable (T_io, sqlite: Q_dtype), and
dtypes are spelled as ATI declares them without pointer and alignment (fp16,
bf16, fp32; sqlite: torch.float16, torch.bfloat16, torch.float32). See
SQLITE_INPUT_COLUMNS and SQLITE_DTYPES.

Every <iface>.txtdb stores one tensor per functional, with one dimension per
tuning key; any number of tuning keys is allowed. "#!" starts a metadata
section, other "#" lines are comments. For kernels the tensor holds
hsaco_options ordinals:

    #! tuning keys
    Max_seqlen_q,16,32,...
    Max_seqlen_k,16,32,...
    #! tuning tensor
    functional,tuning keys->hsaco_options
    1,
    <one line per Max_seqlen_q, one hsaco_options ordinal per Max_seqlen_k>
    2,
    ...

For operators the tensor holds the backend precedence instead:

    #! precedence
    bits_per_slot,4
    chars_per_slot,1
    #! tuning tensor
    functional,tuning keys->precedence

A precedence is a sequence of slots, each holding a backend ordinal (bord) as
a base62 number (digits 0-9, A-Z, a-z) of chars_per_slot characters, most
significant digit first and zero-padded. With chars_per_slot 1, each character
is one backend and bords range from 1 to 61. Slots are little endian: the
most preferred backend is the last slot, and a precedence is read right to
left. bord 0 is reserved for "no backend" and never written. bords (3, 2, 1),
most preferred first, are written as `123`.

bits_per_slot does not change how a precedence is written. It is the slot
width the code generator uses to pack a precedence into an integer. Every
bord in the file must fit both bits_per_slot and chars_per_slot. With
bits_per_slot 4 and chars_per_slot 1, a precedence is identical to the
uppercase hex of the packed integer.

op_backend.csv maps the name @ati.backend declared to its bord. bords are
private to the tuning database and need not match the C++ backend enum.

Tensor rows are run-length encoded: n consecutive copies of a value v, blank
included, are written as v[n] when that is shorter, e.g. 104[10]. Runs never
cross lines.

The first line of a tensor file is a shebang, #!/usr/bin/env -S python -m
aotriton.database.interpreter. It documents the parser of the file, which is
not executable. Readers skip a first line starting with "#!/".

Incremental changes: <iface>.<N>-<name>.dbpatch next to <iface>.txtdb overlays
it, named like git format-patch output: N is zero-padded to 4 digits, <name>
describes the change. A patch uses the same tensor format, and its tuning keys
and metadata sections must equal the base's. Patches apply in order of N.
Each functional listed in a patch overrides the base's cells, where a blank
patch cell keeps the base value; functionals absent from the base are added.
The base is often another arch's tensor, named by a base reference, e.g.
amd/gfx1150/attn_fwd.txtdb holding only "#! base" and ../gfx1100/attn_fwd.txtdb,
plus amd/gfx1150/attn_fwd.0001-gfx1150.dbpatch with what gfx1150 tuned on its
own.

A patch may store its tensor sparse instead, listing each value once with the
cells it goes to, as 0-based indices into the tuning key ladders:

    #! sparse tuning tensor
    functional,hsaco_options<-coordinates
    13,
    104<-0{0}5{3}
    15<-0{2,3}1{6}2{4,5}3{2-4,6,7}

A group of cells gives one index per tuning key but the last, then a {list}
of indices and inclusive ranges for the last key. A group is thus part of one
line of the dense tensor: 3{2-4,6,7} is row 3, columns 2-4, 6 and 7; with
three tuning keys, 0,3{2,3} is (0,3,2) and (0,3,3). Writers emit one group
per line, ranges for runs of 3 or more, and no spaces; readers accept
spaces.
Cells not listed keep the base value, and a cell may be listed only once.
Only patches may be sparse.
'''

import argparse
import csv
import itertools
import os
import re
import sqlite3
from pathlib import Path

INPUTS = 'inputs$'
OP = 'op$'
PSELS = 'tuned_kernel$'
COPTS = 'compiler_options$'
LUT_KEYS = ('max_seqlen_q', 'max_seqlen_k')
METADATA = '.metadata'
TENSOR_KEYS = '#! tuning keys'
TENSOR_SECTION = '#! tuning tensor'
SPARSE_TENSOR_SECTION = '#! sparse tuning tensor'
SPARSE_FUNCTIONAL = re.compile(r'(\d+),')
SPARSE_VALUE = re.compile(r'([^\s<]+)\s*<-\s*(.*)')
_INDEX_LIST = r'\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*'
# A group of cells: one index per tuning key but the last, then a {list} for
# the last key, e.g. 3{2-4,6,7}, i.e. part of one line of the dense tensor
SPARSE_GROUP = re.compile(rf'(\d+(?:,\d+)*)?\{{({_INDEX_LIST})\}}')
TENSOR_HEADER = 'functional,tuning keys->{}'
SPARSE_TENSOR_HEADER = 'functional,{}<-coordinates'
TENSOR_TARGETS = ('hsaco_options', 'precedence')
PRECEDENCE_SECTION = '#! precedence'
PRECEDENCE_BITS = 4
PRECEDENCE_CHARS = 1
BASE62 = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
PATCH_SUFFIX = '.dbpatch'
# First line of a tensor file: documents the program that parses it
SHEBANG = '#!/usr/bin/env -S python -m aotriton.database.interpreter'
BASE_SECTION = 'base'
RLE_RUN = re.compile(r'(.*)\[(\d+)\]')
# Not consumed by codegen, and 0.0 in every row
UNSTORED_COLUMNS = ['op$tflops']
# sqlite names a tensor axis <signature_name>_dtype and spells its dtype the
# torch way. functional.csv uses the ATI type variable and the ATI dtype.
SQLITE_INPUT_COLUMNS = {'Q_dtype': 'T_io'}
SQLITE_DTYPES = {
    'torch.float16': 'fp16',
    'torch.bfloat16': 'bf16',
    'torch.float32': 'fp32',
}
# functional.csv columns follow the kernel signature, which is also ATI's axis
# order. Operators follow the kernel that owns their functional axes. Columns
# that are not kernel arguments go last.
_FWD_SIGNATURE = ['T_io', 'BLOCK_DMODEL', 'PADDED_HEAD', 'ENABLE_DROPOUT', 'CAUSAL_TYPE',
                  'BIAS_TYPE', 'USE_ALIBI', 'INT8', 'INT8_KV', 'USE_P_SCALE']
_BWD_SIGNATURE = ['T_io', 'BLOCK_DMODEL', 'CAUSAL_TYPE', 'ENABLE_DROPOUT', 'PADDED_HEAD',
                  'BIAS_TYPE']
FUNCTIONAL_COLUMNS = {
    'attn_fwd': _FWD_SIGNATURE,
    'op_attn_fwd': _FWD_SIGNATURE,
    'bwd_kernel_dk_dv': _BWD_SIGNATURE,
    'bwd_kernel_dq': _BWD_SIGNATURE,
    'bwd_kernel_fuse': _BWD_SIGNATURE,
    'op_attn_bwd': _BWD_SIGNATURE,
}
# sqlite op$best1st/op$best2nd store the C++ backend enum value, which is the
# @ati.backend index in modules/flash/aot/__init__.py
SQLITE_OP_BACKENDS = {
    'op_attn_fwd': ['triton', 'aiter', 'flyc'],
    'op_attn_bwd': ['triton_split', 'triton_fuse', 'aiter', 'flyc'],
}

def parse():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--kernel_databases', type=Path, nargs='*', default=[],
                   help='Per-kernel sqlite shards, e.g. amd/gfx950/attn_fwd.sqlite3')
    p.add_argument('--op_database', type=Path, default=None)
    p.add_argument('--arch', default=None, help='Only export this architecture')
    p.add_argument('--vendor', default='amd')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--drop_columns', nargs='*', default=[],
                   help='Columns not to export')
    p.add_argument('--patch_on', nargs='*', default=[],
                   help='<arch>=<base arch>: base references to the base arch\'s files, and '
                        'store <arch> as <iface>.0001-<arch>.dbpatch, '
                        'e.g. gfx1150=gfx1100')
    p.add_argument('--sparse_patch', nargs='*', default=[],
                   help='Arches of --patch_on whose patches are written as '
                        'sparse tuning tensors')
    p.add_argument('--verify', action='store_true',
                   help='Load the CSVs back and compare with the sqlite input')
    return p.parse_args()

def is_lut_key(col):
    return col.removeprefix(INPUTS).lower() in LUT_KEYS

def functional_from_sqlite(row):
    '''
    {sqlite input column without inputs$: value} -> functional.csv row
    '''
    ret = {}
    for c, v in row.items():
        if c in SQLITE_INPUT_COLUMNS:
            c, v = SQLITE_INPUT_COLUMNS[c], SQLITE_DTYPES[v]
        ret[c] = v
    return ret

def functional_to_sqlite(row):
    '''
    Inverse of functional_from_sqlite
    '''
    columns = {v: k for k, v in SQLITE_INPUT_COLUMNS.items()}
    dtypes = {v: k for k, v in SQLITE_DTYPES.items()}
    ret = {}
    for c, v in row.items():
        if c in columns:
            c, v = columns[c], dtypes[v]
        ret[c] = v
    return ret

class OrdinalTable(object):
    '''
    A CSV table with implicit ordinals. Rows are dicts of the columns they
    define.
    '''
    def __init__(self, path):
        self._path = path
        self._columns = []
        self._rows = []
        self._index = {}
        if path.is_file():
            with open(path, newline='') as f:
                reader = csv.reader(f)
                self._columns = next(reader)
                for line in reader:
                    self._append({k: v for k, v in zip(self._columns, line) if v != ''})

    def _append(self, row):
        key = tuple(sorted(row.items()))
        self._rows.append(row)
        self._index[key] = len(self._rows)    # Implicit ordinal, first data row is 1
        return self._index[key]

    def lookup(self, row):
        row = {k: str(v) for k, v in row.items()}
        key = tuple(sorted(row.items()))
        if key in self._index:
            return self._index[key]
        for k in row:
            if k not in self._columns:
                self._columns.append(k)
        return self._append(row)

    def get(self, ordinal):
        return self._rows[ordinal - 1]

    def order_columns(self, columns, order):
        '''
        Set the column order of a new table: columns listed in order first
        (in that order), then the rest. An existing table keeps its order.
        '''
        if self._columns:
            return
        self._columns = [c for c in order if c in columns] + [c for c in columns if c not in order]

    def write(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, 'w', newline='') as f:
            w = csv.writer(f, lineterminator='\n')
            w.writerow(self._columns)
            for row in self._rows:
                w.writerow([row.get(c, '') for c in self._columns])

class OpBackendTable(object):
    '''
    name,bord rows. bord 0 is reserved for "no backend".
    '''
    def __init__(self, path):
        self._path = path
        self._bords = {}
        if path.is_file():
            with open(path, newline='') as f:
                for row in csv.DictReader(f):
                    self._bords[row['name']] = int(row['bord'])
        self._names = {b: n for n, b in self._bords.items()}

    def lookup(self, name):
        if name not in self._bords:
            new_bord = max(self._bords.values(), default=0) + 1
            self._bords[name] = new_bord
            self._names[new_bord] = name
        return self._bords[name]

    def get(self, bord):
        return self._names[bord]

    def bords(self):
        return self._bords.values()

    def write(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, 'w', newline='') as f:
            w = csv.writer(f, lineterminator='\n')
            w.writerow(['name', 'bord'])
            for name, bord in sorted(self._bords.items(), key=lambda x: x[1]):
                w.writerow([name, bord])

class Tables(object):
    '''
    All tables under .metadata/. Per-iface tables are opened on first use.
    '''
    PER_IFACE = ('functional', 'psels', 'hsaco_options', 'op_backend')

    def __init__(self, root):
        self._meta = root / METADATA
        self.copts = OrdinalTable(self._meta / 'copts.csv')
        self._iface = {}

    def __call__(self, iface, name):
        assert name in self.PER_IFACE, name
        key = (iface, name)
        if key not in self._iface:
            cls = OpBackendTable if name == 'op_backend' else OrdinalTable
            self._iface[key] = cls(self._meta / iface / f'{name}.csv')
        return self._iface[key]

    def write(self):
        self.copts.write()
        for t in self._iface.values():
            t.write()

class Exporter(object):
    def __init__(self, args):
        self._args = args
        self.tables = Tables(args.output)
        self.patch_on = dict(x.split('=') for x in args.patch_on)
        self.sparse_patch = set(args.sparse_patch)
        assert self.sparse_patch <= set(self.patch_on), '--sparse_patch arches must be in --patch_on'

    def kernel_csv(self, gpu, iface):
        arch, mod = gpu.split('_')
        d = self._args.output / self._args.vendor / arch
        if mod != 'mod0':
            d = d / mod
        d.mkdir(parents=True, exist_ok=True)
        return d / f'{iface}.txtdb'

    @staticmethod
    def read_table(dbc, table):
        cols = [r[1] for r in dbc.execute(f"PRAGMA table_info('{table}')") if r[1] != 'id']
        quoted = ','.join('"' + c + '"' for c in cols)
        return cols, dbc.execute(f"SELECT {quoted} FROM '{table}'").fetchall()

    def export_table(self, dbc, table, iface):
        cols, rows = self.read_table(dbc, table)
        cols = [c for c in cols if c not in self._args.drop_columns + UNSTORED_COLUMNS]
        icol = {c: i for i, c in enumerate(cols)}
        func_cols = [c for c in cols if c.startswith(INPUTS) and not is_lut_key(c)]
        lut_cols = [c for c in cols if c.startswith(INPUTS) and is_lut_key(c)]
        psel_cols = [c for c in cols if c.startswith(PSELS)]
        copt_cols = [c for c in cols if c.startswith(COPTS)]
        op_cols = [c for c in cols if c.startswith(OP)]
        rest = set(cols) - set(func_cols + lut_cols + psel_cols + copt_cols + op_cols + ['gpu'])
        assert not rest, f'{table}: unknown columns {rest}'
        def sub(row, subcols, prefix):
            return {c.removeprefix(prefix): row[icol[c]] for c in subcols}
        if self._args.arch is not None:
            rows = [r for r in rows if r[icol['gpu']].startswith(self._args.arch + '_')]
        functional = self.tables(iface, 'functional')
        functional.order_columns([SQLITE_INPUT_COLUMNS.get(c.removeprefix(INPUTS), c.removeprefix(INPUTS))
                                  for c in func_cols],
                                 FUNCTIONAL_COLUMNS.get(iface, []))
        # Sort so functional ordinals are assigned deterministically
        func_order = sorted({tuple(r[icol[c]] for c in func_cols) for r in rows})
        for ftuple in func_order:
            functional.lookup(functional_from_sqlite({c.removeprefix(INPUTS): v for c, v in zip(func_cols, ftuple)}))
        if op_cols:
            op_backend = self.tables(iface, 'op_backend')
            for name in SQLITE_OP_BACKENDS[iface]:
                op_backend.lookup(name)
        else:
            psels = self.tables(iface, 'psels')
            hsaco_options = self.tables(iface, 'hsaco_options')
        by_gpu = {}
        for r in rows:
            fid = functional.lookup(functional_from_sqlite(sub(r, func_cols, INPUTS)))
            lut = tuple(r[icol[c]] for c in lut_cols)
            if op_cols:
                value = self.precedence(op_backend, iface,
                                        r[icol[OP + 'best1st']], r[icol[OP + 'best2nd']])
            else:
                value = hsaco_options.lookup({
                    'psels': psels.lookup(sub(r, psel_cols, PSELS)),
                    'copts': self.tables.copts.lookup(sub(r, copt_cols, COPTS)),
                })
            by_gpu.setdefault(r[icol['gpu']], []).append((fid, lut, value))
        lut_names = [c.removeprefix(INPUTS) for c in lut_cols]
        for gpu, entries in by_gpu.items():
            fn = self.kernel_csv(gpu, iface)
            ladders = None
            sparse = False
            if fn.parent.name in self.patch_on:
                # A patch shares the base's ladders, whichever cells it sets
                base = self._args.output / self._args.vendor / self.patch_on[fn.parent.name] / fn.name
                ladders = [[int(v) for v in ladder] for _, ladder in load_tensor(base).keys]
                sparse = fn.parent.name in self.sparse_patch
                fn = patch_name(fn, 1, fn.parent.name)
            with open(fn, 'w', newline='') as f:
                if op_cols:
                    assert all(b < (1 << PRECEDENCE_BITS) for b in op_backend.bords()), \
                        f'{iface}: bords do not fit bits_per_slot {PRECEDENCE_BITS}'
                    write_tensor(f, lut_names, entries, 'precedence',
                                 {PRECEDENCE_SECTION: [['bits_per_slot', PRECEDENCE_BITS],
                                                       ['chars_per_slot', PRECEDENCE_CHARS]]},
                                 ladders=ladders, sparse=sparse)
                else:
                    write_tensor(f, lut_names, entries, 'hsaco_options', ladders=ladders, sparse=sparse)
            print(f'{fn}: {len(entries)} entries')

    @staticmethod
    def precedence(op_backend, iface, best1st, best2nd):
        # best2nd repeats best1st when there is no runner-up
        backends = [best1st] if best2nd == best1st else [best1st, best2nd]
        names = [SQLITE_OP_BACKENDS[iface][b] for b in backends]
        return encode_precedence([op_backend.lookup(n) for n in names])

    def sqlite_tables(self, dbf):
        dbc = sqlite3.connect(dbf)
        for table, in dbc.execute("SELECT name FROM sqlite_master WHERE type='table'"):
            _, iface = table.split('$')
            yield dbc, table, iface

    def run(self):
        dbfs = list(self._args.kernel_databases)
        if self._args.op_database:
            dbfs.append(self._args.op_database)
        for dbf in dbfs:
            for dbc, table, iface in self.sqlite_tables(dbf):
                self.export_table(dbc, table, iface)
        self.tables.write()
        self.link_bases()

    def link_bases(self):
        '''
        For each --patch_on arch, a base reference to every <iface>.txtdb of
        its base arch. An arch without sqlite data only gets the references.
        '''
        vendor_dir = self._args.output / self._args.vendor
        for arch, base in self.patch_on.items():
            (vendor_dir / arch).mkdir(parents=True, exist_ok=True)
            for target in sorted((vendor_dir / base).glob('*.txtdb')):
                ref = vendor_dir / arch / target.name
                rel = (Path('..') / base / target.name).as_posix()
                content = f'{SHEBANG}\n#! {BASE_SECTION}\n{rel}\n'
                if ref.is_file() and not ref.is_symlink() and ref.read_text() == content:
                    continue
                ref.unlink(missing_ok=True)
                ref.write_text(content)
                print(f'{ref}: base {rel}')

def encode_precedence(bords, chars=PRECEDENCE_CHARS):
    '''
    bords, most preferred first -> base62 string, most preferred last, with
    chars characters per slot
    '''
    slots = []
    for b in bords:
        assert 0 < b < len(BASE62) ** chars, f'bord {b} does not fit {chars} base62 characters'
        digits = ''
        for _ in range(chars):
            b, d = divmod(b, len(BASE62))
            digits = BASE62[d] + digits
        slots.append(digits)
    return ''.join(reversed(slots))

def decode_precedence(text, chars=PRECEDENCE_CHARS):
    '''
    Inverse of encode_precedence
    '''
    assert text, 'empty precedence'
    assert len(text) % chars == 0, f'precedence {text} is not a multiple of {chars} characters'
    bords = []
    for i in reversed(range(0, len(text), chars)):
        b = 0
        for c in text[i:i + chars]:
            assert c in BASE62, f'{c!r} in precedence {text} is not a base62 digit'
            b = b * len(BASE62) + BASE62.index(c)
        bords.append(b)
    assert 0 not in bords, f'"no backend" inside precedence {text}'
    return bords

def rle_encode(cells):
    out = []
    for value, run in itertools.groupby(str(c) for c in cells):
        n = len(list(run))
        packed = f'{value}[{n}]'
        # n values take n - 1 separating commas
        out += [packed] if len(packed) < n * len(value) + n - 1 else [value] * n
    return out

def rle_decode(fields):
    out = []
    for field in fields:
        m = RLE_RUN.fullmatch(field)
        if m:
            out += [m[1]] * int(m[2])
        else:
            out.append(field)
    return out

def write_tensor(f, lut_names, entries, target, metadata={}, ladders=None, sparse=False):
    '''
    entries: [(functional ordinal, tuning key values, value of target)]
    metadata: {section: [fields]}, written before the tuning tensor
    ladders: the tuning key ladders, by default the values found in entries
    Each functional is followed by its tuning tensor in row-major order, one
    line per combination of all tuning keys but the last. Missing entries are
    left blank. With 3 or more tuning keys, a blank line separates the slices
    of the keys before the last two. With sparse, each functional lists its values instead, in the
    order of their first cell, each with its cells in row-major order.
    '''
    if ladders is None:
        ladders = [sorted({lut[i] for _, lut, _ in entries}) for i in range(len(lut_names))]
    cells = {}
    for fid, lut, tid in entries:
        cells.setdefault(fid, {})[lut] = tid
    print(SHEBANG, file=f)
    print(TENSOR_KEYS, file=f)
    for name, ladder in zip(lut_names, ladders):
        print(','.join([name] + [str(v) for v in ladder]), file=f)
    for section, lines in metadata.items():
        print(section, file=f)
        for fields in lines:
            print(','.join(str(x) for x in fields), file=f)
    print(SPARSE_TENSOR_SECTION if sparse else TENSOR_SECTION, file=f)
    print((SPARSE_TENSOR_HEADER if sparse else TENSOR_HEADER).format(target), file=f)
    for fid in sorted(cells):
        print(f'{fid},', file=f)
        if sparse:
            groups = {}
            for index in itertools.product(*(range(len(ladder)) for ladder in ladders)):
                lut = tuple(ladder[i] for ladder, i in zip(ladders, index))
                if lut in cells[fid]:
                    groups.setdefault(cells[fid][lut], []).append(index)
            for value, indices in groups.items():
                print(f'{value}<-{format_sparse_cells(indices)}', file=f)
            continue
        for n, prefix in enumerate(itertools.product(*ladders[:-1])):
            if len(ladders) > 2 and n > 0 and n % len(ladders[-2]) == 0:
                print(file=f)
            line = [cells[fid].get((*prefix, v), '') for v in ladders[-1]]
            print(','.join(rle_encode(line)), file=f)

def format_index_list(indices):
    '''
    Sorted indices -> "2-4,6,7": runs of 3 or more become inclusive ranges
    '''
    parts = []
    i = 0
    while i < len(indices):
        j = i
        while j + 1 < len(indices) and indices[j + 1] == indices[j] + 1:
            j += 1
        if j - i >= 2:
            parts.append(f'{indices[i]}-{indices[j]}')
        else:
            parts += [str(x) for x in indices[i:j + 1]]
        i = j + 1
    return ','.join(parts)

def format_sparse_cells(indices):
    '''
    Cells in row-major order -> groups, one per combination of all tuning keys
    but the last, i.e. per dense line, e.g. [(0,2),(0,3),(3,2)] -> "0{2,3}3{2}"
    '''
    groups = {}
    for index in indices:
        groups.setdefault(index[:-1], []).append(index[-1])
    return ''.join(','.join(map(str, prefix)) + f'{{{format_index_list(last)}}}'
                   for prefix, last in groups.items())

def parse_index_list(text):
    '''
    "2-4,6,7" -> [2, 3, 4, 6, 7]
    '''
    indices = []
    for part in text.split(','):
        a, _, b = part.partition('-')
        a, b = int(a), int(b or a)
        assert a <= b, f'empty range {part}'
        indices += range(a, b + 1)
    return indices

def parse_sparse_cells(text):
    '''
    Inverse of format_sparse_cells, also accepting the groups of a line in
    any order and repeated. Yields index tuples.
    '''
    text = ''.join(text.split())    # readers tolerate whitespace
    groups = list(SPARSE_GROUP.finditer(text))
    assert groups and ''.join(m[0] for m in groups) == text, f'malformed cells {text}'
    for m in groups:
        prefix = tuple(int(i) for i in m[1].split(',')) if m[1] else ()
        for last in parse_index_list(m[2]):
            yield (*prefix, last)

def read_sections(fn):
    '''
    Split a file into {section: [fields]}. "#!" starts a metadata section,
    other "#" lines are comments. A first line starting with "#!/" is a
    shebang and is skipped; section names never start with "/".
    '''
    sections = {}
    current = None
    with open(fn) as f:
        for n, line in enumerate(f):
            line = line.rstrip('\n')
            if n == 0 and line.startswith('#!/'):
                continue
            if line.startswith('#!'):
                current = line[2:].strip()
                continue
            if line.startswith('#') or not line.strip():
                continue
            sections.setdefault(current, []).append(line.split(','))
    return sections

def patch_name(fn, n, name):
    # git format-patch style: <iface>.<NNNN>-<name>.dbpatch
    return fn.with_name(f'{fn.stem}.{n:04d}-{name}{PATCH_SUFFIX}')

def patch_files(fn):
    '''
    The patches of <iface>.txtdb, in order of N
    '''
    pattern = re.compile(re.escape(fn.stem) + r'\.(\d{4,})-(.+)' + re.escape(PATCH_SUFFIX))
    found = {}
    for p in fn.parent.glob(f'{fn.stem}.*{PATCH_SUFFIX}'):
        m = pattern.fullmatch(p.name)
        assert m, f'{p}: a patch must be named <iface>.<NNNN>-<name>{PATCH_SUFFIX}'
        n = int(m[1])
        assert n not in found, f'{p} and {found.get(n)} have the same patch number'
        found[n] = p
    return [found[n] for n in sorted(found)]

class Tensor(object):
    '''
    A parsed tensor CSV: tuning keys, metadata sections, target, and
    {functional ordinal: [[cell]]} with one row per combination of all tuning keys
    but the last.
    '''
    def __init__(self, fn):
        self.fn = fn
        sections = read_sections(fn)
        self.keys = [(line[0], line[1:]) for line in sections.pop('tuning keys')]
        self.sparse = 'sparse tuning tensor' in sections
        assert not (self.sparse and 'tuning tensor' in sections), f'{fn}: both dense and sparse tensors'
        header, *lines = sections.pop('sparse tuning tensor' if self.sparse else 'tuning tensor')
        headers = {(SPARSE_TENSOR_HEADER if self.sparse else TENSOR_HEADER).format(t): t
                   for t in TENSOR_TARGETS}
        assert ','.join(header) in headers, f'{fn}: unsupported tensor header {",".join(header)}'
        self.target = headers[','.join(header)]
        self.metadata = sections
        nrows = 1
        for _, ladder in self.keys[:-1]:
            nrows *= len(ladder)
        self.cells = {}
        if self.sparse:
            self._parse_sparse(lines, nrows)
            return
        for i in range(0, len(lines), nrows + 1):
            fline = lines[i]
            assert len(fline) == 2 and fline[1] == '', f'{fn}: expect "<functional>," got {fline}'
            rows = [rle_decode(row) for row in lines[i + 1:i + 1 + nrows]]
            assert len(rows) == nrows, f'{fn}: functional {fline[0]} is truncated'
            for row in rows:
                assert len(row) == len(self.keys[-1][1]), f'{fn}: functional {fline[0]} has a malformed row {row}'
            self.cells[int(fline[0])] = rows

    def _parse_sparse(self, lines, nrows):
        sizes = [len(ladder) for _, ladder in self.keys]
        rows = None
        for fields in lines:
            text = ','.join(fields)   # read_sections split the line on commas
            m = SPARSE_FUNCTIONAL.fullmatch(text)
            if m:
                rows = [[''] * sizes[-1] for _ in range(nrows)]
                self.cells[int(m[1])] = rows
                continue
            m = SPARSE_VALUE.fullmatch(text)
            assert m and rows is not None, f'{self.fn}: expect "<functional>," or "<value><-<cells>" got {text}'
            value = m[1]
            try:
                indices = list(parse_sparse_cells(m[2]))
            except AssertionError as e:
                raise AssertionError(f'{self.fn}: {e}') from e
            for index in indices:
                assert len(index) == len(sizes), \
                    f'{self.fn}: cell {tuple(index)} has {len(index)} indices, expected {len(sizes)}'
                assert all(0 <= i < n for i, n in zip(index, sizes)), \
                    f'{self.fn}: cell {tuple(index)} outside the tuning keys'
                row = 0
                for i, n in zip(index[:-1], sizes[:-1]):
                    row = row * n + i
                assert rows[row][index[-1]] == '', f'{self.fn}: cell {tuple(index)} listed twice'
                rows[row][index[-1]] = value

    def apply(self, patch):
        assert patch.keys == self.keys, f'{patch.fn}: tuning keys differ from {self.fn}'
        assert patch.target == self.target, f'{patch.fn}: target differs from {self.fn}'
        assert patch.metadata == self.metadata, f'{patch.fn}: metadata differs from {self.fn}'
        for fid, rows in patch.cells.items():
            if fid not in self.cells:
                self.cells[fid] = rows
                continue
            self.cells[fid] = [[p if p != '' else b for p, b in zip(prow, brow)]
                               for prow, brow in zip(rows, self.cells[fid])]

def base_reference(fn):
    '''
    The file named by the "#! base" section of <iface>.txtdb fn, or None
    '''
    sections = read_sections(fn)
    if BASE_SECTION not in sections:
        return None
    (line,) = sections[BASE_SECTION]
    return Path(os.path.normpath(Path(fn).parent / ','.join(line)))

def load_tensor(fn):
    '''
    Load <iface>.txtdb and apply its patches. A base reference loads its base,
    with the base's patches, then applies its own.
    '''
    base = base_reference(fn)
    if base is not None:
        tensor = load_tensor(base)
        for patch in patch_files(fn):
            tensor.apply(Tensor(patch))
        return tensor
    tensor = Tensor(fn)
    assert not tensor.sparse, f'{fn}: only patches may be sparse'
    for patch in patch_files(fn):
        tensor.apply(Tensor(patch))
    return tensor

def load_tensor_csv(fn, tables):
    '''
    Expand a tensor CSV, with its patches applied, back into rows using the
    sqlite column names. Used for --verify; can serve as the reference reader
    for codegen.
    '''
    iface = fn.stem
    arch_dir = fn.parent
    mod = 'mod0'
    if arch_dir.name.startswith('mod'):
        mod = arch_dir.name
        arch_dir = arch_dir.parent
    gpu = f'{arch_dir.name}_{mod}'
    tensor = load_tensor(fn)
    def functional(fid):
        row = functional_to_sqlite(tables(iface, 'functional').get(fid))
        return {INPUTS + c: x for c, x in row.items()}
    def hsaco_options(tid):
        hsaco = tables(iface, 'hsaco_options').get(int(tid))
        return {
            **{PSELS + c: x for c, x in tables(iface, 'psels').get(int(hsaco['psels'])).items()},
            **{COPTS + c: x for c, x in tables.copts.get(int(hsaco['copts'])).items()},
        }
    def precedence(text):
        section = dict(tensor.metadata['precedence'])
        bits = int(section['bits_per_slot'])
        bords = decode_precedence(text, int(section['chars_per_slot']))
        assert all(b < (1 << bits) for b in bords), f'{fn}: precedence {text} does not fit bits_per_slot {bits}'
        names = [tables(iface, 'op_backend').get(b) for b in bords]
        backends = [str(SQLITE_OP_BACKENDS[iface].index(n)) for n in names]
        return {OP + 'best1st': backends[0], OP + 'best2nd': backends[min(1, len(backends) - 1)]}
    expand = {'hsaco_options': hsaco_options, 'precedence': precedence}[tensor.target]
    names = [INPUTS + name for name, _ in tensor.keys]
    ladders = [ladder for _, ladder in tensor.keys]
    for fid, rows in sorted(tensor.cells.items()):
        frow = {'gpu': gpu, **functional(fid)}
        for prefix, row in zip(itertools.product(*ladders[:-1]), rows, strict=True):
            for last, tid in zip(ladders[-1], row, strict=True):
                if tid == '':
                    continue
                out = dict(frow)
                out.update(zip(names, (*prefix, last)))
                out.update(expand(tid))
                yield out

def verify(args):
    '''
    Compare every (iface, gpu) in the sqlite input with the CSV tree. A
    patched arch is expected to hold its base arch's rows, overridden by its
    own sqlite rows.
    '''
    root = args.output
    tables = Tables(root)
    patch_on = dict(x.split('=') for x in args.patch_on)
    dbfs = list(args.kernel_databases) + ([args.op_database] if args.op_database else [])
    columns = {}
    expected = {}
    for dbf in dbfs:
        dbc = sqlite3.connect(dbf)
        for table, in dbc.execute("SELECT name FROM sqlite_master WHERE type='table'"):
            _, iface = table.split('$')
            cols, rows = Exporter.read_table(dbc, table)
            cols = [c for c in cols if c not in args.drop_columns + UNSTORED_COLUMNS]
            assert columns.setdefault(iface, cols) == cols, f'{table}: columns differ across databases'
            gi = cols.index('gpu')
            for r in rows:
                if args.arch is not None and not r[gi].startswith(args.arch + '_'):
                    continue
                expected.setdefault((iface, r[gi]), {})[tuple(str(r[cols.index(c)]) for c in cols)] = None
    for arch, base in patch_on.items():
        for iface, cols in columns.items():
            nkeys = sum(1 for c in cols if c == 'gpu' or c.startswith(INPUTS))
            # The merge keys on the leading gpu + inputs$ columns
            assert all(c == 'gpu' or c.startswith(INPUTS) for c in cols[:nkeys]), cols
            gpu = f'{arch}_mod0'
            merged = {}
            for row in expected.get((iface, f'{base}_mod0'), {}):
                merged[(gpu, *row[1:nkeys])] = (gpu, *row[1:])
            for row in expected.get((iface, gpu), {}):
                merged[row[:nkeys]] = row
            if merged:
                expected[(iface, gpu)] = dict.fromkeys(merged.values())
    ok = True
    for (iface, gpu), rows in sorted(expected.items()):
        arch, mod = gpu.split('_')
        d = root / args.vendor / arch
        if mod != 'mod0':
            d = d / mod
        fn = d / f'{iface}.txtdb'
        cols = columns[iface]
        actual = set()
        if fn.exists():
            actual = {tuple(row[c] for c in cols) for row in load_tensor_csv(fn, tables)}
        match = set(rows) == actual
        ok = ok and match
        note = ' (patched)' if arch in patch_on else ''
        print(f'verify {iface} {gpu}{note}: {len(rows)} rows {"OK" if match else "MISMATCH"}')
    assert ok, 'Round trip mismatch'

def main():
    args = parse()
    Exporter(args).run()
    if args.verify:
        verify(args)

if __name__ == '__main__':
    main()
