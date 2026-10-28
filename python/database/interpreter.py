# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

'''
Interpreter of the tuning tensor files of the human readable tuning database,
see docs/Human Readable Tuning Database Format.md.

Tensor files (`<iface>.csv`) and patches (`<iface>.<NNNN>-<name>.dbpatch`) start
with a shebang naming this module as their parser (the files are not
executable). It exports a file's tensors:

    python -m aotriton.database.interpreter amd/gfx942/attn_fwd.csv --output x.npz

A base reference such as amd/gfx1150/attn_fwd.csv ("#! base" naming
../gfx1100/attn_fwd.csv) exports gfx1100's tensor with gfx1150's patches. A
patch exports alone, as a sparse tensor.

Library:

    load(path)              <iface>.csv, with its patches applied -> TuningTensor
    load(patch)             .dbpatch alone -> SparsePatch
    load(patch, base=base)  .dbpatch applied into base
    apply(base, patch)      SparsePatch applied into base, in place
    save_npz(obj, path)     TuningTensor or SparsePatch -> .npz

Cells are hsaco_options ordinals for kernels. For operators they are the
precedence packed into an integer: each bord takes bits_per_slot bits, the most
preferred one in the lowest bits. 0 marks a missing cell in both, since
ordinals start at 1 and bord 0 means "no backend". Cells must fit 63 bits. The
dtype of a loaded tensor is the smallest unsigned integer type that holds
every cell.

Only numpy is needed, and importing this module has no side effects.
'''

import argparse
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

TENSOR_KEYS = 'tuning keys'
TENSOR_SECTION = 'tuning tensor'
SPARSE_TENSOR_SECTION = 'sparse tuning tensor'
PRECEDENCE_SECTION = 'precedence'
BASE_SECTION = 'base'
TENSOR_HEADER = 'functional,tuning keys->{}'
SPARSE_TENSOR_HEADER = 'functional,{}<-coordinates'
TENSOR_TARGETS = ('hsaco_options', 'precedence')
PATCH_SUFFIX = '.dbpatch'
BASE62 = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
RLE_RUN = re.compile(r'(.*)\[(\d+)\]')
ORDINAL = re.compile(r'[0-9]+')
SPARSE_FUNCTIONAL = re.compile(r'([0-9]+),')
SPARSE_VALUE = re.compile(r'([^\s<]+)\s*<-\s*(.*)')
_INDEX_LIST = r'\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*'
# A group of cells: one index per tuning key but the last, then a {list} for
# the last key, e.g. 3{2-4,6,7}, i.e. part of one line of the dense tensor
SPARSE_GROUP = re.compile(rf'(\d+(?:,\d+)*)?\{{({_INDEX_LIST})\}}')
MAX_CELL = (1 << 63) - 1
# Names in the .npz that a tuning key must not shadow
NPZ_RESERVED = ('tensor', 'functional', 'sparse', 'tuning_keys', 'target',
                'bits_per_slot', 'chars_per_slot')


class TensorFileError(ValueError):
    pass


@dataclass(kw_only=True)
class _Layout:
    '''
    What a tensor and a patch must share for the patch to apply
    '''
    tuning_keys: list[str]
    ladders: list[tuple[int, ...]]
    target: str
    bits_per_slot: int | None
    chars_per_slot: int | None
    # {section: [fields]} of all sections but the tensor itself
    metadata: dict = field(repr=False)


@dataclass(kw_only=True)
class TuningTensor(_Layout):
    '''
    functional: 1-D, the functional ordinal at each index of axis 0 of tensor
    tensor: rank 1 + number of tuning keys, 0 = missing cell
    '''
    functional: np.ndarray
    tensor: np.ndarray


@dataclass(kw_only=True)
class SparsePatch(_Layout):
    '''
    sparse: N x (1 + number of tuning keys + 1) integer matrix, one row per
    cell, (functional ordinal, index into each ladder, value)
    '''
    sparse: np.ndarray


def _fail(fn, msg):
    raise TensorFileError(f'{fn}: {msg}')


def encode_precedence(bords, chars=1):
    '''
    bords, most preferred first -> base62 string, most preferred last, with
    chars characters per slot. The inverse of decode_precedence.
    '''
    slots = []
    for b in bords:
        if not 0 < b < len(BASE62) ** chars:
            raise ValueError(f'bord {b} does not fit {chars} base62 characters')
        digits = ''
        for _ in range(chars):
            b, d = divmod(b, len(BASE62))
            digits = BASE62[d] + digits
        slots.append(digits)
    return ''.join(reversed(slots))


def decode_precedence(text, chars=1):
    '''
    base62 string -> bords, most preferred first
    '''
    if not text:
        raise ValueError('empty precedence')
    if len(text) % chars:
        raise ValueError(f'precedence {text} is not a multiple of {chars} characters')
    bords = []
    for i in reversed(range(0, len(text), chars)):
        b = 0
        for c in text[i:i + chars]:
            if c not in BASE62:
                raise ValueError(f'{c!r} in precedence {text} is not a base62 digit')
            b = b * len(BASE62) + BASE62.index(c)
        bords.append(b)
    if 0 in bords:
        raise ValueError(f'"no backend" inside precedence {text}')
    return bords


def pack_precedence(bords, bits):
    '''
    bords, most preferred first -> integer, the most preferred in the lowest bits
    '''
    packed = 0
    for rank, b in enumerate(bords):
        if not 0 < b < (1 << bits):
            raise ValueError(f'bord {b} does not fit bits_per_slot {bits}')
        packed |= b << (rank * bits)
    return packed


def rle_decode(fields):
    out = []
    for f in fields:
        m = RLE_RUN.fullmatch(f)
        if m:
            out += [m[1]] * int(m[2])
        else:
            out.append(f)
    return out


def parse_index_list(text):
    '''
    "2-4,6,7" -> [2, 3, 4, 6, 7]
    '''
    indices = []
    for part in text.split(','):
        a, _, b = part.partition('-')
        a, b = int(a), int(b or a)
        if a > b:
            raise ValueError(f'empty range {part}')
        indices += range(a, b + 1)
    return indices


def parse_sparse_cells(text):
    '''
    "0{2,3}3{2}" -> index tuples (0, 2), (0, 3), (3, 2)
    '''
    text = ''.join(text.split())    # readers tolerate whitespace
    groups = list(SPARSE_GROUP.finditer(text))
    if not groups or ''.join(m[0] for m in groups) != text:
        raise ValueError(f'malformed cells {text}')
    for m in groups:
        prefix = tuple(int(i) for i in m[1].split(',')) if m[1] else ()
        for last in parse_index_list(m[2]):
            yield (*prefix, last)


def smallest_uint(maxval):
    for dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
        if maxval <= np.iinfo(dtype).max:
            return np.dtype(dtype)
    raise ValueError(f'{maxval} does not fit an unsigned 64-bit integer')


def read_sections(fn):
    '''
    Split a file into {section: [fields]}. "#!" starts a metadata section,
    other "#" lines are comments. A first line starting with "#!/" is a
    shebang and is skipped; section names never start with "/".
    '''
    sections = {}
    current = None
    with open(fn, encoding='utf-8') as f:
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


def patch_files(fn):
    '''
    The patches of <iface>.csv, in order of <NNNN>, looked up next to fn.
    '''
    fn = Path(fn)
    pattern = re.compile(re.escape(fn.stem) + r'\.(\d{4,})-(.+)' + re.escape(PATCH_SUFFIX))
    found = {}
    for p in fn.parent.glob(f'{fn.stem}.*{PATCH_SUFFIX}'):
        m = pattern.fullmatch(p.name)
        if not m:
            _fail(p, f'a patch must be named <iface>.<NNNN>-<name>{PATCH_SUFFIX}')
        n = int(m[1])
        if n in found:
            _fail(p, f'{found[n]} has the same patch number')
        found[n] = p
    return [found[n] for n in sorted(found)]


def _parse_layout(fn, sections):
    '''
    Pop the tuning keys, the tensor section and the precedence section off
    sections. Returns (layout kwargs, sparse, tensor lines).
    '''
    if TENSOR_KEYS not in sections:
        _fail(fn, f'no "#! {TENSOR_KEYS}" section')
    names, ladders = [], []
    for line in sections.pop(TENSOR_KEYS):
        name, *values = line
        try:
            ladder = tuple(int(v) for v in values)
        except ValueError:
            _fail(fn, f'tuning key {name} has a ladder that is not integers: {values}')
        if not ladder or any(a >= b for a, b in zip(ladder, ladder[1:])):
            _fail(fn, f'tuning key {name} needs a non-empty, strictly ascending ladder')
        if name in names:
            _fail(fn, f'duplicate tuning key {name}')
        names.append(name)
        ladders.append(ladder)
    if not names:
        _fail(fn, f'empty "#! {TENSOR_KEYS}" section')
    sparse = SPARSE_TENSOR_SECTION in sections
    if sparse and TENSOR_SECTION in sections:
        _fail(fn, 'both dense and sparse tensors')
    section = SPARSE_TENSOR_SECTION if sparse else TENSOR_SECTION
    if section not in sections:
        _fail(fn, f'no "#! {TENSOR_SECTION}" section')
    header, *lines = sections.pop(section)
    headers = {(SPARSE_TENSOR_HEADER if sparse else TENSOR_HEADER).format(t): t
               for t in TENSOR_TARGETS}
    if ','.join(header) not in headers:
        _fail(fn, f'unsupported tensor header {",".join(header)}')
    target = headers[','.join(header)]
    bits = chars = None
    if target == 'precedence':
        entries = {}
        for line in sections.get(PRECEDENCE_SECTION, []):
            if len(line) != 2 or line[0] in entries:
                _fail(fn, f'malformed "#! {PRECEDENCE_SECTION}" entry {",".join(line)}')
            entries[line[0]] = line[1]
        if set(entries) != {'bits_per_slot', 'chars_per_slot'}:
            _fail(fn, f'"#! {PRECEDENCE_SECTION}" needs bits_per_slot and chars_per_slot')
        try:
            bits, chars = int(entries['bits_per_slot']), int(entries['chars_per_slot'])
        except ValueError:
            _fail(fn, f'"#! {PRECEDENCE_SECTION}" entries are not integers')
        if bits < 1 or chars < 1:
            _fail(fn, f'"#! {PRECEDENCE_SECTION}" entries must be positive')
    elif PRECEDENCE_SECTION in sections:
        _fail(fn, f'"#! {PRECEDENCE_SECTION}" in a {target} tensor')
    layout = dict(tuning_keys=names, ladders=ladders, target=target,
                  bits_per_slot=bits, chars_per_slot=chars, metadata=sections)
    return layout, sparse, lines


def _cell_parser(fn, layout):
    '''
    Returns text -> int, for hsaco_options ordinals and precedences, cached
    '''
    target, bits, chars = layout['target'], layout['bits_per_slot'], layout['chars_per_slot']
    cache = {}
    def parse(text):
        value = cache.get(text)
        if value is not None:
            return value
        try:
            if target == 'hsaco_options':
                if not ORDINAL.fullmatch(text) or int(text) < 1:
                    raise ValueError(f'{text!r} is not an hsaco_options ordinal')
                value = int(text)
            else:
                bords = decode_precedence(text, chars)
                value = pack_precedence(bords, bits)
            if value > MAX_CELL:
                raise ValueError(f'{text} does not fit 63 bits')
        except ValueError as e:
            _fail(fn, str(e))
        cache[text] = value
        return value
    return parse


def _parse_dense(fn, layout, lines):
    '''
    -> (functional ordinals, array of 1 + number of tuning keys dimensions)
    '''
    parse = _cell_parser(fn, layout)
    ladders = layout['ladders']
    nrows = 1
    for ladder in ladders[:-1]:
        nrows *= len(ladder)
    functional, blocks = [], []
    for i in range(0, len(lines), nrows + 1):
        fline = lines[i]
        if len(fline) != 2 or fline[1] != '' or not ORDINAL.fullmatch(fline[0]):
            _fail(fn, f'expect "<functional>," got {",".join(fline)}')
        fid = int(fline[0])
        if fid in functional:
            _fail(fn, f'functional {fid} listed twice')
        rows = [rle_decode(row) for row in lines[i + 1:i + 1 + nrows]]
        if len(rows) != nrows:
            _fail(fn, f'functional {fid} is truncated')
        for row in rows:
            if len(row) != len(ladders[-1]):
                _fail(fn, f'functional {fid} has a malformed row {row}')
        functional.append(fid)
        blocks.append([[parse(c) if c != '' else 0 for c in row] for row in rows])
    cells = np.array(blocks, dtype=np.int64).reshape(len(functional), *[len(ladder) for ladder in ladders])
    return np.array(functional, dtype=np.int64), cells


def _parse_sparse(fn, layout, lines):
    '''
    -> N x (1 + number of tuning keys + 1) matrix. Functionals that list no
    cell carry no data and are dropped.
    '''
    parse = _cell_parser(fn, layout)
    sizes = tuple(len(ladder) for ladder in layout['ladders'])
    seen = set()
    rows = []
    cells = None
    for fields in lines:
        text = ','.join(fields)     # read_sections split the line on commas
        m = SPARSE_FUNCTIONAL.fullmatch(text)
        if m:
            fid = int(m[1])
            if fid in seen:
                _fail(fn, f'functional {fid} listed twice')
            seen.add(fid)
            cells = set()
            continue
        m = SPARSE_VALUE.fullmatch(text)
        if not m or cells is None:
            _fail(fn, f'expect "<functional>," or "<value><-<cells>" got {text}')
        value = parse(m[1])
        try:
            indices = list(parse_sparse_cells(m[2]))
        except ValueError as e:
            _fail(fn, str(e))
        for index in indices:
            if len(index) != len(sizes):
                _fail(fn, f'cell {index} has {len(index)} indices, expected {len(sizes)}')
            if not all(0 <= i < n for i, n in zip(index, sizes)):
                _fail(fn, f'cell {index} outside the tuning keys')
            if index in cells:
                _fail(fn, f'cell {index} listed twice')
            cells.add(index)
            rows.append((fid, *index, value))
    return np.array(rows, dtype=np.int64).reshape(-1, len(sizes) + 2)


def _dense_to_sparse(functional, cells):
    '''
    Dense patch, blank (0) = keep -> the matrix of the cells it sets
    '''
    nz = np.nonzero(cells)
    return np.column_stack([functional[nz[0]], *nz[1:], cells[nz]]).astype(np.int64)


def _check_compatible(base, patch, fn=None):
    fn = fn or 'patch'
    if base.tuning_keys != patch.tuning_keys or base.ladders != patch.ladders:
        _fail(fn, 'tuning keys differ from the base')
    if base.target != patch.target:
        _fail(fn, 'target differs from the base')
    if base.metadata != patch.metadata:
        _fail(fn, 'metadata differs from the base')


def apply(base, patch):
    '''
    Update base in place with the cells of patch. Functionals new to base are
    appended along axis 0 first, which reallocates base.tensor, as does a
    patch value too large for its dtype. Returns base.
    '''
    _check_compatible(base, patch)
    s = patch.sparse
    sizes = base.tensor.shape[1:]
    if s.ndim != 2 or s.shape[1] != len(sizes) + 2:
        raise TensorFileError(f'sparse tensor needs {len(sizes) + 2} columns')
    if s.shape[0] == 0:
        return base
    index, values = s[:, 1:-1], s[:, -1]
    if (index < 0).any() or (index >= np.array(sizes)).any():
        raise TensorFileError('sparse tensor has cells outside the tuning keys')
    if (values < 1).any():
        raise TensorFileError('sparse tensor has cells without a value')
    fids = s[:, 0]
    # new functionals, in order of first appearance
    _, first = np.unique(fids, return_index=True)
    new = fids[np.sort(first)]
    new = new[~np.isin(new, base.functional)]
    dtype = base.tensor.dtype
    maxval = int(values.max())
    if maxval > np.iinfo(dtype).max:
        dtype = smallest_uint(maxval)
    if len(new) or dtype != base.tensor.dtype:
        base.tensor = np.concatenate([base.tensor.astype(dtype, copy=False),
                                      np.zeros((len(new), *sizes), dtype=dtype)])
        base.functional = np.concatenate([base.functional, new])
    order = np.argsort(base.functional)
    f = order[np.searchsorted(base.functional, fids, sorter=order)]
    base.tensor[(f, *index.T)] = values
    return base


def _narrow(tensor):
    '''
    Smallest unsigned dtype that holds every cell
    '''
    dtype = smallest_uint(int(tensor.max()) if tensor.size else 0)
    return tensor.astype(dtype, copy=False)


def _base_reference(fn, sections):
    '''
    The file named by the "#! base" section of <iface>.csv fn, or None. A base
    reference holds nothing else: it shares the base's effective tensor, and
    its own changes are patches next to it. The path is relative to fn's
    directory and names a file with the same <iface>.csv name.
    '''
    if BASE_SECTION not in sections:
        return None
    lines = sections.pop(BASE_SECTION)
    if len(lines) != 1:
        _fail(fn, f'"#! {BASE_SECTION}" needs exactly one path')
    if sections:
        _fail(fn, f'a base reference holds nothing but "#! {BASE_SECTION}", found {sorted(sections, key=str)}')
    if fn.suffix == PATCH_SUFFIX:
        _fail(fn, f'a patch cannot be a base reference')
    target = Path(os.path.normpath(fn.parent / ','.join(lines[0])))
    if target.name != fn.name:
        _fail(fn, f'base {target} must be the same interface, {fn.name}')
    if not target.is_file():
        _fail(fn, f'base {target} does not exist')
    return target


def base_chain(path):
    '''
    [path, its base, the base's base, ...], the last one holding the tensor
    '''
    chain = [Path(path)]
    while True:
        base = _base_reference(chain[-1], read_sections(chain[-1]))
        if base is None:
            return chain
        if base in chain:
            _fail(chain[0], f'base references form a cycle: {" -> ".join(map(str, chain + [base]))}')
        chain.append(base)


def applied_patches(path):
    '''
    The patches load(path) applies, in order: those of the base, recursively,
    then those next to path.
    '''
    return [p for fn in reversed(base_chain(path)) for p in patch_files(fn)]


def source_files(path):
    '''
    Every file the effective tensor of path depends on, in a fixed order: the
    base chain from path to the tensor, then the applied patches.
    '''
    return base_chain(path) + applied_patches(path)


def _load_one(path):
    fn = Path(path)
    sections = read_sections(fn)
    if BASE_SECTION in sections:
        _base_reference(fn, sections)
        _fail(fn, 'is a base reference; load() follows it')
    layout, sparse, lines = _parse_layout(fn, sections)
    patch = fn.suffix == PATCH_SUFFIX
    if sparse and not patch:
        _fail(fn, 'only patches may be sparse')
    if sparse:
        return SparsePatch(**layout, sparse=_parse_sparse(fn, layout, lines))
    functional, cells = _parse_dense(fn, layout, lines)
    if patch:
        return SparsePatch(**layout, sparse=_dense_to_sparse(functional, cells))
    return TuningTensor(**layout, functional=functional, tensor=_narrow(cells))


def load(path, base=None):
    '''
    path is an <iface>.csv, or a patch of it.

    - <iface>.csv: a TuningTensor with its patches applied. A base reference
      ("#! base") loads its base's effective tensor, including the base's own
      patches, then applies the patches next to path.
    - patch, base=None: the patch alone, as a SparsePatch
    - patch, with base: applied into base, which is returned
    '''
    path = Path(path)
    if path.suffix == PATCH_SUFFIX:
        loaded = _load_one(path)
        if base is None:
            return loaded
        try:
            return apply(base, loaded)
        except TensorFileError as e:
            raise TensorFileError(f'{path}: {e}') from e
    if base is not None:
        raise ValueError(f'{path}: base is for patches; only a {PATCH_SUFFIX} applies into one')
    chain = base_chain(path)
    loaded = _load_one(chain[-1])
    for p in applied_patches(path):
        load(p, base=loaded)
    loaded.tensor = _narrow(loaded.tensor)
    return loaded


def save_npz(obj, path):
    '''
    TuningTensor: tensor, functional, tuning_keys (names in axis order), one
    ladder per tuning key named after it, target, and bits_per_slot and
    chars_per_slot for precedences. SparsePatch: sparse instead of tensor and
    functional. Compressed, runs of 0 cost little.
    '''
    arrays = {}
    if isinstance(obj, SparsePatch):
        arrays['sparse'] = obj.sparse
    else:
        arrays['tensor'] = obj.tensor
        arrays['functional'] = obj.functional
    arrays['tuning_keys'] = np.array(obj.tuning_keys)
    for name, ladder in zip(obj.tuning_keys, obj.ladders):
        if name in NPZ_RESERVED:
            raise ValueError(f'tuning key {name} is reserved in the .npz')
        arrays[name] = np.array(ladder, dtype=np.int64)
    arrays['target'] = np.array(obj.target)
    if obj.bits_per_slot is not None:
        arrays['bits_per_slot'] = np.array(obj.bits_per_slot)
        arrays['chars_per_slot'] = np.array(obj.chars_per_slot)
    np.savez_compressed(path, **arrays)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Export a tuning tensor file, with its patches applied, or a patch alone, to .npz')
    parser.add_argument('path', help='<iface>.csv or <iface>.<NNNN>-<name>.dbpatch')
    parser.add_argument('--output', default=None, help='.npz file to write, prompted when omitted')
    args = parser.parse_args(argv)
    obj = load(args.path)
    output = args.output
    if output is None:
        output = input('Output .npz file name: ').strip()
    save_npz(obj, output)


if __name__ == '__main__':
    main()
