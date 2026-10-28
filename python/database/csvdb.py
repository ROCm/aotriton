# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

'''
The code generator's reader of the CSV tuning database.

A tuning database is part of its module, at <module>/database, where the
module directory's name is the family name (e.g. modules/flash/database). Its
format is specified in docs/Human Readable Tuning Database Format.md. Tensor
files are parsed by the interpreter (interpreter.py), so the format is parsed
in one place; this module adds the .metadata/ tables and the lookup of one
functional.
'''

import csv
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from . import interpreter

VENDOR = 'amd'
METADATA = '.metadata'


def _read_table(path):
    '''
    A .metadata/ table as a list of {column: value}; row i has ordinal i + 1.
    Blank cells are left out.
    '''
    with open(path, newline='') as f:
        return [{k: v for k, v in row.items() if v != ''} for row in csv.DictReader(f)]


def _db_value(choice):
    '''
    The database spelling of a functional choice: the ATI dtype without pointer
    and alignment for tensors (*fp16:16 -> fp16), 0/1 for bools, else str().
    '''
    if getattr(choice, 'is_tensor', False):
        return str(choice.triton_compile_signature).lstrip('*').partition(':')[0]
    value = getattr(choice, 'triton_compile_signature', choice)
    if isinstance(value, (bool, np.bool_)):
        return str(int(value))
    return str(value)


def tensor_path(gpu, iface):
    '''
    gfx942_mod0 -> amd/gfx942/<iface>.csv, gfx942_mod1 -> amd/gfx942/mod1/<iface>.csv
    '''
    arch, mod = gpu.split('_')
    d = Path(VENDOR) / arch
    if mod != 'mod0':
        d = d / mod
    return d / f'{iface}.csv'


@dataclass
class Slab:
    '''
    The tuning tensor of one functional on one gpu: cells has one dimension
    per tuning key, 0 = missing.
    '''
    gpu: str
    path: Path                  # relative to the family database
    tuning_keys: list[str]
    ladders: list[tuple[int, ...]]
    cells: np.ndarray
    patches: list[Path]         # patches with a block for this functional


@dataclass
class TuningView:
    '''
    What the database holds for one functional, one Slab per gpu the
    functional is optimized for (None where that gpu has no entry).
    '''
    functional: object
    iface: str
    ordinal: int | None
    fallback: bool
    target: str | None
    bits_per_slot: int | None
    slabs: list[Slab | None]
    database: 'Database' = field(repr=False)

    @property
    def empty(self):
        return all(s is None or not s.cells.any() for s in self.slabs)

    @property
    def source(self):
        '''
        The "// tuning:" comment lines (without "// "), one per gpu, plus one
        per patch that may have contributed. "possible": the patch has a block
        for this functional, but may still leave this LUT's cells to the base.
        '''
        sig = self.functional.unified_signature
        lines = []
        for slab in self.slabs:
            if slab is None:
                continue
            fb = ', fallback' if self.fallback else ''
            lines.append(f'tuning: {slab.path.as_posix()} functional {self.ordinal}{fb} ({sig})')
            lines += [f'tuning: possible patch {p.as_posix()}' for p in slab.patches]
        if not lines:
            lines.append(f'tuning: no entry for {self.iface} ({sig})')
        return lines

    def backends(self, packed):
        '''
        A packed precedence -> the backend names, most preferred first
        '''
        assert self.target == 'precedence'
        mask = (1 << self.bits_per_slot) - 1
        names = []
        packed = int(packed)
        while packed:
            names.append(self.database.op_backend(self.iface)[packed & mask])
            packed >>= self.bits_per_slot
        return names

    def hsaco_option(self, ordinal):
        '''
        An hsaco_options ordinal -> ({psel: int}, {copt: int})
        '''
        assert self.target == 'hsaco_options'
        return self.database.hsaco_option(self.iface, int(ordinal))


def tuning_source(view, functional):
    '''
    The "// tuning:" comment of a generated tune file, as one string whose
    lines after the first start with "// " (the template supplies the first)
    '''
    if view is None:
        lines = [f'tuning: no database for {functional.meta_object.NAME} ({functional.unified_signature})']
    else:
        lines = view.source
    return '\n// '.join(lines)


class Database(object):
    '''
    One module's CSV tuning database, at <module>/database. Tables and tensor
    files are loaded on first use and cached for the process.
    '''
    _cache = {}

    @classmethod
    def for_module(cls, module_dir):
        path = Path(module_dir).resolve() / 'database'
        key = path.as_posix()
        if key not in cls._cache:
            cls._cache[key] = cls(path)
        return cls._cache[key]

    def __init__(self, path):
        self._path = Path(path)
        self._tables = {}
        self._tensors = {}
        self._functional_index = {}
        self._patch_ordinals = {}

    def _table(self, *parts):
        if parts not in self._tables:
            self._tables[parts] = _read_table(self._path / METADATA / Path(*parts))
        return self._tables[parts]

    def op_backend(self, iface):
        '''
        {bord: backend name}
        '''
        key = ('op_backend', iface)
        if key not in self._tables:
            self._tables[key] = {int(r['bord']): r['name'] for r in self._table(iface, 'op_backend.csv')}
        return self._tables[key]

    def hsaco_option(self, iface, ordinal):
        hsaco = self._table(iface, 'hsaco_options.csv')[ordinal - 1]
        psels = self._table(iface, 'psels.csv')[int(hsaco['psels']) - 1]
        copts = self._table('copts.csv')[int(hsaco['copts']) - 1]
        return ({k: int(v) for k, v in psels.items()}, {k: int(v) for k, v in copts.items()})

    def _tensor(self, rel):
        if rel not in self._tensors:
            path = self._path / rel
            self._tensors[rel] = interpreter.load(path) if path.exists() else None
        return self._tensors[rel]

    def _patches_with(self, rel, ordinal):
        patches = []
        for p in interpreter.patch_files(self._path / rel):
            if p not in self._patch_ordinals:
                self._patch_ordinals[p] = set(interpreter.load(p).sparse[:, 0].tolist())
            if ordinal in self._patch_ordinals[p]:
                patches.append(p.relative_to(self._path))
        return patches

    def _ordinal(self, iface, columns_values):
        '''
        The functional ordinal whose row matches {column: value} on those
        columns, or None. Other columns (single-choice axes) are not compared.
        '''
        columns = tuple(sorted(columns_values))
        key = (iface, columns)
        if key not in self._functional_index:
            index = {}
            path = self._path / METADATA / iface / 'functional.csv'
            for ordinal, row in enumerate(_read_table(path) if path.exists() else [], start=1):
                missing = [c for c in columns if c not in row]
                assert not missing, f'{path}: no column {missing} for the functional axes'
                k = tuple(row[c] for c in columns)
                assert k not in index, f'{path}: functionals {index.get(k)} and {ordinal} differ only in non-axis columns'
                index[k] = ordinal
            self._functional_index[key] = index
        return self._functional_index[key].get(tuple(columns_values[c] for c in columns))

    @staticmethod
    def _columns(functional, choices):
        '''
        {signature_name: choice} -> {functional.csv column: database value}.
        A tensor axis is named after its ATI type variable (T_io), other
        axes after their argument.
        '''
        axes = {ax.signature_name: ax for ax in functional.meta_object.axes_multi}
        out = {}
        for name, choice in choices.items():
            ax = axes[name]
            out[ax.var_name if ax.kind == 'tensor' else name] = _db_value(choice)
        return out

    def lookup(self, functional):
        '''
        The TuningView of functional, or None when the database has no tuning
        tensor of its interface for any of its gpus.
        '''
        meta = functional.meta_object
        iface = meta.NAME
        rels = [tensor_path(gpu, iface) for gpu in functional.optimized_for]
        tensors = [self._tensor(rel) for rel in rels]
        if all(t is None for t in tensors):
            return None
        view = None
        for fallback, choices in ((False, functional.compact_choices), (True, functional.fallback_choices)):
            ordinal = self._ordinal(iface, self._columns(functional, choices))
            view = self._view(functional, iface, ordinal, fallback, rels, tensors)
            if not view.empty:
                break
        return view

    def _view(self, functional, iface, ordinal, fallback, rels, tensors):
        slabs = []
        target = bits = None
        for gpu, rel, t in zip(functional.optimized_for, rels, tensors):
            if t is None:
                slabs.append(None)
                continue
            assert target in (None, t.target), f'{rel}: target {t.target} differs from {target}'
            target, bits = t.target, t.bits_per_slot
            index = np.flatnonzero(t.functional == ordinal) if ordinal is not None else []
            if len(index) == 0:
                slabs.append(None)
                continue
            slabs.append(Slab(gpu=gpu, path=rel, tuning_keys=t.tuning_keys, ladders=t.ladders,
                              cells=t.tensor[index[0]], patches=self._patches_with(rel, ordinal)))
        return TuningView(functional=functional, iface=iface, ordinal=ordinal, fallback=fallback,
                          target=target, bits_per_slot=bits, slabs=slabs, database=self)
