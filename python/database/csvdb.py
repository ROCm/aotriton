# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

'''
The code generator's reader of the CSV tuning database.

A tuning database is part of its module, at <module>/database, where the
module directory's name is the family name (e.g. modules/flash/database). A
build may use an external tuning database instead, see
tuning_database_dir(). The format is specified in
docs/Human Readable Tuning Database Format.md. Tensor
files are parsed by the interpreter (interpreter.py), so the format is parsed
in one place; this module adds the .metadata/ tables and the lookup of one
functional.
'''

import csv
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
import numpy as np
from . import interpreter
from ..gpu_targets import gpu2archmod

VENDOR = 'amd'
METADATA = '.metadata'


def _read_table(path):
    '''
    A .metadata/ table as {ordinal: {column: value}}. The header is line 0, so
    the first row has ordinal 1; look rows up by ordinal, never by position.
    Blank cells are left out.
    '''
    with open(path, newline='') as f:
        rows = csv.DictReader(f)
        return {ordinal: {k: v for k, v in row.items() if v != ''}
                for ordinal, row in enumerate(rows, start=1)}


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
    gfx942_mod0 -> amd/gfx942/<iface>.txtdb, gfx942_mod1 -> amd/gfx942/mod1/<iface>.txtdb
    '''
    arch, mod = gpu2archmod(gpu)
    d = Path(VENDOR) / arch
    if mod != 0:
        d = d / f'mod{mod}'
    return d / f'{iface}.txtdb'


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


def tuning_database_dir(module_dir, external=None):
    '''
    Where the tuning database of a module is: <external>/<family>/database when
    an external tuning database is given and has that family, else
    <module>/database. The family is the module directory's name.

    The external tuning database (--external_tuning_database) lets a build use a
    database from outside the modules, e.g. one exported from Tuner v3.5 to
    evaluate before committing it. Do not remove it; see CLAUDE.md.
    '''
    module_dir = Path(module_dir)
    if external is not None:
        candidate = Path(external) / module_dir.name / 'database'
        if candidate.is_dir():
            return candidate
    return module_dir / 'database'


class Database(object):
    '''
    One module's CSV tuning database, at <module>/database. Tables and tensor
    files are loaded on first use and cached for the process.
    '''
    _cache = {}

    @classmethod
    def for_module(cls, module_dir, external=None):
        path = tuning_database_dir(module_dir, external).resolve()
        key = path.as_posix()
        if key not in cls._cache:
            cls._cache[key] = cls(path)
        return cls._cache[key]

    def __init__(self, path):
        self._path = Path(path)
        self._tables = {}
        self._tensors = {}
        self._functional_index = {}
        self._hsaco_options = {}
        self._patch_ordinals = {}
        self._applied_patches = {}

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
            rows = self._table(iface, 'op_backend.csv').values()
            self._tables[key] = {int(r['bord']): r['name'] for r in rows}
        return self._tables[key]

    def hsaco_option(self, iface, ordinal):
        '''
        ({psel: int}, {copt: int}) of an hsaco_options ordinal, as read-only
        mappings. Each interface's table is decoded once, on first use: the
        code generator looks up the same few hundred ordinals for every
        tuned cell.
        '''
        if iface not in self._hsaco_options:
            psels = self._table(iface, 'psels.csv')
            copts = self._table('copts.csv')
            def decode(row):
                return MappingProxyType({k: int(v) for k, v in row.items()})
            def gen():
                for ordinal, hsaco in self._table(iface, 'hsaco_options.csv').items():
                    psel = psels[int(hsaco['psels'])]
                    copt = copts[int(hsaco['copts'])]
                    yield ordinal, (decode(psel), decode(copt))
            self._hsaco_options[iface] = dict(gen())
        return self._hsaco_options[iface][ordinal]

    def _tensor(self, rel):
        if rel not in self._tensors:
            path = self._path / rel
            self._tensors[rel] = interpreter.load(path) if path.exists() else None
        return self._tensors[rel]

    def _patches_with(self, rel, ordinal):
        # Resolving the patches reads every file of the base chain, the whole
        # base tensor included; do it once per tensor file, not per functional.
        if rel not in self._applied_patches:
            self._applied_patches[rel] = interpreter.applied_patches(self._path / rel)
        patches = []
        for p in self._applied_patches[rel]:
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
            for ordinal, row in (_read_table(path) if path.exists() else {}).items():
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
            # TuningView.backends() decodes every slab with one bits_per_slot
            assert target is None or bits == t.bits_per_slot, \
                f'{rel}: bits_per_slot {t.bits_per_slot} differs from {bits}'
            target, bits = t.target, t.bits_per_slot
            index = np.flatnonzero(t.functional == ordinal) if ordinal is not None else []
            if len(index) == 0:
                slabs.append(None)
                continue
            slabs.append(Slab(gpu=gpu,
                              path=rel,
                              tuning_keys=t.tuning_keys,
                              ladders=t.ladders,
                              cells=t.tensor[index[0]],
                              patches=self._patches_with(rel, ordinal)))
        return TuningView(functional=functional,
                          iface=iface,
                          ordinal=ordinal,
                          fallback=fallback,
                          target=target,
                          bits_per_slot=bits,
                          slabs=slabs,
                          database=self)
