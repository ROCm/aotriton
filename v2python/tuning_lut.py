# Copyright © 2023-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from .kernel_signature import KernelSignature
from .kernel_desc import get_template
import numpy as np
import itertools
import io
import shutil
import sys
from pathlib import Path
import os
SKIPPED_LUT_CHECK = os.getenv('AOTRITON_SKIP_LUT_CHECK', default='').split(',')

GPU_TO_DIRECTORY = {
    'MI200'  : 'amd-gfx90a',
    'MI300X' : 'amd-gfx942',
    'Navi31' : 'amd-gfx110x',
    'Navi32' : 'amd-gfx110x',
    'Unidentified' : 'amd-gfx950',
    'Unidentified02'    : 'amd-gfx120x',
}

GPU_TO_CLUSTER_SUFFIX = {
    'MI200'  : 'MI200',
    'MI300X' : 'MI300X',
    'Navi31' : 'Navi3x',
    'Navi32' : 'Navi3x',
    'Unidentified'      : 'Unidentified',
    'Unidentified02'    : 'Unidentified02',
}

class MissingLutEntry(Exception):
    def __init__(self, gpu, kdesc, ofn, fsels, lut_tensor):
        self.gpu = gpu
        self.kdesc = kdesc
        self.ofn = ofn
        self.fsels = fsels
        self.lut_tensor = lut_tensor

    def __repr__(self):
        return f'{ofn} fsels={self._fsels} has broken tuning table:\n{lut_tensor}'

    def get_missing_lut_entries(self) -> "list[str]":
        return self.kdesc.get_missing_lut_entries(self.gpu, self.lut_tensor, self.fsels)

class KernelTuningEntryForFunctionalOnGPU(object):
    LUT_TEMPLATE = get_template('autotune_table_entry.cc')
    BIN_INDEX_SUFFIX = '_binned_index'

    def __init__(self,
                 kdesc : 'KernelDescription',
                 dba : 'KernelTuningDatabaseForArch',
                 fsels : 'list[ArgumentSelection]',
                 columns : 'list[str]', # Column names, None when using Json/Schemaless DB
                 rows : 'list[dict] or list[tuple]', # Rows, list of dicts when using Json/Schemaless DB
                 autotune_keys: 'list[tuple[str, Binning]]',
                 perf_meta : 'list[ArgumentSelection]'):
        self._kdesc = kdesc
        self._dba = dba
        self._fsels = fsels
        # print(f'{self._fsels=}')
        self._lut_dic = {}
        self._autotune_keys = autotune_keys
        self._autotune_key_values = { key : set() for key, _ in autotune_keys } if autotune_keys is not None else None
        self._autotune_key_class = { key : klass for key, klass in autotune_keys } if autotune_keys is not None else None
        self._sigs = []
        self._sig_dict = {}
        self._feature_disabled = self._kdesc.is_functional_disabled_on_gpu(self._dba.gpu, self._fsels)
        if autotune_keys is None or self._feature_disabled:
            self._lut_dtype = np.int8
            self._lut_cdtype = f'int8_t'
            self._lut_tensor = np.array([0], dtype=np.int8)
            self._lut_cshape = ''.join([f'[{s}]' for s in self._lut_tensor.shape])
            self._untuned = True
            # print(f'{dba.is_passthrough_tuning()=}')
            if dba.is_passthrough_tuning():
                # print(f'KernelTuningEntryForFunctionalOnGPU.__init__ {len(rows)=}')
                for row in rows:
                    psels, compiler_options = dba.craft_perf_selection(columns, row, perf_meta)
                    self._allocate_sig(psels, compiler_options)
                self._lut_dic[0] = 0
            else:
                default_psels, default_co = dba.craft_perf_selection(None, None, perf_meta)
                self._lut_dic[0] = self._allocate_sig(default_psels, default_co)[0]
            return
        self._untuned = False
        # print(f'KernelTuningEntryForFunctionalOnGPU {fsels=}')
        # print(f'{rows=}')
        for row in rows:
            # fs_atk_values = self.extract_autotune_key_values(tinfo)
            fs_atk_values = self.extract_autotune_key_values(columns, row)
            # print(f'{fs_atk_values=}')
            psels, compiler_options = dba.craft_perf_selection(columns, row, perf_meta)
            self._lut_dic[fs_atk_values] = self._allocate_sig(psels, compiler_options)[0]
        assert self._sigs
        self._lut_tensor = None

    def extract_autotune_key_values(self, columns, row):
        assert not self._untuned
        return tuple([self.track_autotune_key_values(columns, row, tup) for tup in self._autotune_keys])

    def track_autotune_key_values(self, columns, row, tup):
        tinputs = self._dba.extract_inputs(columns, row)
        key = tup[0]
        value = tinputs[key]
        self._autotune_key_values[key].add(value)
        return value

    def _allocate_sig(self, psels, compiler_options):
        sig = KernelSignature(self._kdesc, self._fsels, psels, compiler_options, self._dba.gpu)
        # print(f'_allocate_sig {sig.compact_signature}')
        compact = sig.compact_signature
        if compact not in self._sig_dict:
            self._sig_dict[compact] = (len(self._sigs), sig)
            self._sigs.append(sig)
        return self._sig_dict[compact]

    def get_lut(self) -> 'tuple[np.ndarray, list[KernelSignature]':
        if self._lut_tensor is None:
            self._build_lut_tensor()
        assert self._lut_tensor is not None
        return self._lut_tensor, self._sigs

    def _build_lut_tensor(self):
        self._autotune_key_buckets = [ klass(self._autotune_key_values[key]) for key, klass in self._autotune_keys ]
        for dtype in [np.int8, np.int16, np.int32, np.int64]:
            if len(self._sigs) < np.iinfo(dtype).max:
                break
        self._lut_dtype = dtype
        self._lut_cdtype = f'int{np.iinfo(dtype).bits}_t'
        self._lut_tensor = np.empty([bucket.nvalues for bucket in self._autotune_key_buckets], dtype=dtype)
        assert self._lut_tensor.size > 0, 'LUT tensor must be non-empty. Empty LUT is not constructed by _build_lut_tensor'
        self._lut_cshape = ''.join([f'[{s}]' for s in self._lut_tensor.shape])
        self._list_of_atk_representatives = [bucket.representatives for bucket in self._autotune_key_buckets]
        list_of_atk_indices = [range(bucket.nvalues) for bucket in self._autotune_key_buckets]
        for indices, atk_values in zip(itertools.product(*list_of_atk_indices),
                                       itertools.product(*self._list_of_atk_representatives)):
            fs_atk_values = tuple(atk_values)
            self._lut_tensor[indices] = self._lut_dic.get(fs_atk_values, -1)
        # FIXME: Debugging
        if False and self._kdesc.SHIM_KERNEL_NAME == 'attn_fwd':
            print(f'_build_lut_tensor {self._autotune_key_values=}')
            print(f'_build_lut_tensor {self._autotune_key_buckets=}')
            print(f'_build_lut_tensor {self._lut_tensor=}', flush=True)

    def gen_kernel_symbols(self, kernel_image_dir):
        for sig in self._sigs:
            # print(f"gen_kernel_symbols {sig.compact_signature=}")
            o = self._kdesc.build_object_file_description(kernel_image_dir, sig)
            yield o.c_identifier_signature, o._hsaco_kernel_path, o

    def codegen_kernel_psels(self, kernel_image_dir):
        lines = []
        for sig in self._sigs:
            lines.append(f'R"xyzw({sig.jsongen_psels()})xyzw"')
        return 'static const char* kernel_psels[] = {\n  ' + ",\n  ".join(lines) + "\n};"

    def codegen_kernel_copts(self, kernel_image_dir):
        lines = []
        for sig in self._sigs:
            lines.append(f'R"xyzw({sig.jsongen_copts()})xyzw"')
        return 'static const char* kernel_copts[] = {\n  ' + ",\n  ".join(lines) + "\n};"

    def codegen_package_path(self, kernel_image_dir):
        for _, _, o in self.gen_kernel_symbols(kernel_image_dir):
            dir_arch = Path(GPU_TO_DIRECTORY[o.target_gpu])
            fonly = o.functional_signature + '_' + GPU_TO_CLUSTER_SUFFIX[o.target_gpu]
            return 'R"xyzw(' + str(dir_arch / o.KERNEL_FAMILY / o.SHIM_KERNEL_NAME / fonly) + ')xyzw"'

    def codegen_kernel_image_objects(self, kernel_image_dir, noimage_mode):
        kernel_image_symbols = []
        for _, _, o in self.gen_kernel_symbols(kernel_image_dir):
            if not noimage_mode and not self._feature_disabled:
                assert o.compiled_files_exist, f'Compiled file {o._hsaco_kernel_path} not exists'
            kernel_image_symbols.append(f'{{ PACKAGE_PATH, "{o.obj.stem}" }},')
        ALIGN = '\n' + 4 * ' '
        return ALIGN.join(kernel_image_symbols)

    def codegen_kernel_image_perfs(self, kernel_image_dir):
        kernel_image_perfs = []
        for sig in self._sigs:
            kernel_image_perfs.append('{ ' + sig.codegen_perf_object() + ' }')
        ALIGN = ',\n' + 4 * ' '
        return ALIGN.join(kernel_image_perfs)

    def write_lut_source(self, library_suffix : str, outdir : 'pathlib.Path', bare_mode, noimage_mode):
        gpu_kernel_image_dir = outdir.parent / f'gpu_kernel_image.{self._kdesc.SHIM_KERNEL_NAME}'
        lut_tensor, sigs = self.get_lut()
        try:
            first_sig = sigs[0]
        except IndexError as e:
            print(f'[DEBUG] {self._fsels=}')
            raise e
        godel_number = first_sig.godel_number
        ofn = outdir / f'{first_sig.functional_signature}_{first_sig.target_gpu}.cc'
        raise_lut_entry = False
        if self._kdesc.FULL_KERNEL_NAME in SKIPPED_LUT_CHECK:
            pass
        elif not self._kdesc.sancheck_lut_tensor(self._dba.gpu, lut_tensor, self._fsels):
            raise_lut_entry = True
        if bare_mode:
            return ofn
        if ofn.exists():
            with open(ofn) as f:
                old_content = f.read()
        else:
            old_content = ''
        mf = io.StringIO()  # Memory File
        d = {
            'library_suffix'        : library_suffix,
            'kernel_psels'          : self.codegen_kernel_psels(gpu_kernel_image_dir),
            'kernel_copts'          : self.codegen_kernel_copts(gpu_kernel_image_dir),
            'kernel_family_name'    : self._kdesc.KERNEL_FAMILY,
            'shim_kernel_name'      : self._kdesc.SHIM_KERNEL_NAME,
            'godel_number'          : godel_number,
            'perf_fields'           : ';\n    '.join(self._kdesc.perf_fields),
            'package_path'          : self.codegen_package_path(gpu_kernel_image_dir),
            'kernel_image_objects'  : self.codegen_kernel_image_objects(gpu_kernel_image_dir,
                                                                        noimage_mode=noimage_mode),
            'kernel_image_perfs'    : self.codegen_kernel_image_perfs(gpu_kernel_image_dir),
            'lut_dtype'             : self._lut_cdtype,
            'lut_shape'             : self._lut_cshape,
            'lut_data'              : self.lut_cdata,
            'param_class_name'      : self._kdesc.param_class_name,
            'binning_autotune_keys' : self.codegen_binning_code(),
            'binned_indices'        : self.codegen_binned_indices(),
            'perf_field_assignment' : self.codegen_perf_assignment(),
            'gpu'                   : self._dba.gpu,
            'arch_number'           : self._dba.arch_number,
            'human_readable_signature' : first_sig.human_readable_signature
        }
        print(self.LUT_TEMPLATE.format_map(d), file=mf)
        mf.seek(0)
        if mf.read() != old_content:
            mf.seek(0)
            with open(ofn, 'w') as of:
                shutil.copyfileobj(mf, of)
        if raise_lut_entry:
            raise MissingLutEntry(self._dba.gpu, self._kdesc, ofn, self._fsels, lut_tensor)
        return ofn

    @property
    def lut_cdata(self):
        lut_tensor, _ = self.get_lut()
        return np.array2string(lut_tensor, separator=',').replace('[', '{').replace(']', '}')
        # cdata = io.StringIO()
        # with np.printoptions(threshold=sys.maxsize):
        #     print(lut_tensor, file=cdata)
        # return cdata.getvalue().replace('[', '{').replace(']', '}')

    def codegen_binning_code(self):
        if self._untuned:
            return ''
        ALIGN = '\n' + 4 * ' '  # Note codegen_binning_lambda already contains ';'
        stmt = []
        for (key, _), bucket in zip(self._autotune_keys, self._autotune_key_buckets):
            stmt += bucket.codegen_binning_lambda(key, out_suffix=self.BIN_INDEX_SUFFIX)
        return ALIGN.join(stmt)

    def codegen_binned_indices(self):
        if self._untuned:
            return '[0]'
        return ''.join([f'[{key}{self.BIN_INDEX_SUFFIX}]' for key, _ in self._autotune_keys])

    def codegen_perf_assignment(self):
        ALIGN = ';\n' + 4 * ' '
        stmt = []
        for meta in self._kdesc._perf_meta:
            for aname in meta.argument_names:
                stmt.append(f'params.{aname} = perf.{aname}')
        return ALIGN.join(stmt)
