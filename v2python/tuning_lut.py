# Copyright © 2023-2025 Advanced Micro Devices, Inc.
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

ARCH_TO_DIRECTORY = {
    'gfx90a'  : 'amd-gfx90a',
    'gfx942' : 'amd-gfx942',
    'gfx1100' : 'amd-gfx110x',
    'gfx1101' : 'amd-gfx110x',
    'gfx1151' : 'amd-gfx115x',
    'gfx950' : 'amd-gfx950',
    'gfx1200'    : 'amd-gfx120x',
    'gfx1201'    : 'amd-gfx120x',
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
        self._lut_dic = { gpu : {} for gpu in self._dba.for_gpus}
        self._autotune_keys = autotune_keys
        self._autotune_key_values = { key : set() for key, _ in autotune_keys } if autotune_keys is not None else None
        self._autotune_key_class = { key : klass for key, klass in autotune_keys } if autotune_keys is not None else None
        self._sigs = []
        self._sig_dict = {}
        self._feature_disabled = self._kdesc.is_functional_disabled_on_arch(self._dba.arch, self._fsels)
        if autotune_keys is None or self._feature_disabled:
            ngpus = len(self._dba.for_gpus)
            self._lut_dtype = np.int8
            self._lut_cdtype = f'int8_t'
            self._lut_tensor = np.zeros([ngpus, 1], dtype=np.int8)
            self._lut_cshape = ''.join([f'[{s}]' for s in self._lut_tensor.shape])
            self._untuned = True
            # print(f'{dba.is_passthrough_tuning()=}')
            if dba.is_passthrough_tuning():
                # print(f'KernelTuningEntryForFunctionalOnGPU.__init__ {len(rows)=}')
                for row in rows:
                    psels, compiler_options = dba.craft_perf_selection(columns, row, perf_meta)
                    self._allocate_sig(psels, compiler_options)
                for gpu in self._dba.for_gpus:
                    self._lut_dic[gpu][0] = 0
            else:
                default_psels, default_co = dba.craft_perf_selection(None, None, perf_meta)
                index = self._allocate_sig(default_psels, default_co)[0]
                for gpu in self._dba.for_gpus:
                    self._lut_dic[gpu][0] = index
            return
        self._untuned = False
        # print(f'KernelTuningEntryForFunctionalOnGPU {fsels=}')
        # print(f'{rows=}')
        gpu_col = self._dba.locate_gpu_col(columns)
        for row in rows:
            # fs_atk_values = self.extract_autotune_key_values(tinfo)
            fs_atk_values = self.extract_autotune_key_values(columns, row)
            # print(f'{fs_atk_values=}')
            psels, compiler_options = dba.craft_perf_selection(columns, row, perf_meta)
            gpu = self._dba.get_gpu_from_row(gpu_col, row)
            self._lut_dic[gpu][fs_atk_values] = self._allocate_sig(psels, compiler_options)[0]
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

    # DispatcherV3 Note
    # Signatures are shared by all GPU mods.
    def _allocate_sig(self, psels, compiler_options):
        sig = KernelSignature(self._kdesc, self._fsels, psels, compiler_options, self._dba.arch)
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
        ngpus = len(self._dba.for_gpus)
        self._lut_dtype = dtype
        self._lut_cdtype = f'int{np.iinfo(dtype).bits}_t'
        self._lut_shape = [ngpus] + [bucket.nvalues for bucket in self._autotune_key_buckets]
        self._lut_tensor = np.empty(self._lut_shape, dtype=dtype)
        assert self._lut_tensor.size > 0, 'LUT tensor must be non-empty. Empty LUT is not constructed by _build_lut_tensor'
        self._lut_cshape = ''.join([f'[{s}]' for s in self._lut_tensor.shape])
        self._list_of_atk_representatives = [bucket.representatives for bucket in self._autotune_key_buckets]
        list_of_atk_indices = [range(bucket.nvalues) for bucket in self._autotune_key_buckets]
        for i, gpu in enumerate(self._dba.for_gpus):
            if i > 0:
                # Assign default values.
                # This allows partial tuning
                self._lut_tensor[i] = self._lut_tensor[0]
            for indices, atk_values in zip(itertools.product(*list_of_atk_indices),
                                           itertools.product(*self._list_of_atk_representatives)):
                fs_atk_values = tuple(atk_values)
                self._lut_tensor[i][indices] = self._lut_dic[gpu].get(fs_atk_values, -1)
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
            dir_arch = Path(ARCH_TO_DIRECTORY[o.target_arch])
            fonly = o.functional_signature + '_' + o.target_arch
            return str(dir_arch / o.KERNEL_FAMILY / o.SHIM_KERNEL_NAME / fonly)

    def codegen_func_name(self, kernel_image_dir):
        for _, _, o in self.gen_kernel_symbols(kernel_image_dir):
            fsel, psel, copts = o.compact_signature_components
            return fsel

    def codegen_arch_name(self, kernel_image_dir):
        for _, _, o in self.gen_kernel_symbols(kernel_image_dir):
            return o.target_arch

    def codegen_compact_kernels(self, kernel_image_dir, package_path, noimage_mode):
        meta_objects = []
        string_dict = {None : 0}
        def register_string(s):
            assert s is not None # string_dict[None] tracks the total size
            if s in string_dict:
                return string_dict[s]
            offset = string_dict[None]
            string_dict[s] = offset
            string_dict[None] = offset + len(s) + 1  # Need a trailing '\0'
            return offset
        for _, _, o in self.gen_kernel_symbols(kernel_image_dir):
            if not noimage_mode and not self._feature_disabled:
                assert o.compiled_files_exist, f'Compiled file {o._hsaco_kernel_path} not exists'
            fsel, psel, copt = o.compact_signature_components
            b2sum_u64, raw = o.blake2b_hash(package_path)
            assert len(b2sum_u64) == 16
            b2sum_u64_hi = b2sum_u64[:8]
            b2sum_u64_lo = b2sum_u64[8:]
            psel_offset = register_string(psel)
            copt_offset = register_string(copt)
            meta_objects.append(f'{{ 0x{b2sum_u64_hi}u, 0x{b2sum_u64_lo}u, {psel_offset}, {copt_offset} }}, // {b2sum_u64} = b2sum -l 64 <<< {raw}')
        assert string_dict[None] < 2 ** 16 - 1, f'Packed string size {string_dict[None]} exceeds uint16_t limit'
        del string_dict[None]
        packed_string = '\n'.join(['"' + s + '\\0"' for s in string_dict])
        ALIGN = '\n' + 4 * ' '
        return packed_string, ALIGN.join(meta_objects)

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
        ofn = outdir / f'{first_sig.functional_signature}_{first_sig.target_arch}.cc'
        raise_lut_entry = False
        if self._kdesc.FULL_KERNEL_NAME in SKIPPED_LUT_CHECK:
            pass
        elif not self._kdesc.sancheck_lut_tensor(self._dba.arch, lut_tensor[0], self._fsels):
            # We only check lut_tensor[0] here because lut_tensor[1:] are
            # initialized from lut_tensor[0] (see _build_lut_tensor)
            raise_lut_entry = True
        if bare_mode:
            return ofn
        if ofn.exists():
            with open(ofn) as f:
                old_content = f.read()
        else:
            old_content = ''
        mf = io.StringIO()  # Memory File
        package_path = self.codegen_package_path(gpu_kernel_image_dir)
        packed_string, meta_objects = self.codegen_compact_kernels(gpu_kernel_image_dir,
                                                                   package_path,
                                                                   noimage_mode=noimage_mode)
        d = {
            'library_suffix'        : library_suffix,
            'kernel_psels'          : self.codegen_kernel_psels(gpu_kernel_image_dir),
            'kernel_copts'          : self.codegen_kernel_copts(gpu_kernel_image_dir),
            'kernel_family_name'    : self._kdesc.KERNEL_FAMILY,
            'shim_kernel_name'      : self._kdesc.SHIM_KERNEL_NAME,
            'godel_number'          : godel_number,
            'perf_fields'           : ';\n    '.join(self._kdesc.perf_fields),
            'package_path'          : package_path,
            'func_name'             : self.codegen_func_name(gpu_kernel_image_dir),
            'arch_name'             : self.codegen_arch_name(gpu_kernel_image_dir),
            'packed_string'         : packed_string,
            'meta_objects'          : meta_objects,
            'kernel_image_perfs'    : self.codegen_kernel_image_perfs(gpu_kernel_image_dir),
            'lut_dtype'             : self._lut_cdtype,
            'lut_shape'             : self._lut_cshape,
            'lut_data'              : self.lut_cdata,
            'param_class_name'      : self._kdesc.param_class_name,
            'deduplicated_lut_function' : self.codegen_deduplicated_lut_function(),
            'deduplicated_pp_args_function_index' : self.codegen_deduplicated_pp_args_function_index(first_sig),
            'perf_field_assignment' : self.codegen_perf_assignment(),
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
            raise MissingLutEntry(self._dba.for_gpus[0], self._kdesc, ofn, self._fsels, lut_tensor)
        return ofn

    @property
    def lut_cdata(self):
        lut_tensor, _ = self.get_lut()
        def fmt(t):
            return np.array2string(t, separator=',').replace('[', '{').replace(']', '}')
        tensor_text_list = []
        for i, gpu in enumerate(self._dba.for_gpus):
            text  = f'\n// GPU {gpu}\n'
            text += fmt(lut_tensor[i])
            text += f'\n// End of GPU {gpu}\n'
            tensor_text_list.append(text)
        return '{' + '\n,\n'.join(tensor_text_list) + '}'
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

    def codegen_deduplicated_lut_function(self):
        d = {
            'param_class_name'      : self._kdesc.param_class_name,
            'lut_dtype'             : self._lut_cdtype,
            'lut_shape'             : self._lut_cshape,
            'binning_autotune_keys' : self.codegen_binning_code(),
            'binned_indices'        : self.codegen_binned_indices(),
        }
        stmt = []
        stmt.append('({param_class_name}& params, int mod_number, {lut_dtype} lut{lut_shape}) {{')
        stmt.append('    {binning_autotune_keys}')
        stmt.append('    return lut[mod_number]{binned_indices};')
        stmt.append('}}')
        ALIGN = '\n'
        lambda_src = ALIGN.join(stmt).format_map(d)
        lut_function_name = self._kdesc.register_code_lut(lambda_src, d['lut_dtype'], d['lut_shape'])
        return lut_function_name

    def codegen_deduplicated_pp_args_function_index(self, first_sig):
        getters = self._kdesc.codegen_kargs_getter_dic();
        fsel_dict = first_sig.build_final_fsel_dict(all_args=True)
        assign_skips = tuple([isinstance(fsel_dict[aname], int) for aname in self._kdesc.KERNEL_DATA_ARGUMENTS])
        hit, findex = self._kdesc.lookup_prepare_args(assign_skips)
        if hit:
            return findex
        fsel_dict = first_sig.build_final_fsel_dict(all_args=True, with_meta=True)
        stmt = []
        for aname in self._kdesc.KERNEL_DATA_ARGUMENTS:
            fval, fsel = fsel_dict[aname]
            assign = getters[aname] + f', // {aname}'
            if isinstance(fval, int):  # isinstance(True, int) == True
                fval_str = str(fval)
                if fsel.is_conditional and hasattr(fsel._selection, 'list_possible_constexpr_values'):
                    all_vals = fsel._selection.list_possible_constexpr_values(first_sig._selections)
                    fval_str = '/'.join(all_vals)
                assign = '// ' + assign + f' as constexpr {fval_str}'
            stmt.append(assign)
        stmt.append('CAST(global_scratch)')
        pfx = '  return { '
        join = '\n' + ' ' * len(pfx)
        sfx = '         };'
        src = pfx + join.join(stmt) + '\n' + sfx
        return self._kdesc.register_prepare_args(assign_skips, src)

    def codegen_perf_assignment(self):
        ALIGN = ';\n' + 4 * ' '
        stmt = []
        for meta in self._kdesc._perf_meta:
            for aname in meta.argument_names:
                stmt.append(f'params.{aname} = perf.{aname}')
        return ALIGN.join(stmt)
