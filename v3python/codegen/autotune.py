# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Generate <family>/autotune.<kernel_name>/<functional>.cc

from ..base import Functional
from ..base import typed_choice as TC
from ..kernel.ksignature import KernelSignature
from .template import get_template
from ..utils import (
    LazyFile,
    dict2json,
    log,
)
from .common import (
    codegen_struct_cfields,
    MissingLutEntry,
    hsaco_filename,
    hsaco_dir,
)
import numpy as np
from .basetune import BaseTuneCodeGenerator
import json

class AutotuneCodeGenerator(BaseTuneCodeGenerator):
    AUTOTUNE_TEMPLATE = get_template('autotune_table_entry.cc')
    BIN_INDEX_SUFFIX = '_binned_index'

    '''
    For generic it accepts Functional instead of a KernelDescription object
    '''
    def __init__(self,
                 args,
                 f : Functional,
                 dataframe_for_tuning : 'pandas.DataFrame | None',
                 parent_repo):
        super().__init__(args, f, dataframe_for_tuning, parent_repo)
        # TODO: support other binning algorithm
        kdesc = self._f.meta_object
        if args.build_for_tuning or self._df is None or self._df.empty:
            log(lambda : f'translate_empty_dataframe for kernel {kdesc.NAME}')
            self._lut_tensor, self._sigs, self._binning_dict = kdesc.translate_empty_dataframe(f)
            # Replace sigs with configs from KernelDescription.gen_autotune_configs
            if args.build_for_tuning and kdesc.is_tunable:
                self._sigs = list(kdesc.gen_signatures_for_tuning(f))
                if args.build_for_tuning_second_pass:
                    image_path = hsaco_dir(args.build_dir, kdesc)
                    def hsaco_compile_successful(ksig : KernelSignature):
                        full = image_path / hsaco_filename(kdesc, ksig)
                        if not full.exists():
                            return False
                        meta = full.with_suffix('.json')
                        if not meta.exists():
                            return False
                        with open(meta) as f:
                            j = json.load(f)
                            return j['compile_status'] == 'Complete'
                    self._sigs = [ ksig for ksig in self._sigs if hsaco_compile_successful(ksig) ]
        else:
            log(lambda : f'translate_dataframe for kernel {kdesc.NAME}')
            self._lut_tensor, self._sigs, self._binning_dict = kdesc.translate_dataframe(f, self._df)
            if not kdesc.sancheck_lut_tensor(f, self._lut_tensor):
                ent = MissingLutEntry(f, self._lut_tensor)
                if args._should_raise_for_lut(f):
                    raise ent
                else:
                    for j in ent.get_missing_lut_entries():
                        print("TUNE_FLASH --entry_from_json Item: ", j)
        assert all([isinstance(k, KernelSignature)] for k in self._sigs)

    def generate(self):
        # Un "self._" section
        args = self._args

        log(lambda : f'Writing to {self._cc_file}')
        with LazyFile(self._cc_file) as fout:
            self.write_autotune_src(fout)
        hsaco_registry = self._parent_repo.get_hsaco_registry('hsaco')
        hsaco_registry.register(self._f, self.all_signatures)

    def write_autotune_src(self, fout):
        f = self._f
        kdesc = f.meta_object
        lut_ctype, lut_cshape, lut_cdata = self.codegen_format_lut(self._lut_tensor)
        # gpu_kernel_image_dir = args.build_dir / f.FAMILY / f'gpu_kernel_image.{f.NAME}'
        package_path = str(f.full_filepack_path)
        meta_hsacos = self.codegen_compact_kernels(kdesc,
                                                   self._sigs,
                                                   package_path)
        d = {
            'kernel_psels'          : self.codegen_kernel_psels(self._sigs),
            'kernel_copts'          : self.codegen_kernel_copts(self._sigs),
            'kernel_family_name'    : kdesc.FAMILY,
            'shim_kernel_name'      : kdesc.NAME,
            'godel_number'          : f.godel_number,
            'perf_fields'           : codegen_struct_cfields(kdesc.perf_cfields, nalign=4),
            'package_path'          : package_path,
            'func_name'             : f.signature_in_func_name,
            'arch_name'             : f.arch,
            'meta_hsacos'           : meta_hsacos,
            'kernel_image_perfs'    : self.codegen_kernel_image_perfs(self._sigs),
            'lut_ctype'             : lut_ctype,
            'lut_cshape'            : lut_cshape,
            'lut_data'              : lut_cdata,
            'param_class_name'      : kdesc.param_class_name,
            'context_class_name'    : kdesc.context_class_name,
            'deduplicated_lut_function' : self.codegen_deduplicated_lut_function(lut_ctype, lut_cshape),
            'deduplicated_pp_args_function_index' : self.codegen_deduplicated_pp_args_function_index(f),
            'perf_field_assignment' : self.codegen_perf_assignment(),
            'arch_number'           : f.arch_number,
            'human_readable_signature' : f.human_readable_signature,
        }
        print(self.AUTOTUNE_TEMPLATE.format_map(d), file=fout)

    def codegen_kernel_psels(self, ksigs):
        lines = []
        for sig in ksigs:
            lines.append(f'R"xyzw({dict2json(sig.perf_cdict)})xyzw"')
        return 'static const char* kernel_psels[] = {\n  ' + ",\n  ".join(lines) + "\n}"

    def codegen_kernel_copts(self, ksigs):
        lines = []
        for sig in ksigs:
            lines.append(f'R"xyzw({dict2json(sig.copt_dict)})xyzw"')
        return 'static const char* kernel_copts[] = {\n  ' + ",\n  ".join(lines) + "\n}"

    def codegen_compact_kernels(self, kdesc, ksigs, package_path):
        meta_hsacos = []
        string_registry = self._parent_repo.get_string_registry('per_kernel_packed_string')
        def register_string(s):
            return string_registry.register(s)
        for sig in ksigs:
            # if not noimage_mode and not self._feature_disabled:
            #     assert o.compiled_files_exist, f'Compiled file {o._hsaco_kernel_path} not exists'
            b2sum_u64, raw = sig.blake2b_hash(package_path)
            u8raw = raw.decode('utf-8')
            assert len(b2sum_u64) == 16
            b2sum_u64_hi = b2sum_u64[:8]
            b2sum_u64_lo = b2sum_u64[8:]
            psel_offset = register_string(sig.perf_signature)
            copt_offset = register_string(sig.copt_signature)
            meta_hsacos.append(f'{{ 0x{b2sum_u64_hi}u, 0x{b2sum_u64_lo}u, {psel_offset}, {copt_offset} }}, // {b2sum_u64} = b2sum -l 64 <<< {u8raw}')
        # assert string_dict[None] < 2 ** 16 - 1, f'Packed string size {string_dict[None]} exceeds uint16_t limit'
        # del string_dict[None]
        ALIGN = '\n' + 4 * ' '
        return ALIGN.join(meta_hsacos)

    def codegen_kernel_image_perfs(self, ksigs):
        kernel_image_perfs = []
        kdesc = self._f.meta_object
        perf_cfields = kdesc.perf_cfields
        def codegen_perf_object(sig):
            dic = sig.perf_compact_dict
            return ', '.join([f'.{field.aname} = {dic[field.aname].infotext}' for field in perf_cfields ])
        for sig in ksigs:
            kernel_image_perfs.append('{ ' + codegen_perf_object(sig) + ' }')
        ALIGN = ',\n' + 4 * ' '
        return ALIGN.join(kernel_image_perfs)

    def codegen_format_lut(self, lut_tensor):
        f = self._f
        max_value = np.max(lut_tensor)
        for dtype in [np.int8, np.int16, np.int32, np.int64]:
            if max_value < np.iinfo(dtype).max:
                break
        ctype =  f'int{np.iinfo(dtype).bits}_t'
        cshape = ''.join([f'[{s}]' for s in lut_tensor.shape])
        def fmt(t):
            return np.array2string(t, separator=',').replace('[', '{').replace(']', '}')
        tensor_text_list = []
        for i, gpu in enumerate(f.optimized_for):
            text  = f'\n// GPU {gpu}\n'
            text += fmt(lut_tensor[i])
            text += f'\n// End of GPU {gpu}\n'
            tensor_text_list.append(text)
        cdata = '{' + '\n,\n'.join(tensor_text_list) + '}'
        return ctype, cshape, cdata

    def codegen_deduplicated_lut_function(self, lut_ctype, lut_cshape):
        kdesc = self._f.meta_object
        d = {
            'param_class_name'      : kdesc.param_class_name,
            'lut_ctype'             : lut_ctype,
            'lut_shape'             : lut_cshape,
            'binning_autotune_keys' : self.codegen_binning_code(),
            'binned_indices'        : self.codegen_binned_indices(),
        }
        lambda_params = '(const {param_class_name}& params, int mod_number, {lut_ctype} lut{lut_shape})'
        stmt = []
        stmt.append(lambda_params + ' {{')
        stmt.append('    {binning_autotune_keys}')
        stmt.append('    return lut[mod_number]{binned_indices};')
        stmt.append('}}')
        ALIGN = '\n'
        lambda_src = ALIGN.join(stmt).format_map(d)
        lut_registry = self._parent_repo.get_function_registry('lut_function')
        lut_params = lambda_params.format_map(d)
        lut_function_pfx = f'{kdesc.NAME}__lut_lambda'
        lut_function_name = lut_registry.register(lambda_src, 'int', lut_function_pfx, lut_params)
        return lut_function_name

    def codegen_binning_code(self):
        if self._binning_dict is None:
            return ''
        ALIGN = '\n' + 4 * ' '  # Note codegen_binning_lambda already contains ';'
        stmt = []
        for key, algo in self._binning_dict.items():
            stmt += algo.codegen_binning_lambda(key, out_suffix=self.BIN_INDEX_SUFFIX)
        return ALIGN.join(stmt)

    def codegen_binned_indices(self):
        if self._binning_dict is None:
            return '[0]'
        return ''.join([f'[{key}{self.BIN_INDEX_SUFFIX}]' for key in self._binning_dict.keys()])

    def codegen_deduplicated_pp_args_function_index(self, functional : Functional):
        kdesc = self._f.meta_object

        pp_registry = self._parent_repo.get_signatured_function_registry('pp_function')
        bind_dict = functional.build_complete_bind_dict(with_resolved_tc=True)
        def _pp_signature(aname):
            bind, tc = bind_dict[aname]
            is_constexpr = isinstance(tc, TC.constexpr_base)
            return is_constexpr
            # tc_value = tc.json_value if is_constexpr else None
            # print(f'\tassign_skips {aname} {is_constexpr=}')
            # return (is_constexpr, tc_value)
        assign_skips = tuple([_pp_signature(aname) for aname in kdesc.KERNEL_DATA_ARGUMENTS])
        if True:
            tc_dict = { aname : tc.triton_compile_signature for aname, (_, tc) in bind_dict.items() }
            log(lambda : f'{functional.compact_signature_noarch=} {assign_skips=} {tc_dict=}')
        hit, findex = pp_registry.contains(assign_skips)
        if hit:
            return findex
        getter_dict = self.codegen_getter(kdesc)
        stmt = []
        for aname in kdesc.KERNEL_DATA_ARGUMENTS:
            bind, tc = bind_dict[aname]
            assign = getter_dict[aname] + f', // {aname}'
            # Comment out constexpr values
            if isinstance(tc, TC.constexpr_base):  # isinstance(True, int) == True
                fmt_val = str(bind.value)
                if bind.param_maybe_conditional:
                    fmt_val = bind.document_conditional_value()
                assign = '// ' + assign + f' as constexpr {fmt_val}'
            stmt.append(assign)
        stmt.append('CAST(global_scratch)')
        pfx = '  return { '
        join = '\n' + ' ' * len(pfx)
        sfx = '         };'
        src = pfx + join.join(stmt) + '\n' + sfx
        return pp_registry.register(assign_skips, src)

    def codegen_getter(self, kdesc):
        d = {}
        for tensor_aname, (stride_anames, _) in kdesc.TENSOR_STRIDE_INPUTS.items():
            tensor_rank = kdesc.get_tensor_rank(tensor_aname)
            for i in range(tensor_rank):
                aname = stride_anames[i]
                d[aname] = f'params.{tensor_aname}->kparam_stride({i})'
        for tp in kdesc.list_functional_params():
            for aname in tp.all_names:
                tc = tp.repr_choice.resolve(aname, tc_dict=None)
                if isinstance(tc, TC.tensor):
                    d[aname] = f'params.{aname}->kparam_data_ptr()'
        for aname in kdesc.KERNEL_DATA_ARGUMENTS:  # TODO: make this general
            if aname in d:
                continue
            d[aname] = f'CAST(&params.{aname})'
        return d

    def codegen_perf_assignment(self):
        kdesc = self._f.meta_object
        ALIGN = ';\n' + 4 * ' '
        stmt = []
        for meta in kdesc.gen_performance_params():
            for aname in meta.all_names:
                stmt.append(f'context.{aname} = perf.{aname}')
        return ALIGN.join(stmt)

    @property
    def all_signatures(self):
        return self._sigs

