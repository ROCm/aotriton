# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Generate <family>/autotune.<kernel_name>/<functional>.cc

from ..template_instantiation.ir import Functional
from ..template_instantiation.ir import typed_choice as TC
from ..template_instantiation.ir.ksignature import KernelSignature
from .template import get_template
from ..utils import (
    LazyFile,
    dict2json,
    log,
)
from .common import (
    codegen_struct_cfields,
    MissingLutEntry,
    hsaco_ondisk_name,
    hsaco_dir,
)
from .basetune import BaseTuneCodeGenerator
import json
import numpy as np

class AutotuneCodeGenerator(BaseTuneCodeGenerator):
    AUTOTUNE_TEMPLATE = get_template('autotune_table_entry.cc')

    '''
    For generic it accepts Functional instead of a KernelDescription object
    '''
    def __init__(self,
                 args,
                 f : Functional,
                 dataframe_for_tuning : 'pandas.DataFrame | None',
                 sql : tuple,
                 parent_repo):
        super().__init__(args, f, dataframe_for_tuning, parent_repo)
        self._sql = sql
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
                        full = image_path / hsaco_ondisk_name(kdesc, ksig)
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
            ok, errors, missing_entries = kdesc.sancheck_lut_tensor(f, self._lut_tensor)
            if not ok:
                if args._should_raise_for_lut(f):
                    raise MissingLutEntry(f, self._lut_tensor)
                else:
                    for j in missing_entries:
                        print(kdesc.NAME, "TUNE_V3BIS testrun Item: ", j)
                        print("  SQL:", self._sql)
                        for err in errors:
                            print("    ERROR:", err)
        assert all(isinstance(k, KernelSignature) for k in self._sigs)

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
        flatzip_path = f.full_flatzip_path.as_posix()
        aks2_entry   = f.filepack_inzip_name
        meta_hsacos = self.codegen_compact_kernels(kdesc,
                                                   self._sigs,
                                                   flatzip_path)
        d = {
            'kernel_psels'          : self.codegen_kernel_psels(self._sigs),
            'kernel_copts'          : self.codegen_kernel_copts(self._sigs),
            'kernel_family_name'    : kdesc.FAMILY,
            'shim_kernel_name'      : kdesc.NAME,
            'godel_number'          : f.godel_number,
            'perf_fields'           : codegen_struct_cfields(kdesc.perf_cfields, nalign=4),
            'flatzip_path'          : flatzip_path,
            'aks2_entry'            : aks2_entry,
            'func_name'             : f.unified_signature,
            'arch_name'             : f.arch,
            'meta_hsacos'           : meta_hsacos,
            'kernel_image_perfs'    : self.codegen_kernel_image_perfs(self._sigs),
            'lut_ctype'             : lut_ctype,
            'lut_cshape'            : lut_cshape,
            'lut_data'              : lut_cdata,
            'context_class_name'    : kdesc.context_class_name,
            'deduplicated_lut_function' : self.codegen_deduplicated_lut_function(lut_ctype, lut_cshape),
            'deduplicated_pp_args_function_index' : self.codegen_deduplicated_pp_args_function_index(f),
            'perf_field_assignment' : self.codegen_perf_assignment(),
            'arch_number'           : f.arch_number,
            'human_readable_signature' : f.human_readable_signature,
            'sql'                   : self._sql,
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

    def codegen_compact_kernels(self, kdesc, ksigs, flatzip_path):
        meta_hsacos = []
        string_registry = self._parent_repo.get_string_registry('per_kernel_packed_string')
        def register_string(s):
            return string_registry.register(s)
        for sig in ksigs:
            # if not noimage_mode and not self._feature_disabled:
            #     assert o.compiled_files_exist, f'Compiled file {o._hsaco_kernel_path} not exists'
            b2sum_u64, raw = sig.blake2b_hash(flatzip_path)
            u8raw = raw.decode('utf-8')
            assert len(b2sum_u64) == 16
            b2sum_u64_hi = b2sum_u64[:8]
            b2sum_u64_lo = b2sum_u64[8:]
            psel_offset = register_string(sig.perf_section)
            copt_offset = register_string(sig.copt_section)
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
        lut_min = int(np.min(lut_tensor))
        lut_max = int(np.max(lut_tensor))
        for dtype in [np.int8, np.int16, np.int32, np.int64]:
            info = np.iinfo(dtype)
            if info.min <= lut_min and lut_max <= info.max:
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

    def codegen_deduplicated_pp_args_function_index(self, functional : Functional):
        kdesc = self._f.meta_object

        pp_registry = self._parent_repo.get_signatured_function_registry('pp_function')
        # IR-neutral: pp_arg_doc(aname) -> (is_constexpr, comment_value). Works for
        # both legacy Bind-backed and ATI Override-backed functionals.
        doc = { larg.aname : functional.pp_arg_doc(larg.aname)
                for larg in kdesc.iter_launch_arguments() }
        assign_skips = tuple([doc[larg.aname][0]
                              for larg in kdesc.iter_launch_arguments()])
        hit, findex = pp_registry.contains(assign_skips)
        if hit:
            return findex
        stmt = []
        for larg in kdesc.iter_launch_arguments():
            is_constexpr, comment_value = doc[larg.aname]
            assign = larg.expr + f', // {larg.aname}'
            # Comment out constexpr values
            if is_constexpr:
                assign = '// ' + assign + f' as constexpr {comment_value}'
            stmt.append(assign)
        stmt.append('CAST(&aux.global_scratch),')
        stmt.append('CAST(&aux.profile_scratch)')
        pfx = '  return { '
        join = '\n' + ' ' * len(pfx)
        sfx = '         };'
        # Do NOT join with ','. There are comment text after the parameter.
        src = pfx + join.join(stmt) + '\n' + sfx
        return pp_registry.register(assign_skips, src)

    def codegen_perf_assignment(self):
        kdesc = self._f.meta_object
        ALIGN = ';\n' + 4 * ' '
        stmt = []
        for aname in kdesc.gen_performance_params():
            stmt.append(f'context.{aname} = perf.{aname}')
        return ALIGN.join(stmt)

    @property
    def all_signatures(self):
        return self._sigs

