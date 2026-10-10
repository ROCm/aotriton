# Copyright © 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Generate <family>/flyc.<kernel_name>.{h,cc}
#
# The flyc analogue of kernel.py's KernelShimGenerator. It mirrors that closely;
# the two places it does not are where flyc's kernarg ABI is not the operator's:
#   (1) pp_args functions take `(const {context_class_name}& context)`, not
#       `(const {param_class_name}& params, const TritonAuxiliaryArguments& aux)`
#       -- a flyc kernel takes exactly its own real parameters, none of them one
#       of the two trailing scratch pointers a Triton kernel takes. A
#       `const auto& params = *context.params;` alias line is prepended so the
#       registered pp_args source (which still refers to `params.*` for
#       tensor/scalar operands) keeps working unmodified.
#   (2) flyc.h declares per-kernel "context helpers" -- host-side computations
#       hand-implemented in csrc/, exactly as grid_calculator() already is -- as
#       const member functions on the context struct, plus mutable scratch
#       members caching their return values for pp_args to take the address of.

from ..template_instantiation.ir import Functional
from .interface import InterfaceGenerator
from .template import get_template
from ..utils import LazyFile, log
from .common import codegen_struct_cfields, codegen_includes
from .flytune import FlycTuneCodeGenerator


# TODO(de-duplication): this generator and codegen/kernel.py's
# KernelShimGenerator are substantially parallel, as are the flyc.{h,cc}
# templates against shim.{h,cc} (measured: ~34 of ~200 lines differ in the .cc,
# ~31 of ~110 in the .h). The duplication is deliberate for now -- flyc's shape
# was still moving -- but it is real debt: a fix or feature applied to one side
# silently misses the other. The `kctl.control_bits` handling is the concrete
# example, appearing three times in each template.
#
# Unify once flyc stops moving. The divergences are all parameterisable: the
# header name, the PP_FUNC signature (flyc has no TritonAuxiliaryArguments),
# one Pon line in lookup_optimal, and the tune namespace.

class FlycShimGenerator(InterfaceGenerator):
    HEADER_TEMPLATE = get_template('flyc.h')
    SOURCE_TEMPLATE = get_template('flyc.cc')
    PFX = 'flyc'

    def create_sub_generator(self, functional: Functional, df: 'pandas.DataFrame', sql: tuple):
        if functional.meta_object.is_functional_disabled(functional):
            log(lambda: f'Functional {functional.godel_number=} disabled')
            return None, False
        return FlycTuneCodeGenerator(self._args, functional, df, sql, self._this_repo), True

    def write_shim_header(self, functionals, fout):
        kdesc = self._iface
        shared_iface = kdesc.SHARED_IFACE is not None
        if shared_iface:
            self._add_iface_for_source(kdesc.SHARED_IFACE)
        shared_iface_family = kdesc.SHARED_IFACE.FAMILY if shared_iface else kdesc.FAMILY
        d = {
            'kernel_family_name'    : kdesc.FAMILY,
            'shim_kernel_name'      : kdesc.NAME,
            'flyc_gpu_symbol'       : kdesc.gpu_symbol_name,
            'param_class_name'      : kdesc.param_class_name,
            'shared_iface_family'   : shared_iface_family,
            'shared_iface'          : 1 if shared_iface else 0,
            'call_options_struct'   : kdesc.SHARED_IFACE.CALL_OPTIONS_NAME if shared_iface else 'void',
            'context_class_name'    : kdesc.context_class_name,
            'metadata_class_name'   : kdesc.metadata_class_name,
            'func_fields'           : codegen_struct_cfields(kdesc.func_cfields, nalign=4),
            'context_helper_declares'        : self.codegen_context_helper_declares(),
            'context_helper_scratch_members' : self.codegen_context_helper_scratch_members(),
            'compiled_rung_table_declares'   : self.codegen_compiled_rung_table_declares(),
            'kernel_table_entry_declares'   : self.codegen_tune_table_entry_declares(functionals),
            'number_of_functionals' : kdesc.godel_number,
            'declare_list_of_deduplicated_lut_functions' : self.codegen_declare_list_of_deduplicated_lut_functions(),
        }
        d['includes'] = codegen_includes(self._hdr_include_repo.get_data())
        print(self.HEADER_TEMPLATE.format_map(d), file=fout)

    def write_shim_source(self, functionals, fout):
        kdesc = self._iface
        shared_iface = kdesc.SHARED_IFACE is not None
        shared_iface_family = kdesc.SHARED_IFACE.FAMILY if shared_iface else kdesc.FAMILY
        list_of_pp_args_function_defs, list_of_pp_args_function_decls, pp_func_num = self.codegen_kernel_arguments()
        d = {
            'shared_iface'        : 1 if shared_iface else 0,
            'shared_iface_family' : shared_iface_family,
            'call_options_struct' : kdesc.SHARED_IFACE.CALL_OPTIONS_NAME if shared_iface else 'void',
            'kernel_family_name'  : kdesc.FAMILY,
            'shim_kernel_name'    : kdesc.NAME,
            'flyc_gpu_symbol'     : kdesc.gpu_symbol_name,
            'param_class_name'    : kdesc.param_class_name,
            'context_class_name'  : kdesc.context_class_name,
            'godel_number_body'   : self.codegen_godel_number_body(),
            'context_helper_evaluate' : self.codegen_context_helper_evaluate(),
            'compiled_rung_table_defs' : self.codegen_compiled_rung_table_defs(functionals),
            'pp_func_num'         : pp_func_num,
            'list_of_pp_args_function_defs'  : list_of_pp_args_function_defs,
            'list_of_pp_args_function_decls' : list_of_pp_args_function_decls,
            'get_archmod_number_body' : self.codegen_archmod_number_body(),
            'number_of_functionals'  : kdesc.godel_number,
            'per_kernel_packed_string'  : self.codegen_per_kernel_packed_string(),
            'kernel_table_entries' : self.codegen_tune_table_entries(functionals),
            'list_of_deduplicated_lut_functions' : self.codegen_list_of_deduplicated_lut_functions(),
        }
        d['includes'] = codegen_includes(self._src_include_repo.get_data())
        print(self.SOURCE_TEMPLATE.format_map(d), file=fout)

    def codegen_per_kernel_packed_string(self):
        # Same "zero enabled functionals never lazily creates the registry"
        # gap as codegen_kernel_arguments above -- an empty packed string is
        # the correct answer, not a crash. But the template substitutes this
        # value directly as the initializer of `const char foo[] =\n{value}\n;`
        # with no surrounding braces (see template/flyc.cc), so the value must
        # itself be a valid C++ string-literal expression -- '""', not ''.
        return self._this_repo.get_data('per_kernel_packed_string', return_none=True) or '""'

    # codegen_declare_compiled_in_features / codegen_define_compiled_in_features
    # used to live here, mirroring kernel.py's KernelShimGenerator (which still
    # has them, and whose Triton-axis binning site in attn_fwd.cc/attn_bwd.cc
    # still calls the equivalent AttnFwdMetadata::get_BLOCK_DMODEL_choices()).
    # There is deliberately no flyc analogue. Such a table would enumerate an
    # axis's DECLARED choices, not the arch-specific COMPILED subset
    # autotune_table actually carries -- and the compiled subset is precisely
    # what a helper rounding a head dimension has to consult. That distinction
    # is what the compiled_<axis> / compiled_<axis>_count arrays
    # (codegen_compiled_rung_table_defs, above) exist to get right, so a second,
    # untrue, unused table would be dead weight at best and a trap for a future
    # caller at worst.

    def _rung_table_params(self):
        """Functional axes that get a compiled-rung table: helper-wired
        (`context_helper_for_functional` answers) and not a plain bool.
        PADDED_HEAD is itself boolean and is derived from the
        rounding decision made against BLOCK_DMODEL's table
        (modules/flash/csrc/<kernel>.cc's hand-written helper), not
        table-driven on its own -- excluded here even once wired, so it gets
        no (degenerate, {false,true}) table of its own."""
        kdesc = self._iface
        params = []
        for tp in kdesc.list_functional_params():
            helper_name = kdesc.context_helper_for_functional(tp.repr_name)
            if helper_name is None:
                continue
            if tp.repr_typed_choice.itype == 'bool':
                continue
            params.append(tp)
        return params

    def codegen_compiled_rung_table_declares(self):
        params = self._rung_table_params()
        if not params:
            return '// no compiled-rung table for this kernel'
        lines = []
        for tp in params:
            lname = tp.repr_name.lower()
            ctype = tp.repr_typed_choice.itype
            lines.append(f'static const {ctype}* const compiled_{lname}[];')
            lines.append(f'static const int compiled_{lname}_count[];')
        return '\n    '.join(lines)

    def codegen_compiled_rung_table_defs(self, functionals):
        """GENERATE the per-arch compiled rung table(s), one per axis
        `_rung_table_params` finds, from the same `functionals` list that fills
        autotune_table -- not from a second, hand-maintained ladder -- so the
        table cannot silently drift from what this kernel actually compiles.

        The property that makes the table trustworthy -- that it equals the set
        of values whose functionals were not disabled -- is asserted here rather
        than claimed: `expected` is recomputed independently, via a fresh
        `gen_functionals` pass filtered by `is_functional_disabled` directly,
        bypassing the `functionals` argument entirely. If some future change
        filtered `functionals` by anything other than that predicate before
        it reached this generator, this assertion -- not a code reviewer --
        would be the one to notice."""
        kdesc = self._iface
        params = self._rung_table_params()
        if not params:
            return ''
        context_class_name = kdesc.context_class_name
        blocks = []
        for tp in params:
            axis = tp.axis
            lname = tp.repr_name.lower()
            ctype = tp.repr_typed_choice.itype
            row_defs = []
            row_names = []
            counts = []
            for arch_number, target_arch in enumerate(self._target_arch_keys):
                surviving = sorted({
                    f.resolved[axis.repr_arg].triton_compile_signature
                    for f in functionals if f.arch == target_arch})
                expected = sorted({
                    f.resolved[axis.repr_arg].triton_compile_signature
                    for f in kdesc.gen_functionals({target_arch: self._target_arch[target_arch]})
                    if not kdesc.is_functional_disabled(f)})
                assert surviving == expected, (
                    f'flyc kernel {kdesc.NAME!r}: compiled-rung table for '
                    f'{tp.repr_name!r} on {target_arch!r} ({surviving}) does not '
                    f'match the independently recomputed not-disabled set '
                    f'({expected}) -- functionals passed to write_shim_source '
                    f'diverged from is_functional_disabled')
                # No non-emptiness check here on purpose: these flyc kernels'
                # own `@ati.disable` predicates reject every arch but their
                # one home arch unconditionally (e.g. `f.arch != 'gfx1201':
                # return True`), so in any REAL multi-arch build `surviving`
                # is legitimately empty for every other arch_number -- that
                # is not a bug, it is autotune_table's null row restated for
                # this table. It is still safe at runtime: a helper reading
                # an empty row can only feed a wrong digit into
                # godel_number(), and godel_number() on that arch already
                # maps every digit to a null tune_func (same predicate), so
                # there is no godel number a wrong rounding could alias into.
                row_name = f'{context_class_name}_compiled_{lname}_{arch_number}'
                row_names.append(row_name)
                # A SENTINEL, not nothing, when the row is empty. `int16_t
                # row[] = {  };` is a zero-length array: a GNU extension, not
                # ISO C++, and -pedantic-errors rejects it -- and the row IS
                # legitimately empty in the common case above (a gfx942-only
                # build compiles no flyc kernel at all, so every one of these
                # rows is empty and nothing configures). The count array stays
                # at the true 0, which is the only thing flyc_round_up_rung
                # reads, so the sentinel is unreachable by construction; -1 is
                # spelled out anyway because it is that function's own
                # "nothing fits" answer, so a row that somehow WERE read still
                # yields the reject path rather than a plausible rung.
                literal = ', '.join(str(v) for v in surviving) if surviving else '-1'
                row_defs.append(f'static constexpr {ctype} {row_name}[] = {{ {literal} }};')
                counts.append(str(len(surviving)))
            ptr_array = (f'const {ctype}* const {context_class_name}::compiled_{lname}[] = '
                         f'{{ {", ".join(row_names)} }};')
            count_array = (f'const int {context_class_name}::compiled_{lname}_count[] = '
                           f'{{ {", ".join(counts)} }};')
            blocks.append('\n'.join(row_defs + [ptr_array, count_array]))
        return '\n\n'.join(blocks)

    def codegen_context_helper_declares(self):
        kdesc = self._iface
        lines = [f'{ctype} {name}() const;' for name, ctype in kdesc.iter_context_helpers()]
        return '\n    '.join(lines)

    def codegen_context_helper_evaluate(self):
        """Fills `[[context_helper_evaluate]]` in `lookup_optimal()`: evaluate
        every context helper exactly once, there.

        `iter_context_helpers` is per-description and `lookup_optimal` is a
        per-description method, so that is the natural home -- one evaluation,
        rather than one per surviving functional sharing a deduplicated pp_args
        registration, which is what evaluating from pp_args would give.

        The slot is spliced directly into the member function
        `[[context_class_name]]::lookup_optimal`, so there is no `context`
        local: `this` is implicit and the members are named directly."""
        kdesc = self._iface
        lines = [f'scratch_params.{name} = {name}();'
                 for name, ctype in kdesc.iter_context_helpers()]
        if not lines:
            return '// no context helpers'
        return '\n    '.join(lines)

    def codegen_context_helper_scratch_members(self):
        """Storage for the ati.context_helper() results.

        Grouped into one struct rather than loose members so the kernarg
        vector's `CAST(&context.scratch_params.<name>)` reads as one namespace,
        and so adding a helper does not add another top-level context field.
        `mutable` because pp_args fills them from a const context, and the
        kernarg vector holds their addresses -- a local would dangle.
        """
        kdesc = self._iface
        helpers = list(kdesc.iter_context_helpers())
        if not helpers:
            return '// no context helpers'
        body = '\n'.join(f'        {ctype} {name};' for name, ctype in helpers)
        return 'mutable struct {\n' + body + '\n    } scratch_params;'

    def codegen_kernel_arguments(self):
        context_class_name = self._iface.context_class_name
        # return_none=True: an arch where every functional of this kernel is
        # disabled (e.g. flyc kernels on an arch that has not been wired up
        # yet) never runs a single FlycTuneCodeGenerator, so the registry is
        # never lazily created. Zero functionals means zero pp_args
        # functions, not a bug -- do not let it raise.
        pp_registry = self._this_repo.get_data('pp_function', return_none=True) or {}
        stmt = []
        array = []
        for pp_signature, (findex, src) in pp_registry.items():
            pp_function_name = f'{self._iface.NAME}_pp_args_{findex}'
            stmt.append(f'static std::vector<void*>')
            stmt.append(f'{pp_function_name}(const {context_class_name}& context) {{')
            stmt.append(f'  const auto& params = *context.params;')
            stmt.append(src)
            stmt.append(f'}}')
            array.append(pp_function_name)
        pp_func_num = len(pp_registry.keys())
        if pp_func_num == 0:
            # Same zero-length-array problem as the rung table above, and the
            # same shape of answer. `PP_FUNC prepare_arguments[ 0 ]` is a GNU
            # extension that -pedantic-errors rejects, and an arch with every
            # functional disabled -- which is EVERY arch in a build that
            # targets neither gfx1201 nor gfx950 -- registers no pp_args at
            # all. One null entry keeps the declaration and the definition
            # well-formed and agreeing on the bound; nothing can index it,
            # because lookup_optimal() rejects the functional before launch()
            # reaches prepare_arguments[pp_args_index] (the autotune table is
            # all-null on that arch too, by the same predicate).
            pp_func_num = 1
            array.append('nullptr')
        return '\n'.join(stmt), ',\n  '.join(array), pp_func_num
