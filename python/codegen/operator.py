# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Generate <family>/op.<op_name>.{h,cc}

import io
from ..template_instantiation.ir import (
    typed_choice as TC,
    Functional,
    Interface,
)
from .interface import InterfaceGenerator
from .template import get_template
from ..utils import (
    LazyFile,
)
from .common import codegen_struct_cfields, codegen_includes
from .optune import OptuneCodeGenerator

class OperatorGenerator(InterfaceGenerator):
    HEADER_TEMPLATE = get_template('op.h')
    SOURCE_TEMPLATE = get_template('op.cc')
    KSHIM_LAUNCHER_TEMPLATE = get_template('kshim_launcher.cc')
    METRO_LAUNCHER_TEMPLATE = get_template('metro_launcher.cc')
    METRO_SNIPPET_TEMPLATE = get_template('snippet/metro_per_kernel.cc')
    IFELSE_SNIPPET_TEMPLATE = get_template('snippet/metro_per_kernel_ifelse.cc')
    METRO_LAUNCH_SNIPPET_TEMPLATE = get_template('snippet/metro_launch_kernel.cc')
    IFELSE_LAUNCH_SNIPPET_TEMPLATE = get_template('snippet/metro_launch_kernel_ifelse.cc')
    PFX = 'iface'

    def create_sub_generator(self, functional : Functional, df : 'pandas.DataFrame', sql : tuple):
        ocg = OptuneCodeGenerator(self._args, functional, df, self._this_repo)
        if not ocg.is_trivial:
            return ocg, True
        else:
            ocg.generate_trivial()
        return None, True

    def codegen_tune_struct_name(self, arch_number, godel_number):
        tt_dict = self._this_repo.get_data('trivial_tunes')
        trivial_enum = tt_dict.get((arch_number, godel_number), None)
        if trivial_enum is None:
            return super().codegen_tune_struct_name(arch_number, godel_number)
        tune_name = self._iface.TUNE_NAME
        is_extern = False
        return f'{tune_name}_{self._iface.NAME}__Trivial_{trivial_enum}', is_extern

    def write_shim_header(self, functionals, fout):
        iface = self._iface
        d = {
            'family_name'           : iface.FAMILY,
            'param_class_name'      : iface.param_class_name,
            'call_options_struct'   : iface.CALL_OPTIONS_NAME,
            'context_class_name'    : iface.context_class_name,
            'func_fields'           : codegen_struct_cfields(iface.func_cfields, nalign=4),
            'list_of_backend_enum'  : self.codegen_backend_enums(nalign=8),
            'fallback_backend'      : iface.fallback_backend.enum_name,
            'total_number_of_backends'      : self._iface.nbackends,
            'optune_table_entry_declares'   : self.codegen_tune_table_entry_declares(functionals),
            'number_of_functionals' : iface.godel_number,
            'declare_list_of_deduplicated_lut_functions' : self.codegen_declare_list_of_deduplicated_lut_functions(),
        }
        print(self.HEADER_TEMPLATE.format_map(d), file=fout)

    def write_shim_source(self, functionals, fout):
        iface = self._iface
        d = {
            'family_name'               : iface.FAMILY,
            'iface_name'                : iface.NAME,
            'param_class_name'          : iface.param_class_name,
            'context_class_name'        : iface.context_class_name,
            'godel_number_body'         : self.codegen_godel_number_body(),
            'get_archmod_number_body'   : self.codegen_archmod_number_body(),
            'def_trivial_tunes'         : self.codegen_trivial_tunes(),
            'optune_table_entries'      : self.codegen_tune_table_entries(functionals),
            'number_of_functionals'     : iface.godel_number,
            'def_backend_launchers'     : self.codegen_launchers(nalign=0),
            'launcher_table_entries'    : self.codegen_launch_table_entries(nalign=4),
            'list_of_deduplicated_lut_functions' : self.codegen_list_of_deduplicated_lut_functions(),
        }
        d['includes'] = codegen_includes(self._src_include_repo.get_data())
        print(self.SOURCE_TEMPLATE.format_map(d), file=fout)

    def codegen_backend_enums(self, nalign):
        stmt = []
        for i, backend in enumerate(self._iface.list_backends()):
            stmt.append(f'{backend.enum_name} = {i}')
        ALIGN = ',\n' + ' ' * nalign
        return ALIGN.join(stmt)

    def codegen_launchers(self, nalign):
        iface = self._iface
        stmt = []
        for backend in iface.list_backends():
            stmt.append(self.codegen_single_launcher(backend, nalign))
        ALIGN = '\n\n'
        return ALIGN.join(stmt)

    def codegen_single_launcher(self, backend : Interface, nalign):
        # Dispatch by CODEGEN_MODULE (duck-typed), not class: a single-kernel (kshim)
        # backend is a triton kernel, the slim affine kernel, or a flyc kernel
        # ('triton'/'affine'/'flyc') -- all three are one kernel behind one
        # context, which is the whole of what the kshim launcher needs. A metro
        # launcher is 'op' and exposes list_kernels().
        cgmod = getattr(backend, 'CODEGEN_MODULE', None)
        if cgmod in ('triton', 'affine', 'flyc'):
            return self.codegen_kshim_launcher(backend, nalign)
        if cgmod == 'op' and hasattr(backend, 'list_kernels'):
            return self.codegen_metro_launcher(backend, nalign)
        assert False, f'Unsupported backend class {backend.__class__}'

    def codegen_kshim_launcher(self, kdesc : Interface, nalign):
        iface = self._iface
        stmt = []
        self._add_iface_for_source(kdesc)
        d = {
            'context_class_name'    : iface.context_class_name,
            'launcher_func_name'    : self.codegen_launcher_func_name(kdesc),
            'backend_context_name'  : kdesc.context_class_name,
        }
        return self.KSHIM_LAUNCHER_TEMPLATE.format_map(d)

    def codegen_metro_launcher(self, metro : Interface, nalign):
        iface = self._iface
        context_class_name = iface.context_class_name
        lookup_stmt = []
        launch_stmt = []

        # FIXME: lookup all and then launch, in case any sub-kernel failed
        for nth, kdesc in enumerate(metro.list_kernels()):
            if hasattr(kdesc, 'if_kernel'):       # a ConditionalKernel step
                self._add_iface_for_source(kdesc.if_kernel)
                d = {
                    'condition'             : f'context.params->{kdesc.if_parameter} {kdesc.if_expr}',
                    'backend_context_name'  : kdesc.if_kernel.context_class_name,
                    'nth_kernel'            : nth,
                }
                if kdesc.else_kernel is None:
                    snippet = self.METRO_SNIPPET_TEMPLATE.format_map(d)
                    launch_snippet = self.METRO_LAUNCH_SNIPPET_TEMPLATE.format_map(d)
                else:
                    self._add_iface_for_source(kdesc.else_kernel)
                    d['else_context_name'] = kdesc.else_kernel.context_class_name
                    snippet = self.IFELSE_SNIPPET_TEMPLATE.format_map(d)
                    launch_snippet = self.IFELSE_LAUNCH_SNIPPET_TEMPLATE.format_map(d)
            else:
                self._add_iface_for_source(kdesc)
                d = {
                    'condition'             : 'true',
                    'backend_context_name'  : kdesc.context_class_name,
                    'nth_kernel'            : nth,
                }
                snippet = self.METRO_SNIPPET_TEMPLATE.format_map(d)
                launch_snippet = self.METRO_LAUNCH_SNIPPET_TEMPLATE.format_map(d)
            lookup_stmt.append(snippet)
            launch_stmt.append(launch_snippet)
        launch_stmt.append('return hipSuccess;')
        d = {
            'context_class_name'    : iface.context_class_name,
            'launcher_func_name'    : self.codegen_launcher_func_name(metro),
            'lookup_every_kernel'   : '\n'.join(lookup_stmt),
            'launch_every_kernel'   : '\n'.join(launch_stmt),
        }
        return self.METRO_LAUNCHER_TEMPLATE.format_map(d)

    def codegen_launcher_func_name(self, backend):
        return f'launcher_for_{backend.enum_name}'

    def codegen_launch_table_entries(self, nalign):
        iface = self._iface
        stmt = [ '&' + self.codegen_launcher_func_name(b) for b in iface.list_backends() ]
        ALIGN = ',\n' + ' ' * nalign
        return ALIGN.join(stmt)

    def codegen_trivial_tunes(self):
        trivial_tunes = self._this_repo.get_data('trivial_tunes')
        uniques = sorted(set(trivial_tunes.values()))
        context_class_name = self._iface.context_class_name
        tune_name = self._iface.TUNE_NAME
        stmt = []
        for trivial_enum in uniques:
            stmt.append(f'int {tune_name}_{self._iface.NAME}__Trivial_{trivial_enum}({context_class_name}& context, int) {{')
            stmt.append(f'    context.backend_index = {context_class_name}::BackendEnum::{trivial_enum};')
            stmt.append(f'    return context.backend_index;')
            stmt.append('}')
            stmt.append('')
        return '\n'.join(stmt)


# --- the same numbering, published -------------------------------------------
#
# `OperatorGenerator.codegen_backend_enums` writes BackendEnum into the INTERNAL
# iface.<op>.h. The index is also an ABI: `attn_options::force_backend_index`
# takes it, so tests and tuning tools name it too -- and until now they named it
# as a bare integer, with the mapping living in comments
# (modules/flash/tune/level_op.py) that nothing checked. Three constants in this
# codebase went that way and drifted: CausalType is declared in
# include/aotriton/flash.h AND by hand in modules/flash/tests/aotriton_flash.py,
# and the two disagree about which members exist.
#
# So the numbering is published from the same list that assigns it, and every
# other spelling derives from this one.
#
# Free functions over an Operator, not methods on the generator: a full build
# fans out per-operator workers, each generating with --selective, so the parent
# process never runs the generator loop and could not collect these from it. The
# operator itself is all they need.


def _public_backend_constant(declared_name):
    """`'triton_fuse'` -> `'kTritonFuse'`.

    The PUBLIC constant, named from what @ati.backend declared and nothing else.
    Deliberately not `backend.enum_name`, which prefixes kMetro_ / kShim_ /
    kSlimAffine_ / kFlyc_ according to how the backend is assembled. That is a
    property of the generator, and a caller choosing a backend has no reason to
    know it -- or to have their code break when a backend is re-shaped from a
    bare kernel into a metro without its identity changing.

    The internal BackendEnum in iface.<op>.h keeps the prefixed spelling: it
    names generated launchers and is where the assembly shape is genuinely the
    subject.
    """
    return 'k' + ''.join(part.capitalize() for part in declared_name.split('_'))


def backend_constants_struct_name(op):
    """`OpAttnFwdBackend` for operator `op_attn_fwd` -- the same derivation
    `param_class_name`/`context_class_name` use, so the names an operator
    contributes stay recognisably one family."""
    return op.context_class_name.removesuffix('Context') + 'Backend'


def codegen_backend_constants(op):
    """The public `struct <Op>Backend { static constexpr int32_t ... }`.

    A struct of `static constexpr int32_t`, not an `enum class`, matching
    CausalType / VarlenType / WindowValue in include/aotriton/flash.h and for the
    reason recorded there: an enum class needs a cast to reach its underlying
    type, and `force_backend_index` is a plain int.

    Member names come from `op.backend_names` -- what @ati.backend declared --
    through `_public_backend_constant`, and the X-macro below is built from the
    same list in the same order, so the published constant and the binding that
    exposes it cannot disagree about a name any more than about a value."""
    name = backend_constants_struct_name(op)
    lines = [f'struct AOTRITON_API {name} {{']
    for i, declared in enumerate(op.backend_names):
        lines.append(f'  static constexpr int32_t {_public_backend_constant(declared)} = {i};')
    lines.append(f'  static constexpr int32_t Max = {op.nbackends};')
    lines.append('};')
    return '\n'.join(lines)


def codegen_backend_constant_xmacro(op):
    """A per-struct X-macro listing that operator's constants and their names.

    Two arguments per row: the C++ constant (`kFlyc`) and the backend's declared
    name (`"flyc"`), the one `@ati.backend` was written with. Both, because a
    caller pinning a backend wants to say which one in the vocabulary the
    description uses, not in enum spelling -- and neither should be transcribed
    by hand into a test.

    Per struct rather than one list of (struct, name) pairs: a binding expanding
    a mixed list would have to pick the right target per row, and the targets are
    different C++ types, so no single expression does it. One macro per struct
    expands against one already-chosen target.

    An X-macro rather than generated pybind: the binding stays hand-written in
    modules/<family>/bindings/, which is where a reader looks for it, but the
    NAMES come from here. A binding listing them literally would be the fourth
    copy of this list, and the third one drifted.
    """
    name = backend_constants_struct_name(op)
    rows = ' \\\n'.join(f'  X({_public_backend_constant(d)}, "{d}")'
                        for d in op.backend_names)
    return f'#define AOTRITON_BACKENDS_{name}(X) \\\n{rows}'
