#!/usr/bin/env python

from pathlib import Path
import json

SOURCE_PATH = Path(__file__).resolve()

def _get_template(name='kernel_shim.cc'):
    with open(SOURCE_PATH.parent.parent / 'csrc' / 'template' / name, 'r') as f:
        return f.read()


class ObjectFileDescription(object):
    SIGNATURE_TO_C = {
        'fp32'  : 'float',
        '*fp32' : 'const float*',
        '*fp16' : 'const __fp16*',
        '*bf16' : 'const __bf16*',
        'i32'   : 'int32_t',
        'i64'   : 'int64_t',
        'u32'   : 'uint32_t',
        'u64'   : 'uint64_t',
    }
    num_warps = 4
    num_stages = 4
    CXX_TEMPLATE = _get_template()
    CXX_HEADER_TEMPLATE_HEADER = _get_template('kernel_shim.header.h')
    CXX_HEADER_TEMPLATE_FOOTER = _get_template('kernel_shim.footer.h')

    def __init__(self,
                 triton_kernel_desc : 'KernelDescription',
                 choice: list[frozenset[str], list[str]],
                 signature_in_list : list[str],
                 hsaco_kernel_path : Path):
        self._triton_kernel_desc = triton_kernel_desc
        self.SHIM_KERNEL_NAME = self._triton_kernel_desc.SHIM_KERNEL_NAME
        self._argument_choices = choice
        self._sigature_in_list = signature_in_list
        self._hsaco_kernel_path = Path(hsaco_kernel_path)
        # self._hsaco_metatdata_path = Path() if triton_metadata_path is None else self._triton_file_path.with_suffix('.json')
        self._hsaco_metatdata_path = self._hsaco_kernel_path.with_suffix('.json')
        if self._hsaco_metatdata_path.exists():
            with self._hsaco_metatdata_path.open('r') as f:
                self._metadata = json.load(f)
        else:
            self._metadata = None

    @property
    def signature_c_mangle(self):
        # * -> ^: Pascal Pointer
        # : -> @: A(@)lign
        mangle_sig = [ str(t).replace('*', 'P').replace(':', 'A') for t in self._sigature_in_list ]
        return '_'.join(mangle_sig)

    @property
    def src(self):
        return self._triton_kernel_desc._triton_file_path

    @property
    def entrance(self):
        return self._triton_kernel_desc._triton_kernel_name

    @property
    def obj(self):
        return self._hsaco_kernel_path

    @property
    def signature(self):
        return ', '.join([ str(t) for t in self._sigature_in_list ])

    def generate_shim_source(self) -> str:
        shim_arguments, casted_shim_parameters = self.compute_c_argument()
        # template_arguments, template_constants = self.compute_template_arguments()
        template_specialization = self.compute_struct_template_specialization(align1=len(self.SHIM_KERNEL_NAME)+1)
        fmt = {
                'hsaco_kernel_name' : self._metadata['name'],
                'incbin_symbol_name' : self.SHIM_KERNEL_NAME + '__' + self.signature_c_mangle,
                'hsaco_kernel_path' : self._hsaco_kernel_path.absolute(),
                'shim_kernel_name' : self.SHIM_KERNEL_NAME,
                'shim_kernel_specialization' : template_specialization,
                'shared_memory_size' : self._metadata['shared'],
                'shim_arguments' : shim_arguments,
                'casted_shim_parameters' : casted_shim_parameters,
               }
        return ObjectFileDescription.CXX_TEMPLATE.format_map(fmt)

    def generate_shim_header_leading(self) -> str:
        fmt = {
                'template_constants': self.compute_struct_template_typenames(),
                'shim_kernel_name': self.SHIM_KERNEL_NAME,
        }
        return ObjectFileDescription.CXX_HEADER_TEMPLATE_HEADER.format_map(fmt)

    def generate_shim_header_member_function(self) -> str:
        TEMPLATE = ' hipError_t operator()(dim3 grid, dim3 block, {shim_arguments}, hipStream_t stream);\n'
        shim_arguments, _ = self.compute_c_argument()
        fmt = {
                'shim_arguments': shim_arguments,
        }
        return TEMPLATE.format_map(fmt)

    def generate_shim_header_closing_struct_define(self) -> str:
        return '};\n\n'

    def generate_shim_header_extern_template(self) -> str:
        TEMPLATE = 'template struct {shim_kernel_name}<{shim_kernel_specialization}>;'
        template_specialization = self.compute_struct_template_specialization(align1=len(self.SHIM_KERNEL_NAME)+17)
        fmt = {
            'shim_kernel_name': self.SHIM_KERNEL_NAME,
            'shim_kernel_specialization': template_specialization,
        }
        return TEMPLATE.format_map(fmt)

    def generate_shim_header_trailing(self) -> str:
        return self.CXX_HEADER_TEMPLATE_FOOTER

    def compute_c_argument(self, align1=23, align2=30):
        arguments = self.get_c_arguments()
        typed_arguments = [f'{self.get_ctype(a)[1]} {a}' for a in arguments]
        casted_arguments = [f'static_cast<void*>(&{a})' for a in arguments]
        ALIGN1 = ',\n' + ' ' * align1
        ALIGN2 = ',\n' + ' ' * align2
        return ALIGN1.join(typed_arguments), ALIGN2.join(casted_arguments)

    def sanitize_type(self, t):
        colon = t.rfind(':')
        colon = None if colon < 0 else colon
        return t[:colon]

    def get_ctype(self, arg_name):
        for k, v in self._argument_choices.items():
            if arg_name in k:
                stemv = self.sanitize_type(str(v))
                not_constant = stemv in self.SIGNATURE_TO_C
                if not_constant:
                    ctype = self.SIGNATURE_TO_C[stemv]
                elif isinstance(v, bool):
                    ctype = 'bool'
                elif isinstance(v, int):
                    ctype = 'int32_t'
                elif isinstance(v, float):
                    ctype = 'float'
                return not_constant, ctype
        assert False, f"cannot find {arg_name}"

    def get_cvalue(self, arg_name):
        for k, v in self._argument_choices.items():
            if arg_name in k:
                if isinstance(v, bool):
                    return 'true' if v else 'false'
                return v
        assert False, f"cannot find {arg_name}"

    def get_c_arguments(self):
        return self._filter_arguments(request_constants=False)

    def get_template_arguments(self):
        return self._filter_arguments(request_constants=True)

    def _filter_arguments(self, request_constants):
        ret = []
        for a in self._triton_kernel_desc.ARGUMENTS:
            not_constant, _ = self.get_ctype(a)
            if request_constants:
                if not not_constant:
                    ret.append(a)
            else:
                if not_constant:
                    ret.append(a)
        return ret

    # Returns things for
    # extern template<int STAGE,
    #                 int BLOCK_M,
    #                 ...
    #                >
    def compute_struct_template_typenames(self, align1=9):
        constants = self._filter_arguments(request_constants=True)
        typed_constants = []
        for a in constants:
            typed_constants.append(f'{self.get_ctype(a)[1]} {a}')
        ALIGN1 = ',\n' + ' ' * align1
        return ALIGN1.join(typed_constants)

    def compute_struct_template_specialization(self, align1=13):
        constants = self._filter_arguments(request_constants=True)
        constant_values_with_hint = []
        for a in constants:
            v = str(self.get_cvalue(a))
            v += f' /* {a} */'
            constant_values_with_hint.append(v)
        ALIGN1 = ',\n' + ' ' * align1
        return ALIGN1.join(constant_values_with_hint)

    """
    def compute_template_arguments(self, align1=1, align2=10):
        arguments = self.get_template_arguments()
        type_choices = self._argument_choices
        typed_arguments = []
        template_constants = []
        typename_allocation = {}
        # do NOT loop over type_choices directly.
        # We want to maintain the order of typename Type#
        for a in self._triton_kernel_desc.ARGUMENTS:
            for k, v in type_choices.items():
                if a not in k:
                    continue
                if isinstance(v, str):
                    allocated = False
                    for k in typename_allocation.keys():
                        if a in k:
                            allocated = True
                            break
                    if not allocated:
                        index = len(typed_arguments)
                        tname = f'typename Type{index}'
                        typed_arguments.append(tname)
                        typename_allocation[k] = tname
                elif isinstance(v, bool):
                    cv = 'true' if v else 'false'
                    template_constants.append(f'bool {a}  = {cv}')
                else:
                    template_constants.append(f'{self.get_ctype(v)} {a}  = {v}')

        ALIGN1 = ',\n' + ' ' * align1
        ALIGN2 = ';\n' + ' ' * align2
        return ALIGN1.join(typed_arguments), ALIGN2.join(template_constants)
    """
