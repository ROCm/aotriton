from .rules import kernels as triton_kernels
import io
import shutil
import argparse
from pathlib import Path

SOURCE_PATH = Path(__file__).resolve()
CSRC = (SOURCE_PATH.parent.parent / 'csrc').absolute()
INCBIN = (SOURCE_PATH.parent.parent / 'third_party/incbin/').absolute()
# COMPILER = SOURCE_PATH.parent / 'compile.py'
COMPILER = 'hipcc'
LINKER = 'ar'

LIBRARY_NAME = 'libaotriton'

def parse():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--build_dir", type=str, default='build/', help="build directory")
    p.add_argument("--archive", action='store_true', help='generate archive library instead of shared library. No linking with dependencies.')
    args = p.parse_args()
    # print(args)
    return args

'''
ShimMakefileGenerator
 +- generate libaotriton.a
 +- KernelShimGenerator
     +- generate kernel launcher header
     +- collect objects
     +- generate kernel launcher source
'''

class Generator(object):

    def __init__(self, args, out):
        self._args = args
        self._out = out
        self._children = []

    @property
    def is_file(self):
        return False

    def generate(self):
        self.write_prelude()
        self.loop_children()
        self.write_body()
        self.write_conclude()

        if self._out is not None:
            self._out.flush()

    @property
    def children_out(self):
        return self._out

    def gen_children(self, out):
        pass

    def write_prelude(self):
        pass

    def loop_children(self):
        for c in self.gen_children(self.children_out):
            c.generate()
            self._children.append(c)

    def write_body(self):
        pass

    def write_conclude(self):
        pass

    @property
    def list_of_output_object_files(self):
        return []

class MakefileGenerator(Generator):
    def __init__(self, args, grand_target, out):
        super().__init__(args, out)
        self._main_content = io.StringIO()
        self._targets = []
        self._grand_target = grand_target
        self._phony = []

    @property
    def children_out(self):
        return self._main_content

    def gen_children(self, out):
        pass

    def write_prelude(self):
        pass

    def write_body(self):
        self._main_content.seek(0)
        shutil.copyfileobj(self._main_content, self._out)

    def write_conclude(self):
        print('.PHONY: ', ' '.join(self._phony), file=self._out)

    def get_all_object_files(self):
        return sum([self.list_of_output_object_files for c in self._children], [])

class ShimMakefileGenerator(MakefileGenerator):

    def __init__(self, args):
        grand_target = LIBRARY_NAME + '.a' if args.archive else '.so'
        build_dir = Path(args.build_dir)
        f = open(build_dir / 'Makefile.shim', 'w')
        super().__init__(args=args, grand_target=grand_target, out=f)

    def __del__(self):
        self._out.close()

    def gen_children(self, out):
        for k in triton_kernels:
            yield KernelShimGenerator(self._args, self.children_out, k)

    def write_prelude(self):
        f = self._out
        super().write_prelude()
        print(f"HIPCC={COMPILER}", file=f)
        print(f"AR={LINKER}", file=f)
        print(f"", file=f)
        print('', file=self._out)
        all_object_files = ' '.join(self.get_all_object_files())
        print(self._grand_target, ': ', all_object_files, file=self._out)
        if self._args.archive:
            print('\t', '${AR} -r ', self._grand_target, file=f)
        else:
            print('\t', COMPILER, ' -shared -fPIC -o ', self._grand_target, file=f)
        print('\n\n', file=f)

class KernelShimGenerator(Generator):
    def __init__(self, args, out, k : 'KernelDescription'):
        super().__init__(args, out)
        self._kdesc = k
        self._shim_path = Path(args.build_dir) / k.KERNEL_FAMILY
        self._shim_path.mkdir(parents=True, exist_ok=True)
        self._fhdr = open(self._shim_path / Path(k.SHIM_KERNEL_NAME + '.h'), 'w')
        self._fsrc = open(self._shim_path / Path(k.SHIM_KERNEL_NAME + '.cc'), 'w')

    def __del__(self):
        self._fhdr.close()
        self._fsrc.close()

    def write_prelude(self):
        self._kdesc.write_launcher_header(self._fhdr)

    def gen_children(self, out):
        k = self._kdesc
        for o in k.gen_all_object_files(self._shim_path, file_name_prefix=k.SHIM_KERNEL_NAME):
            yield ObjectShimCodeGenerator(self._args, k, o)

    def write_conclude(self):
        self._kdesc.write_launcher_source(self._fsrc, [c._odesc for c in self._children])

class ObjectShimCodeGenerator(Generator):
    def __init__(self, args, k, o):
        super().__init__(args, None)
        self._kdesc = k
        self._odesc = o

    def get_all_object_files(self):
        return [self._odesc._hsaco_kernel_path.with_suffix('.o')] if self._odesc.compiled_files_exist else []

    def loop_children(self):
        pass

def main():
    args = parse()
    gen = ShimMakefileGenerator(args)
    gen.generate()

def gen_cc_from_object(args, o : 'ObjectFileDescription', makefile):
    src = o._hsaco_kernel_path.with_suffix('.cc')
    hdr = Path(o.SHIM_KERNEL_NAME).with_suffix('.h')
    with open(src, 'w') as f:
        print(o.generate_shim_source(), file=f)
    target = o._hsaco_kernel_path.with_suffix('.o')
    print(target.name, ': ', src.name, ' ', hdr.name, file=makefile)
    cmd  = '$(HIPCC) ' + f'{src.absolute()} -I{CSRC} -I{INCBIN} -c -fPIC -std=c++17'
    print('\t', cmd, '\n', file=makefile)
    return target.name

def gen_from_kernel(args, k, build_dir, makefile, generate_separate_so=False):
    all_targets = []
    object_rules = io.StringIO()
    shim_common_header = io.StringIO()
    shim_member_function_decls = set()
    shim_extern_template_decls = set()

    all_object_files = k.get_object_files(build_dir, prefix=k.SHIM_KERNEL_NAME)
    for o in all_object_files:
        if shim_common_header.tell() == 0:
            print(o.generate_shim_header_leading(), file=shim_common_header)
        all_targets.append(gen_cc_from_object(args, o, object_rules))
        shim_member_function_decls.add(o.generate_shim_header_member_function())

    for member_function_decl in shim_member_function_decls:
        print(member_function_decl, file=shim_common_header)
    print(o.generate_shim_header_closing_struct_define(), file=shim_common_header)

    for o in all_object_files:
        shim_extern_template_decls.add(o.generate_shim_header_extern_template())
    for template_decl in shim_extern_template_decls:
        print(template_decl, file=shim_common_header)

    print(o.generate_shim_header_trailing(), file=shim_common_header)
    with (build_dir / o.SHIM_KERNEL_NAME).with_suffix('.h').open('w') as hdrf:
        shim_common_header.seek(0)
        shutil.copyfileobj(shim_common_header, hdrf)

    if not generate_separate_so:
        object_rules.seek(0)
        shutil.copyfileobj(object_rules, makefile)
        return k.SHIM_KERNEL_NAME, all_targets

    output_so = k.SHIM_KERNEL_NAME
    output_so += '.a'  if args.archive else '.so'
    def print_all_o():
        for t in all_targets:
            print(t, end=' ', file=makefile)
    print(output_so, ': ', end='', file=makefile)
    print_all_o()
    if args.archive:
        print('\n\t', '${AR} -r ', output_so, end=' ', file=makefile)
    else:
        print('\n\t', COMPILER, ' -shared -fPIC -o ', output_so, end=' ', file=makefile)
    print_all_o()
    print('\n\n', file=makefile)
    object_rules.seek(0)
    shutil.copyfileobj(object_rules, makefile)
    return output_so, []

def oldmain(generate_separate_so=False):
    args = parse()
    build_dir = Path(args.build_dir)
    with open(build_dir / 'Makefile.shim', 'w') as f:
        makefile_content = io.StringIO()
        print(f"HIPCC={COMPILER}", file=makefile_content)
        print(f"AR={LINKER}", file=makefile_content)
        print(f"", file=makefile_content)
        per_kernel_targets = []
        all_o_files = []
        for k in rules.kernels:
            this_kernel_target, this_kernel_o_files = gen_from_kernel(args, k, build_dir, makefile_content, generate_separate_so=generate_separate_so)
            per_kernel_targets.append(this_kernel_target)
            all_o_files += this_kernel_o_files
        print('libaotriton: ', end='', file=f)
        if generate_separate_so:
            for t in per_kernel_targets:
                print(t, end=' ', file=f)
            print('\n', file=f)
        else:
            output_so = 'libaotriton.a' if args.archive else 'libaotriton.so'
            def print_all_o():
                for t in all_o_files:
                    print(t, end=' ', file=f)
            print_all_o()
            if args.archive:
                print('\n\t', '${AR} -r ', output_so, end=' ', file=f)
            else:
                print('\n\t', COMPILER, ' -shared -fPIC -o ', output_so, end=' ', file=f)
            print_all_o()
            print('\n\n', file=f)
        makefile_content.seek(0)
        shutil.copyfileobj(makefile_content, f)
        if generate_separate_so:
            print('.PHONY: libaotriton', file=f)
        else:
            print('.PHONY: libaotriton ', ' '.join(per_kernel_targets), file=f)

if __name__ == '__main__':
    main()
