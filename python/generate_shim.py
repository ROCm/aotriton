import rules
import io
import shutil
from pathlib import Path

SOURCE_PATH = Path(__file__).resolve()
CSRC = (SOURCE_PATH.parent.parent / 'csrc').absolute()
INCBIN = (SOURCE_PATH.parent.parent / 'third_party/incbin/').absolute()
# COMPILER = SOURCE_PATH.parent / 'compile.py'
COMPILER = 'hipcc'

def gen_cc_from_object(o : 'ObjectFileDescription', makefile):
    src = o._hsaco_kernel_path.with_suffix('.cc')
    hdr = Path(o.SHIM_KERNEL_NAME).with_suffix('.h')
    with open(src, 'w') as f:
        print(o.generate_shim_source(), file=f)
    target = o._hsaco_kernel_path.with_suffix('.o')
    print(target.name, ': ', src.name, ' ', hdr.name, file=makefile)
    cmd  = f'{COMPILER} {src.absolute()} -I{CSRC} -I{INCBIN} -c -fPIC'
    print('\t', cmd, '\n', file=makefile)
    return target.name

def gen_from_kernel(k, build_dir, makefile):
    all_targets = []
    object_rules = io.StringIO()
    shim_common_header = io.StringIO()
    shim_member_function_decls = set()

    all_object_files = k.get_object_files(build_dir, prefix=k.SHIM_KERNEL_NAME)
    for o in all_object_files:
        if shim_common_header.tell() == 0:
            print(o.generate_shim_header_leading(), file=shim_common_header)
        all_targets.append(gen_cc_from_object(o, object_rules))
        shim_member_function_decls.add(o.generate_shim_header_member_function())
    for member_function_decl in shim_member_function_decls:
        print(member_function_decl, file=shim_common_header)
    print(o.generate_shim_header_closing_struct_define(), file=shim_common_header)
    for o in all_object_files:
        print(o.generate_shim_header_extern_template(), file=shim_common_header)
    print(o.generate_shim_header_trailing(), file=shim_common_header)
    with (build_dir / o.SHIM_KERNEL_NAME).with_suffix('.h').open('w') as hdrf:
        shim_common_header.seek(0)
        shutil.copyfileobj(shim_common_header, hdrf)

    output_so = k.SHIM_KERNEL_NAME + '.so'
    def print_all_o():
        for t in all_targets:
            print(t, end=' ', file=makefile)
    print(output_so, ': ', end='', file=makefile)
    print_all_o()
    print('\n\t', COMPILER, ' -shared -fPIC -o ', output_so, end=' ', file=makefile)
    print_all_o()
    print('\n\n', file=makefile)
    object_rules.seek(0)
    shutil.copyfileobj(object_rules, makefile)
    print('.PHONY: all', file=makefile)

def main():
    build_dir = Path('build/')
    with open(build_dir / 'Makefile.shim', 'w') as f:
        for k in rules.kernels:
            gen_from_kernel(k, build_dir, f)

if __name__ == '__main__':
    main()
