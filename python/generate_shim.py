import rules
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

def parse():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--build_dir", type=str, default='build/', help="build directory")
    p.add_argument("--archive", action='store_true', help='generate archive library instead of shared library. No linking with dependencies.')
    args = p.parse_args()
    # print(args)
    return args

def gen_cc_from_object(args, o : 'ObjectFileDescription', makefile):
    src = o._hsaco_kernel_path.with_suffix('.cc')
    hdr = Path(o.SHIM_KERNEL_NAME).with_suffix('.h')
    with open(src, 'w') as f:
        print(o.generate_shim_source(), file=f)
    target = o._hsaco_kernel_path.with_suffix('.o')
    print(target.name, ': ', src.name, ' ', hdr.name, file=makefile)
    cmd  = '$(HIPCC) ' + f'{src.absolute()} -I{CSRC} -I{INCBIN} -c -fPIC'
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

def main(generate_separate_so=False):
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
        print('liboort: ', end='', file=f)
        if generate_separate_so:
            for t in per_kernel_targets:
                print(t, end=' ', file=f)
            print('\n', file=f)
        else:
            output_so = 'liboort.a' if args.archive else 'liboort.so'
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
            print('.PHONY: liboort', file=f)
        else:
            print('.PHONY: liboort ', ' '.join(per_kernel_targets), file=f)

if __name__ == '__main__':
    main()
