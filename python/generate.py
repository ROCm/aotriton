import rules
import io
import shutil
from pathlib import Path

SOURCE_PATH = Path(__file__).resolve()
COMPILER = SOURCE_PATH.parent / 'compile.py'

def gen_from_object(o : 'ObjectFileDescription', makefile):
    target = o._hsaco_kernel_path.name
    print(target, ':', file=makefile)
    cmd  = f'{COMPILER} {o.src.absolute()} --kernel_name {o.entrance} -o {o.obj.absolute()}'
    cmd += f' -g 1,1,1 --num_warps {o.num_warps} --num_stages {o.num_stages}'
    cmd += f" --signature '{o.signature}'"
    print('\t', cmd, '\n', file=makefile)
    return target

def gen_from_kernel(k, p, makefile):
    all_targets = []
    object_rules = io.StringIO()
    for o in k.get_object_files(p, prefix=k.SHIM_KERNEL_NAME):
        all_targets.append(gen_from_object(o, object_rules))
    print('all: ', end='', file=makefile)
    for t in all_targets:
        print(t, end=' ', file=makefile)
    print('\n\n', file=makefile)
    object_rules.seek(0)
    shutil.copyfileobj(object_rules, makefile)
    print('.PHONY: all', file=makefile)

def main():
    build_dir = Path('build/')
    with open(build_dir / 'Makefile.compile', 'w') as f:
        for k in rules.kernels:
            gen_from_kernel(k, build_dir, f)

if __name__ == '__main__':
    main()
