from .rules import kernels as triton_kernels
from .kernel_tuning_database import KernelTuningDatabase
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
    p.add_argument("--target_gpus", type=str, default=None, nargs='*',
                   help="Ahead of Time (AOT) Compile Architecture. PyTorch is required for autodetection if --targets is missing.")
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
        return
        yield

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
    AUTOTUNE_TABLE_PATH = 'autotune_table'

    def __init__(self, args, out, k : 'KernelDescription'):
        super().__init__(args, out)
        # Shim code and functional dispatcher
        self._kdesc = k
        self._kdesc.set_target_gpus(args.target_gpus)
        self._shim_path = Path(args.build_dir) / k.KERNEL_FAMILY
        self._shim_path.mkdir(parents=True, exist_ok=True)
        self._fhdr = open(self._shim_path / Path(k.SHIM_KERNEL_NAME + '.h'), 'w')
        self._fsrc = open(self._shim_path / Path(k.SHIM_KERNEL_NAME + '.cc'), 'w')
        # Autotune dispatcher
        self._autotune_path = Path(args.build_dir) / k.KERNEL_FAMILY / f'autotune.{k.SHIM_KERNEL_NAME}'
        self._autotune_path.mkdir(parents=True, exist_ok=True)
        self._ktd = KernelTuningDatabase(SOURCE_PATH.parent / 'rules', k.SHIM_KERNEL_NAME)

    def __del__(self):
        self._fhdr.close()
        self._fsrc.close()

    def write_prelude(self):
        self._kdesc.write_launcher_header(self._fhdr)

    def gen_children(self, out):
        k = self._kdesc
        p = self._shim_path
        args = self._args
        for gpu, fsels, lut in k.gen_tuned_kernel_lut(self._ktd):
            yield AutotuneCodeGenerator(args, self._autotune_path, k, gpu, fsels, lut)

        ktd = KernelTuningDatabase(SOURCE_PATH.parent / 'rules', k.SHIM_KERNEL_NAME)
        for o in k.gen_all_object_files(p, tuned_db=ktd):
            yield ObjectShimCodeGenerator(self._args, k, o)

    def write_conclude(self):
        self._kdesc.write_launcher_source(self._fsrc, [c._odesc for c in self._children if isinstance(c, ObjectShimCodeGenerator)])

class AutotuneCodeGenerator(Generator):
    def __init__(self, args, outdir, k, gpu, fsels, lut):
        super().__init__(args, None)
        self._outdir = outdir
        self._kdesc = k
        self._gpu = gpu
        self._fsels = fsels
        self._lut = lut

    def write_body(self):
        self._lut.write_lut_source(self._outdir)

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

if __name__ == '__main__':
    main()
