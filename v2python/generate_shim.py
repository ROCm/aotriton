# Copyright © 2023-2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from .rules import kernels as triton_kernels
from .tuning_database import KernelTuningDatabase
from .tuning_lut import MissingLutEntry
import io
import shutil
import argparse
from pathlib import Path

SOURCE_PATH = Path(__file__).resolve()
CSRC = (SOURCE_PATH.parent.parent / 'v2src').absolute()
INCBIN = (SOURCE_PATH.parent.parent / 'third_party/incbin/').absolute()
COMMON_INCLUDE = (SOURCE_PATH.parent.parent / 'include/').absolute()
# COMPILER = SOURCE_PATH.parent / 'compile.py'
COMPILER = 'hipcc'
LINKER = 'ar'

LIBRARY_NAME = 'libaotriton_v2'

class NoWriteIfNoUpdateFile(object):
    def __init__(self, ofn : Path):
        self._ofn = ofn
        self._old_content = ''

    @property
    def path(self):
        return self._ofn

    def __enter__(self):
        self._mf = io.StringIO()
        if self._ofn.exists():
            with open(self._ofn) as f:
                self._old_content = f.read()
        return self._mf

    @property
    def memory_file(self):
        return self._mf

    def __exit__(self, exc_type, exc_val, exc_tb):
        mf = self.memory_file
        mf.seek(0)
        if mf.read() != self._old_content:
            mf.seek(0)
            with open(self.path, 'w') as of:
                shutil.copyfileobj(mf, of)

def parse():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--target_gpus", type=str, default=None, nargs='*',
                   help="Ahead of Time (AOT) Compile Architecture. PyTorch is required for autodetection if --targets is missing.")
    p.add_argument("--build_dir", type=Path, default='build/', help="build directory")
    p.add_argument("--archive_only", action='store_true', help='Only generate archive library instead of shared library. No linking with dependencies.')
    p.add_argument("--library_suffix", type=str, default='', help="Add suffix to the library name 'aotriton' to avoid symbol conflicts")
    p.add_argument("--bare_mode", action='store_true', help="Instead of generating a proper Makefile, only generate a list of source files and leave the remaining tasks to cmake.")
    p.add_argument("--noimage_mode", action='store_true', help="Expect the GPU kernel images are built separately.")
    p.add_argument("--build_for_tuning", action='store_true', help="Include all GPU kernels in the dispatcher for performance tuning.")
    p.add_argument("--build_for_tuning_but_skip_kernel", type=str, default='', nargs='*',
                   help="Excluse certain GPU kernels for performance tuning when --build_for_tuning=True.")
    p.add_argument("--verbose", action='store_true', help="Print debugging messages")
    p.add_argument("--lut_sanity_check", action='store_true', help="Do not raise exceptions when the look up table (lut) is incomplete.")
    args = p.parse_args()
    args._sanity_check_exceptions = []
    args.build_for_tuning_but_skip_kernel = args.build_for_tuning_but_skip_kernel
    # print(args)
    return args

def is_tuning_on_for_kernel(args, k : 'KernelDescription'):
    if not args.build_for_tuning:
        return False
    elif k.FULL_KERNEL_NAME in args.build_for_tuning_but_skip_kernel:
        return False
    else:
        return True

'''
ShimMakefileGenerator
 +- generate libaotriton.a
 +- KernelShimGenerator
     +- generate param and context classes
     +- AutotuneCodeGenerator for each functional variant of the kernel
         +- generate lut entries under autotune.<KERNEL_NAME>/<functional_signature>.cc
         +- generate corresponding makefile rule
     +- collect objects (ObjectShimCodeGenerator) for linking
     +- write Makefile rules for param and context classes
     +- generate param and context class implementations
'''

class Generator(object):

    def __init__(self, args, out):
        self._args = args
        self._out = out
        self._children = []

    @property
    def is_file(self):
        return False

    @property
    def is_bare(self):
        return self._args.bare_mode

    def verbose(self, *args, **kwargs):
        if self._args.verbose:
            print(*args, **kwargs)

    @property
    def build_root(self):
        return self._args.build_dir

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

class MakefileSegmentGenerator(Generator):

    def __init__(self, args, out):
        super().__init__(args, out);
        self._cc_cmd = '$(HIPCC) $(EXTRA_COMPILER_OPTIONS) '
        self._cc_cmd += f' -I{INCBIN} -I{COMMON_INCLUDE} -fPIC -std=c++20'

    @property
    def list_of_self_object_files(self) -> 'list[Path]':
        return []

    @property
    def list_of_child_object_files(self) -> 'list[Path]':
        return sum([c.list_of_output_object_files for c in self._children], [])

    @property
    def list_of_output_object_files(self) -> 'list[Path]':
        return self.list_of_self_object_files + self.list_of_child_object_files

class MakefileGenerator(MakefileSegmentGenerator):
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
        if self.is_bare:
            return
        print('.PHONY: ', ' '.join(self._phony), file=self._out)

class ShimMakefileGenerator(MakefileGenerator):

    def __init__(self, args):
        # grand_target = LIBRARY_NAME + '.a' if args.archive else '.so'
        grand_target = LIBRARY_NAME
        self._build_dir = args.build_dir
        if args.bare_mode:  # CAVEAT: .is_bare is unavailable at the moment
            f = open(self._build_dir / 'Bare.shim', 'w')
        else:
            f = open(self._build_dir / 'Makefile.shim', 'w')
        arf = open(self._build_dir / 'ar.txt', 'w')
        super().__init__(args=args, grand_target=grand_target, out=f)
        self._library_suffixes = ['.a']  if args.archive_only else ['.a', '.so']
        self._arf = arf

    def __del__(self):
        self._out.close()
        self._arf.close()

    def gen_children(self, out):
        for k in triton_kernels:
            yield KernelShimGenerator(self._args, self.children_out, k)
        yield SourceBuilder(self._args, self.children_out)

    def write_prelude(self):
        f = self._out
        super().write_prelude()
        if self.is_bare:
            return
        print(f"HIPCC={COMPILER}", file=f)
        print(f"AR={LINKER}", file=f)
        print(f"EXTRA_COMPILER_OPTIONS=-O0 -g -ggdb3", file=f)
        # print(f"EXTRA_COMPILER_OPTIONS=", file=f)
        print(f"", file=f)
        print('', file=self._out)
        print(self._grand_target, ':', ' '.join([f'{LIBRARY_NAME}{s}' for s in self._library_suffixes]), '\n\n', file=self._out)

    def write_conclude(self):
        if self.is_bare:
            return
        f = self._out
        all_object_files = ' '.join([str(p) for p in self.list_of_output_object_files])
        for s in self._library_suffixes:
            fn = f'{LIBRARY_NAME}{s}'
            print(fn, ': ', all_object_files, file=self._out)
            if s == '.a':
                print('\t', '${AR} -r ', fn, '@ar.txt', file=f)
                print(all_object_files, file=self._arf)
            if s == '.so':
                print('\t', COMPILER, ' -g -shared -fPIC -o ', fn, all_object_files, file=f)
            print('\n\n', file=f)

    '''
    @property
    def _object_relative_paths(self):
        return [str(op.relative_to(self._build_dir)) for op in self.list_of_output_object_files]
    '''

class SourceBuilder(MakefileSegmentGenerator):
    DIR = 'v2src'

    def __init__(self, args, out):
        super().__init__(args, out)
        self._build_dir = args.build_dir
        self._srcdir = Path(CSRC)
        self._outdir = self._build_dir / self.DIR
        self._outdir.mkdir(parents=True, exist_ok=True)
        self._objpaths = []

    def write_body(self):
        for cfn in self._srcdir.rglob("*.cc"):
            rpath = cfn.relative_to(self._srcdir)
            if 'template/' in str(rpath):
                print(f'Skip {rpath=}')
                continue
            ofn = (self._outdir / rpath).with_suffix('.o')
            ofn.parent.mkdir(parents=True, exist_ok=True)
            makefile_target = ofn.relative_to(self._build_dir)
            self._objpaths.append(makefile_target)
            if self.is_bare:
                print(str(cfn.absolute()), file=self._out)
            else:
                print(makefile_target, ':', str(cfn.absolute()), file=self._out)
                cmd = self._cc_cmd + f' {cfn.absolute()} -I{self._build_dir.absolute()} -o {ofn.absolute()} -c'
                print('\t', cmd, '\n', file=self._out)

    @property
    def list_of_self_object_files(self) -> 'list[Path]':
        return self._objpaths

class KernelShimGenerator(MakefileSegmentGenerator):
    AUTOTUNE_TABLE_PATH = 'autotune_table'

    def __init__(self, args, out, k : 'KernelDescription'):
        super().__init__(args, out)
        # Shim code and functional dispatcher
        self._kdesc = k
        self._kdesc.set_target_gpus(args.target_gpus)
        self._shim_path = args.build_dir / k.KERNEL_FAMILY
        self._shim_path.mkdir(parents=True, exist_ok=True)
        self._shim_hdr = NoWriteIfNoUpdateFile(self._shim_path / Path(self.SHIM_FILE_STEM + '.h'))
        self._shim_src = NoWriteIfNoUpdateFile(self._shim_hdr.path.with_suffix('.cc'))
        # Autotune dispatcher
        self._autotune_path = args.build_dir / k.KERNEL_FAMILY / f'autotune.{k.SHIM_KERNEL_NAME}'
        self._autotune_path.mkdir(parents=True, exist_ok=True)
        self._tuning = is_tuning_on_for_kernel(self._args, k)
        self._ktd = KernelTuningDatabase(args.build_dir,
                                         k,
                                         build_for_tuning=self._tuning)
        self._objpaths = []

    @property
    def SHIM_FILE_STEM(self):
        return 'shim.' + self._kdesc.SHIM_KERNEL_NAME

    def write_body(self):
        ofn = self._shim_src.path.with_suffix('.o')
        makefile_target = ofn.relative_to(self.build_root)
        if self.is_bare:
            print(str(self._shim_src.path.absolute()), file=self._out)
        else:
            print(makefile_target, ':', str(self._shim_hdr.path.absolute()), str(self._shim_src.path.absolute()), file=self._out)
            cmd  = self._cc_cmd + f' {self._shim_src.path.absolute()} -o {ofn.absolute()} -c -fPIC -std=c++20'
            print('\t', cmd, '\n', file=self._out)
        self._objpaths.append(makefile_target)

    def gen_children(self, out):
        k = self._kdesc
        p = self._shim_path / f'gpu_kernel_image.{k.SHIM_KERNEL_NAME}'
        args = self._args
        debug_counter = 0
        for arch, fsels, lut in k.gen_tuned_kernel_lut(self._ktd):
            # print(f'KernelShimGenerator.gen_children {fsels=}')
            yield AutotuneCodeGenerator(args, self.children_out, self._autotune_path, k, arch, fsels, lut)
            '''
            debug_counter +=1
            if debug_counter >= 2:
                break
            '''

        if self.is_bare:
            return
        for o in k.gen_all_object_files(p, tuned_db=self._ktd, sancheck_fileexists=not self._tuning and not args.noimage_mode):
            yield ObjectShimCodeGenerator(self._args, k, o)

    def write_conclude(self):
        if self.is_bare:
            return
        objs = [c._odesc for c in self._children if isinstance(c, ObjectShimCodeGenerator)]
        with self._shim_hdr as fhdr, self._shim_src as fsrc:
            self._kdesc.write_shim_header(fhdr, objs)
            self._kdesc.write_shim_source(fsrc, objs, noimage_mode=self._args.noimage_mode)

    @property
    def list_of_self_object_files(self) -> 'list[Path]':
        return self._objpaths

class AutotuneCodeGenerator(MakefileSegmentGenerator):
    def __init__(self, args, fileout, outdir, k, arch, fsels, lut):
        super().__init__(args, fileout)
        self._build_dir = args.build_dir
        self._outdir = outdir
        self._kdesc = k
        self._arch = arch
        self._fsels = fsels
        self._lut = lut

    def write_body(self):
        self.verbose('AutotuneCodeGenerator')
        do_raise = None
        # Write the code to file
        try:
            self._ofn = self._lut.write_lut_source(self._args.library_suffix,
                                                   self._outdir,
                                                   bare_mode=self.is_bare,
                                                   noimage_mode=self._args.noimage_mode)
        except MissingLutEntry as e:
            self._ofn = e.ofn  # regardless --build_for_tuning or not
            if not is_tuning_on_for_kernel(self._args, self._kdesc):
                do_raise = e
                print('raise', e)
                self._args._sanity_check_exceptions.append(e)
                # import ipdb; ipdb.set_trace()
                for j in e.get_missing_lut_entries():
                    print("TUNE_FLASH --entry_from_json Item: ", j)
            else:
                pass  # Ignore MissingLutEntry when --build_for_tuning
        self.verbose(f'\t lut = {self._fsels}')
        self.verbose(f'\t ofn = {self._ofn}')
        self._obj_fn = self._ofn.with_suffix('.o')
        self._makefile_target = self._obj_fn.relative_to(self._build_dir)
        # Write the Makefile segment
        if self.is_bare:
            print(str(self._ofn.absolute()), file=self._out)
        else:
            print('#', self._fsels, file=self._out)
            print(self._makefile_target, ':', self._ofn.relative_to(self._build_dir), file=self._out)
            cmd  = self._cc_cmd + f' {self._ofn.absolute()} -o {self._obj_fn.absolute()} -c'
            print('\t', cmd, '\n', file=self._out)
        if do_raise and not self._args.lut_sanity_check:
            raise do_raise

    @property
    def list_of_self_object_files(self) -> 'list[Path]':
        return [self._makefile_target]

# FIXME: a better name.
#        This class name is legacy and now it's only used to store
#        ObjectFileDescription objects to keep record of metadata for compiled
#        kernels
class ObjectShimCodeGenerator(Generator):
    def __init__(self, args, k, o):
        super().__init__(args, None)
        self._kdesc = k
        self._odesc = o

    '''
    def get_all_object_files(self):
        return [self._odesc._hsaco_kernel_path.with_suffix('.o')] if self._odesc.compiled_files_exist else []
    '''

    def loop_children(self):
        pass

    @property
    def list_of_output_object_files(self) -> 'list[Path]':
        return []

def main():
    args = parse()
    gen = ShimMakefileGenerator(args)
    gen.generate()
    for e in args._sanity_check_exceptions:
        raise e

if __name__ == '__main__':
    main()
