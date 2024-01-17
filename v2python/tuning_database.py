import json
import pathlib
from copy import deepcopy
from collections import defaultdict
from .kernel_argument import TunedArgument
from .gpu_targets import AOTRITON_GPU_ARCH_TUNING_STRING
from .tuning_lut import KernelTuningEntryForFunctionalOnGPU

'''
Used in conjunction with PARTIALLY_TUNED_FUNCTIONALS

Commonly enabling functionals will cost extra resources,
and thus make the fallback turing information unusable
'''
class TuningDowngrader(object):
    def __init__(self, matching_list):
        self._matching_list = matching_list

    @staticmethod
    def create_from_kdesc(k : 'KernelDescription'):
        if not hasattr(k, 'DOWNGRADER'):
            return False
        return TuningDowngrader(k.DOWNGRADER)

    def match(self, matching, fallback_applied_fsels):
        iterator = iter(matching)
        while True:
            key = next(iterator, None)
            value = next(iterator, None)
            if key is None or value is None:
                break
            all_matched = True
            for fsel in fallback_applied_fsels:
                if not fsel.meta.has_argument(key):
                    all_matched = False
                    break
                if fsel.argument_value != value:
                    all_matched = False
                    break
            if all_matched:
                return True
        return False

    def lookup_patcher(self, fallback_applied_fsels):
        for matching, tuned_kernel_patcher in self._matching_list:
            if self.match(matching, fallback_applied_fsels):
                def patcher(tinfo):
                    print(f"Downgrade kernel from {tinfo['tuned_kernel']} {tinfo['compiler_options']}", end=' ')
                    tuned_kernel_patcher(tinfo['tuned_kernel'], tinfo['compiler_options'])
                    print(f"into {tinfo['tuned_kernel']} {tinfo['compiler_options']}")
                    return tinfo
                return patcher
        return None

'''
Note: unlike KernelDescription, whose constants will be specialized for EVERY kernel.
      KernelTuningDatabase(ForArch) should work for all KernelDescription instances.

      Therefore the index of the database can only be built when seeing the
      first set of ArgumentSelection objects, because the json object itself
      has zero information about the triton kernel.
'''
class KernelTuningDatabaseForArch(object):
    def __init__(self, f, downgrader=None):
        self._j = json.load(f)
        self._arch = self._j['arch']
        self._gpu = None
        self._index = None
        self._index_matching_keys = None
        self._lut = {}
        self._downgrader = downgrader

    @property
    def arch(self):
        return self._arch

    def set_gpu(self, gpu, index):
        self._gpu = gpu
        self._arch_number = index
        return self

    def select(self, fsels : 'list[ArgumentSelection]', perf_meta : 'list[ArgumentMetadata]'):
        if self._index is None:
            self._build_db_index(fsels)
        yield from self._select_from_index(fsels, perf_meta)

    def _init_matching_keys(self, fsels):
        self._index_matching_keys = []
        self._fsel_positions = []
        tinput = self._j['tune_info'][0]['inputs']
        for fsel in fsels:
            mfsel = fsel.meta
            if mfsel.nchoices <= 1:
                continue
            key_detected = None
            if mfsel.is_tensor:
                for tensor_name in fsel.argument_names:
                    tensor_key = f'{tensor_name}.dtype'
                    if tensor_key in tinput:
                        key_detected = tensor_key
                        break
            elif mfsel.is_type:
                key_detected = None # TODO
            elif mfsel.is_feature:
                for aname in fsel.argument_names:
                    if aname in tinput:
                        key_detected = aname
                        break
                assert key_detected is not None
            if key_detected is not None:
                self._index_matching_keys.append(key_detected)
                self._fsel_positions.append(fsel.meta.first_apperance)
        # print(f'{self._index_matching_keys=}')

    def extract_keys_from_json(self, ti):
        keys = [ti['inputs'][k] for k in self._index_matching_keys]
        def convert(value):
            if isinstance(value, str) and value.startswith('torch.'):
                if value == 'torch.float16':
                    return '*fp16:16'
                elif value == 'torch.bfloat16':
                    return '*bf16:16'
                else:
                    assert False, f'Unknown datatype {value}'
            return value
        return tuple(map(convert, keys))

    def extract_keys_from_fsels(self, fsels, use_fallback_for_partially_tuned=False):
        keys = {}
        fallback_applied = []
        # print(f'{len(self._fsel_positions)=}')
        for fsel in fsels:
            try:
                # print(f'extract_keys_from_fsels {fsel.argument_names} {fsel.meta.first_apperance}')
                offset = self._fsel_positions.index(fsel.meta.first_apperance)
                if use_fallback_for_partially_tuned and fsel.meta.incomplete_tuning:
                    value = fsel.meta.fallback_tuning_value
                    fallback_applied.append(fsel)
                else:
                    value = fsel.argument_value
                keys[offset] = value
                # print(f'keys[{offset}] = {value}')
            except ValueError:
                pass
        l = [keys[offset] for offset in range(len(self._fsel_positions))]
        # print(f'{l=}')
        return tuple(l), fallback_applied

    def _build_db_index(self, fsels):
        self._init_matching_keys(fsels)
        self._index = defaultdict(list)
        self._index_dedup = defaultdict(list)
        for ti in self._j['tune_info']:
            tup = self.extract_keys_from_json(ti)
            # print(f'_build_db_index {tup}')
            self._index[tup].append(ti)
            is_dup = False
            for eti in self._index_dedup[tup]:
                # print(f'{eti=}')
                if eti['tuned_kernel'] == ti['tuned_kernel'] and eti['compiler_options'] == ti['compiler_options']:
                    is_dup = True
                    break
            if not is_dup:
                self._index_dedup[tup].append(ti)
        if False:  # debug
            tup=('*fp16:16', 1, 16, True, True)
            print(f'_build_db_index {self._index[tup]=} {self._index_dedup[tup]=}')

    def _select_from_index(self,
                           fsels : 'list[ArgumentSelection]',
                           perf_meta : 'list[ArgumentMetadata]',
                           no_duplicate=True):
        indexed = self.lookup_tuning_info(fsels, with_duplicates=not no_duplicate)
        assert indexed
        for tinfo in indexed:
            yield self._craft_perf_selection(tinfo, perf_meta)

    def _craft_perf_selection(self, tinfo : dict, perf_meta: 'list[ArgumentSelection]'):
        ps = dict(tinfo['tuned_kernel'])
        co = dict(tinfo['compiler_options'])
        if 'waves_per_eu' in ps:
            co['waves_per_eu'] = ps['waves_per_eu']
            # co['_debug'] = dict(tinfo)
            del ps['waves_per_eu']
        return [TunedArgument(meta, ps[meta.argument_names[0]]) for meta in perf_meta], co

    def get_lut(self,
                kdesc : 'KernelDescription',
                autotune_keys : 'list[tuple[str, Binning]]',
                fsels : 'list[ArgumentSelection]',
                perf_meta : 'list[ArgumentMetadata]'):
        if self._index is None:
            self._build_db_index(fsels)
        tup, _  = self.extract_keys_from_fsels(fsels)
        if tup not in self._lut:
            indexed = self.lookup_tuning_info(fsels)
            # print(f'{tup=}')
            assert indexed
            self._lut[tup] = KernelTuningEntryForFunctionalOnGPU(kdesc, self, fsels, indexed,
                                                                 autotune_keys, perf_meta)
        return self._lut[tup]

    def lookup_tuning_info(self, fsels, with_duplicates=True):
        tup, _ = self.extract_keys_from_fsels(fsels)
        if tup in self._index:
            return self._index[tup] if with_duplicates else self._index_dedup[tup]
        fallback_tup, fallback_applied_fsels = self.extract_keys_from_fsels(fsels, use_fallback_for_partially_tuned=True)
        print(f'Functionals {tup} cannot be found in tuning db, use {fallback_tup} instead')
        assert fallback_tup in self._index
        tuning_info = self._index[fallback_tup] if with_duplicates else self._index_dedup[fallback_tup]
        return self.downgrade(fallback_applied_fsels, tuning_info)

    def downgrade(self, fallback_applied_fsels, tuning_info):
        if self._downgrader is None:
            return tuning_info
        patcher = self._downgrader.lookup_patcher(fallback_applied_fsels)
        if patcher is None:
            return tuning_info
        return [patcher(deepcopy(tune)) for tune in tuning_info]

class KernelTuningDatabase(object):
    def __init__(self, tune_info_dir : pathlib.Path, k : 'KernelDescription'):
        self.arch_dict = {}
        td = pathlib.Path(tune_info_dir) # in case tune_info_dir is str
        # print(f"Tryint to probe KernelTuningDatabase inside {td}")
        downgrader = TuningDowngrader.create_from_kdesc(k)
        for fn in td.glob(f'tune-{k.SHIM_KERNEL_NAME}-*.json'):
            with open(fn) as f:
                dba = KernelTuningDatabaseForArch(f, downgrader)
            self.arch_dict[dba.arch] = dba

    def select_gpu(self, gpu, index):
        arch = AOTRITON_GPU_ARCH_TUNING_STRING[gpu]
        return self.arch_dict[arch].set_gpu(gpu, index)

    @property
    def empty(self):
        return not self.arch_dict
