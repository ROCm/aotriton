import json
import pathlib
from collections import defaultdict
from .kernel_argument import TunedArgument
from .gpu_targets import AOTRITON_GPU_ARCH_TUNING_STRING

'''
Note: unlike KernelDescription, whose constants will be specialized for EVERY kernel.
      KernelTuningDatabase(ForArch) should work for all KernelDescription instances.

      Therefore the index of the database can only be built when seeing the
      first set of ArgumentSelection objects, because the json object itself
      has zero information about the triton kernel.
'''
class KernelTuningDatabaseForArch(object):
    def __init__(self, f):
        self._j = json.load(f)
        self._arch = self._j['arch']
        self._index = None
        self._index_matching_keys = None

    @property
    def arch(self):
        return self._arch

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
        return frozenset(map(convert, keys))

    def extract_keys_from_fsels(self, fsels):
        keys = {}
        for fsel in fsels:
            try:
                offset = self._fsel_positions.index(fsel.meta.first_apperance)
                value = fsel.argument_value
                keys[offset] = value
            except ValueError:
                pass
        return frozenset([keys[offset] for offset in range(len(self._fsel_positions))])

    def _build_db_index(self, fsels):
        self._init_matching_keys(fsels)
        self._index = defaultdict(list)
        self._index_dedup = defaultdict(list)
        for ti in self._j['tune_info']:
            tup = self.extract_keys_from_json(ti)
            self._index[tup].append(ti)
            is_dup = False
            for eti in self._index_dedup[tup]:
                # print(f'{eti=}')
                if eti['tuned_kernel'] == ti['tuned_kernel'] and eti['compiler_options'] == ti['compiler_options']:
                    is_dup = True
                    break
            if not is_dup:
                self._index_dedup[tup].append(ti)

    def _select_from_index(self,
                           fsels : 'list[ArgumentSelection]',
                           perf_meta : 'list[ArgumentMetadata]',
                           no_duplicate=True):
        tup = self.extract_keys_from_fsels(fsels)
        if no_duplicate:
            indexed = self._index_dedup[tup]
        else:
            indexed = self._index[tup]
        for tinfo in indexed:
            ps = dict(tinfo['tuned_kernel'])
            co = dict(tinfo['compiler_options'])
            if 'waves_per_eu' in ps:
                co['waves_per_eu'] = ps['waves_per_eu']
                del ps['waves_per_eu']
                # co['_debug'] = dict(tinfo)
            yield self._craft_perf_selection(ps, perf_meta), co

    def _craft_perf_selection(self, ps : dict, perf_meta: 'list[ArgumentSelection]'):
        return [TunedArgument(meta, ps[meta.argument_names[0]]) for meta in perf_meta]

class KernelTuningDatabase(object):
    def __init__(self, tune_info_dir : pathlib.Path, kernel_name : str):
        self.arch_dict = {}
        td = pathlib.Path(tune_info_dir) # in case tune_info_dir is str
        # print(f"Tryint to probe KernelTuningDatabase inside {td}")
        for fn in td.glob(f'tune-{kernel_name}-*.json'):
            with open(fn) as f:
                dba = KernelTuningDatabaseForArch(f)
            self.arch_dict[dba.arch] = dba

    def select_gpu(self, gpu):
        arch = AOTRITON_GPU_ARCH_TUNING_STRING[gpu]
        return self.arch_dict[arch]

    @property
    def empty(self):
        return not self.arch_dict
