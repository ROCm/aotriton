class KernelSignature(object):
    def __init__(self,
                 kdesc : 'KernelDescription',
                 func_selections : 'tuple[ArgumentSelection]',
                 perf_selections : 'tuple[ArgumentSelection]'):
        self._kdesc = kdesc
        self._func_selections = func_selections
        self._perf_selections = perf_selections
        self._selections = list(func_selections) + list(perf_selections)

    @property
    def godel_number(self):
        return sum([s.godel_number for s in self._func_selections])

    @property
    def compact_signature(self):
        lf = [s.compact_signature for s in self._func_selections]
        lp = [s.compact_signature for s in self._perf_selections]
        sf = ','.join([x for x in lf if x is not None])
        sp = ','.join([x for x in lp if x is not None])
        return 'F__' + sf + '__P__' + sp

    @property
    def functional_signature(self):
        lf = [s.compact_signature for s in self._func_selections]
        sf = ','.join([x for x in lf if x is not None])
        return 'FONLY__' + sf + '__'

    @property
    def arguments(self):
        return self._kdesc.ARGUMENTS

    @property
    def triton_api_signature_list(self) -> 'list[str]':
        sig = {}
        [s.update_triton_api_signature(sig) for s in self._selections]
        l = [None] * len(self.arguments)
        for k, v in sig.items():
            l[k] = v
        return l
