# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

'''
The tuned cells of a TuningView (csvdb.py), laid out for a generated LUT.

A LUT has one slab per gpu the functional is optimized for and one dimension
per tuning key; along a key it has one bucket per value tuned on any gpu.
Kernels and operators decode the cells differently (hsaco options vs backend
precedences) and store different entries, but lay out the LUT the same way.
'''

import numpy as np


class TunedCells(object):
    '''
    Every tuned cell of a TuningView, decoded.

    cells: (gpu index, tuning key values, decode(cell)) per tuned cell
    key_values: per tuning key, the sorted values tuned on any gpu (the LUT's
                buckets along that key)
    '''

    def __init__(self, view, key_names, decode):
        nkeys = len(key_names)
        self.nslabs = len(view.slabs)
        self.cells = []
        for i, slab in enumerate(view.slabs):
            if slab is None:
                continue
            assert slab.tuning_keys == key_names, \
                f'{slab.path}: tuning keys {slab.tuning_keys}, expected {key_names}'
            for index in zip(*np.nonzero(slab.cells)):
                keys = [slab.ladders[d][index[d]] for d in range(nkeys)]
                self.cells.append((i, keys, decode(slab.cells[index])))
        self.key_values = [sorted({keys[d] for _, keys, _ in self.cells}) for d in range(nkeys)]

    @property
    def values(self):
        return [value for _, _, value in self.cells]

    def fill(self, entries):
        '''
        The LUT (int32, -1 = not tuned) holding entries[j] at the bucket of
        cells[j]. Slabs after the first start as a copy of the first, so a gpu
        without a tuned cell somewhere takes the first gpu's entry.
        '''
        shape = [self.nslabs] + [len(v) for v in self.key_values]
        lut = np.full(shape, -1, dtype=np.int32)
        buckets = [{v: j for j, v in enumerate(values)} for values in self.key_values]
        for i in range(self.nslabs):
            if i > 0:
                lut[i] = lut[0]
            for (gpu, keys, _), entry in zip(self.cells, entries):
                if gpu == i:
                    lut[(i, *[buckets[d][k] for d, k in enumerate(keys)])] = entry
        return lut
