# Copyright © 2023-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT


class Binning(object):
    pass

class BinningLessOrEqual(Binning):

    def __init__(self, bin_representatives):
        self._bin_representatives = sorted(list(set(bin_representatives)))

    @property
    def nvalues(self):
        return len(self._bin_representatives)

    @property
    def representatives(self):
        return self._bin_representatives

    def codegen_binning_lambda(self, key, out_suffix):
        out = f'{key}{out_suffix}'
        stmt = []
        stmt.append(f'auto {out} = [] (int x) {{')
        for index, rep in enumerate(self._bin_representatives):
            stmt.append(f'    if (x <= {rep}) return {index};')
        stmt.append(f'    return {len(self._bin_representatives)-1};')
        stmt.append(f'}}(params.{key});')
        return stmt

class BinningExact(Binning):
    pass
