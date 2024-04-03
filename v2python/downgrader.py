# Copyright © 2023-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

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
            return None
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

