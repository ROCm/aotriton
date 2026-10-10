# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from .sqlite import Factory as SqliteFactory

FACTORIES = [
    SqliteFactory,
]

class Factories(object):

    '''
    Currently only support sqlite:
        sqlite: requires tuning_database.sqlite3 under path
        other: requires a config file, for example: pg.json
    '''
    _cache = {}

    @staticmethod
    def create_factory(path):
        key = path.as_posix()
        if key in Factories._cache:
            return Factories._cache[key]
        for fac in FACTORIES:
            if (path / fac.SIGNATURE_FILE).exists():
                instance = fac(path)
                Factories._cache[key] = instance
                return instance
            print(f'Cannot find {path/fac.SIGNATURE_FILE}')
        assert False, 'database.Factories.create_factory failed. Database file missing?'
