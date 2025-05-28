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
    @staticmethod
    def create_factory(path):
        for fac in FACTORIES:
            if (path / fac.SIGNATURE_FILE).exists():
                return fac(path)
        assert False, 'database.Factories.create_factory failed. Database file missing?'
