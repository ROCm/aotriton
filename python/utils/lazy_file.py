# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import shutil
import io
from pathlib import Path

# LazyFile: a class to support lazy write to disk file
#           The file is only updated when the content changes
# Was named as NoWriteIfNoUpdateFile (very verbose)
class LazyFile(object):
    # When True, all instances skip actual disk writes (content is still
    # generated in memory so the rest of the pipeline runs normally).
    suppress_writes: bool = False

    def __init__(self, ofn : Path):
        self._ofn = ofn
        self._old_content = ''

    @property
    def path(self):
        return self._ofn

    def __enter__(self):
        self._mf = io.StringIO()
        if not LazyFile.suppress_writes and self._ofn.exists():
            with open(self._ofn) as f:
                self._old_content = f.read()
        return self._mf

    @property
    def memory_file(self):
        return self._mf

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None or LazyFile.suppress_writes:
            return  # do not flush partial buffer on exception
        mf = self.memory_file
        mf.seek(0)
        if mf.read() != self._old_content:
            mf.seek(0)
            with open(self.path, 'w') as of:
                shutil.copyfileobj(mf, of)

