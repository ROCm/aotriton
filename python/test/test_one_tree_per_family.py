# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One process generates one family from one module tree.

Two trees can both supply a `flash` (the real `modules/` and this directory's
`fakefamily/` do), and both can be parsed. Only one can be GENERATED: every
generated path is keyed by the family name alone, so the second tree would
overwrite the first's output rather than sit beside it.

That makes the process-global family -> aot binding unambiguous, which is what
`load_family_aot` promises its callers -- notably `ir/triton/kdesc.py`'s
sancheck back-edge, which has a family name and no parser handle. Before this
was rejected the binding was last-writer-wins, and that back-edge answered a
first-tree description with the second tree's package.
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from aotriton.codegen.parser import Parser, load_family_aot

FAKE_MODULES = Path(__file__).resolve().parent / 'fakefamily'


def test_reloading_the_same_tree_is_idempotent():
    first = Parser(FAKE_MODULES).load_family_aot('flash')
    again = Parser(FAKE_MODULES).load_family_aot('flash')
    assert again is first                      # the sys.modules entry, reused
    assert load_family_aot('flash') is first


def test_a_second_tree_for_one_family_is_refused(tmp_path):
    # A byte-identical copy, which is the benign case: even here the two are
    # different module objects generating to one path, so the ambiguity is real
    # and the refusal does not depend on the trees having diverged.
    twin = tmp_path / 'fakefamily'
    shutil.copytree(FAKE_MODULES, twin)

    first = Parser(FAKE_MODULES).load_family_aot('flash')
    with pytest.raises(RuntimeError, match='already loaded'):
        Parser(twin).load_family_aot('flash')
    assert load_family_aot('flash') is first   # the refusal changed nothing
