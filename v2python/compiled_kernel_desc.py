#!/usr/bin/env python

from pathlib import Path
import json

SOURCE_PATH = (Path(__file__) / '..' / 'v2src').resolve()

class CompiledKernelDescription(object):

    def __init__(self,
                 source_desc : 'KernelDescription',
                 choice: 'list[frozenset[str], list[str]]',
                 signature_in_list : 'list[str]',
                 object_file_path : Path):
        self._source_desc = source_desc
        self._argument_choices = choice
        self._sigature_in_list

