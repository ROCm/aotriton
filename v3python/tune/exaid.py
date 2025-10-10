#!/usr/bin/env python

import sys
import os
from pathlib import Path
from .testrun import main as testrun_entry
from .utils import safe_readline
import subprocess
import importlib
import errno
from pathlib import Path
import json

CURRENT_FILE_PATH = Path(__file__).resolve()
AOTRITON_ROOT = CURRENT_FILE_PATH.parent.parent.parent.absolute()

def first(line, sep=" "):
    seps = line.split(sep, maxsplit=1)
    if len(seps) > 1:
        return seps
    return seps[0], None

class ExaidSubprocessNotOK(RuntimeError):
    def __init__(self, stdout: str|None, stderr: str|None):
        self.stdout = stdout
        self.stderr = stderr

class ExaidProxy(object):
    ENTRY = testrun_entry
    def __init__(self, module_name, gpu_id):
        self._module_name = module_name
        self._gpu_id = gpu_id
        self._process = None
        self._last_error = None

    def get_base_dir(self):
        return AOTRITON_ROOT.as_posix()

    @property
    def process(self):
        if self._process is None:
            args = ['python', '-m', 'v3python.tune.testrun',
                    self._module_name, '--gpu', str(self._gpu_id)]
            self._process = subprocess.Popen(args,
                                             stdin=subprocess.PIPE,
                                             stdout=subprocess.PIPE,
                                             cwd=self.get_base_dir(),
                                             text=True)
        return self._process

    def write(self, *objects, sep=' '):
        print(*objects, sep=sep, file=self.process.stdin, flush=True)
        print('[exaid] sending command to worker process: ', *objects, sep=sep, flush=True)

    def readinfo(self, *, timeout=10):
        (line, eno, error_msg) = safe_readline(self.process, timeout=timeout)
        if eno != 0:
            if eno == errno.ETIMEDOUT:
                self._process.kill()
            self._process.wait()
            del self._process
            self._process = None
            raise OSError(eno, error_msg)
        ret, info = first(line)
        if ret != "OK":
            raise ExaidSubprocessNotOK(line, error_msg)
        return info

class ExaidWorker(object):
    TMPFS_LOCATION = Path('/dev/shm/aotriton-tuner')
    _cache = {}

    def __init__(self, module_name: str, gpu_id: int):
        self._module_name = module_name
        self._module = None
        self._gpu_id = gpu_id
        self._proxy = None

    @property
    def module(self):
        if self._module is None:
            self._module = importlib.import_module('.' + self._module_name, package='v3python.tune')
        return self._module

    @property
    def tmpfs(self) -> Path:
        return self.TMPFS_LOCATION

    @property
    def proxy(self):
        if self._proxy is None:
            self._proxy = ExaidProxy(self._module_name, self._gpu_id)
        return self._proxy

    def entry_from_dict(self, entry_dict: dict):
        tune = self.module.TuneDesc()
        return tune.ENTRY_CLASS.from_dict(entry_dict)

    def get_tmpfs_for(self, entry_dict):
        return self.TMPFS_LOCATION / self.entry_from_dict(entry_dict).as_posix()

    def prepare_data(self, entry_dict: dict, workdir: Path):
        entry = self.entry_from_dict(entry_dict)
        self.proxy.write('prepare_data', entry.as_text(), workdir.as_posix())
        return self.proxy.readinfo(timeout=30)

    def probe(self, workdir: Path):
        self.proxy.write('probe', workdir.as_posix())
        return json.loads(self.proxy.readinfo())

    def benchmark(self, workdir: Path, kname: str, hsaco_id: int):
        self.proxy.write('benchmark', workdir.as_posix(), f'{kname}={hsaco_id}')
        return json.loads(self.proxy.readinfo())

def exaid_create(module_name, gpu_id):
    key = (module_name, gpu_id)
    if key not in ExaidWorker._cache:
        ExaidWorker._cache[key] = ExaidWorker(module_name, gpu_id)
    return ExaidWorker._cache[key]
