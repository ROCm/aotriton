#!/usr/bin/env python
# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

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
import logging

logger = logging.getLogger(__name__)

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
            logger.info(f"Starting exaid worker process: module={self._module_name}, gpu={self._gpu_id}")
            self._process = subprocess.Popen(args,
                                             stdin=subprocess.PIPE,
                                             stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE,
                                             cwd=self.get_base_dir())
            logger.info(f"Exaid worker process started: pid={self._process.pid}")
        return self._process

    def write(self, *objects, sep=' '):
        cmd = sep.join(str(o) for o in objects)
        logger.info(f"Sending command to worker (pid={self.process.pid}): {cmd}")
        line = cmd + '\n'
        self.process.stdin.write(line.encode('utf-8'))
        self.process.stdin.flush()

    def readinfo(self, *, timeout: int | float = 10):
        logger.info(f"Waiting for response from worker (pid={self.process.pid}, timeout={timeout}s)")
        while True:
            (line, eno, error_msg) = safe_readline(self.process, timeout=timeout)
            if eno != 0 or line is None:
                if eno == errno.ETIMEDOUT:
                    logger.error(f"Worker timeout after {timeout}s, killing process (pid={self._process.pid})")
                    self._process.kill()
                elif line is None:
                    logger.error(f"Worker closed stdout unexpectedly (pid={self._process.pid})")
                else:
                    logger.error(f"Worker error (pid={self._process.pid}, errno={eno}): {error_msg}")
                self._process.wait()
                stdout = self._process.stdout.read().decode('utf-8', errors='replace')
                stderr = self._process.stderr.read().decode('utf-8', errors='replace')
                logger.error(f"Worker stdout: {stdout}")
                logger.error(f"Worker stderr: {stderr}")
                del self._process
                self._process = None
                error_desc = error_msg if error_msg else "stdout closed unexpectedly"
                raise OSError(eno if eno != 0 else errno.EPIPE, error_desc + "\nSTDOUT:\n" + stdout + "\nSTDERR:\n" + stderr)
            ret, info = first(line)
            if ret == "OVERHEATING:":
                logger.warning(f"Worker overheating warning: {line}")
                continue
            if ret != "OK":
                logger.error(f"Worker returned non-OK status: {line}")
                raise ExaidSubprocessNotOK(line, error_msg)
            logger.info(f"Received response from worker (pid={self.process.pid}): {ret} {info}")
            break
        return info

    def join(self):
        if self._process is None:
            return
        pid = self._process.pid
        try:
            self._process.wait(0.2)
            logger.info(f"Worker process exited cleanly (pid={pid})")
        except subprocess.TimeoutExpired:
            logger.warning(f"Worker process did not exit in 0.2s, killing (pid={pid})")
            self._process.kill()
            self._process.wait()
            logger.info(f"Worker process killed (pid={pid})")
            del self._process
            self._process = None

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
        logger.info(f"prepare_data: entry={entry_dict}, workdir={workdir}")
        entry = self.entry_from_dict(entry_dict)
        self.proxy.write('prepare_data', entry.as_text(), workdir.as_posix())
        result = self.proxy.readinfo(timeout=120)
        logger.info(f"prepare_data completed: {result}")
        return result

    def probe(self, workdir: Path):
        logger.info(f"probe: workdir={workdir}")
        self.proxy.write('probe', workdir.as_posix())
        result = json.loads(self.proxy.readinfo())
        logger.info(f"probe completed: found {len(result)} kernels")
        return result

    def benchmark(self, workdir: Path, kname: str, hsaco_index: int):
        logger.info(f"benchmark: workdir={workdir}, kernel={kname}, hsaco_index={hsaco_index}")
        self.proxy.write('benchmark', workdir.as_posix(), f'{kname}={hsaco_index}')
        result = json.loads(self.proxy.readinfo())
        logger.info(f"benchmark completed: {kname}[{hsaco_index}] result={result.get('result', 'unknown')}")
        return result

    def exit(self):
        self.proxy.write("exit")
        self.proxy.join()

def exaid_create(module_name, gpu_id):
    key = (module_name, gpu_id)
    if key not in ExaidWorker._cache:
        ExaidWorker._cache[key] = ExaidWorker(module_name, gpu_id)
    return ExaidWorker._cache[key]

def exaid_exitall():
    for _, exaid in ExaidWorker._cache.items():
        exaid.exit()
    ExaidWorker._cache = {}
