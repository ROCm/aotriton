#!/usr/bin/env python
# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
import logging
from pathlib import Path

from aotriton.pgdef.exaid import ExaidProxy, ExaidSubprocessNotOK

logger = logging.getLogger(__name__)


class ExaidTunerWorker(object):
    """The tune application's runner (type 'tuner'): a programmatic consumer of
    a `testrun` subprocess, one persistent process per (module, GPU).

    Handlers get one from `ctx.runner('tuner', module_name)`; the node passes
    this class to `run_node_role(runners={'tuner': ExaidTunerWorker})`. Unlike
    `testrun`, callers here (handlers.py) DO carry a level -- the task
    category's second level, 'kernel' or 'op' -- but it is not state of this
    class: it is threaded into `probe()` as a per-call filter argument
    (Revision note 3).

    Startup is ~2 s (torch plus a GPU context), so the process is reused across
    tasks. TMPFS_LOCATION is the tuner's scratch space for prepared data.
    """

    TMPFS_LOCATION = Path('/dev/shm/aotriton-tuner')

    def __init__(self, module_name: str, gpu_id: int):
        self._module_name = module_name
        self._module = None
        self._gpu_id = gpu_id
        self._proxy = None

    def _spawn_argv(self):
        return ['python', '-m', 'aotriton.tune.testrun',
                self._module_name, '--gpu', str(self._gpu_id)]

    @property
    def module(self):
        if self._module is None:
            from .registry import load_tune_module
            self._module = load_tune_module(self._module_name)
        return self._module

    @property
    def tmpfs(self) -> Path:
        return self.TMPFS_LOCATION

    @property
    def proxy(self):
        if self._proxy is None:
            self._proxy = ExaidProxy(self._spawn_argv)
        return self._proxy

    def entry_from_dict(self, entry_dict: dict):
        tune = self.module.TuneDesc()
        return tune.ENTRY_CLASS.from_dict(entry_dict)

    def get_tmpfs_for(self, entry_dict):
        return self.TMPFS_LOCATION / self.entry_from_dict(entry_dict).as_posix()

    def prepare_data(self, entry_dict: dict, workdir: Path, extra_im_texts: list[str] = []):
        logger.info(f"prepare_data: entry={entry_dict}, workdir={workdir}, extra_ims={len(extra_im_texts)}")
        workdir.mkdir(parents=True, exist_ok=True)
        entry = self.entry_from_dict(entry_dict)
        self.proxy.write('prepare_data', entry.as_text(), workdir.as_posix(), *extra_im_texts)
        result = self.proxy.readinfo(timeout=600)
        logger.info(f"prepare_data completed: {result}")
        return result

    def probe(self, workdir: Path, arch: str | None = None, tuning_level: str | None = None):
        """
        Args:
            tuning_level: optional per-call filter (e.g. 'kernel' or 'op'),
                NOT worker state -- passed straight through to `testrun`'s
                `probe` command as a third wire token. Callers driving a task
                (e.g. the ProbeHandler) pass the category's second level
                here so a worker only probes what its container's library can
                serve; filtering happens on this call, before `testrun`
                attempts any import, not by post-hoc filtering this method's
                return value.
        """
        logger.info(f"probe: workdir={workdir} arch={arch} tuning_level={tuning_level}")
        self.proxy.write('probe', workdir.as_posix(), arch or '', tuning_level or '')
        # TODO: Add 'PARTIAL:' to wire protocol for long outputs.
        result = json.loads(self.proxy.readinfo(timeout=30))
        logger.info(f"probe completed: found {len(result)} kernels")
        return result

    def benchmark(self, workdir: Path, impl_selector):
        """
        Args:
            impl_selector: an `aotriton.tune.tdesc.ImplSelector` instance
                identifying the impl variant to benchmark. Its `as_text()`
                form (e.g. 'attn_fwd=3' or 'op.attn_fwd=1') is the DSL the
                `testrun.py` worker process's `benchmark` command parses back
                into an `ImplSelector` via `ImplSelector.parse_text()`.
        """
        logger.info(f"benchmark: workdir={workdir}, impl_selector={impl_selector.as_text()}")
        self.proxy.write('benchmark', workdir.as_posix(), impl_selector.as_text())
        # TODO: Need a keep alive protocol for long experiments
        result = json.loads(self.proxy.readinfo(timeout=300))
        logger.info(f"benchmark completed: {impl_selector.as_text()} "
                   f"result={result.get('result', 'unknown')}")
        return result

    def exit(self):
        self.proxy.shutdown()
