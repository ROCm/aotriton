# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
ExaidPerfmonRunner: perfmon's runner type ('perfmon'), one per (module, GPU).

It drives a subject's native `bin/runner` through `perfmon/launch_runner.sh`,
which speaks the pgdef.exaid line protocol:

    enumerate <entry_pon> <iface>            OK {"backends": [names]}
    measure <entry_pon> <iface> <backend|->  OK {stats...} | UNAVAILABLE <msg> | error
    platform                                 OK {"vram_total_bytes", ...}

The runner is keyed by module only, never by subject: NodeContext keeps every
runner it creates, and one runner per subject would leave one native process
(and one GPU context) per subject alive. Instead `use_subject()` selects among
at most MAX_LIVE (2) native processes keyed by (subject, arch), evicting the
least recently used. Dispatch groups tasks by subject, so evictions are rare;
the second process absorbs the interleaving at a subject boundary.

Only Python's side of the protocol is here; the thermal gate is passed to the
runner by the environment (AOTRITON_THERMAL_GATE), which the process inherits.
"""

import json
import logging
import platform as _platform
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from aotriton.pgdef import paths
from aotriton.pgdef.exaid import ExaidProxy, ExaidSubprocessNotOK

logger = logging.getLogger(__name__)

#: The runner's answer to a measure whose backend cannot run this entry here.
UNAVAILABLE = 'UNAVAILABLE'

#: Wire token of the unforced operator in `measure`.
UNFORCED = '-'

#: Where subjects are installed, relative to the workdir (the manifest's
#: install subtree).
INSTALL_SUBTREE = Path('installed') / 'perfmon'


class BackendUnavailable(Exception):
    """This release or arch cannot run this (entry, backend): the adapter
    said unsupported, or the untimed first launch of a forced backend failed.
    Recorded as excluded, not as no data. A failing unforced operator is an
    error instead."""


def subject_dir(arch: str, subject: str) -> Path:
    return paths.workdir() / INSTALL_SUBTREE / arch / subject


def launcher() -> Path:
    """`perfmon/launch_runner.sh` of the checkout the node runs from: the
    configured source directory, else the current directory (the node
    scripts cd into the checkout). It is never installed with the package."""
    candidates = [paths.src_dir(), Path.cwd()]
    for root in candidates:
        path = root / 'perfmon' / 'launch_runner.sh'
        if path.is_file():
            return path
    raise FileNotFoundError(f'perfmon/launch_runner.sh not found under any of {[str(c) for c in candidates]}; '
                            'set AOTRITON_SRC to the checkout')


@lru_cache(maxsize=1)
def host_facts() -> dict:
    """Facts about the host that the container can see. Host names are not
    recorded; RAM speed and the driver version come from the host inventory."""
    facts: dict = {'kernel': _platform.release(), 'cpu': None, 'os': None, 'gpus_per_node': None}
    try:
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):
                facts['cpu'] = line.split(':', 1)[1].strip()
                break
    except OSError:
        pass
    try:
        for line in Path('/etc/os-release').read_text().splitlines():
            if line.startswith('PRETTY_NAME='):
                facts['os'] = line.split('=', 1)[1].strip().strip('"')
                break
    except OSError:
        pass
    try:
        # KFD topology nodes with a nonzero gpu_id are GPUs (zero is a CPU node).
        nodes = Path('/sys/class/kfd/kfd/topology/nodes').glob('*/gpu_id')
        facts['gpus_per_node'] = sum(1 for p in nodes if p.read_text().strip() not in ('', '0'))
    except OSError:
        pass
    return facts


@dataclass
class _Live:
    """One (subject, arch) runner slot: its proxy and the platform facts."""
    proxy: ExaidProxy | None = None
    platform: dict | None = None


class ExaidPerfmonRunner:
    """Handlers get one from `ctx.runner('perfmon', module)`; the node passes
    this class as `run_node_role(runners={'perfmon': ExaidPerfmonRunner})`."""

    #: The first command after a (re)launch also pays for HIP init.
    PLATFORM_TIMEOUT = 120
    ENUMERATE_TIMEOUT = 120
    #: Per reply line. The thermal gate's OVERHEATING keep-alives each start a
    #: new wait, so cooling down does not count against it.
    MEASURE_TIMEOUT = 600

    #: Native processes kept alive per GPU: the one being served and the one
    #: before it. Two covers a boundary between two subjects, where the
    #: messages of the last task of one and the first task of the next
    #: interleave; a third is never needed. An idle runner holds only its HIP
    #: context (tensors are released after each measure), so one spare is
    #: cheap, while a restart pays HIP init and a platform query.
    MAX_LIVE = 2

    def __init__(self, module: str, gpu_id: int):
        self._module = module
        self._gpu_id = gpu_id
        self._subject: str | None = None
        self._arch: str | None = None
        #: (subject, arch) -> its proxy (None until first used) and platform
        #: facts, least recently used first.
        self._live: OrderedDict[tuple, _Live] = OrderedDict()

    @property
    def subject(self) -> str | None:
        return self._subject

    def _argv_for(self, subject: str | None, arch: str | None):
        def spawn_argv() -> list[str]:
            if subject is None:
                raise RuntimeError('use_subject() must be called before the runner is launched')
            argv = [launcher().as_posix(),
                    '--subject', subject,
                    '--module', self._module,
                    '--gpu', str(self._gpu_id)]
            if arch is not None:
                # The runner identity is (subject, arch): a subject installed
                # under two arches is ambiguous to the launcher without it.
                argv += ['--arch', arch]
            return argv
        return spawn_argv

    def _current(self) -> '_Live':
        key = (self._subject, self._arch)
        live = self._live.get(key)
        if live is None:
            live = self._live[key] = _Live()
        return live

    @property
    def proxy(self) -> ExaidProxy:
        live = self._current()
        if live.proxy is None:
            live.proxy = ExaidProxy(self._argv_for(self._subject, self._arch))
        return live.proxy

    def use_subject(self, subject: str, arch: str | None = None) -> None:
        """Serve (`subject`, `arch`) from now on. With `arch`, check first that
        the subject is installed on this node: a missing subject is the node's
        problem, not a result. Switching to a runner still alive is free; a
        new one evicts the least recently used when MAX_LIVE are alive."""
        if arch is not None and not subject_dir(arch, subject).is_dir():
            raise FileNotFoundError(f'subject {subject} is not installed for {arch}: '
                                    f'{subject_dir(arch, subject)}')
        key = (subject, arch)
        if key in self._live:
            self._live.move_to_end(key)
        else:
            while len(self._live) >= self.MAX_LIVE:
                old_key, old = self._live.popitem(last=False)
                logger.info(f'GPU {self._gpu_id}: evicting the runner of {old_key[0]} ({old_key[1]})')
                if old.proxy is not None:
                    old.proxy.shutdown()
            logger.info(f'GPU {self._gpu_id}: switching runner from {self._subject} to {subject}')
            self._live[key] = _Live()
        self._subject = subject
        self._arch = arch

    def platform(self) -> dict:
        """The runner's platform facts, asked once per live runner."""
        live = self._current()
        if live.platform is None:
            self.proxy.write('platform')
            live.platform = json.loads(self.proxy.readinfo(timeout=self.PLATFORM_TIMEOUT))
        return live.platform

    def enumerate(self, entry_pon: str, iface: str) -> list[str]:
        """The canonical names of the backends the subject can force for
        `iface` ([] when its release cannot force backends)."""
        self.proxy.write('enumerate', entry_pon, iface)
        reply = json.loads(self.proxy.readinfo(timeout=self.ENUMERATE_TIMEOUT))
        return list(reply['backends'])

    def measure(self, entry_pon: str, iface: str, backend: str | None) -> dict:
        """Time `iface` forced to `backend` (None: unforced). Returns the
        runner's statistics. Raises BackendUnavailable for an UNAVAILABLE
        reply, ExaidSubprocessNotOK for any other error reply, and OSError
        when the runner dies or times out."""
        self.proxy.write('measure', entry_pon, iface, UNFORCED if backend is None else backend)
        try:
            return json.loads(self.proxy.readinfo(timeout=self.MEASURE_TIMEOUT))
        except ExaidSubprocessNotOK as e:
            line = (e.stdout or '').strip()
            if line == UNAVAILABLE or line.startswith(UNAVAILABLE + ' '):
                raise BackendUnavailable(line[len(UNAVAILABLE):].strip()) from e
            raise

    def exit(self) -> None:
        for live in self._live.values():
            if live.proxy is not None:
                live.proxy.shutdown()

