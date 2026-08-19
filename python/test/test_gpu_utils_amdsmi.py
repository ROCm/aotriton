# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Temperature-sensor selection in `aotriton.tune.gpu_utils`, against a stub AMD-SMI.

Which sensor a GPU implements varies by ASIC and the two known cases are
opposites: gfx942 answers junction and rejects edge, gfx1151 answers edge and
rejects junction (ROCm/aotriton#188, where an unhandled junction query aborted
tuning outright). `_pick_temp_sensor` therefore probes instead of assuming, and
this file pins that branching down.

Stubbed rather than run against real hardware, in the same spirit as this
suite's fake kernels: it needs no GPU, and it can present a gfx1151 to the code
under test on a machine that has none. The flip side is that it validates our
model of AMD-SMI, not AMD-SMI -- the behaviours encoded in `_FakeAmdSmi` were
read off real hardware and a future ROCm could diverge without this noticing.
Everything else in gpu_utils (BDF resolution, VRAM units, init refcounting,
visible-device masks) needs a real amdsmi + GPU and is not covered here.
"""

import importlib
import sys
import types

import pytest

# ---------------------------------------------------------------- stub AMD-SMI

AMDSMI_STATUS_NOT_SUPPORTED = 2


class _FakeSensor:
    """Stands in for amdsmi.AmdSmiTemperatureType members (`.name` is logged)."""

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f'<sensor {self.name}>'


class _FakeAmdSmiLibraryException(Exception):
    def __init__(self, err_code):
        super().__init__(err_code)
        self.err_code = abs(err_code)

    def get_error_code(self):
        return self.err_code


class _FakeAmdSmi:
    """Minimal AMD-SMI whose sensor support is settable per ASIC."""

    def __init__(self):
        self.AmdSmiTemperatureType = types.SimpleNamespace(
            EDGE=_FakeSensor('EDGE'), JUNCTION=_FakeSensor('HOTSPOT'))
        self.AmdSmiTemperatureMetric = types.SimpleNamespace(CURRENT='current')
        self.AmdSmiLibraryException = _FakeAmdSmiLibraryException
        self.amdsmi_wrapper = types.SimpleNamespace(
            AMDSMI_STATUS_NOT_SUPPORTED=AMDSMI_STATUS_NOT_SUPPORTED)
        # sensor -> reading; anything absent raises NOT_SUPPORTED
        self.supported = {}
        # override to raise a different error code from amdsmi_get_temp_metric
        self.error_code = None
        self.reads = 0

    # -- lifecycle (refcounted, like the real library) --
    def amdsmi_init(self):
        pass

    def amdsmi_shut_down(self):
        pass

    def amdsmi_get_processor_handle_from_bdf(self, bdf):
        return f'handle:{bdf}'

    def amdsmi_get_processor_handles(self):
        return ['handle:0000:65:00.0']

    # -- the call under test --
    def amdsmi_get_temp_metric(self, handle, sensor, metric):
        self.reads += 1
        if self.error_code is not None:
            raise _FakeAmdSmiLibraryException(self.error_code)
        if sensor not in self.supported:
            raise _FakeAmdSmiLibraryException(AMDSMI_STATUS_NOT_SUPPORTED)
        return self.supported[sensor]


def _fake_torch():
    """Enough torch for gpu_utils to import and for `_bdf_of` to work."""
    torch = types.ModuleType('torch')

    class Tensor:  # `torch.Tensor | None` annotations evaluate at def time
        pass

    torch.Tensor = Tensor
    torch.cuda = types.SimpleNamespace(
        get_device_properties=lambda i: types.SimpleNamespace(
            pci_domain_id=0, pci_bus_id=0x65, pci_device_id=0))
    return torch


def _fake_pyaotriton():
    mod = types.ModuleType('pyaotriton')
    for name in ('get_name_suffix', 'T0', 'T1', 'T2', 'T4', 'DType', 'Stream',
                 'hipError_t', 'hipGetLastError', 'HipMemory',
                 'hipDeviceSynchronize'):
        setattr(mod, name, object())
    return mod


@pytest.fixture
def gpu_utils():
    """Import gpu_utils against the stubs, and undo it afterwards.

    `_TEMP_SENSORS` is built at import time from `amdsmi.AmdSmiTemperatureType`,
    so the stub has to be in `sys.modules` before the import, and any real
    gpu_utils another test already imported has to be evicted first.
    """
    saved = {name: sys.modules.get(name)
             for name in ('torch', 'amdsmi', 'pyaotriton',
                          'aotriton.tune.gpu_utils')}
    fake = _FakeAmdSmi()
    sys.modules['torch'] = _fake_torch()
    sys.modules['amdsmi'] = fake
    sys.modules['pyaotriton'] = _fake_pyaotriton()
    sys.modules.pop('aotriton.tune.gpu_utils', None)
    try:
        mod = importlib.import_module('aotriton.tune.gpu_utils')
        mod.default_device_id = lambda: 0
        yield mod, fake
    finally:
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def _reset(mod):
    """Clear the cached device so each case re-probes."""
    if mod._amdsmi_stack is not None:
        mod._amdsmi_stack.close()
    mod._amdsmi_stack = None
    mod._amdsmi_device_id = None
    mod._amdsmi_handle = None
    mod._amdsmi_sensor = None


# --------------------------------------------------------------------- tests

def test_junction_preferred_when_supported(gpu_utils):
    """gfx942: junction answers, edge does not -- junction must win."""
    mod, fake = gpu_utils
    T = fake.AmdSmiTemperatureType
    fake.supported = {T.JUNCTION: 39}
    _reset(mod)

    handle, sensor = mod._own_amdsmi_device(0)

    assert sensor is T.JUNCTION
    assert mod._get_temperature_amdsmi(handle, sensor) == 39


def test_falls_back_to_edge_when_junction_unsupported(gpu_utils, capsys):
    """gfx1151: junction raises NOT_SUPPORTED, edge answers (ROCm/aotriton#188)."""
    mod, fake = gpu_utils
    T = fake.AmdSmiTemperatureType
    fake.supported = {T.EDGE: 30}
    _reset(mod)

    handle, sensor = mod._own_amdsmi_device(0)

    assert sensor is T.EDGE
    assert mod._get_temperature_amdsmi(handle, sensor) == 30
    # the fallback is meant to be visible, not silent
    assert 'HOTSPOT' in capsys.readouterr().out


def test_junction_still_wins_when_both_supported(gpu_utils):
    """Preference order, not merely 'whichever answers first'."""
    mod, fake = gpu_utils
    T = fake.AmdSmiTemperatureType
    fake.supported = {T.EDGE: 30, T.JUNCTION: 39}
    _reset(mod)

    _handle, sensor = mod._own_amdsmi_device(0)

    assert sensor is T.JUNCTION


def test_raises_when_no_sensor_is_supported(gpu_utils):
    """Loud: a silent skip would leave an overheating GPU unthrottled."""
    mod, fake = gpu_utils
    fake.supported = {}
    _reset(mod)

    with pytest.raises(RuntimeError, match='implements none'):
        mod._own_amdsmi_device(0)
    assert mod._amdsmi_stack is None, 'must not cache a half-built device'


def test_unrelated_amdsmi_error_is_not_swallowed(gpu_utils):
    """Only NOT_SUPPORTED means 'try the next sensor'."""
    mod, fake = gpu_utils
    fake.supported = {fake.AmdSmiTemperatureType.JUNCTION: 39}
    fake.error_code = 11  # anything that is not NOT_SUPPORTED
    _reset(mod)

    with pytest.raises(fake.AmdSmiLibraryException) as excinfo:
        mod._own_amdsmi_device(0)
    assert excinfo.value.get_error_code() == 11


def test_sensor_probe_is_cached(gpu_utils):
    """Probing once per worker, not once per poll."""
    mod, fake = gpu_utils
    T = fake.AmdSmiTemperatureType
    fake.supported = {T.EDGE: 30}  # junction fails first -> 2 reads to probe
    _reset(mod)

    handle, sensor = mod._own_amdsmi_device(0)
    after_probe = fake.reads
    for _ in range(5):
        mod._own_amdsmi_device(0)
        mod._get_temperature_amdsmi(handle, sensor)

    assert fake.reads - after_probe == 5, 'cached device must not re-probe'
