#!/usr/bin/env python
# Copyright © 2023-2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import pytest
from .exaid import exaid_create, ExaidSubprocessNotOK
import shutil
import json
import torch

@pytest.mark.parametrize('gpu_id', [3])
@pytest.mark.parametrize('module', ['flash'])
def test_exaid(module, gpu_id):
    exaid = exaid_create(module, gpu_id)
    entry = {"dtype": "float16", "hdim": 32, "seqlen_q": 256, "seqlen_k": 128, "causal": False, "dropout_p": 0.5, "bias_type": 0}
    tmpdir = exaid.get_tmpfs_for(entry)
    exaid.prepare_data(entry, tmpdir)
    assert (tmpdir / 'entry.json').is_file()
    with open(tmpdir / 'entry.json') as f:
        entry_json = json.load(f)
    for test in entry_json["tests"]:
        test_name = test["test_name"]
        d = torch.load(test["pt_file"])
        for tname, te in d["bidi_outputs"].items():
            if te is None:
                continue
            ref_error = te[1]
            assert ref_error > 0, f"Test {test_name} tensor {tname} has ref_error == 0.0"
    kernel_dict = exaid.probe(tmpdir)
    KNAMES = [ 'attn_fwd', 'bwd_kernel_dk_dv', 'bwd_kernel_dq', 'bwd_kernel_fuse' ]
    for kname in KNAMES:
        assert kname in kernel_dict.keys()
        hsaco_id = 0
        result_data = exaid.benchmark(tmpdir, kname, hsaco_id)
        assert result_data
        assert result_data['impl_selection']['kernel_name'] == kname
        assert result_data['impl_selection']['hsaco_index'] == hsaco_id
        assert "adiffs" in result_data
    shutil.rmtree(tmpdir)

if __name__ == '__main__':
    main()

