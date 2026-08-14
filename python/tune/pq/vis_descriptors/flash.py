# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Python-side descriptor for flash attention kernels/ops.
# Drives query construction in visperf.py.
#
# dims entries: (sql_expression, alias)
#   sql_expression: fragment used in SELECT and GROUP BY (cast inline if needed)
#   alias: column name in result rows and the JS descriptor

FLASH_DESCRIPTOR: dict = {
    'id': 'flash',
    'label': 'Flash Attention',

    # Unified schema (modular-tune.md §4.3/§4.7): kernel-level and op-level
    # rows share the same best_tuning_results/tuning_results tables and the
    # same iface_name column; tuning_level ('kernel' | 'op') is the sole
    # discriminator, since iface_name collides across levels (e.g. bare
    # 'attn_fwd' is valid at both levels -- see modules/flash/tune/level_op.py).
    'kernels': ['attn_fwd', 'bwd_kernel_dk_dv', 'bwd_kernel_dq', 'bwd_kernel_fuse'],
    'kernel_table': 'best_tuning_results',
    'name_col': 'iface_name',

    # op-level iface_names are bare, never '<name>_op' -- the ImplSelector
    # DSL's 'op.' prefix is surface syntax only and never reaches storage.
    'ops': ['attn_fwd', 'attn_bwd'],

    # Dimensions extracted from task_config->entry.
    # Each entry: (sql_expression, alias)
    # seqlen_q and seqlen_k are always included as the matrix axes.
    'dims': [
        ("task_config->'entry'->>'dtype'",                                               'dtype'),
        ("(task_config->'entry'->>'hdim')::int",                                         'hdim'),
        ("(task_config->'entry'->>'seqlen_q')::int",                                     'seqlen_q'),
        ("(task_config->'entry'->>'seqlen_k')::int",                                     'seqlen_k'),
        ("CASE WHEN (task_config->'entry'->>'causal')::bool THEN 1 ELSE 0 END",         'causal'),
        ("(task_config->'entry'->>'bias_type')::int",                                    'bias_type'),
        ("CASE WHEN (task_config->'entry'->>'dropout_p')::float > 0 THEN 1 ELSE 0 END", 'dropout'),
    ],

    'matrix_axes': ('seqlen_q', 'seqlen_k'),
}
