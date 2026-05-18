-- most_accurate_tuning_results: plain table (replaces materialized view)
-- Populated by .tune/bin/recreate_materialized_view (full) or
-- .tune/bin/update_materialized_view (incremental by task_id).
-- Copyright © 2026 Advanced Micro Devices, Inc.
-- SPDX-License-Identifier: MIT

CREATE TABLE IF NOT EXISTS most_accurate_tuning_results (
    task_id              BIGINT  NOT NULL,
    arch                 TEXT    NOT NULL,
    task_config          JSONB   NOT NULL,
    kernel_name          TEXT    NOT NULL,
    test_case            TEXT    NOT NULL,
    tensor_name          TEXT    NOT NULL,
    target_fudge_factor  FLOAT,
    absolute_error       FLOAT
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_most_accurate_tuning_results_lookup
    ON most_accurate_tuning_results (task_id, kernel_name, test_case, tensor_name);

-- most_accurate_optune_results: op-mode counterpart, populated by
-- recreate_materialized_view / update_materialized_view --tuning_mode op.
CREATE TABLE IF NOT EXISTS most_accurate_optune_results (
    task_id              BIGINT  NOT NULL,
    arch                 TEXT    NOT NULL,
    task_config          JSONB   NOT NULL,
    op_name              TEXT    NOT NULL,
    test_case            TEXT    NOT NULL,
    tensor_name          TEXT    NOT NULL,
    target_fudge_factor  FLOAT,
    absolute_error       FLOAT
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_most_accurate_optune_results_lookup
    ON most_accurate_optune_results (task_id, op_name, test_case, tensor_name);
