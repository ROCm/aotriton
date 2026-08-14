-- most_accurate_tuning_results: plain table (replaces materialized view)
-- Populated by .tune/bin/recreate_materialized_view (full) or
-- .tune/bin/update_materialized_view (incremental by task_id).
-- Copyright © 2026 Advanced Micro Devices, Inc.
-- SPDX-License-Identifier: MIT
--
-- Phase 2 unification (modular-tune.md §4.3/§4.7): single table, keyed by
-- (task_id, iface_name, test_case, tensor_name); `tuning_level` denormalized
-- (see schema.sql's tuning_results comment for why) rather than a separate
-- most_accurate_optune_results table.

CREATE TABLE IF NOT EXISTS most_accurate_tuning_results (
    task_id              BIGINT  NOT NULL,
    arch                 TEXT    NOT NULL,
    tuning_level         TEXT    NOT NULL,
    task_config          JSONB   NOT NULL,
    iface_name           TEXT    NOT NULL,
    test_case            TEXT    NOT NULL,
    tensor_name          TEXT    NOT NULL,
    target_fudge_factor  FLOAT,
    absolute_error       FLOAT
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_most_accurate_tuning_results_lookup
    ON most_accurate_tuning_results (task_id, iface_name, test_case, tensor_name);
