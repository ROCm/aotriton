-- Materialized views for the AOTriton tuning database
-- Copyright © 2026 Advanced Micro Devices, Inc.
-- SPDX-License-Identifier: MIT
--
-- PostgreSQL does not support CREATE OR REPLACE for materialized views.
-- Run this file (via schema.sql or directly) to create all materialized
-- views. To recreate after a schema change, drop in reverse dependency
-- order and re-run:
--
--   DROP MATERIALIZED VIEW IF EXISTS most_accurate_tuning_results;
--   \ir materialized_views.sql
--
-- Note: best_tuning_results is a plain table populated by
--   v3python/tune/pq/compute_best_results.py (Python multiprocessing).

-- ============================================================
-- most_accurate_tuning_results
-- For each (task_id, kernel_name, test_case, tensor_name):
-- the minimum absolute_error across all hsaco_index values.
-- Reads result_data JSONB inline; PostgreSQL pipelines the
-- jsonb_each fan-out with GROUP BY without materialising rows.
-- result_data schema: v3python/tune/pq/tuning_results_schema_sample.txt
-- ============================================================

CREATE MATERIALIZED VIEW IF NOT EXISTS most_accurate_tuning_results AS
SELECT
    tr.task_id,
    tq.arch,
    tq.task_config,
    tr.kernel_name,
    test_case.key                               AS test_case,
    tensor.key                                  AS tensor_name,
    MIN((tensor.value->>0)::float)              AS target_fudge_factor,
    MIN((tensor.value->>1)::float)              AS absolute_error
FROM tuning_results tr
JOIN task_queue tq ON tq.id = tr.task_id
CROSS JOIN LATERAL jsonb_each(tr.result_data->'adiffs') AS test_case(key, value)
CROSS JOIN LATERAL jsonb_each(test_case.value)           AS tensor(key, value)
WHERE tr.result_data IS NOT NULL
GROUP BY tr.task_id, tq.arch, tq.task_config, tr.kernel_name, test_case.key, tensor.key;

-- Matches the join key used by compute_best_results.py.
CREATE INDEX IF NOT EXISTS idx_most_accurate_tuning_results_lookup
    ON most_accurate_tuning_results (task_id, kernel_name, test_case, tensor_name);
