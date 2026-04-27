-- Materialized view: best_tuning_results
-- Copyright © 2026 Advanced Micro Devices, Inc.
-- SPDX-License-Identifier: MIT
--
-- Unnests result_data.adiffs and aggregates over all hsaco_index values,
-- producing one row per (task_id, kernel_name, test_case, tensor_name)
-- with the minimum absolute_error across all hsacos.
-- Adiff leaf array layout: [fudge_factor, absolute_error, reference_error]
-- absolute_error may be NULL for NaN results.
--
-- Refresh before building the tuning database:
--   REFRESH MATERIALIZED VIEW best_tuning_results;

DROP MATERIALIZED VIEW IF EXISTS best_tuning_results;

CREATE MATERIALIZED VIEW best_tuning_results AS
SELECT
    tq.id          AS task_id,
    tq.arch,
    tq.task_config,
    tr.kernel_name,
    test_case.key  AS test_case,
    tensor.key     AS tensor_name,
    MIN((tensor.value->>1)::float) AS absolute_error
FROM tuning_results tr
CROSS JOIN LATERAL jsonb_each(tr.result_data->'adiffs') AS test_case(key, value)
CROSS JOIN LATERAL jsonb_each(test_case.value)           AS tensor(key, value)
JOIN task_queue tq ON tq.id = tr.task_id
WHERE tr.result_data IS NOT NULL
GROUP BY tq.id, tq.arch, tq.task_config, tr.kernel_name, test_case.key, tensor.key;

CREATE INDEX idx_best_tuning_results_lookup
    ON best_tuning_results (arch, kernel_name, task_id);
