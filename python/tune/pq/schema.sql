-- Tuner v3.5 PostgreSQL Queue Schema
-- Copyright © 2026 Advanced Micro Devices, Inc.
-- SPDX-License-Identifier: MIT
--
-- Phase 2 (modularization unification, modular-tune.md §4.3/§4.7): the
-- Phase-1 `flash`/`flash_op` module-name split is gone. `module` now always
-- names a FAMILY ('flash', ...); the axis that used to be spelled as a
-- `_op`-suffixed module name is `tuning_level` ('kernel' | 'op'), stored as
-- its own NOT NULL column everywhere the old code matched on
-- `module (NOT) LIKE '%_op'`.
--
-- `iface_name`/`impl_index` replace the old `kernel_name`/`hsaco_index` and
-- `op_name`/`backend_index` column pairs (ImplSelector, modular-tune.md §4.2):
-- one interface-name/variant-index column pair shared by every tuning level,
-- not two level-specific ones.
--
-- Fresh schema, no migration (modular-tune.md, PR B directive): existing
-- `optune_results` / `best_optune_results` / `most_accurate_optune_results`
-- tables are gone outright, not ALTER'd. Re-initialize and re-tune.

-- Parent table (partitioned by architecture)
CREATE TABLE IF NOT EXISTS task_queue (
    id BIGSERIAL,
    arch TEXT NOT NULL,
    module TEXT NOT NULL,
    tuning_level TEXT NOT NULL CHECK (tuning_level IN ('kernel', 'op')),
    task_config JSONB NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',  -- pending/running/completed/failed/cancelled
    priority INT DEFAULT 5,
    worker_id TEXT,
    node_hostname TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    error TEXT,
    retry_count INT DEFAULT 0,
    PRIMARY KEY (id, arch)
) PARTITION BY LIST (arch);

-- Worker heartbeat table (for monitoring and health checks)
CREATE TABLE IF NOT EXISTS worker_heartbeat (
    node_hostname TEXT NOT NULL,
    worker_name TEXT NOT NULL,
    arch TEXT NOT NULL,
    last_heartbeat TIMESTAMP NOT NULL DEFAULT NOW(),
    status TEXT NOT NULL DEFAULT 'active',  -- active/idle/dead
    tasks_completed INT DEFAULT 0,
    tasks_failed INT DEFAULT 0,
    PRIMARY KEY (node_hostname, worker_name)
);

CREATE INDEX IF NOT EXISTS idx_worker_heartbeat_alive
    ON worker_heartbeat (last_heartbeat DESC)
    WHERE status = 'active';

-- Tuning results table (stores individual per-impl-variant benchmark
-- results). Unified: this single table replaces the former
-- tuning_results/optune_results pair -- one row per (task_id, iface_name,
-- impl_index) regardless of tuning_level. `tuning_level` is denormalized
-- here (not just on task_queue) because this table is frequently queried by
-- (iface_name, impl_index) alone, without a task_queue join, and
-- `iface_name` collides across levels (highest-risk area #2 of
-- modular-tune.md: e.g. 'attn_fwd' is a valid iface_name at both the kernel
-- and op level).
CREATE TABLE IF NOT EXISTS tuning_results (
    id BIGSERIAL PRIMARY KEY,
    task_id BIGINT NOT NULL,
    tuning_level TEXT NOT NULL CHECK (tuning_level IN ('kernel', 'op')),
    iface_name TEXT NOT NULL,
    impl_index INT NOT NULL,
    result TEXT NOT NULL,  -- OK/NotOK/crash/ERROR
    result_data JSONB,
    error JSONB,
    gpu_id INT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_tuning_results_task
    ON tuning_results (task_id, iface_name, impl_index);

CREATE INDEX IF NOT EXISTS idx_tuning_results_iface
    ON tuning_results (tuning_level, iface_name, result);

-- Utility views for monitoring
CREATE OR REPLACE VIEW queue_progress AS
SELECT
    arch,
    COUNT(*) FILTER (WHERE status = 'pending') as pending,
    COUNT(*) FILTER (WHERE status = 'running') as running,
    COUNT(*) FILTER (WHERE status = 'completed') as completed,
    COUNT(*) FILTER (WHERE status = 'failed') as failed,
    COUNT(*) FILTER (WHERE status = 'cancelled') as cancelled,
    COUNT(*) as total,
    ROUND(100.0 * COUNT(*) FILTER (WHERE status = 'completed') / NULLIF(COUNT(*), 0), 2) as pct_complete
FROM task_queue
GROUP BY arch
ORDER BY arch;

CREATE OR REPLACE VIEW kernel_queue_progress AS
SELECT
    arch,
    COUNT(*) FILTER (WHERE status = 'pending') as pending,
    COUNT(*) FILTER (WHERE status = 'running') as running,
    COUNT(*) FILTER (WHERE status = 'completed') as completed,
    COUNT(*) FILTER (WHERE status = 'failed') as failed,
    COUNT(*) FILTER (WHERE status = 'cancelled') as cancelled,
    COUNT(*) as total,
    ROUND(100.0 * COUNT(*) FILTER (WHERE status = 'completed') / NULLIF(COUNT(*), 0), 2) as pct_complete
FROM task_queue
WHERE tuning_level = 'kernel'
GROUP BY arch
ORDER BY arch;

CREATE OR REPLACE VIEW op_queue_progress AS
SELECT
    arch,
    COUNT(*) FILTER (WHERE status = 'pending') as pending,
    COUNT(*) FILTER (WHERE status = 'running') as running,
    COUNT(*) FILTER (WHERE status = 'completed') as completed,
    COUNT(*) FILTER (WHERE status = 'failed') as failed,
    COUNT(*) FILTER (WHERE status = 'cancelled') as cancelled,
    COUNT(*) as total,
    ROUND(100.0 * COUNT(*) FILTER (WHERE status = 'completed') / NULLIF(COUNT(*), 0), 2) as pct_complete
FROM task_queue
WHERE tuning_level = 'op'
GROUP BY arch
ORDER BY arch;

CREATE OR REPLACE VIEW worker_health AS
SELECT
    node_hostname,
    worker_name,
    arch,
    status,
    tasks_completed,
    tasks_failed,
    EXTRACT(EPOCH FROM (NOW() - last_heartbeat)) as seconds_since_heartbeat,
    CASE
        WHEN EXTRACT(EPOCH FROM (NOW() - last_heartbeat)) < 60 THEN 'healthy'
        WHEN EXTRACT(EPOCH FROM (NOW() - last_heartbeat)) < 300 THEN 'stale'
        ELSE 'dead'
    END as health_status
FROM worker_heartbeat
ORDER BY last_heartbeat DESC;

CREATE OR REPLACE VIEW task_timing_stats AS
SELECT
    arch,
    module,
    tuning_level,
    COUNT(*) as completed_tasks,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM (completed_at - started_at))) as median_duration_sec,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM (completed_at - started_at))) as p95_duration_sec,
    AVG(EXTRACT(EPOCH FROM (completed_at - started_at))) as avg_duration_sec
FROM task_queue
WHERE status = 'completed' AND completed_at IS NOT NULL
GROUP BY arch, module, tuning_level;

CREATE OR REPLACE VIEW stale_tasks AS
SELECT
    id,
    arch,
    worker_id,
    node_hostname,
    EXTRACT(EPOCH FROM (NOW() - started_at)) / 3600 as hours_running
FROM task_queue
WHERE status = 'running'
  AND EXTRACT(EPOCH FROM (NOW() - started_at)) > 7200  -- Running > 2 hours
ORDER BY started_at ASC;

CREATE OR REPLACE VIEW completion_eta AS
WITH stats AS (
    SELECT
        arch,
        COUNT(*) FILTER (WHERE status = 'pending') as remaining,
        COUNT(*) FILTER (WHERE status = 'running') as active_workers,
        AVG(EXTRACT(EPOCH FROM (completed_at - started_at)))
            FILTER (WHERE status = 'completed' AND completed_at > NOW() - INTERVAL '1 hour')
            as avg_task_duration_sec
    FROM task_queue
    GROUP BY arch
)
SELECT
    arch,
    remaining,
    active_workers,
    avg_task_duration_sec,
    CASE
        WHEN active_workers > 0 AND avg_task_duration_sec IS NOT NULL THEN
            ROUND((remaining * avg_task_duration_sec / active_workers) / 3600, 2)
        ELSE NULL
    END as eta_hours
FROM stats
WHERE remaining > 0;

-- Function to create partition for an architecture
CREATE OR REPLACE FUNCTION create_arch_partition(arch_name TEXT)
RETURNS VOID AS $$
DECLARE
    partition_name TEXT;
BEGIN
    partition_name := 'task_queue_' || arch_name;

    -- Create partition if it doesn't exist
    EXECUTE format(
        'CREATE TABLE IF NOT EXISTS %I PARTITION OF task_queue FOR VALUES IN (%L)',
        partition_name, arch_name
    );

    -- Create indexes on the partition. tuning_level is the leading column
    -- (alongside status) since fetch_tasks() always filters on both --
    -- a kernel worker must never claim an op task and vice versa (F16).
    EXECUTE format(
        'CREATE INDEX IF NOT EXISTS %I ON %I (tuning_level, status, priority DESC, id ASC) WHERE status = %L',
        partition_name || '_fetch', partition_name, 'pending'
    );

    EXECUTE format(
        'CREATE INDEX IF NOT EXISTS %I ON %I (worker_id, status)',
        partition_name || '_worker', partition_name
    );

    EXECUTE format(
        'CREATE INDEX IF NOT EXISTS %I ON %I (created_at DESC)',
        partition_name || '_created', partition_name
    );
END;
$$ LANGUAGE plpgsql;

-- best_tuning_results: plain table populated by compute_best_results.py.
-- Unified: replaces the former best_tuning_results/best_optune_results pair.
-- For each (task_id, iface_name): fastest impl_index meeting the accuracy
-- threshold relative to most_accurate_tuning_results.
CREATE TABLE IF NOT EXISTS best_tuning_results (
    task_id      BIGINT    NOT NULL,
    arch         TEXT      NOT NULL,
    tuning_level TEXT      NOT NULL CHECK (tuning_level IN ('kernel', 'op')),
    task_config  JSONB     NOT NULL,
    iface_name   TEXT      NOT NULL,
    impl_index   INT       NOT NULL,
    median_time  FLOAT     NOT NULL,
    impl_desc    JSONB,
    computed_at  TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (task_id, iface_name)
);

CREATE INDEX IF NOT EXISTS idx_best_tuning_results_lookup
    ON best_tuning_results (arch, tuning_level, iface_name, task_id);

-- Extra unit tests associated with a task, populated by reset_broken_to_pending
-- when re-queuing entries that failed pytest correctness checks.
-- Rows accumulate across passes and are never deleted by reset_to_pending.
CREATE TABLE IF NOT EXISTS task_extra_uts (
    id          BIGSERIAL PRIMARY KEY,
    task_id     BIGINT  NOT NULL,
    im_text     TEXT    NOT NULL CHECK (im_text NOT LIKE '% %'),
    active      BOOLEAN NOT NULL DEFAULT TRUE,
    created_at  TIMESTAMP DEFAULT NOW(),
    UNIQUE (task_id, im_text)
);

CREATE INDEX IF NOT EXISTS idx_task_extra_uts_task_id
    ON task_extra_uts (task_id);
