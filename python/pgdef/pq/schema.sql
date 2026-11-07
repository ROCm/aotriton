-- Copyright © 2026 Advanced Micro Devices, Inc.
-- SPDX-License-Identifier: MIT
--
-- pgdef: the engine's PostgreSQL schema.
--
-- Plain, idempotent SQL (CREATE ... IF NOT EXISTS / CREATE OR REPLACE): no psql
-- meta-commands, no PL/pgSQL, so `.def/bin/initdb` can feed it to psql and a
-- test can feed it to psycopg. Applied FIRST; every application's own
-- python/<app>/pq/schema.sql references these tables and is applied after.
--
-- Everything lives in schema `pgdef`; nothing goes in `public`.
--
-- Design rule: columns are for routing and scheduling, jsonb is for the rest.
--   payload    the application's immutable input
--   appcookie  the application's mutable progress (NULL until first written)
--   logistics  the engine's maintenance state; only pgdef/pq/logistics.py
--              writes it. Times inside are epoch seconds (float8), because
--              text::timestamptz is not IMMUTABLE and could not be indexed.
--
-- Partitions (pgdef.task_<pool>, pgdef.report_<pool>) are created on demand by
-- dispatch, which calls pgdef.ensure_pool(pool) (defined at the end of this
-- file) before inserting into a pool. There is no DEFAULT partition.

CREATE SCHEMA IF NOT EXISTS pgdef;

CREATE TABLE IF NOT EXISTS pgdef.task (
    id          BIGSERIAL,
    pool        TEXT NOT NULL,     -- where it runs: tune uses the arch (gfx950)
    subpool     TEXT,              -- NULL = any machine of the pool; else e.g. 'MI355'
    cat_level0  TEXT NOT NULL,     -- category level 0: the application ('tune')
    cat_level1  TEXT,              -- category level 1: NULL = generic; tune: 'kernel' | 'op'
    payload     JSONB NOT NULL,    -- application input; immutable after dispatch
    appcookie   JSONB,             -- application progress state; NULL until written
    priority    INT NOT NULL DEFAULT 5,
    status      TEXT NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending', 'running', 'done', 'failed', 'cancelled')),
    logistics   JSONB NOT NULL DEFAULT '{"v": 1}',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- `id` alone is the identity. pool is in the key only because PostgreSQL
    -- requires the partition key in a partitioned table's primary key, and a
    -- foreign key must reference exactly such a key.
    PRIMARY KEY (id, pool)
) PARTITION BY LIST (pool);

-- Indexes on the parent are created on every partition, present and future.
CREATE INDEX IF NOT EXISTS task_claim_idx
    ON pgdef.task (cat_level0, priority DESC, id) WHERE status = 'pending';

CREATE INDEX IF NOT EXISTS task_lease_idx
    ON pgdef.task (((logistics->'lease'->>'until')::float8)) WHERE status = 'running';

CREATE INDEX IF NOT EXISTS task_created_idx
    ON pgdef.task (created_at DESC);

-- Raw per-item results: one row per tuned impl, measured backend, ... The
-- application defines the vocabulary of `rkey` and `status`.
CREATE TABLE IF NOT EXISTS pgdef.report (
    id          BIGSERIAL,
    task_id     BIGINT NOT NULL,
    pool        TEXT NOT NULL,
    cat_level0  TEXT NOT NULL,     -- copied from the task, for per-application filtering
    attempt     INT NOT NULL,      -- informational; the task's logistics.attempt
    rkey        JSONB NOT NULL,    -- which item: tune {"cat_level1","iface","impl"}
    status      TEXT NOT NULL,     -- OK / NotOK / crash / ERROR ...
    payload     JSONB,
    error       JSONB,
    worker      TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (id, pool),
    FOREIGN KEY (task_id, pool) REFERENCES pgdef.task (id, pool) ON DELETE CASCADE
) PARTITION BY LIST (pool);

CREATE INDEX IF NOT EXISTS report_task_idx
    ON pgdef.report (task_id, pool);

CREATE INDEX IF NOT EXISTS report_category_idx
    ON pgdef.report (cat_level0, status);

-- One row per PG reader, upserted by the reader itself on claim, lease
-- renewal and idle poll.
CREATE TABLE IF NOT EXISTS pgdef.worker (
    worker_id    TEXT PRIMARY KEY,   -- '<host>/<reader-n>'
    host         TEXT NOT NULL,
    pool         TEXT NOT NULL,
    subpool      TEXT,
    cat_level0   TEXT NOT NULL,
    cat_level1   TEXT,
    last_seen    TIMESTAMPTZ NOT NULL DEFAULT now(),
    current_task BIGINT
);

-- ---------------------------------------------------------------------------
-- Views. They read only pgdef tables, so the webui can show progress for any
-- application without knowing it.
-- ---------------------------------------------------------------------------

-- logistics projected into named columns, for operators.
CREATE OR REPLACE VIEW pgdef.task_status AS
SELECT
    id, pool, subpool, cat_level0, cat_level1, priority, status, created_at,
    COALESCE((logistics->>'attempt')::int, 0)          AS attempt,
    (logistics->'retry'->>'max_attempts')::int         AS max_attempts,
    logistics->'lease'->>'owner'                       AS lease_owner,
    to_timestamp((logistics->'lease'->>'until')::float8) AS lease_until,
    to_timestamp((logistics->>'started')::float8)      AS started_at,
    to_timestamp((logistics->>'finished')::float8)     AS finished_at,
    jsonb_array_length(COALESCE(logistics->'errors', '[]'::jsonb)) AS error_count,
    logistics->'errors'->(-1)                          AS last_error
FROM pgdef.task;

CREATE OR REPLACE VIEW pgdef.queue_progress AS
SELECT
    pool, subpool, cat_level0, cat_level1,
    COUNT(*) FILTER (WHERE status = 'pending')   AS pending,
    COUNT(*) FILTER (WHERE status = 'running')   AS running,
    COUNT(*) FILTER (WHERE status = 'done')      AS done,
    COUNT(*) FILTER (WHERE status = 'failed')    AS failed,
    COUNT(*) FILTER (WHERE status = 'cancelled') AS cancelled,
    COUNT(*)                                     AS total,
    ROUND(100.0 * COUNT(*) FILTER (WHERE status = 'done') / NULLIF(COUNT(*), 0), 2) AS pct_complete
FROM pgdef.task
GROUP BY pool, subpool, cat_level0, cat_level1;

CREATE OR REPLACE VIEW pgdef.worker_health AS
SELECT
    worker_id, host, pool, subpool, cat_level0, cat_level1, current_task, last_seen,
    EXTRACT(EPOCH FROM (now() - last_seen)) AS seconds_since_seen,
    CASE
        WHEN EXTRACT(EPOCH FROM (now() - last_seen)) < 60  THEN 'healthy'
        WHEN EXTRACT(EPOCH FROM (now() - last_seen)) < 300 THEN 'stale'
        ELSE 'dead'
    END AS health_status
FROM pgdef.worker;

-- Running tasks whose lease has lapsed and that no reader has reaped yet.
CREATE OR REPLACE VIEW pgdef.expired_leases AS
SELECT
    id, pool, subpool, cat_level0, cat_level1,
    logistics->'lease'->>'owner' AS lease_owner,
    extract(epoch from now()) - (logistics->'lease'->>'until')::float8 AS seconds_expired
FROM pgdef.task
WHERE status = 'running'
  AND (logistics->'lease'->>'until')::float8 < extract(epoch from now());

-- Duration statistics of finished tasks, per application.
CREATE OR REPLACE VIEW pgdef.task_timing_stats AS
SELECT
    pool, cat_level0, cat_level1,
    COUNT(*) AS done_tasks,
    PERCENTILE_CONT(0.5)  WITHIN GROUP (ORDER BY d) AS median_duration_sec,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY d) AS p95_duration_sec,
    AVG(d)                AS avg_duration_sec
FROM (
    SELECT pool, cat_level0, cat_level1,
           (logistics->>'finished')::float8 - (logistics->>'started')::float8 AS d
    FROM pgdef.task
    WHERE status = 'done'
      AND (logistics->>'finished') IS NOT NULL AND (logistics->>'started') IS NOT NULL
) t
GROUP BY pool, cat_level0, cat_level1;

CREATE OR REPLACE VIEW pgdef.completion_eta AS
WITH stats AS (
    SELECT
        pool, cat_level0, cat_level1,
        COUNT(*) FILTER (WHERE status = 'pending') AS remaining,
        COUNT(*) FILTER (WHERE status = 'running') AS active_workers,
        AVG((logistics->>'finished')::float8 - (logistics->>'started')::float8)
            FILTER (WHERE status = 'done'
                    AND (logistics->>'finished')::float8 > extract(epoch from now()) - 3600)
            AS avg_task_duration_sec
    FROM pgdef.task
    GROUP BY pool, cat_level0, cat_level1
)
SELECT
    pool, cat_level0, cat_level1, remaining, active_workers, avg_task_duration_sec,
    CASE
        WHEN active_workers > 0 AND avg_task_duration_sec IS NOT NULL THEN
            ROUND((remaining * avg_task_duration_sec / active_workers / 3600)::numeric, 2)
        ELSE NULL
    END AS eta_hours
FROM stats
WHERE remaining > 0;

-- pgdef.ensure_pool(pool): create the pool's leaves of pgdef.task and
-- pgdef.report, if absent. Dispatch calls it with the pool as a bound
-- argument (DDL takes no parameters, so the leaf name is built here): the name
-- is validated and quoted with %I. A leaf that exists is skipped before any
-- DDL, so the parent is not locked; a concurrent creation is tolerated.
CREATE OR REPLACE FUNCTION pgdef.ensure_pool(pool text) RETURNS void
LANGUAGE plpgsql AS $$
DECLARE
    parent text;
BEGIN
    IF pool IS NULL OR pool !~ '^[a-z0-9_]+$' THEN
        RAISE EXCEPTION 'invalid pool %: expected ^[a-z0-9_]+$', quote_literal(pool);
    END IF;
    FOREACH parent IN ARRAY ARRAY['task', 'report'] LOOP
        CONTINUE WHEN to_regclass(format('pgdef.%I', parent || '_' || pool)) IS NOT NULL;
        BEGIN
            EXECUTE format('CREATE TABLE IF NOT EXISTS pgdef.%I PARTITION OF pgdef.%I FOR VALUES IN (%L)',
                           parent || '_' || pool, parent, pool);
        EXCEPTION WHEN duplicate_table OR unique_violation THEN
            NULL;       -- another dispatcher created it first
        END;
    END LOOP;
END
$$;
