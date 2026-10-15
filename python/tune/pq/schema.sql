-- Copyright © 2026 Advanced Micro Devices, Inc.
-- SPDX-License-Identifier: MIT
--
-- Application "tune": tables, views and indexes over pgdef.
--
-- Applied by .tune/bin/initdb AFTER python/pgdef/pq/schema.sql (it references
-- the engine's tables). Plain, idempotent SQL; everything lives in schema
-- `tune`, nothing in `public`.
--
-- Storage mapping: tune stores its application name as cat_level0 = 'tune' and
-- its tuning level ('kernel'|'op') as cat_level1 in pgdef (pgdef.task,
-- pgdef.report.rkey) and in tune's own tables.
--
-- The raw per-impl results stay in the generic pgdef.report, one row per tuned
-- impl, with rkey = {"cat_level1": "kernel"|"op", "iface": ..., "impl": N},
-- status = OK | NotOK | crash | ERROR, payload = the benchmark data. tune adds
-- a typed view over them and the derived tables below.
--
-- The tuning level (column `cat_level1`): 'kernel' tunes hsaco variants, 'op'
-- tunes backends.
-- `iface_name` collides across levels (attn_fwd exists at both), so every
-- query over these tables filters on cat_level1.
--
-- Tables derived from reports reference pgdef.report with ON DELETE CASCADE, so
-- a generic reset (which purges reports) needs no tune code. Tables that are
-- rebuilt wholesale (best_tuning_results) need nothing.

CREATE SCHEMA IF NOT EXISTS tune;

-- Typed view over tune's rows of pgdef.report. iface_name/impl_index are
-- ImplSelector's pair (one column pair for every level).
CREATE OR REPLACE VIEW tune.tuning_results AS
SELECT
    r.id,
    r.task_id,
    r.pool                    AS arch,
    r.rkey->>'cat_level1'     AS cat_level1,
    r.rkey->>'iface'          AS iface_name,
    (r.rkey->>'impl')::int    AS impl_index,
    r.status                  AS result,      -- OK/NotOK/crash/ERROR
    r.payload                 AS result_data,
    r.error,
    r.worker,                                 -- 'gpu-<n>' that ran the benchmark
    r.attempt,
    r.created_at
FROM pgdef.report r
WHERE r.cat_level0 = 'tune';

-- Expression indexes on the engine table, partial on this application's rows.
CREATE INDEX IF NOT EXISTS report_tune_task_idx
    ON pgdef.report (task_id, (rkey->>'iface'), ((rkey->>'impl')::int))
    WHERE cat_level0 = 'tune';

CREATE INDEX IF NOT EXISTS report_tune_iface_idx
    ON pgdef.report ((rkey->>'cat_level1'), (rkey->>'iface'), status)
    WHERE cat_level0 = 'tune';

-- best_tuning_results: plain table populated by compute_best_results.py. For
-- each (task_id, iface_name): the fastest impl_index meeting the accuracy
-- threshold relative to most_accurate_tuning_results.
CREATE TABLE IF NOT EXISTS tune.best_tuning_results (
    task_id      BIGINT      NOT NULL,
    arch         TEXT        NOT NULL,
    cat_level1   TEXT        NOT NULL CHECK (cat_level1 IN ('kernel', 'op')),
    payload      JSONB       NOT NULL,      -- the task's payload (module, entry, ...)
    iface_name   TEXT        NOT NULL,
    impl_index   INT         NOT NULL,
    median_time  FLOAT       NOT NULL,
    impl_desc    JSONB,
    computed_at  TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (task_id, iface_name)
);

CREATE INDEX IF NOT EXISTS idx_best_tuning_results_lookup
    ON tune.best_tuning_results (arch, cat_level1, iface_name, task_id);

-- most_accurate_tuning_results: plain table, populated by
-- populate_most_accurate.py (full) or incrementally by task_id. One row per
-- (task, iface, test case, tensor), aggregated over that task's impl reports.
--
-- report_id anchors the row to ONE report of its task (any of them: a reset
-- purges all of a task's reports together, so which one is immaterial). The
-- foreign key is what makes a generic reset clear this table with no tune code.
CREATE TABLE IF NOT EXISTS tune.most_accurate_tuning_results (
    report_id            BIGINT  NOT NULL,
    task_id              BIGINT  NOT NULL,
    arch                 TEXT    NOT NULL,
    cat_level1           TEXT    NOT NULL CHECK (cat_level1 IN ('kernel', 'op')),
    payload              JSONB   NOT NULL,
    iface_name           TEXT    NOT NULL,
    test_case            TEXT    NOT NULL,
    tensor_name          TEXT    NOT NULL,
    target_fudge_factor  FLOAT,
    absolute_error       FLOAT,
    FOREIGN KEY (report_id, arch) REFERENCES pgdef.report (id, pool) ON DELETE CASCADE
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_most_accurate_tuning_results_lookup
    ON tune.most_accurate_tuning_results (task_id, iface_name, test_case, tensor_name);

-- Extra unit tests associated with a task, populated by reset_broken_to_pending
-- when re-queuing entries that failed pytest correctness checks. Rows
-- accumulate across passes and are never deleted by a reset (a reset updates
-- the task row; only deleting the task cascades here).
CREATE TABLE IF NOT EXISTS tune.extra_uts (
    id          BIGSERIAL PRIMARY KEY,
    task_id     BIGINT  NOT NULL,
    pool        TEXT    NOT NULL,
    im_text     TEXT    NOT NULL CHECK (im_text NOT LIKE '% %'),
    active      BOOLEAN NOT NULL DEFAULT TRUE,
    created_at  TIMESTAMPTZ DEFAULT now(),
    UNIQUE (task_id, im_text),
    FOREIGN KEY (task_id, pool) REFERENCES pgdef.task (id, pool) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_extra_uts_task_id
    ON tune.extra_uts (task_id);
