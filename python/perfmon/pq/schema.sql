-- Copyright © 2026 Advanced Micro Devices, Inc.
-- SPDX-License-Identifier: MIT
--
-- Application "perfmon": views over pgdef.
--
-- Applied by .tune/bin/initdb AFTER python/pgdef/pq/schema.sql (it references
-- the engine's tables). Plain, idempotent SQL; everything lives in schema
-- `perfmon`, nothing in `public`. perfmon has no table of its own: every
-- measurement is a row of pgdef.report.
--
-- Storage mapping: perfmon stores its application name as cat_level0 =
-- 'perfmon' and leaves cat_level1 NULL. The task's pool is the arch, its
-- subpool the variant (NULL: any machine of the arch, variant 'any'). The
-- task payload is {"module", "subject", "entry"} (plus "ifaces" for an LLM
-- case measured forward only). A report's rkey is {"iface", "backend"}, with
-- backend NULL for the unforced operator; its status is OK, NotOK, crash or
-- excluded; its payload holds the statistics under "stats" (microseconds),
-- the platform, the thermal snapshot and how the time was taken.

CREATE SCHEMA IF NOT EXISTS perfmon;

-- One row per measurement. The entry stays jsonb: its fields are the
-- family's, and this view serves every family.
CREATE OR REPLACE VIEW perfmon.measurement AS
SELECT
    r.id,
    r.task_id,
    r.pool                                    AS arch,
    COALESCE(t.subpool, 'any')                AS variant,
    t.payload->>'subject'                     AS subject,
    t.payload->>'module'                      AS module,
    r.rkey->>'iface'                          AS iface,
    r.rkey->>'backend'                        AS backend,    -- NULL: unforced
    t.payload->'entry'                        AS entry,
    r.status,                                                -- OK/NotOK/crash/excluded
    (r.payload->'stats'->>'median')::float8   AS median_us,
    (r.payload->'stats'->>'mean')::float8     AS mean_us,
    (r.payload->'stats'->>'stddev')::float8   AS stddev_us,
    (r.payload->'stats'->>'min')::float8      AS min_us,
    (r.payload->'stats'->>'p05')::float8      AS p05_us,
    (r.payload->'stats'->>'p95')::float8      AS p95_us,
    r.payload->>'timing_method'               AS timing_method,
    (r.payload->>'n')::int                    AS iterations,
    (r.payload->>'l2_flush')::int             AS l2_flush,
    r.payload->'platform'                     AS platform,
    r.payload->'thermal'                      AS thermal,
    r.payload->'harness'                      AS harness,
    r.error,
    r.worker,
    r.attempt,
    r.created_at
FROM pgdef.report r
JOIN pgdef.task t ON t.id = r.task_id AND t.pool = r.pool
WHERE r.cat_level0 = 'perfmon';

-- The newest measurement of each point. When a point was dispatched twice the
-- newest wins; older rows are never averaged in, which would launder a
-- known-bad measurement.
CREATE OR REPLACE VIEW perfmon.latest_measurement AS
SELECT DISTINCT ON (arch, variant, module, subject, entry, iface, backend) *
FROM perfmon.measurement
ORDER BY arch, variant, module, subject, entry, iface, backend, created_at DESC, id DESC;

-- Expression index on the engine table, partial on this application's rows.
CREATE INDEX IF NOT EXISTS report_perfmon_task_idx
    ON pgdef.report (task_id, (rkey->>'iface'), (rkey->>'backend'))
    WHERE cat_level0 = 'perfmon';
