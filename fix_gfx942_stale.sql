-- Reset stale gfx942 tasks back to pending
-- "Stale" = status 'running' for more than 2 hours
-- Run order: delete tuning_results first, then reset task_queue

BEGIN;

DELETE FROM tuning_results
WHERE task_id IN (
    SELECT id FROM task_queue
    WHERE arch = 'gfx942'
      AND status = 'running'
      AND EXTRACT(EPOCH FROM (NOW() - started_at)) > 7200
);

UPDATE task_queue
SET status = 'pending',
    worker_id = NULL,
    node_hostname = NULL,
    started_at = NULL,
    error = NULL
WHERE arch = 'gfx942'
  AND status = 'running'
  AND EXTRACT(EPOCH FROM (NOW() - started_at)) > 7200;

COMMIT;
