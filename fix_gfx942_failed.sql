-- Reset all failed gfx942 tasks back to pending for retry
-- Failed tasks have status='failed', completed_at set, error populated

BEGIN;

DELETE FROM tuning_results
WHERE task_id IN (
    SELECT id FROM task_queue
    WHERE arch = 'gfx942'
      AND status = 'failed'
);

UPDATE task_queue
SET status = 'pending',
    worker_id = NULL,
    node_hostname = NULL,
    started_at = NULL,
    completed_at = NULL,
    error = NULL
WHERE arch = 'gfx942'
  AND status = 'failed';

COMMIT;
