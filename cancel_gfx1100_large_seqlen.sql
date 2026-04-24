-- Cancel gfx1100 tasks with seqlen_q > 2048 or seqlen_k > 2048
-- seqlen_q/k live at task_config->'entry'->>'seqlen_q/k'
-- "cancelled" means the task was deliberately skipped and kept for archive purposes.
-- It is NOT an error state. Workers never claim cancelled tasks and reset/retry
-- workflows must not touch them. The error column is reused here to store the
-- human-readable cancellation reason.

BEGIN;

UPDATE task_queue
SET status = 'cancelled',
    completed_at = NOW(),
    error = 'cancelled: seqlen_q or seqlen_k exceeds 2048 (not supported on gfx1100)'
WHERE arch = 'gfx1100'
  AND (
      (task_config->'entry'->>'seqlen_q')::int > 2048
      OR (task_config->'entry'->>'seqlen_k')::int > 2048
  );

COMMIT;
