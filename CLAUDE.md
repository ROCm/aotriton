# Project Instructions for Claude Code

## PostgreSQL Connection Configuration

**CRITICAL: NEVER ADD ANY DATABASE NAME TO THE CONFIGURATION OF CONNECTION TO PGSQL. WE NEVER USE IT AND YOU MUST USE THE DEFAULT.**

When creating or modifying PostgreSQL connection parameters:
- DO NOT include `dbname` field in connection parameter dictionaries
- DO NOT add database name to psycopg.connect() calls
- Let psycopg use the default database (same as username)
- Connection params should only contain: `host`, `port`, `user`, `password`

## Database Access Pattern (pq Package)

The `v3python/tune/pq/` package is the **centralized database access library** for all database operations. This ensures CRUD operations are consistent with the schema.

### Key Principles:

1. **pq handles database reads/writes, NOT connections**
   - All functions in `pq/` accept a database connection object (`conn`), not connection parameters
   - Connection management is the caller's responsibility
   - Connections are created once per worker/process and reused

2. **Use pq functions for all database operations**
   - `pq/queue.py` - TaskQueue class for task_queue table operations (fetch, mark_completed, mark_failed, etc.)
   - `pq/results.py` - Functions for tuning_results table (save_tuning_result, get_task_results)
   - DO NOT write raw SQL in other packages (e.g., localq) - use pq functions instead

3. **Pattern Example:**

```python
# GOOD - Persistent connection per worker, use pq functions
import psycopg
from v3python.tune.pq.queue import TaskQueue
from v3python.tune.pq.results import save_tuning_result

class MyWorker:
    def __init__(self, conn_params):
        # Create connection once per worker process
        self.db_conn = psycopg.connect(**conn_params, autocommit=True)
    
    def run(self):
        # Reuse connection for all operations
        task_queue = TaskQueue(self.db_conn)
        
        while self.running:
            tasks = task_queue.fetch_tasks('gfx942', batch_size=10)
            for task in tasks:
                # ... process task ...
                save_tuning_result(task.id, report, self.db_conn)
                task_queue.mark_completed(task.id, arch='gfx942')
    
    def shutdown(self):
        # Close connection on worker shutdown
        if self.db_conn:
            self.db_conn.close()
```

```python
# BAD - Creates new connection per operation
from v3python.tune.pq.queue import TaskQueue

# Don't do this - creates connection internally
task_queue = TaskQueue(conn_params)  # WRONG
```

```python
# BAD - Raw SQL in application code
with db_conn.cursor() as cur:
    cur.execute("""
        UPDATE task_queue
        SET status = 'completed'
        WHERE id = %s
    """, (task_id,))
# Should use: task_queue.mark_completed(task_id, arch)
```

### When to Add Functions to pq:

If you need to perform a database operation:
1. Check if a function exists in `pq/queue.py` or `pq/results.py`
2. If not, add a new function to the appropriate pq module
3. DO NOT write raw SQL in other packages

This keeps all database operations centralized and schema-consistent.
