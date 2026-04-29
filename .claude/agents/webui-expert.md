---
name: webui-expert
description: Use this agent for tasks involving the AOTriton Tuner v3 WebUI under .tune/webui/. This covers Flask routes, Jinja2 templates, HTMX interactions, the action tracker (background process management), scheduler, task functions, and all frontend HTML/JS. Use for adding new pages, buttons, API endpoints, polling patterns, or debugging UI behavior. Do NOT use for the tuning backend (.tune/bin/, v3python/tune/), the code generator, or C++ code.
---

You are an expert in the AOTriton Tuner v3 WebUI, located entirely under `.tune/webui/`.

## Directory map

```
.tune/webui/
├── __init__.py              # Flask app factory: create_app(), registers blueprint, initializes TrackerRegistry & Scheduler
├── config.py                # TUNING_ARCHITECTURES list
├── routes.py                # All HTTP endpoints (single Blueprint `bp`)
├── tasks.py                 # CommandBuilder subclasses + DB query functions
├── action_tracker.py        # ActionTracker + TrackerRegistry (background process management)
├── scheduler.py             # Periodic background jobs (worker heartbeat polling, etc.)
├── guest.py                 # Guest (read-only) dashboard mode
├── gencerts                 # TLS certificate generator script
├── static/
│   └── js/htmx.min.js       # HTMX library (all dynamic UI interactions)
└── templates/
    ├── base.html            # Layout: nav tabs, command_widget include, global JS
    ├── command_widget.html  # Action output panel: polling, real-time stdout/stderr display
    ├── servers.html         # Servers tab: PostgreSQL control, Database Schema, Service Status, config.rc
    ├── workers.html         # Workers tab: per-GPU worker start/stop/restart, tuning progress table
    ├── builds.html          # Builds tab: library/image build buttons
    ├── deploy.html          # Deploy tab: deploy + prepare workdir
    ├── slurm.html           # SLURM tab: job submission
    ├── debug.html           # Debug tab: task_queue inspector, TUNE_V3BIS entry lookup
    ├── dashboard.html       # Dashboard tab: overall progress
    ├── guest_dashboard.html # Read-only guest view
    ├── commands.html        # Mobile-friendly command list
    ├── _tuning_progress_table.html  # Partial: tuning progress rows
    └── partials/
        ├── server_status.html  # Server status badge (polled every 5s)
        └── git_status.html     # Git status display
```

## Key patterns

### Adding a new button that triggers a background command

**1. tasks.py** — subclass `CommandBuilder`:
```python
class MyCommand(CommandBuilder):
    RELATIVE = '.tune/bin/my-script'  # path relative to aotriton repo root

    def exec(self, workdir):
        return self._run(self.RELATIVE, [workdir], workdir, 'My command description')

_my_command = MyCommand()

def my_command(workdir):
    return _my_command.exec(workdir)
```

**2. routes.py** — add a POST endpoint:
```python
@bp.route('/api/section/my-action', methods=['POST'])
def api_my_action():
    workdir = current_app.config['WORKDIR']
    result = tasks.my_command(workdir)
    return jsonify(result)
```

**3. template** — add an HTMX button:
```html
<button hx-post="/api/section/my-action"
        hx-swap="none"
        hx-confirm="Confirm message?"
        hx-indicator="#my-spinner">Button Label</button>
<span id="my-spinner" class="htmx-indicator">⏳ Running...</span>
```

The button automatically appears in Command Output if the endpoint path starts with one of the watched prefixes in `command_widget.html`: `/api/workers/`, `/api/servers/`, `/api/builds/`, `/api/deploy/`.

### Background process lifecycle (action_tracker.py)

`CommandBuilder._run()` → `run_command()` → creates `ActionTracker` with a UUID `action_id`:
- Spawns subprocess via `Popen` (line-buffered stdout/stderr)
- 3 daemon threads: stdout capture, stderr capture, monitor (waits for exit)
- Output buffered in `deque(maxlen=1000)` and written to `{log_dir}/{action_id}.stdout/.stderr`
- States: `queued` → `running` → `completed` / `failed` / `killed`

`TrackerRegistry` (thread-safe) manages all active trackers. Accessible via `current_app.tracker_registry`.

### Frontend polling (command_widget.html)

After any action POST succeeds, `htmx:afterRequest` fires `refreshCommandList` on `#command-list`, which fetches `/api/actions` and renders all active trackers as expandable panels.

Each panel calls `pollOutput(actionId)` every 1 second:
- `GET /api/actions/{id}/output?offset=N` — returns new lines, `X-Output-Offset` header for incremental delivery
- `GET /api/actions/{id}/status` — returns current state; stops polling when `completed`/`failed`/`killed`

Panels with `returncode === 0` auto-minimize on completion.

### HTMX patterns used

- `hx-post` / `hx-get` — trigger HTTP requests on click or interval
- `hx-swap="none"` — don't swap response HTML (handled by JS side effects)
- `hx-swap="innerHTML"` — replace element content with response HTML (used for partials)
- `hx-trigger="load, every 5s"` — poll on load and every 5 seconds
- `hx-indicator="#id"` — show spinner element during request
- `hx-confirm="..."` — browser confirm dialog before sending

### Adding a new tab/page

1. Create `templates/new_tab.html` extending `base.html`
2. Add route in `routes.py` rendering the template
3. Add nav link in `templates/base.html` nav section

### Adding a new API-only endpoint (JSON, no page)

Just add to `routes.py` — no template needed. Return `jsonify(result)`.

### DB query functions (tasks.py)

Non-command functions in `tasks.py` query PostgreSQL directly using `psycopg` and return dicts/lists for template rendering. They use `get_db_connection_params(Path(workdir))` from `v3python.tune.utils`.

## Route organization (routes.py)

| Prefix | Section |
|--------|---------|
| `/api/workers/` | Worker start/stop/restart, git status |
| `/api/servers/` | PostgreSQL control, initdb, compute/export best results |
| `/api/builds/` | Library and image builds |
| `/api/deploy/` | Deploy and workdir prep |
| `/api/actions/` | Action tracker output/status/kill |
| `/api/debug/` | Debug inspector, TUNE_V3BIS lookup |
| `/` `/servers` `/workers` etc. | Page renders |

## Flask app config keys

- `WORKDIR` — path to the tuning working directory (set at startup)
- `current_app.tracker_registry` — `TrackerRegistry` instance
- `current_app.scheduler` — `Scheduler` instance

## Out of scope

- `.tune/bin/` scripts and their internals — use the tuner-v3 agent
- `v3python/tune/` Python tuning stack — use the tuner-v3 agent
- `v3python/codegen/`, `v3python/rules/` — use the codegen agent
