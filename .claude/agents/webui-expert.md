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

The button automatically appears in Command Output if the endpoint path starts with one of the watched prefixes in `command_widget.html`: `/api/workers/`, `/api/servers/`, `/api/builds/`, `/api/deploy/`, `/api/testing/`.

**CRITICAL: use a bare `<button hx-post=...>`, never `<form hx-post=...>`, for action-triggering buttons.** When wrapped in a `<form>`, `event.detail.pathInfo.requestPath` is undefined in the `htmx:afterRequest` listener, so `isActionTrigger` never matches and the Command Output panel is never created. If the button needs to submit input values, use `hx-include="#container-id"` pointing to a plain `<div>` wrapping the inputs — do NOT use a `<form>`.

```html
<!-- GOOD: bare button + hx-include -->
<div id="my-inputs-{{ hostname }}">
    <input type="number" name="pass_num" value="0">
    <select name="backend">...</select>
</div>
<button hx-post="/api/testing/{{ hostname }}/run-test"
        hx-swap="none"
        hx-include="#my-inputs-{{ hostname }}">Run Test</button>

<!-- BAD: form wrapper breaks Command Output panel creation -->
<form hx-post="/api/testing/{{ hostname }}/run-test" hx-swap="none">
    <input name="pass_num">
    <button type="submit">Run Test</button>
</form>
```

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

### hx-indicator must NOT be used with hostname-derived IDs

Worker hostnames can be IP addresses (e.g. `10.216.51.98`). HTMX's `hx-indicator` uses `document.querySelectorAll`, which treats dots as CSS class selectors — so `#spinner-10.216.51.98` throws `DOMException: not a valid selector`.

**Never use `hx-indicator` or `hx-include` on per-host buttons**, and never put a hostname-derived string in an ID that is referenced by a CSS selector. Just omit `hx-indicator`. For `hx-include`, use `hx-vals='js:{...}'` with `getElementById` instead — `getElementById` does not use CSS selector parsing and is safe with dots.

`getElementById` in plain JS is fine with dots since it does not use CSS selector parsing.

```html
<!-- BAD: crashes with IP hostnames (dots = CSS class selectors) -->
<button hx-post="/api/deploy/{{ worker.hostname }}"
        hx-indicator="#spinner-{{ worker.hostname }}"
        hx-include="#inputs-{{ worker.hostname }}">Deploy</button>

<!-- GOOD: no hx-indicator, use hx-vals+js for input values -->
<button hx-post="/api/testing/{{ worker.hostname }}/run-test"
        hx-swap="none"
        hx-vals='js:{pass_num: document.getElementById("pass-{{ worker.hostname }}").value,
                     backend: document.getElementById("backend-{{ worker.hostname }}").value}'>
    Run Test
</button>
```

### Synchronous reads must NOT use the action tracker

Use `subprocess.run(capture_output=True)` for synchronous SSH reads (e.g. reading a remote file). Do NOT use `run_command()` / the action tracker for these — that creates a Command Output panel for a trivial read, which is confusing and noisy.

```python
# GOOD: synchronous read
result = subprocess.run(['ssh', hostname, 'cat /remote/file'], capture_output=True, text=True)
return {'status': 'ok', 'content': result.stdout}

# BAD: pollutes Command Output with a read operation
return run_command(['ssh', hostname, 'cat /remote/file'], ...)
```

### Role-based host filtering (workers.db)

Workers have roles via the `worker_roles(hostname, role_name)` junction table. Role names: `Tuner`, `Builder`, `Tester`.

- In templates: show controls only for hosts with the relevant role using `{% if worker.is_tuner %}...{% else %}<td></td><td></td>{% endif %}`. Use strikethrough on the hostname (`style="text-decoration: line-through;"`) when a host lacks the role — do not change background color.
- Bulk actions iterate only role-filtered hosts via `get_tuner_hostnames()` / `get_tester_hostnames()` in tasks.py, never all workers.
- Role toggle is a plain `fetch()` POST (not HTMX) followed by `location.reload()` so the row re-renders correctly.

### Command Output widget: always use textarea + JS polling, never HTMX swap

The Command Output widget (`command_widget.html`) deliberately avoids HTMX swaps for output updates. Every output panel is a `<textarea readonly>` whose content is appended by plain `fetch()` polling in JS.

**Why:** HTMX `hx-swap` replaces DOM nodes. If the command list were re-rendered via HTMX swap, every refresh would destroy all existing textarea values, reset scroll positions, and tear down any in-progress poll loops. This is unacceptable when multiple commands are running simultaneously.

**The correct pattern:**
1. `#command-list` uses `hx-swap="none"` — HTMX fires the request and calls `htmx:afterRequest`, but never touches the DOM itself.
2. The `htmx:afterRequest` listener reads `event.detail.xhr.response` (raw JSON) and **only creates panels for action IDs not already in the DOM** (`getElementById('panel-' + id)`). Existing panels are left untouched.
3. Each new panel contains a `<textarea id="output-{id}" data-offset="0">`. JS `pollOutput()` calls `GET /api/actions/{id}/output?offset=N`, appends the returned text to `textarea.value`, and updates `data-offset` for incremental delivery.
4. Status updates (`GET /api/actions/{id}/status`) are also plain `fetch()` — they update the status badge and kill-button visibility by direct DOM manipulation, not by re-rendering.

```html
<!-- GOOD: textarea updated by JS, not HTMX -->
<textarea readonly id="output-${actionId}" data-offset="0"></textarea>
<script>
function pollOutput(actionId) {
    const ta = document.getElementById('output-' + actionId);
    fetch(`/api/actions/${actionId}/output?offset=${ta.dataset.offset}`)
        .then(r => { ta.dataset.offset = r.headers.get('X-Output-Offset'); return r.text(); })
        .then(text => { ta.value += text; /* auto-scroll */ });
}
</script>

<!-- BAD: re-rendering via hx-swap destroys running panel state -->
<div hx-get="/api/actions" hx-trigger="every 1s" hx-swap="innerHTML">...</div>
```

**Auto-minimize:** panels with `returncode === 0` call `toggleMinimize()` on completion, which sets `height: 0` on the textarea via a CSS class. This keeps the UI clean without removing the panel.

### Do not add custom styles to `<pre>` tags

Never add `style="..."` attributes or inline CSS to `<pre>` elements. Leave `<pre>` tags unstyled; rely on the stylesheet for any formatting.

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

## Notes for `/perf` (visperf)

- The arch dropdown is populated by `v3python.tune.pq.visperf.get_available_archs`,
  which queries `kernel_table` only. This is intentional and complete:
  every AOTriton op is backed by one or more Triton kernels, so any arch
  the library is built for shows up in `best_tuning_results`. Do not
  "fix" this by unioning `best_optune_results` — the union is dead SQL.

## Out of scope

- `.tune/bin/` scripts and their internals — use the tuner-v3 agent
- `v3python/tune/` Python tuning stack — use the tuner-v3 agent
- `v3python/codegen/`, `v3python/rules/` — use the codegen agent
