# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Print an application's handler graph from the handlers' EMITS declarations.

    python -m aotriton.pgdef.localq.graph tune

The graph is implicit (a handler's return value is the next message), but every
edge is declared next to its handler and checked at runtime, so this output can
not drift from the code.

The command line tool is a developer convenience, and the one place that
imports an application by name: it loads `aotriton.<app>.handlers`, which by
convention exports `HANDLERS` (the handler classes, including the start
handler) and `START` (the start handler class). Nothing in the engine does that
at run time; `run_node_role` is handed these objects by the application.
"""

import argparse
import importlib

from .handlers import BROKER_CLASSES, HandlerSet


def render(handler_classes, start=None) -> str:
    """ASCII tree of the graph, from `start` (default: the first handler).

    A handler reached a second time is printed once more, marked `(see above)`,
    and not expanded, so joins and loops stay readable. Handlers never reached
    from `start` are listed at the end: dead code, or a start handler that
    was left out.
    """
    hs = HandlerSet(handler_classes)
    app_classes = [c for c in handler_classes]
    root = (start or app_classes[0]).CLASS
    lines: list[str] = []
    seen: set[str] = set()

    def label(cls_name: str) -> str:
        if cls_name in BROKER_CLASSES:
            return f'{cls_name}  [broker: releases the PG reader]'
        h = hs.by_class[cls_name]
        return f'{cls_name}  [{h.QUEUE}]'

    def walk(cls_name: str, prefix: str, last: bool, top: bool):
        connector = '' if top else ('`-- ' if last else '|-- ')
        again = cls_name in seen
        lines.append(f'{prefix}{connector}{label(cls_name)}' + ('  (see above)' if again else ''))
        if again or cls_name in BROKER_CLASSES:
            return
        seen.add(cls_name)
        targets = hs.by_class[cls_name].EMITS
        child_prefix = prefix if top else prefix + ('    ' if last else '|   ')
        for i, target in enumerate(targets):
            walk(target, child_prefix, i == len(targets) - 1, False)

    walk(root, '', True, True)
    # The engine failure path is implicit for every handler; show it once.
    unreached = [c.CLASS for c in app_classes if c.CLASS not in seen]
    if unreached:
        lines.append('')
        lines.append('not reachable from the start handler: ' + ', '.join(unreached))
    return '\n'.join(lines)


def load_application(app: str):
    """(handler classes, start class) of `aotriton.<app>.handlers`."""
    module = importlib.import_module(app if '.' in app else f'aotriton.{app}.handlers')
    return list(module.HANDLERS), getattr(module, 'START', None)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('app', help="application name ('tune'), or a dotted module path with HANDLERS")
    args = parser.parse_args(argv)
    handlers, start = load_application(args.app)
    print(render(handlers, start))


if __name__ == '__main__':
    main()
