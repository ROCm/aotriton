# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Shared AST function-parameter introspection.

A kernel description needs to read a kernel file's parameter list without
importing or executing it -- the file may import a DSL the reader deliberately
does not have. This module holds the AST mechanics: one place that walks a
kernel file's syntax tree, and one place that turns a `FunctionDef`'s `args`
into a signature-order name list.

Callers differ only in the predicate that identifies the definition and in
whether nested scopes are searched. A flyc kernel is located by NAME inside a
vendored file, and may sit in a nested scope, so it walks; a Triton kernel is
located by name at module scope, so it does not.

This module has no dependency on `decorators/` or `ir/` (and neither of those
imports it back into the other) precisely so it can sit underneath both without
inverting the existing `ir` <- `decorators` layering.
"""

import ast


class AstParamError(Exception):
    """A source file has no function definition matching the predicate, or the
    match uses `*args`/`**kwargs`, which cannot be introspected into a fixed
    ARGUMENTS order."""


def find_functions(tree, predicate, *, walk=False):
    """`FunctionDef`/`AsyncFunctionDef` nodes in `tree` (an `ast.parse` result)
    for which `predicate(node)` is truthy, in source order.

    `walk=False` (the default) looks at top-level definitions only.
    `walk=True` visits every nested scope too (`ast.walk`) -- needed when the
    match is not guaranteed to sit at the top level."""
    nodes = ast.walk(tree) if walk else tree.body
    return [n for n in nodes
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and predicate(n)]


def collect_params(fn, *, what):
    """The plain parameter names of `fn` (`posonlyargs + args + kwonlyargs`,
    in signature order) -- no import, no execution. Raises `AstParamError` if
    `fn` uses `*args`/`**kwargs`: ATI cannot introspect those into a fixed
    ARGUMENTS order. `what` names the caller's context for the error
    message."""
    a = fn.args
    if a.vararg is not None or a.kwarg is not None:
        raise AstParamError(
            f"{what} uses *args/**kwargs, which ATI cannot introspect into a "
            f"fixed ARGUMENTS order")
    return [p.arg for p in (a.posonlyargs + a.args + a.kwonlyargs)]


def find_one_function(tree, predicate, *, walk=False, what):
    """`find_functions(tree, predicate, walk=walk)`, requiring the match to be
    unique. Raises `AstParamError` naming `what` and how many matches were
    found (0 or more than 1) -- used by a caller that locates its def by a
    predicate with no independent way to disambiguate multiple hits."""
    matches = find_functions(tree, predicate, walk=walk)
    if len(matches) != 1:
        raise AstParamError(
            f"{what}: expected exactly one matching function definition, "
            f"found {len(matches)}")
    return matches[0]

