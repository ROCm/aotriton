# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Metro-kernel transpiler (executive plan Step 5.5; agent-plans/ati_rev1.md §5).

`@ati.metro_kernel` lets an operator's collaborating kernels be wired with
ordinary Python if/else:

    @ati.metro_kernel
    def metro_fwd(params):
        attn_fwd(params)                               # by-name wiring
        if params.encoded_softmax.data_ptr() != 0:     # 0/None -> nullptr
            debug_simulate_encoded_softmax(params, R=params.encoded_softmax)

We PARSE the function's AST and never execute it (so both if/else branches are
visible — no __bool__-on-proxy problem). The grammar is narrow: bare sub-kernel
calls, if/elif/else, and `ast.Compare` conditions of the form
`params.<NAME>[.data_ptr()] <op> <literal>`. Anything else is a build-time error.

Output is a MetroPlan (steps); the operator builder lowers it to the existing
MetroKernel / ConditionalKernel IR, so the C++ launcher codegen is untouched.
"""

import ast
import inspect

# Comparison operator -> C++ rendering.
_CMP = {
    ast.Eq: '==', ast.NotEq: '!=', ast.Gt: '>', ast.Lt: '<',
    ast.GtE: '>=', ast.LtE: '<=',
}


class MetroError(Exception):
    """A build-time error in a @ati.metro_kernel body (out-of-grammar construct
    or unsupported condition). Names the metro and the offending source."""


class Call:
    """A sub-kernel invocation: just the sub-kernel name. Argument wiring is NOT a
    metro concern — it lives on the sub-kernel's own @ati.tensor/@ati.scalar
    `wires_to=` decorator (rev0 §4.3), so a metro call is plain `kernel(params)`."""
    __slots__ = ('kernel',)

    def __init__(self, kernel):
        self.kernel = kernel

    def __repr__(self):
        return f'Call({self.kernel!r})'


class Cond:
    """An if/else step: a condition (if_parameter, if_expr) and the then/else
    sub-plans (each a list of steps)."""
    __slots__ = ('if_parameter', 'if_expr', 'then', 'orelse')

    def __init__(self, if_parameter, if_expr, then, orelse):
        self.if_parameter = if_parameter
        self.if_expr = if_expr
        self.then = then
        self.orelse = orelse

    def __repr__(self):
        return (f'Cond({self.if_parameter!r}, {self.if_expr!r}, '
                f'then={self.then}, orelse={self.orelse})')


class MetroPlan:
    """The transpiled metro: its name, the params variable, and an ordered list of
    steps (Call | Cond)."""
    __slots__ = ('name', 'params_name', 'steps')

    def __init__(self, name, params_name, steps):
        self.name = name
        self.params_name = params_name
        self.steps = steps

    def __repr__(self):
        return f'MetroPlan({self.name!r}, steps={self.steps})'


def _err(name, node, msg):
    return MetroError(f'metro {name!r}: {msg} (line {getattr(node, "lineno", "?")})')


def _param_attr(node, params_name, name):
    """If `node` is `<params>.<X>` return X, else None."""
    if (isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == params_name):
        return node.attr
    return None


def _literal(node):
    """The python value of a literal node (numbers, None, True/False)."""
    if isinstance(node, ast.Constant):
        return node.value
    raise ValueError('not a literal')


def _lower_condition(name, test, params_name):
    """An ast.Compare `params.<NAME>[.data_ptr()] <op> <literal>` ->
    (if_parameter, if_expr). Raises MetroError for anything outside the grammar."""
    if not isinstance(test, ast.Compare) or len(test.ops) != 1:
        raise _err(name, test, 'condition must be a single comparison '
                               'params.<NAME>[.data_ptr()] <op> <literal>')
    op = type(test.ops[0])
    if op not in _CMP:
        raise _err(name, test, f'unsupported comparison operator {op.__name__}')
    left, right = test.left, test.comparators[0]
    try:
        rhs = _literal(right)
    except ValueError:
        raise _err(name, test, 'right-hand side of the condition must be a literal')

    # left is either params.<NAME> or params.<NAME>.data_ptr()
    is_ptr = False
    if (isinstance(left, ast.Call) and isinstance(left.func, ast.Attribute)
            and left.func.attr == 'data_ptr' and not left.args):
        is_ptr = True
        left = left.func.value
    pname = _param_attr(left, params_name, None)
    if pname is None:
        raise _err(name, test, 'condition must read params.<NAME>'
                               '[.data_ptr()]')

    cmp = _CMP[op]
    if is_ptr:
        # pointer null-check: 0/None -> nullptr
        if rhs in (0, None):
            return pname, f'->data_ptr() {cmp} nullptr'
        raise _err(name, test, 'data_ptr() comparison must be against 0 or None')
    return pname, f'{cmp} {rhs}'


def _lower_call(name, node, params_name):
    """An expression statement `kernel(params)` -> Call. Keyword renames are no
    longer accepted here — argument wiring is declared on the sub-kernel's
    `wires_to=` decorator, not in the metro body (rev0 §4.3)."""
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
        raise _err(name, node, 'only sub-kernel calls and if/else are allowed')
    call = node.value
    if not isinstance(call.func, ast.Name):
        raise _err(name, call, 'sub-kernel must be a bare name')
    kernel = call.func.id
    # positional arg must be exactly `params`
    if len(call.args) != 1 or not (isinstance(call.args[0], ast.Name)
                                   and call.args[0].id == params_name):
        raise _err(name, call, f'sub-kernel call takes exactly the positional '
                               f'{params_name!r}')
    if call.keywords:
        raise _err(name, call, 'sub-kernel call takes no keyword arguments; '
                               'declare argument wiring with wires_to= on the '
                               "sub-kernel's @ati.tensor/@ati.scalar instead")
    return Call(kernel)


def _lower_body(name, body, params_name):
    steps = []
    for stmt in body:
        if isinstance(stmt, ast.If):
            if_parameter, if_expr = _lower_condition(name, stmt.test, params_name)
            then = _lower_body(name, stmt.body, params_name)
            orelse = _lower_body(name, stmt.orelse, params_name) if stmt.orelse else []
            steps.append(Cond(if_parameter, if_expr, then, orelse))
        elif isinstance(stmt, ast.Pass):
            continue
        else:
            steps.append(_lower_call(name, stmt, params_name))
    return steps


def transpile(fn) -> MetroPlan:
    """Parse a @ati.metro_kernel function into a MetroPlan. Never executes it."""
    src = inspect.getsource(fn)
    src = _dedent(src)
    mod = ast.parse(src)
    fdef = next((n for n in mod.body if isinstance(n, ast.FunctionDef)), None)
    if fdef is None:
        raise MetroError('metro: no function definition found')
    args = fdef.args
    if (len(args.args) != 1 or args.vararg or args.kwarg or args.kwonlyargs):
        raise MetroError(f'metro {fdef.name!r}: must take exactly one parameter '
                         f'(the params object)')
    params_name = args.args[0].arg
    steps = _lower_body(fdef.name, fdef.body, params_name)
    return MetroPlan(fdef.name, params_name, steps)


def _dedent(src):
    import textwrap
    return textwrap.dedent(src)


def metro_kernel(fn):
    """@ati.metro_kernel: transpile the function body (never executed) into a
    MetroPlan, attached as fn.__ati_metro__. Returns the function untouched so the
    operator builder can read the plan."""
    fn.__ati_metro__ = transpile(fn)
    return fn


def lower_plan(plan, kernel_map, metro_factory, conditional_factory):
    """Lower a MetroPlan to the existing MetroKernel/ConditionalKernel IR.

    kernel_map:          {sub-kernel name -> KernelDescription object}.
    metro_factory:       callable(steps:list) -> MetroKernel (the lowered backend
                         list). Argument wiring is NOT threaded here — it lives on
                         each sub-kernel's kdesc (wires_to=, rev0 §4.3).
    conditional_factory: the ConditionalKernel class/callable
                         (if_parameter, if_expr, if_kernel, else_kernel).

    Each Cond branch must be a single sub-kernel call (the C++ if/else launcher
    template supports one kernel per branch); a multi-step branch is an error.
    """
    def resolve(call):
        if call.kernel not in kernel_map:
            raise MetroError(
                f'metro {plan.name!r}: unknown sub-kernel {call.kernel!r}; '
                f'known: {sorted(kernel_map)}')
        return kernel_map[call.kernel]

    def one_call(name, branch, which):
        if len(branch) != 1 or not isinstance(branch[0], Call):
            raise MetroError(
                f'metro {plan.name!r}: the {which} branch of a condition must be a '
                f'single sub-kernel call')
        return branch[0]

    steps = []
    for step in plan.steps:
        if isinstance(step, Call):
            steps.append(resolve(step))
        else:  # Cond
            if_call = one_call(plan.name, step.then, 'if')
            else_kernel = None
            if step.orelse:
                else_kernel = resolve(one_call(plan.name, step.orelse, 'else'))
            steps.append(conditional_factory(step.if_parameter, step.if_expr,
                                             resolve(if_call), else_kernel))
    return metro_factory(steps)
