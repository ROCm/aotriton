# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from .functional import Functional
# from .ttype import (
#     typename_t,
#     ConditionalValue,
# )
# from .guesstype import (
#     create_tensor_type,
#     guess_tparam_type,
#     guess_vparam_type,
# )
from . import typed_choice
from .typed_choice import ConditionalChoice
from .conditional_value import (
    ConditionalConstexpr,
    ConditionalDeferredConstexpr,
    ConditionalDeferredElseTensor,
)
from .parameter import (
    TemplateParameter,
    PerformanceTemplateParameter,
)

# TODO: Move common items b/w KernelDescription and Operator here
class Tunable(object):
    pass

