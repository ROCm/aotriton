{
    const auto& opts = context.call_options
#if [[returns]]
    context.params->[[target]] = (*opts.[[hook_function]])(opts.[[cookie]]);
#else
    (*opts.[[hook_function]])(opts.[[cookie]]);
#endif
}
