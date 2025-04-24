// Launch Kernel [[shim_kernel_name]]
[[shim_ns]]::[[param_class_name]] individual_params_[[call_index]] = {
  .params = &params,
};
[[shim_ns]]::[[context_class_name]] individual_context_[[call_index]];
// FIXME: Replace with Individual Kernels
individual_context_[[call_index]].grid_calculator = context.grid_calculators[ [[shim_kernel_enum]] ];
err = context.lookup_optimal(params, gpu);
if (err != hipSuccess)
  return err;
err = context.launch(params, stream);
if (err != hipSuccess)
  return err;
