{
    context.selected_pp_args = &context::pp_direct_kernel_args_for_[[direct_kernel_args]];
    context.kernel_on_device = kernel_cluster.get([[kernel_obj_index]]);
    context.affine_kernel_function_name = R"xyzwwzyx([[mangled_name]])xyzwwzyx";
    context.package_path = R"xyzwwzyx([[package_path]])xyzwwzyx";
    context.arch_name = R"xyzwwzyx([[arch]])xyzwwzyx";
    return hipSuccess;
}
