  if (launch_condition[[nth_kernel]]_if) {
    err = bcontext[[nth_kernel]]_if.launch(stream);
    if (err != hipSuccess)
      return err;
  }
  if (launch_condition[[nth_kernel]]_else) {
    err = bcontext[[nth_kernel]]_else.launch(stream);
    if (err != hipSuccess)
      return err;
  }
