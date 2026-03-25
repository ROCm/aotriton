  if (launch_condition[[nth_kernel]]) {
    err = bcontext[[nth_kernel]].launch(stream);
    if (err != hipSuccess)
      return err;
  }
  if (launch_condition[[nth_kernel]]_else) {
    err = bcontext[[nth_kernel]]_else.launch(stream);
    if (err != hipSuccess)
      return err;
  }
