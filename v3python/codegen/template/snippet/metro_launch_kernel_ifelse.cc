  if (condition[[nth_kernel]]) {
    err = bcontext[[nth_kernel]]_if.launch(stream);
    if (err != hipSuccess)
      return err;
  } else {
    err = bcontext[[nth_kernel]]_else.launch(stream);
    if (err != hipSuccess)
      return err;
  }
