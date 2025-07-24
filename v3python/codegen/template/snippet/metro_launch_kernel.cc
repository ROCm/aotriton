  if (condition[[nth_kernel]]) {
    err = bcontext[[nth_kernel]].launch(stream);
    if (err != hipSuccess)
        return err;
  }
