    err = bcontext[[nth_kernel]].launch(stream);
    if (err != hipSuccess)
        return err;
