Simple example scripts for the `testrun` REPL. Pipe one in, e.g.

    .tune/remote/testrun_direct flash --tuning_mode kernel < tt3_1.txt

An impl is selected with `[<tuning_level>.]<iface_name>=<impl_index>`.
Unprefixed names are kernel-level interfaces (`attn_fwd=3`,
`bwd_kernel_dk_dv=0`); operators take an `op.` prefix (`op.attn_fwd=1`).

`tt3_*` and `tt4_*` are kernel-level, `tt5_*` operator-level. Do not mix the
two in one session: a container carries exactly one pyaotriton build, and
kernel-level and operator-level tuning need different ones. Pass the matching
`--tuning_mode` to `testrun_direct`, which is what selects that build.

`probe <dir>` accepts optional `[arch]` and `[tuning_level filter]` tokens;
with neither, it reports every impl it can resolve plus a reason for each it
cannot.
