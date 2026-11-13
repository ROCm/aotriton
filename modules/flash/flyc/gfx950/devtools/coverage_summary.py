#!/usr/bin/env python3
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Summarize a gfx950 flyc coverage sweep (`coverage_sweep.py`): status matrices, fastest builds, failures.

Prints, per (kernel, head_dim):

1. Status counts per candidate and pass.
2. Per candidate, a seqlen_q x seqlen_k matrix of the check pass aggregated
   over dtype, mask, feature and layout: `.` every record OK, otherwise the
   worst status's letter (F FAULT, W WRONG, B BUILD_ERROR, R REFUSED,
   N NOT_BUILDABLE, S SKIPPED, ? missing) -- with `--by_group`, one matrix per
   (dtype, mask, feature) instead.
3. The fastest correct candidate per cell: a candidate is correct at a cell when
   its check records for *both* layouts are OK. Per group, the winner's
   letter per cell plus its speedup over the policy's pick; with `--full`,
   every group's matrix, otherwise a count of wins per candidate and the cells
   where the policy pick is not the winner by more than `--noise`.
4. Every non-OK record grouped by (candidate, symptom), so a failure can be
   triaged as a bug (fix the kernel) or a limitation (make the policy not
   choose that build for those inputs).

    python coverage_summary.py ~/scratch/coverage/pilot-hd128 [--by_group] [--full]
"""

import argparse
import collections
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import coverage_space as space  # noqa: E402

LETTER = {
    "OK": ".",
    "WRONG": "W",
    "FAULT": "F",
    "BUILD_ERROR": "B",
    "REFUSED": "R",
    "NOT_BUILDABLE": "N",
    "SKIPPED": "S",
}
# Worst first: the matrix cell shows the worst status among its records.
SEVERITY = ("FAULT", "WRONG", "BUILD_ERROR", "REFUSED", "NOT_BUILDABLE", "SKIPPED", "OK")


def worst(statuses):
    for s in SEVERITY:
        if s in statuses:
            return s
    return None


def latest(recs):
    """One record per key, the last one written (a resumed run may rewrite none, but be safe)."""
    out = {}
    for r in sorted(recs, key=lambda r: r.get("ts", 0)):
        out[space.record_key(r)] = r
    return list(out.values())


def symptom(r):
    """A short, shape-independent description of why a record is not OK."""
    st = r["status"]
    err = r.get("error") or ""
    if st == "WRONG":
        parts = []
        for piece in err.split("; "):
            m = re.match(r"(\w+) (has \d+ NaN|err|not \+inf)", piece)
            if m:
                what = "NaN" if "NaN" in m.group(2) else ("dead-row LSE" if "inf" in m.group(2) else "err")
                parts.append(f"{m.group(1)} {what}" if what != "dead-row LSE" else "LSE dead rows not +inf")
        return f"WRONG ({severity(r)}): " + ", ".join(sorted(set(parts)))
    if st == "FAULT":
        tail = (r.get("stderr") or "") + " " + err
        for pat in ("Memory access fault", "hipErrorIllegalAddress", "illegal memory", "hang", "page not present",
                    "Segmentation fault", "Aborted"):
            if pat.lower() in tail.lower():
                return f"FAULT: {pat}"
        return "FAULT: " + err[:60]
    # Strip numbers so one refusal reason groups across shapes.
    msg = re.sub(r"\d+", "#", err.splitlines()[0] if err else "")
    return f"{st}: {msg[:140]}"


def severity(r):
    """How far past tolerance a WRONG record is: marginal (<= 2x), moderate (<= 10x), gross, or NaN.

    Recomputes each output's tolerance the way `coverage_worker` judged it.
    A marginal failure may be a tolerance question; a gross one is a bug or
    an unwritten output.
    """
    if any((r.get("nan") or {}).values()):
        return "NaN"
    errs, refm = r.get("errs") or {}, r.get("ref_max") or {}
    worst_ratio = 0.0
    for k, e in errs.items():
        if k == "delta" or e is None:
            continue
        if r["kernel"] == "fwd":
            tol = 1.5e-2 * max(1.0, refm.get("O", 1.0)) if k == "O" else 1e-2
        else:
            grads = [v for g, v in refm.items() if g != "delta"]
            tol = 2e-2 * max(refm.get(k, 0.0), 1e-3 * max(grads, default=0.0), 1e-6)
        worst_ratio = max(worst_ratio, e / tol)
    if "dead rows" in (r.get("error") or "") and worst_ratio <= 1:
        return "dead-row LSE"
    return "marginal" if worst_ratio <= 2 else "moderate" if worst_ratio <= 10 else "gross"


def fmt_cell_key(r):
    return f"{r['dtype']}/{r['mask']}/{r['feature']}/{r['layout']} {r['sq']}x{r['sk']}"


def matrix(seqlens, cell_fn, width=2):
    head = "   Sq\\Sk " + "".join(f"{s:>{width + 4}}" for s in seqlens)
    lines = [head]
    for sq in seqlens:
        lines.append(f"  {sq:>6} " + "".join(f"{cell_fn(sq, sk):>{width + 4}}" for sk in seqlens))
    return "\n".join(lines)


def abbrev(names):
    """Single-letter-ish abbreviations for candidate names, stable by sorted order."""
    out = {}
    for i, n in enumerate(sorted(names)):
        out[n] = "0123456789abcdefghijklmnopqrstuvwxyz"[i]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="\n".join(__doc__.splitlines()[1:]))
    p.add_argument("out_dir")
    p.add_argument("--by_group", action="store_true", help="status matrices per (dtype, mask, feature)")
    p.add_argument("--full", action="store_true", help="fastest-candidate matrix for every group")
    p.add_argument("--noise", type=float, default=0.02, help="speedups within this fraction are ties")
    p.add_argument("--max_cells", type=int, default=12, help="cells listed per failure symptom")
    args = p.parse_args()

    recs = latest(space.load_records(args.out_dir))
    if not recs:
        sys.exit(f"no records in {args.out_dir}")
    seqlens = sorted({r["sq"] for r in recs if r.get("sq") is not None})
    by_khd = collections.defaultdict(list)
    for r in recs:
        by_khd[(r["kernel"], r["hd"])].append(r)

    for (kernel, hd), rs in sorted(by_khd.items()):
        print(f"\n{'=' * 100}\n{kernel} head_dim {hd}\n{'=' * 100}")
        cands = sorted({r["candidate"] for r in rs if r["pass"] != "compile"})

        # 1. counts
        print("\n-- status counts (candidate x pass) --")
        cnt = collections.Counter((r["candidate"], r["pass"], r["status"]) for r in rs)
        names = sorted({(c, ps) for c, ps, _ in cnt})
        print(f"  {'candidate':<34} {'pass':<8} " + " ".join(f"{s:>13}" for s in space.STATUSES))
        for c, ps in names:
            print(f"  {c:<34} {ps:<8} " + " ".join(f"{cnt.get((c, ps, s), 0):>13}" for s in space.STATUSES))

        # compile stats: one line per build
        comp = [r for r in rs if r["pass"] == "compile"]
        if comp:
            print("\n-- builds (compile pass) --")
            n_ok = sum(r["status"] == "OK" for r in comp)
            t = sum(r.get("compile_s") or 0 for r in comp)
            print(f"  {n_ok}/{len(comp)} builds OK, {t:.0f} s compile time summed over builds")

        # register stats per candidate (from check records)
        st = {}
        for r in rs:
            if r["pass"] == "check" and r.get("stats"):
                st.setdefault((r["candidate"], r["dtype"], r["mask"], r["feature"], r.get("build_id")), r["stats"])
        if st:
            print("\n-- register / LDS per build (max over dtype/mask/feature; spill = vgpr+sgpr spills) --")
            agg = collections.defaultdict(lambda: collections.defaultdict(int))
            for (c, *_rest, bid), s in st.items():
                flat = s if "vgpr_count" in s else {k: max((v or {}).get(k, 0) for v in s.values())
                                                    for k in ("vgpr_count", "agpr_count", "sgpr_count",
                                                              "vgpr_spill_count", "sgpr_spill_count",
                                                              "group_segment_fixed_size",
                                                              "private_segment_fixed_size")}
                key = bid or c
                for k, v in flat.items():
                    agg[key][k] = max(agg[key][k], v or 0)
            for key, s in sorted(agg.items()):
                print(f"  {key:<34} vgpr {s['vgpr_count']:>3} agpr {s['agpr_count']:>3} sgpr {s['sgpr_count']:>3} "
                      f"spill v{s['vgpr_spill_count']}/s{s['sgpr_spill_count']} "
                      f"lds {s['group_segment_fixed_size']:>6} scratch {s['private_segment_fixed_size']}")

        # 2. status matrices (check pass)
        chk = [r for r in rs if r["pass"] == "check"]
        if args.by_group:
            keys = sorted({(r["dtype"], r["mask"], r["feature"]) for r in chk})
        else:
            keys = [None]
        for c in cands:
            alias = {r.get("error") for r in chk if r["candidate"] == c}
            if all(r["status"] == "SKIPPED" for r in chk if r["candidate"] == c) and len(alias) == 1:
                print(f"\n-- check status: {c}: every record SKIPPED ({alias.pop()}) --")
                continue
            for gk in keys:
                sel = [r for r in chk if r["candidate"] == c and (gk is None or (r["dtype"], r["mask"],
                                                                                 r["feature"]) == gk)]
                if not sel:
                    continue
                cell = collections.defaultdict(set)
                for r in sel:
                    cell[(r["sq"], r["sk"])].add(r["status"])
                bad = sum(1 for r in sel if r["status"] != "OK")
                label = c if gk is None else f"{c} {'/'.join(gk)}"
                print(f"\n-- check status: {label}  ({len(sel)} records, {bad} not OK) --")
                print(matrix(seqlens, lambda sq, sk: LETTER.get(worst(cell.get((sq, sk), set())), "?")))

        # 3. fastest correct candidate per cell
        tim = [r for r in rs if r["pass"] == "time" and r["status"] == "OK" and r.get("us")]
        if tim:
            ok_check = collections.defaultdict(set)  # (group, sq, sk, cand) -> statuses over layouts
            for r in chk:
                ok_check[(r["dtype"], r["mask"], r["feature"], r["sq"], r["sk"], r["candidate"])].add(r["status"])
            ab = abbrev(cands)
            print("\n-- fastest correct candidate (OK in both layouts) --")
            print("   legend: " + ", ".join(f"{v}={k}" for k, v in sorted(ab.items(), key=lambda kv: kv[1])))
            wins = collections.Counter()
            misses = []
            no_correct = []
            groups = sorted({(r["dtype"], r["mask"], r["feature"]) for r in tim})
            for gk in groups:
                cellmap = {}
                for sq in seqlens:
                    for sk in seqlens:
                        here = [r for r in tim if (r["dtype"], r["mask"], r["feature"], r["sq"], r["sk"]) == (*gk, sq, sk)]
                        if not here:
                            continue
                        good = [r for r in here if ok_check.get((*gk, sq, sk, r["candidate"])) == {"OK"}]
                        pick = [r for r in here if r.get("policy_pick")]
                        if not good:
                            no_correct.append(f"{'/'.join(gk)} {sq}x{sk}")
                            cellmap[(sq, sk)] = "!"
                            continue
                        best = min(good, key=lambda r: r["us"])
                        wins[best["candidate"]] += 1
                        sp = None
                        if pick:
                            sp = pick[0]["us"] / best["us"]
                            pick_ok = ok_check.get((*gk, sq, sk, pick[0]["candidate"])) == {"OK"}
                            if sp > 1 + args.noise or not pick_ok:
                                misses.append((sp, f"{'/'.join(gk)} {sq}x{sk}", pick[0]["candidate"], best["candidate"],
                                               pick[0]["us"], best["us"], best["tf"], pick_ok))
                        cellmap[(sq, sk)] = ab[best["candidate"]] + ("*" if sp and sp > 1 + args.noise else "")
                if args.full:
                    print(f"\n   {'/'.join(gk)}   (* = beats the policy pick by more than {args.noise:.0%};"
                          " ! = no correct candidate)")
                    print(matrix(seqlens, lambda sq, sk: cellmap.get((sq, sk), "-"), width=3))
            print("\n   wins per candidate: " + ", ".join(f"{c} {n}" for c, n in wins.most_common()))
            if no_correct:
                print(f"   cells with no correct timed candidate ({len(no_correct)}): "
                      + ", ".join(no_correct[:args.max_cells * 2]) + (" ..." if len(no_correct) > 2 * args.max_cells else ""))
            if misses:
                print(f"\n   policy pick slower than the fastest correct candidate by > {args.noise:.0%}, or"
                      f" not correct ({len(misses)} cells; largest first):")
                print(f"   {'cell':<42} {'policy pick':<24} {'fastest correct':<24} {'pick us':>9} {'best us':>9}"
                      f" {'x':>6} {'best TF':>8}")
                for sp, cellk, pk, bk, pus, bus, btf, pick_ok in sorted(misses, key=lambda m: -m[0])[:60]:
                    print(f"   {cellk:<42} {pk + ('' if pick_ok else ' (WRONG)'):<24} {bk:<24} {pus:>9.1f} {bus:>9.1f}"
                          f" {sp:>6.2f} {btf:>8.1f}")

            # policy pick TF overview: per group, the policy pick's TF on the diagonal
            print("\n   policy-pick TF on the square cells (Sq=Sk), per group:")
            print("   " + f"{'group':<30}" + "".join(f"{s:>8}" for s in seqlens))
            for gk in groups:
                row = []
                for s in seqlens:
                    pr = [r for r in tim if (r["dtype"], r["mask"], r["feature"], r["sq"], r["sk"]) == (*gk, s, s)
                          and r.get("policy_pick")]
                    row.append(f"{pr[0]['tf']:>8.1f}" if pr else f"{'-':>8}")
                print("   " + f"{'/'.join(gk):<30}" + "".join(row))

        # 4. failures grouped by (candidate, symptom)
        nonok = [r for r in rs if r["status"] != "OK" and r["pass"] != "compile"]
        if nonok:
            print("\n-- non-OK records by (candidate, symptom) --")
            grp = collections.defaultdict(list)
            for r in nonok:
                grp[(r["candidate"], r["pass"], symptom(r))].append(r)
            for (c, ps, sym), lst in sorted(grp.items(), key=lambda kv: (kv[0][0], -len(kv[1]))):
                print(f"\n  [{c}] {ps} {sym}  -- {len(lst)} records")
                axes = collections.Counter()
                for r in lst:
                    axes.update([f"dtype={r['dtype']}", f"mask={r['mask']}", f"feature={r['feature']}",
                                 f"layout={r.get('layout')}", f"sq={r.get('sq')}", f"sk={r.get('sk')}"])
                print("     axis values: " + ", ".join(f"{k}:{v}" for k, v in sorted(axes.items())))
                def _cell_order(r):
                    return (r["dtype"], r["mask"], r["feature"], r.get("layout") or "", r.get("sq") or 0, r.get("sk") or 0)

                for r in sorted(lst, key=_cell_order)[:args.max_cells]:
                    detail = r.get("error") or ""
                    if r["status"] == "WRONG" and r.get("errs"):
                        detail = " ".join(f"{k}={v:.2e}" for k, v in r["errs"].items())
                    print(f"     {fmt_cell_key(r):<46} {detail[:120]}")
                if len(lst) > args.max_cells:
                    print(f"     ... {len(lst) - args.max_cells} more")


if __name__ == "__main__":
    main()
