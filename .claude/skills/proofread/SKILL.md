---
name: proofread
description: Proofread a PR description held in a .md file (e.g. pr237.md, usually untracked). Fixes typos, grammar and formatting in place and lists them first, then reports content-level problems last — claims the code contradicts, inverted explanations, misclassified sections. Use whenever asked to proofread, review or check over a PR description, PR draft, or similar .md write-up.
---

# Proofreading a PR description

The author wants a clean file and a reply they can read from the bottom up.

## Reply order — least important FIRST

**The author reads the tail of a long reply first.** So the reply runs in
ascending order of importance: the mechanical fixes open it, the content
problems close it, and the single most serious finding is the last thing on
the page. This inverts the usual habit of leading with the headline — do not
"fix" it back.

```
1. Fixed in place   <- mechanical, already applied, skimmable
2. Content findings <- ascending severity, most serious LAST
```

If there are no content findings, say so in one line at the end.

## 1. Fix in place, then list briefly

Apply directly with `Edit` — never ask permission for these. Then list them
at the TOP of the reply, compressed: a one-line summary plus counts beats an
itemised inventory ("3 typos, 2 missing articles, reflowed 4 lines >78ch").
Name individual fixes only where the author might disagree with the choice.

- Typos, misspellings, grammar, missing or wrong articles
- Wrapping: reflow to 78 characters (`awk 'length>78' file.md` to find them)
- Punctuation consistency *within the document* — pick whichever convention
  the majority of bullets already use and make the rest match
- Stray double blank lines, trailing whitespace, inconsistent list markers
- Inconsistent backticking or capitalisation of identifiers that appear
  elsewhere in the same file

## 2. Report last — do not fix unilaterally

State these at the END of the reply, with the correction you would make.
Ascending severity, so the worst one lands last. Keep each one short.

Listed here in the same ascending order the reply should use:

- **Budget overruns** per prwriter's line/word ceilings, when substantial.
- **Unsupported numbers.** House style wants the command or measurement
  behind any perf/scale claim.
- **Misclassification** against `.claude/agents/prwriter.md`: a headline
  behaviour change buried in Minor Changes, a real caveat with no Known
  Issues entry, sections out of canonical order.
- **Technical explanations that are wrong or misleading** even where not
  strictly false, e.g. naming the wrong mechanism for a real symptom.
- **Claims the code contradicts.** The most valuable thing this skill does,
  so it goes last where the author reads first. A description can confidently
  assert the opposite of what the diff does — an inverted flag polarity, a
  mode claimed not to apply when it does, a symbol that was never touched.

## Verify before flagging a content error

Never flag a technical claim from memory or from earlier conversation. Read
the source or the diff first:

```
git diff <base>..HEAD          # what the PR actually changes
grep -n '<symbol>' <file>      # what the symbol actually does
```

Quote the file:line you checked in the reply. A wrong correction costs more
than a missed typo.

## Respect the author's edits

- The file changes between turns. **Re-read it immediately before editing** —
  the author frequently rewrites sections after asking.
- If the author deleted a section, it was deliberate. Do not restore it; note
  its absence once at most, then leave it.
- If the author overrules a finding, drop it and do not re-raise it. Their
  common and correct objection: *"the diff makes this obvious"* — prwriter
  says to omit detail that is self-describing from the code, so a one-line
  mechanical fix genuinely does not need a `why`.

## Housekeeping

These files are usually untracked scratch (`pr<N>.md` at the repo root).
Do not `git add` them, do not commit them, and never add them to
`.gitignore` — see `CLAUDE.md`.
