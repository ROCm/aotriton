# PR Description Writing Instructions

This document provides guidelines for writing PR descriptions in the AOTriton project based on analysis of historical PRs merged into `upstream/main`.

## Overall Structure

A PR description should follow this structure:

1. **Overview Section** - High-level summary of what the PR does and why
2. **Major Changes Section** - Detailed list of significant changes
3. **Minor Changes Section** (optional) - List of smaller changes
4. **Known Issues/Problems Section** (optional) - Document any limitations or issues

## 1. Overview Section

Start with a header `# Overview` followed by:

- **What**: A clear, concise explanation of what this PR does (1-3 paragraphs)
- **Why**: The motivation or context for the change
- **Impact**: Performance improvements, new features, or API changes if applicable

### Examples of good overview sections:

```markdown
# Overview

This PR enables kernel pipelining and XCD (compute die) remapping
optimizations for gfx950 GPUs (MI350 series), significantly improving
performance of Flash Attention kernels.

This update delivers 904 TFLOPS performance on MI355X with mainline Triton JIT,
improved from 753 TFLOPS with the same Triton JIT process without pipelining.
```

```markdown
# Overview

This PR changes the API for varlen inputs and expects compact logsumexp
(LSE) tensor, following Tri's FlashAttention API.

Previously for varlen inputs with `B` sequences, `H` heads, AOTriton
expects a regular-sized LSE Tensor: `(B*H, Max_sequence_length)`.
However this approach requires padding at last dimension and wasting
memory when the sequence lengths change drastically within a batch.
```

## 2. Major Changes Section

Use header `## Major Changes` or `# Major Changes` followed by a bulleted list.

### Formatting Guidelines:

- Use `* [category] Description` format where category indicates the affected component:
  - `[kernel]` - Triton kernel changes
  - `[api]` - API changes (mark as `**BREAKING**` if breaking)
  - `[shim]` - C++ shim layer changes
  - `[build]` - Build system changes
  - `[test]` - Test infrastructure changes
  - `[db]` - Database changes
  - `[codegen]` - Code generation changes
  - `[rules]` - Kernel selection rules
  - `[tritonsrc]` - Triton source code changes
  - `[ci]` - CI/CD changes
  - `[binding]` - Python binding changes
  - `[compiler]` - Compiler changes
  - `[tune]` - Tuning system changes

- Use sub-bullets with `+` for details under each major point
- Use inline code formatting for:
  - Function/class names: `` `attn_fwd` ``
  - File names: `` `config.h` ``
  - Env vars: `` `AOTRITON_SKIP_LUT_CHECK` ``
  - Options/flags: `` `num_stages=2` ``
  - Constants: `` `VarlenType::PaddedVarlen` ``

### Example:

```markdown
## Major Changes

* [api] **BREAKING** Use compact LSE for varlen inputs
* [kernel] Use compact LSE and Delta Tensor
* [kernel] `head_dim` argument in all SDPA Triton kernels is replaced by
  `hdim_qk` and `hdim_vo` arguments to support this feature.
* [shim] The dispatcher will use the last dimensions of `Q` and `V` tensor as
  `hdim_qk` and `hdim_vo`
* [test] Add test cases to test `hdim_qk != hdim_vo`. New cases are added to
  + `test_fast` case (`FOR_RELEASE=0`)
  + `test_hdim_qk_ne_vo` (`FOR_RELEASE=3`)
```

## 3. Minor Changes Section

Use header `## Minor Changes` or `# Minor Changes`.

Follow similar categorization and formatting as Major Changes, but for:
- Refactoring that doesn't change functionality
- Documentation updates
- Small bug fixes
- Tool improvements
- Code cleanup

### Example:

```markdown
## Minor Changes

* [test] Adjust unit tests for varlen compact LSE tensor
* [docs] Update comments about input dimensions in V2 header file
* [build] Replace `git log -1 --format=%H` with `git rev-parse HEAD`
```

## 4. Known Issues/Problems Section

Use header `## Known Issues`, `## Known Problems`, or `# Known Issues`.

Document:
- Features not yet implemented
- Database updates pending
- Platform-specific issues
- Performance concerns
- Test coverage gaps
- Temporary workarounds needed

### Example:

```markdown
## Known Issues

* [db] The operator tuning database is not updated.
* [test] The number of functionals supported by FWD AITER ASM kernels are
  fairly limited.
* [aiter] The following AITER ASM functions need more investigation:
  - "Group mode" AITER ASM kernels to support varlen
  - MQA/GQA support
```

## 5. Additional Notes (Optional)

If needed, add a `## Notes` section for important clarifications:

```markdown
## Notes

* The code uses `hdim_vo` instead of `hdim_v` for better alignment with
  `hdim_qk`, and emphasizes the output tensor (`o`) should follow Tensor `V`'s
  head dimension.
```

## Special Considerations

### Breaking API Changes
- Mark as `**BREAKING**` in the item description
- Clearly explain the old vs new behavior
- Provide migration guidance if applicable

### Performance Changes
- Include concrete numbers when available
- Specify test conditions/commands used to measure
- Use TFLOPS or other relevant metrics

### Database Updates
- Always note if database is updated or pending update
- Mention affected architectures

### Deprecation
- **Ignore v2src and v2python**: These directories are deprecated
- Do not document changes to deprecated components unless they affect non-deprecated code

### Code Examples
Use triple backticks for:
- Command examples
- Configuration snippets
- Code samples

### Links and References
- Link to related issues: `This fixes #98`
- Reference external docs when relevant

## Tone and Style

- Be concise but complete
- Use technical terminology correctly
- Focus on **what** changed and **why** it matters
- Avoid redundant information
- Use imperative mood for changes ("Add support", not "Added support")
- Group related changes together under the same bullet

## Final Checklist

Before submitting a PR description:

- [ ] Overview explains the "what" and "why"
- [ ] Major changes are properly categorized
- [ ] Breaking changes are clearly marked
- [ ] Performance impacts are quantified when available
- [ ] Known issues are documented
- [ ] Code/file names use inline code formatting
- [ ] No changes to v2src/v2python are documented (deprecated)
