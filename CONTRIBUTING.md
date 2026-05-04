# Contributing to semble

Thanks for your interest in semble. This document explains how contributions work and what we expect.

## tl;dr

- **Bug fix or typo?** Open a PR directly.
- **New feature or behaviour change?** Open an issue first to discuss with us.
- **AI-generated PRs** will be closed without review if they weren't discussed beforehand.

---

## Discuss before building

Our libraries are small and focused by design. We care a lot about keeping it that way. Before you invest time writing code for a new feature, please open an issue describing:

- What problem you're solving
- Why it belongs in semble (as opposed to a wrapper or separate tool)
- What API or behaviour change it would involve
- A minimal (code) example of how it would work

**PRs that add features without a prior issue will be closed.**

## What we welcome

- Bug fixes (with a test that reproduces the issue)
- Documentation improvements and example fixes

## What we generally won't accept

- Large new features that haven't been discussed
- Features that significantly expand the scope of the library
- Dependency additions
- AI-generated code dumps with no context or discussion

## Opening a good issue

If you found a bug, include:
- semble version (`pip show semble`)
- Python version
- A minimal reproducible example
- What you expected vs. what happened

If you want a feature, include the things listed under "Discuss before building" above.

## Pull request checklist

Before opening a PR:

- [ ] Run `make test` and confirm all tests pass
- [ ] Run `make lint` and `make typecheck`
- [ ] Run `make fix` to auto-fix any lint issues
- [ ] If you added behaviour, add or update tests
- [ ] If you changed a public API, update the docstrings
- [ ] Keep the diff focused (one logical change per PR)

You can also run `make pre-commit` to run all checks at once.

## Code style

- We use `ruff` for formatting and linting
- We use `mypy` for type checking and expect all new code to be fully typed
- Keep things simple; we prefer readable over clever

## A note on AI-assisted contributions

We don't have a blanket policy against AI tools (we also use them ourselves). But we do expect:

1. **You understand what you're submitting.** If you ran an agent against the repo and opened a PR with the output, you should be able to explain what it does.
2. **The contribution was discussed first.** An AI generating code for an agreed-on, well-scoped issue is fine. An AI inventing features and opening a PR is not.
3. **Tests and quality are your responsibility.** "The AI wrote it" is not a substitute for correctness.

PRs that appear to be unreviewed AI output (large scope, multiple unrelated files touched, no prior discussion, new deps) will be closed with a pointer to this document.

---

Questions? Open an issue.
