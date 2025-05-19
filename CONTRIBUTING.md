# Contributing to llm-d-benchmark

## Governance Structure

`llm-d-benchmark` adopts the following hierarchical technical governance structure:

- A community of **contributors** who file issues and submit pull requests
- A body of **core maintainers** who own `llm-d-benchmark` overall and drive its development
- A **lead core maintainer** who is the catch-all decision maker when consensus cannot be reached by core maintainers

All contributions are expected to follow `llm-d-benchmark` design principles and best practices, as enforced by core maintainers. While high-quality pull requests are appreciated and encouraged, all maintainers reserve the right to prioritize their own work over code reviews at-will, hence contributors should not expect their work to be reviewed promptly.

Contributors can maximize the chances of their work being accepted by maintainers by meeting a high quality bar before sending a PR to maintainers.

### Core maintainers

The core maintainers lead the development of `llm-d-benchmark` and define the benchmarking infrastructure and strategy for the broader `llm-d project`. Their responsibilities include:

- Proposing, implementing and reviewing load profiles, parameter configurations, run rules, data collections, and analysis of workloads to `llm-d`
- Enforcing code quality standards and adherence to core design principles

The core maintainers should publicly articulate their decision-making, and share the reasoning behind their decisions, vetoes, and dispute resolution.

List of core maintainers can be found in the [OWNERS](OWNERS) file.

### Lead core maintainer

When core maintainers cannot come to a consensus, a publicly declared lead maintainer is expected to settle the debate and make executive decisions.

The Lead Core Maintainer should publicly articulate their decision-making, and give a clear reasoning for their decisions.

The Lead Core Maintainer is also responsible for confirming or removing core maintainers.

#### Lead maintainer (as of 05/13/2025)

- [Marcio Silva](https://github.com/maugustosilva)

### Decision Making

#### Uncontroversial Changes

We are committed to accepting functional bug fixes that meet our quality standards – and include minimized unit tests to avoid future regressions. Performance improvements generally fall under the same category, with the caveat that they may be rejected if the trade-off between usefulness and complexity is deemed unfavorable by core maintainers. Design changes that neither fix known functional nor performance issues are automatically considered controversial.

#### Controversial Changes

More controversial design changes (e.g., breaking changes to workload profiles, load generators, run rules or data collection and analisys tools) are evaluated on a case-by-case basis under the subjective judgment of core maintainers.

## Submitting a Pull Request

We welcome contributions to any aspect of `llm-d-benchmark`! If you have a bug fix, feature request, or improvement, please submit a pull request (PR) to the repository.

Before submitting a pull request, please ensure that you have following dependencies installed and set up:

- [pre-commit](https://pre-commit.com/)

Then run:

```./util/setup_precommit.sh```

For every Pull Request submitted, ensure the following steps have been done:

1. [Sign your commits](https://docs.github.com/en/authentication/managing-commit-signature-verification/signing-commits)
2. Make sure that [pre-commit](https://pre-commit.com/) hook has been run, and passes, before push the contents of your PR.