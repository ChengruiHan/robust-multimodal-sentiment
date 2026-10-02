# Local validation

The code/documentation release was checked on macOS arm64 on 2026-10-02.
Root synthetic tests ran on Python 3.11.15 and 3.12.13. Module arithmetic tests
used an existing Python 3.11.15 / NumPy 2.2.6 environment, separately from the
locked Python 3.12 / NumPy 2.1.3 training environment.

| Check | Result |
| --- | --- |
| Root synthetic metric/asset tests | 5 passed on Python 3.11 and 3.12 |
| Feature aggregation/review tests | 4 passed on Python 3.11 |
| Explanation and width tests | 9 passed locally on Python 3.11 |
| Public-checkout simulation | Original external sample test skips when absent; remaining explanation tests pass |
| Markdown checks | Code fences, math delimiters, table widths and local link targets checked |
| Python and shell syntax | Python sources compiled; three shell runners parsed successfully |
| Publication scope | Code and documentation only; no research records or private manifests in the public branch |

Root tests use synthetic numbers and temporary files. The explanation suite
includes a schema check for an optional original local sample; it explicitly
skips when absent. The public CI does not require original research data.

## Verification limits

The STATE-MSA PyTorch contract suite was not run locally because PyTorch was
unavailable. GitHub Actions configures a CPU job using NumPy 2.1.3 and PyTorch
2.5.1, plus standard-library jobs on Python 3.11/3.12. A configured workflow
alone does not confirm remote check success; consult the PR checks.

No training, full checkpoint inference, video extraction or source-time alignment
was rerun. Research records remain outside the public release. Optional private
re-scoring utilities verify numeric consistency of separately supplied records;
they are not a claim that users can reproduce original results from this checkout.

See [DATA](DATA.md) and [REPRODUCE](REPRODUCE.md) for external asset requirements
and the commands for each verification or reproduction path.
