# iOS Repair Benchmark

Versioned XcodeFixBench task corpus.

```text
fixtures/   synthetic source snapshots materialized into deterministic Git repositories
dev/        open task contracts, Gold Patches, Negative Patches and replay specs
```

Current executable task: [`keyboard-layout-001`](dev/keyboard-layout-001/).

Inspect its contract without running Xcode:

```bash
uv run xcodefix task --task keyboard-layout-001
```

Run the complete Gold Patch path on the pinned local simulator:

```bash
uv run xcodefix run \
  --task keyboard-layout-001 \
  --candidate gold \
  --approve-patch
```

Without `--approve-patch`, the run reproduces and diagnoses the Bug, inspects the candidate, then stops at `approval_required` before modifying the isolated workspace.
