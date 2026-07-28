# keyboard-layout-001

First executable XcodeFixBench development task.

The failure is deliberately narrower than a production third-party-keyboard incident. It deterministically reproduces the causal mechanism:

```text
notification snapshot = 325
layout guide later = 521
second notification = absent
broken panel = 325
mismatch = 196
```

The Gold Patch changes only `KeyboardFixture/KeyboardHeightResolver.swift` and follows the current guide overlap. The negative hard-coded patch returns `521`; it fixes the showcased number but must fail the independent `384` oracle.

Evidence boundary:

- This is a `synthetic-seeded` UIKit fixture, not production source.
- Simulator replay proves the deterministic mechanism and the Harness path.
- It does not prove real-device third-party input-method behavior.
