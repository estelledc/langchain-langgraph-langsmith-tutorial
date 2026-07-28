# Keyboard Layout 001 fixture

Synthetic UIKit fixture for the stale-notification-versus-final-guide geometry bug.

- One keyboard notification reports `325`.
- The simulated layout guide later reaches `521` without a second notification.
- The broken resolver keeps the panel at `325`.
- The correct repair follows the current guide overlap for system keyboards while preserving business-panel and hidden states.

The committed Xcode project is generated deterministically:

```bash
python3 scripts/generate_project.py
xcodebuild -project KeyboardFixture.xcodeproj -scheme KeyboardFixture -list
```

This fixture contains no production or company source code. Its bundle identifier, UI, state machine and tests are purpose-built for XcodeFixBench.
