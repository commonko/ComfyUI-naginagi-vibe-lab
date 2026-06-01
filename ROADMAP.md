# ComfyUI-HumanGate Roadmap

## v0.1 Stop Behavior

`Stop` is implemented by raising `HumanGateUserStop`.

Reason:

- It reliably stops the current workflow from inside a custom node.
- It does not depend on private ComfyUI executor internals.
- It works in the same route/session architecture used by Pause and Chooser.

Known UI effect:

- ComfyUI displays the raised exception as an Error Report.
- This is an intentional cancellation, not an unexpected node crash.

## v0.2 Investigation: Non-Error Cancellation

Investigate whether current ComfyUI exposes a stable public API for one of:

- interrupting the active execution from a custom node,
- marking a prompt as user-cancelled without an exception,
- blocking or cancelling downstream execution without Error Report UI,
- returning a cancellation status through the executor.

Acceptance criteria for replacing the v0.1 path:

- Stop halts the running prompt without rendering as a node error.
- The behavior works with Nodes 2.0 and API mode failure semantics are documented.
- The implementation does not depend on private or version-fragile executor fields.
- `HumanGateUserStop` remains as fallback for older ComfyUI versions.
