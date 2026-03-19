# Changelog

## v0.1.32

- add protocol compatibility for CLI protocol versions `2..=3`
- add protocol v3 event coverage and v2/v3 tool/permission compatibility handling
- add typed session RPC helpers for mode, model, plan, workspace, agent, fleet, compaction, and session logging
- add `Session::disconnect()` and align shutdown semantics with upstream SDK behavior
- tighten tests and release docs for the expanded parity surface
