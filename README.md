# copilot-sdk (Rust)

Rust SDK for interacting with the GitHub Copilot CLI agent runtime (JSON-RPC over stdio or TCP).

This is a Rust port of the upstream SDKs and is currently in technical preview.

## Requirements

- Rust 1.85+ (Edition 2024)
- GitHub Copilot CLI installed and authenticated
- `copilot` available in `PATH`, or set `COPILOT_CLI_PATH` to the CLI executable/script

## Install

Once published, add:

```toml
[dependencies]
copilot-sdk = "0.1.32"
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
```

For development from this repository:

```toml
[dependencies]
copilot-sdk = { path = "." }
```

## Quick Start

```rust
use copilot_sdk::{Client, SessionConfig};

#[tokio::main]
async fn main() -> copilot_sdk::Result<()> {
    let client = Client::builder().build()?;
    client.start().await?;

    let session = client.create_session(SessionConfig::default()).await?;
    let response = session.send_and_collect("Hello!", None).await?;
    println!("{}", response);

    client.stop().await;
    Ok(())
}
```

## Features

The Rust SDK already covers the main SDK workflows:

- stdio and TCP transports
- session create/resume/list/delete plus foreground-session APIs
- streaming events, hooks, custom tools, and ask-user/user input forwarding
- MCP servers, custom agents, attachments, BYOK, reasoning effort, and infinite sessions
- typed session RPC helpers for model, mode, plan, workspace, agent, fleet, compaction, and log flows

### Protocol compatibility

Startup accepts the same CLI protocol range documented upstream: `2..=3`.

### Infinite Sessions

Automatic context window management that compacts conversation history when approaching token limits:

```rust
let config = SessionConfig {
    infinite_sessions: Some(InfiniteSessionConfig::enabled()),
    ..Default::default()
};
```

Typed helpers are available for workspace state too: `plan_read`, `plan_update`,
`plan_delete`, `workspace_list_files`, `workspace_read_file`, and
`workspace_create_file`.

### Custom Tools

Register tools that the assistant can invoke:

```rust
session.register_tool_with_handler(
    Tool::builder("get_weather", "Get current weather")
        .string_param("city", "City name", true)
        .build(),
    |invocation| async move {
        let city: String = invocation.arg("city")?;
        Ok(ToolResult::text(format!("Weather in {}: Sunny, 72°F", city)))
    },
).await;
```

### Client Utilities

```rust
let status = client.get_status().await?;       // CLI version info
let auth = client.get_auth_status().await?;    // Authentication state
let models = client.list_models().await?;      // Available models

let session = client.create_session(SessionConfig::default()).await?;
let mode = session.get_mode().await?;
session.set_mode(copilot_sdk::AgentMode::Plan).await?;
let current_model = session.get_current_model().await?;
session.switch_model("claude-sonnet-4.5", Some("high")).await?;
session
    .log("Switched model", Some(copilot_sdk::SessionLogLevel::Info), None)
    .await?;
session.disconnect().await?;
```

### Hooks and ask-user

See:

- `examples/hooks.rs`
- `examples/user_input.rs`
- `examples/permission_callback.rs`

### BYOK (Bring Your Own Key)

Use your own API keys with compatible providers:

```rust
let config = SessionConfig {
    provider: Some(ProviderConfig {
        base_url: Some("https://api.openai.com/v1".into()),
        api_key: Some("sk-...".into()),
        ..Default::default()
    }),
    ..Default::default()
};
```

## Examples

```bash
cargo run --example basic_chat
cargo run --example tool_usage
cargo run --example streaming
```

## Development

### Setup

Enable pre-commit hooks to catch formatting/linting issues before push:

```bash
git config core.hooksPath .githooks
```

### Commands

```bash
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```

E2E tests (real Copilot CLI):

```bash
cargo test --features e2e -- --test-threads=1
```

Snapshot conformance tests (optional, against upstream YAML snapshots):

```bash
cargo test --features snapshots --test snapshot_conformance
```

Set `COPILOT_SDK_RUST_SNAPSHOT_DIR` or `UPSTREAM_SNAPSHOTS` to point at `copilot-sdk/test/snapshots` if it cannot be auto-detected.

## Notes

- Supports stdio (spawned CLI) and TCP (spawned or external server).
- Protocol negotiation accepts CLI protocol versions `2..=3`.

## License

MIT License - see [LICENSE](LICENSE).

## Related

- Upstream SDKs: https://github.com/github/copilot-sdk
