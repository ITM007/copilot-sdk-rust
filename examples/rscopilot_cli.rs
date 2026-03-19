// Copyright (c) 2026 Elias Bachaalany
// SPDX-License-Identifier: MIT

//! A tiny Rust REPL around the Copilot SDK.
//!
//! Run:
//! `cargo run --example rscopilot-cli`
//!
//! Commands:
//! - `/exit` or `/quit`: exit
//! - `/clear`: destroy the current session and create a new one
//! - `/approvals ask`: prompt on each permission request
//! - `/approvals bypass all`: auto-approve all permission requests (no prompts)

use copilot_sdk::{
    Client, LogLevel, PermissionRequest, PermissionRequestResult, SessionConfig,
    SessionEventData, SystemMessageConfig, SystemMessageMode, find_copilot_cli,
};
use std::io::{self, Write};
use std::sync::{
    Arc,
    atomic::{AtomicU8, Ordering},
};

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ApprovalMode {
    Ask = 0,
    BypassAll = 1,
}

impl ApprovalMode {
    fn from_args(args: &[&str]) -> Option<Self> {
        match args {
            ["ask"] => Some(Self::Ask),
            ["bypass", "all"] | ["bypass-all"] | ["bypassall"] => Some(Self::BypassAll),
            _ => None,
        }
    }
}

fn prompt_approve(req: &PermissionRequest) -> bool {
    eprintln!();
    eprintln!(
        "[permission request] kind={} tool_call_id={:?}",
        req.kind, req.tool_call_id
    );
    if !req.extension_data.is_empty() {
        if let Ok(pretty) = serde_json::to_string_pretty(&req.extension_data) {
            eprintln!("{pretty}");
        }
    }

    eprint!("Approve? [y/N]: ");
    let _ = io::stderr().flush();

    let mut line = String::new();
    if io::stdin().read_line(&mut line).is_err() {
        return false;
    }
    matches!(line.trim().to_ascii_lowercase().as_str(), "y" | "yes")
}

async fn create_session(
    client: &Client,
    approval_mode: Arc<AtomicU8>,
) -> copilot_sdk::Result<(Arc<copilot_sdk::Session>, copilot_sdk::EventSubscription)> {
    let session = client
        .create_session(SessionConfig {
            streaming: true,
            system_message: Some(SystemMessageConfig {
                mode: Some(SystemMessageMode::Replace),
                content: Some(
                    "You are a helpful Copilot agent. Keep replies concise and actionable."
                        .to_string(),
                ),
            }),
            ..Default::default()
        })
        .await?;

    let handler_mode = approval_mode.clone();
    session
        .register_permission_handler(move |req| {
            match handler_mode.load(Ordering::Relaxed) {
                m if m == ApprovalMode::BypassAll as u8 => {
                    PermissionRequestResult::approved()
                }
                _ => {
                    if prompt_approve(req) {
                        PermissionRequestResult::approved()
                    } else {
                        PermissionRequestResult::denied()
                    }
                }
            }
        })
        .await;

    let events = session.subscribe();
    Ok((session, events))
}

fn print_help(current_mode: ApprovalMode) {
    println!();
    println!("Commands:");
    println!("  /exit | /quit            Exit");
    println!("  /clear                   Destroy session and create a new one");
    println!("  /approvals               Show current mode");
    println!("  /approvals ask           Prompt for approvals");
    println!("  /approvals bypass all    Auto-approve everything (dangerous)");
    println!();
    println!("Current approvals mode: {current_mode:?}");
    println!();
}

#[tokio::main]
async fn main() -> copilot_sdk::Result<()> {
    println!("=== rscopilot-cli (Copilot SDK Rust REPL) ===\n");

    if find_copilot_cli().is_none() {
        println!(
            "Copilot CLI not found. Set `COPILOT_CLI_PATH` or install/authenticate `copilot`."
        );
        return Ok(());
    }

    let client = Client::builder().log_level(LogLevel::Info).build()?;
    client.start().await?;

    let approval_mode = Arc::new(AtomicU8::new(ApprovalMode::Ask as u8));
    let (mut session, mut events) =
        create_session(&client, approval_mode.clone()).await?;
    println!("Session: {}", session.session_id());
    print_help(ApprovalMode::Ask);

    let stdin = io::stdin();
    loop {
        print!("you> ");
        io::stdout().flush().ok();

        let mut line = String::new();
        if stdin.read_line(&mut line).unwrap_or(0) == 0 {
            break;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        if let Some(cmd) = line.strip_prefix('/') {
            let parts: Vec<&str> = cmd.split_whitespace().collect();
            let (name, args) = parts.split_first().unwrap_or((&"", &[][..]));

            match *name {
                "exit" | "quit" => break,
                "clear" => {
                    println!("Clearing session...");
                    let _ = session.destroy().await;
                    (session, events) =
                        create_session(&client, approval_mode.clone()).await?;
                    println!("Session: {}", session.session_id());
                }
                "approvals" => {
                    if args.is_empty() {
                        let mode = match approval_mode.load(Ordering::Relaxed) {
                            m if m == ApprovalMode::BypassAll as u8 => {
                                ApprovalMode::BypassAll
                            }
                            _ => ApprovalMode::Ask,
                        };
                        print_help(mode);
                        continue;
                    }

                    let Some(mode) = ApprovalMode::from_args(args) else {
                        println!("Usage: /approvals ask | /approvals bypass all");
                        continue;
                    };
                    approval_mode.store(mode as u8, Ordering::Relaxed);
                    println!("Approvals mode set to: {mode:?}");
                }
                "help" => {
                    let mode = match approval_mode.load(Ordering::Relaxed) {
                        m if m == ApprovalMode::BypassAll as u8 => {
                            ApprovalMode::BypassAll
                        }
                        _ => ApprovalMode::Ask,
                    };
                    print_help(mode);
                }
                _ => {
                    println!("Unknown command: /{name}. Try /help.");
                }
            }

            continue;
        }

        session.send(line).await?;

        print!("assistant> ");
        io::stdout().flush().ok();

        let mut printed_delta = false;
        loop {
            match events.recv().await {
                Ok(event) => match &event.data {
                    SessionEventData::AssistantMessageDelta(d) => {
                        printed_delta = true;
                        print!("{}", d.delta_content);
                        io::stdout().flush().ok();
                    }
                    SessionEventData::AssistantMessage(msg) => {
                        if !printed_delta {
                            print!("{}", msg.content);
                            io::stdout().flush().ok();
                        }
                    }
                    SessionEventData::ToolExecutionStart(t) => {
                        println!("\n[tool start] {} ({})", t.tool_name, t.tool_call_id);
                        print!("assistant> ");
                        io::stdout().flush().ok();
                    }
                    SessionEventData::ToolExecutionComplete(t) => {
                        println!(
                            "\n[tool done] {} success={}",
                            t.tool_call_id, t.success
                        );
                        if let Some(err) = &t.error {
                            eprintln!("[tool error] {}", err.message);
                        }
                        print!("assistant> ");
                        io::stdout().flush().ok();
                    }
                    SessionEventData::SessionIdle(_) => {
                        println!();
                        break;
                    }
                    SessionEventData::SessionError(err) => {
                        eprintln!("\n[session error] {}", err.message);
                        break;
                    }
                    _ => {}
                },
                Err(e) => {
                    eprintln!("\n[event stream error] {e:?}");
                    break;
                }
            }
        }
    }

    let _ = session.destroy().await;
    client.stop().await;
    Ok(())
}
