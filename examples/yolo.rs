// Copyright (c) 2026 Elias Bachaalany
// SPDX-License-Identifier: MIT

//! YOLO mode example - fully autonomous agent with all permissions auto-approved.
//!
//! This example demonstrates:
//! - Custom system prompt configuration
//! - Auto-approving all tool permissions (no user prompts)
//! - Multi-turn conversation with full tool access
//!
//! ⚠️ WARNING: This grants the agent full access to execute tools without confirmation.
//! Use with caution in production environments.

use copilot_sdk::{
    Client, PermissionRequestResult, SessionConfig, SystemMessageConfig,
    SystemMessageMode,
};

#[tokio::main]
async fn main() -> copilot_sdk::Result<()> {
    println!("=== YOLO Mode Example ===");
    println!("Auto-approving all permissions. Use with caution.\n");

    let client = Client::builder().use_stdio(true).build()?;
    client.start().await?;

    // Create session with custom system prompt
    let session = client
        .create_session(SessionConfig {
            system_message: Some(SystemMessageConfig {
                mode: Some(SystemMessageMode::Replace),
                content: Some(
                    "You are a skilled software engineering assistant. \
                     Execute tasks efficiently and provide clear, actionable responses. \
                     When using tools, proceed without hesitation."
                        .to_string(),
                ),
            }),
            ..Default::default()
        })
        .await?;

    println!("Session: {}\n", session.session_id());

    // Auto-approve all permission requests (YOLO mode)
    session
        .register_permission_handler(|req| {
            println!("[Auto-approved: {}]", req.kind);
            PermissionRequestResult::approved()
        })
        .await;

    // Multi-turn conversation
    let prompts = [
        "Hello! I'm testing your capabilities.",
        "What tools do you have access to?",
        "List the files in the current directory.",
    ];

    for prompt in prompts {
        println!("You: {}\n", prompt);
        let response = session.send_and_collect(prompt, None).await?;
        println!("Assistant: {}\n", response);
        println!("---\n");
    }

    client.stop().await;
    println!("Done!");
    Ok(())
}
