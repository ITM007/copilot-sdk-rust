// Copyright (c) 2026 Elias Bachaalany
// SPDX-License-Identifier: MIT

//! Compaction events example demonstrating session compaction monitoring.
//!
//! This example shows how to:
//! 1. Configure infinite sessions with a low compaction threshold
//! 2. Subscribe to SessionCompactionStart/Complete events
//! 3. Monitor context usage via SessionUsageInfo events
//! 4. Track compaction progress in real-time

use copilot_sdk::{Client, InfiniteSessionConfig, SessionConfig, SessionEventData};
use std::io::Write;
use std::sync::Arc;

#[tokio::main]
async fn main() -> copilot_sdk::Result<()> {
    let client = Client::builder().build()?;

    println!("=== Compaction Events Example ===\n");
    println!("This example monitors context compaction in real-time.");
    println!("When the context window fills up, the session automatically");
    println!("compacts (summarizes) older messages to free up space.\n");

    client.start().await?;

    // Configure session with a low compaction threshold
    let config = SessionConfig {
        streaming: true,
        infinite_sessions: Some(InfiniteSessionConfig::with_thresholds(0.10, 0.0)),
        ..Default::default()
    };

    let session = client.create_session(config).await?;
    println!("Session created with low compaction threshold (10%)");
    println!("Compaction events will appear when context usage exceeds the threshold.\n");

    let idle_notify = Arc::new(tokio::sync::Notify::new());
    let idle_clone = Arc::clone(&idle_notify);

    let _unsub = session
        .on(move |event| match &event.data {
            SessionEventData::AssistantMessageDelta(delta) => {
                print!("{}", delta.delta_content);
                std::io::stdout().flush().unwrap();
            }
            SessionEventData::AssistantMessage(msg) => {
                if !msg.content.is_empty() {
                    println!("\nAssistant: {}", msg.content);
                }
            }
            SessionEventData::SessionUsageInfo(usage) => {
                let pct = if usage.token_limit > 0.0 {
                    (usage.current_tokens / usage.token_limit * 100.0) as i64
                } else {
                    0
                };
                println!(
                    "\n[Usage] Tokens: {}/{} ({}%)  Messages: {}",
                    usage.current_tokens as i64,
                    usage.token_limit as i64,
                    pct,
                    usage.messages_length as i64,
                );
            }
            SessionEventData::SessionCompactionStart(_) => {
                println!("\n*** COMPACTION STARTED ***");
                println!("    Context is being summarized to free up space...");
            }
            SessionEventData::SessionCompactionComplete(complete) => {
                println!("\n*** COMPACTION COMPLETE ***");
                println!(
                    "    Success: {}",
                    if complete.success { "yes" } else { "no" }
                );

                if let Some(ref error) = complete.error {
                    println!("    Error: {error}");
                }

                if let (Some(pre), Some(post)) = (
                    complete.pre_compaction_tokens,
                    complete.post_compaction_tokens,
                ) {
                    println!("    Tokens: {} -> {}", pre as i64, post as i64);
                }

                if let (Some(pre_msgs), Some(post_msgs)) = (
                    complete.pre_compaction_messages_length,
                    complete.post_compaction_messages_length,
                ) {
                    println!("    Messages: {} -> {}", pre_msgs as i64, post_msgs as i64);
                }

                if let Some(ref tokens) = complete.compaction_tokens_used {
                    println!(
                        "    Compaction cost: in={} out={} cached={}",
                        tokens.input as i64,
                        tokens.output as i64,
                        tokens.cached_input as i64,
                    );
                }
            }
            SessionEventData::SessionError(error) => {
                eprintln!("\nError: {}", error.message);
            }
            SessionEventData::SessionIdle(_) => {
                idle_clone.notify_one();
            }
            _ => {}
        })
        .await;

    println!("Send multiple messages to build up context and trigger compaction.");
    println!("Type 'quit' to exit.\n");
    print!("> ");
    std::io::stdout().flush().unwrap();

    let stdin = std::io::stdin();
    let mut line = String::new();

    loop {
        line.clear();
        if stdin.read_line(&mut line).unwrap() == 0 {
            break;
        }
        let input = line.trim();

        if input == "quit" || input == "exit" {
            break;
        }
        if input.is_empty() {
            print!("> ");
            std::io::stdout().flush().unwrap();
            continue;
        }

        session.send(input).await?;
        idle_notify.notified().await;
        print!("\n> ");
        std::io::stdout().flush().unwrap();
    }

    println!("\nCleaning up...");
    session.destroy().await?;
    client.stop().await;

    Ok(())
}
