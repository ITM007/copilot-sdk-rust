// Copyright (c) 2026 Elias Bachaalany
// SPDX-License-Identifier: MIT

//! Attachments example demonstrating file and directory attachments.

use copilot_sdk::{
    AttachmentType, Client, MessageOptions, SessionConfig, SessionEventData,
    UserMessageAttachment,
};
use std::fs;
use std::io::{self, Write};

#[tokio::main]
async fn main() -> copilot_sdk::Result<()> {
    println!("=== Attachments Example ===\n");

    // Create sample file
    let temp_dir = std::env::temp_dir().join("copilot-rust-attachments");
    fs::create_dir_all(&temp_dir).ok();
    let sample_file = temp_dir.join("sample.rs");
    fs::write(
        &sample_file,
        r#"
fn divide(a: i32, b: i32) -> i32 {
    a / b  // Bug: no zero check
}
"#,
    )
    .ok();

    let client = Client::builder().use_stdio(true).build()?;
    client.start().await?;

    let session = client.create_session(SessionConfig::default()).await?;
    let mut events = session.subscribe();

    // Send with file attachment
    let opts = MessageOptions {
        prompt: "Review this code for bugs.".to_string(),
        attachments: Some(vec![UserMessageAttachment {
            attachment_type: AttachmentType::File,
            path: sample_file.to_string_lossy().to_string(),
            display_name: "sample.rs".to_string(),
        }]),
        mode: None,
    };

    println!("Sending with attachment: {}\n", sample_file.display());
    session.send(opts).await?;

    print!("Assistant: ");
    io::stdout().flush().unwrap();
    while let Ok(event) = events.recv().await {
        match &event.data {
            SessionEventData::AssistantMessageDelta(d) => {
                print!("{}", d.delta_content);
                io::stdout().flush().unwrap();
            }
            SessionEventData::SessionIdle(_) => {
                println!();
                break;
            }
            _ => {}
        }
    }

    // Cleanup
    fs::remove_file(&sample_file).ok();
    fs::remove_dir(&temp_dir).ok();

    client.stop().await;
    Ok(())
}
