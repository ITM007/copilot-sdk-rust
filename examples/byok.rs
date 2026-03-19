// Copyright (c) 2026 Elias Bachaalany
// SPDX-License-Identifier: MIT

//! BYOK (Bring Your Own Key) example demonstrating custom provider configuration.

use copilot_sdk::{
    AzureOptions, Client, ProviderConfig, SessionConfig, SessionEventData,
};
use std::env;
use std::io::{self, Write};

#[tokio::main]
async fn main() -> copilot_sdk::Result<()> {
    println!("=== BYOK Example ===\n");

    let client = Client::builder().use_stdio(true).build()?;
    client.start().await?;

    // Get API key from environment
    let api_key = env::var("OPENAI_API_KEY").ok();

    // OpenAI provider config
    let openai_provider = ProviderConfig {
        base_url: "https://api.openai.com/v1".to_string(),
        provider_type: Some("openai".to_string()),
        wire_api: Some("openai".to_string()),
        api_key: api_key.clone(),
        bearer_token: None,
        azure: None,
    };

    // Azure provider config (example)
    let _azure_provider = ProviderConfig {
        base_url: "https://your-resource.openai.azure.com".to_string(),
        provider_type: Some("azure".to_string()),
        wire_api: Some("openai".to_string()),
        api_key: env::var("AZURE_OPENAI_API_KEY").ok(),
        bearer_token: None,
        azure: Some(AzureOptions {
            api_version: Some("2024-02-15-preview".to_string()),
        }),
    };

    let config = SessionConfig {
        provider: if api_key.is_some() {
            Some(openai_provider)
        } else {
            None
        },
        model: Some("gpt-4".to_string()),
        ..Default::default()
    };

    if config.provider.is_some() {
        println!("Using custom OpenAI provider");
    } else {
        println!("No OPENAI_API_KEY, using default Copilot provider\n");
    }

    let session = client.create_session(config).await?;
    let mut events = session.subscribe();

    session.send("What model are you?").await?;

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

    client.stop().await;
    Ok(())
}
