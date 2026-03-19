// Copyright (c) 2026 Elias Bachaalany
// SPDX-License-Identifier: MIT

//! List models example demonstrating model discovery and capabilities.
//!
//! This example shows how to:
//! 1. Enumerate available models via client.list_models()
//! 2. Inspect model capabilities (vision support, context window)
//! 3. Display model limits and policy information

use copilot_sdk::Client;

#[tokio::main]
async fn main() -> copilot_sdk::Result<()> {
    let client = Client::builder().build()?;

    println!("=== List Models Example ===\n");
    println!("Discovering available models and their capabilities...\n");

    client.start().await?;

    let models = client.list_models().await?;

    println!("Found {} model(s):\n", models.len());

    let mut vision_count = 0;

    for (i, model) in models.iter().enumerate() {
        println!("  {}. {} [{}]", i + 1, model.name, model.id);

        // Context window
        println!(
            "     Context window: {} tokens",
            model.capabilities.limits.max_context_window_tokens
        );

        if let Some(max_prompt) = model.capabilities.limits.max_prompt_tokens {
            println!("     Max prompt: {} tokens", max_prompt);
        }

        // Vision support
        if model.capabilities.supports.vision {
            println!("     Vision: SUPPORTED");
            vision_count += 1;

            if let Some(vision) = &model.capabilities.limits.vision {
                if !vision.supported_media_types.is_empty() {
                    println!(
                        "       Media types: {}",
                        vision.supported_media_types.join(", ")
                    );
                }

                if vision.max_prompt_images > 0 {
                    println!(
                        "       Max images per prompt: {}",
                        vision.max_prompt_images
                    );
                }

                if vision.max_prompt_image_size > 0 {
                    println!(
                        "       Max image size: {} bytes",
                        vision.max_prompt_image_size
                    );
                }
            }
        } else {
            println!("     Vision: not supported");
        }

        // Policy info
        if let Some(policy) = &model.policy {
            println!("     Policy: {}", policy.state);
        }

        println!();
    }

    println!(
        "Summary: {} of {} model(s) support vision.",
        vision_count,
        models.len()
    );

    client.stop().await;

    Ok(())
}
