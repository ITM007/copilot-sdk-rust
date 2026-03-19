// Copyright (c) 2026 Elias Bachaalany
// SPDX-License-Identifier: MIT

//! Session management for the Copilot SDK.
//!
//! A session represents a conversation with the Copilot CLI.

use crate::error::{CopilotError, Result};
use crate::events::{SessionEvent, SessionEventData};
use crate::types::{
    AgentMode, ErrorOccurredHookInput, MessageOptions, PermissionRequest,
    PermissionRequestResult, PostToolUseHookInput, PreToolUseHookInput,
    SessionAgentGetCurrentResult, SessionAgentInfo, SessionAgentListResult,
    SessionAgentSelectResult, SessionCompactionCompactResult, SessionEndHookInput,
    SessionFleetStartResult, SessionHooks, SessionLogLevel, SessionLogResult,
    SessionModeGetResult, SessionModeSetResult, SessionModelGetCurrentResult,
    SessionModelSwitchToResult, SessionPlanReadResult, SessionStartHookInput,
    SessionWorkspaceListFilesResult, SessionWorkspaceReadFileResult, Tool,
    ToolResultObject, UserInputInvocation, UserInputRequest, UserInputResponse,
    UserPromptSubmittedHookInput,
};
use serde::de::DeserializeOwned;
use serde_json::Value;
use std::collections::HashMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use tokio::sync::{RwLock, broadcast};

// =============================================================================
// Event Handler Types
// =============================================================================

/// Handler for session events.
pub type EventHandler = Arc<dyn Fn(&SessionEvent) + Send + Sync>;

/// Handler for permission requests.
pub type PermissionHandler =
    Arc<dyn Fn(&PermissionRequest) -> PermissionRequestResult + Send + Sync>;

/// Handler for tool invocations.
pub type ToolHandler = Arc<dyn Fn(&str, &Value) -> ToolResultObject + Send + Sync>;

/// Handler for user input requests.
pub type UserInputHandler = Arc<
    dyn Fn(&UserInputRequest, &UserInputInvocation) -> UserInputResponse + Send + Sync,
>;

/// Type alias for the invoke future.
pub type InvokeFuture =
    std::pin::Pin<Box<dyn std::future::Future<Output = Result<Value>> + Send>>;

type InvokeFn = dyn Fn(&str, Option<Value>) -> InvokeFuture + Send + Sync;

fn parse_response<T: DeserializeOwned>(method: &str, result: Value) -> Result<T> {
    serde_json::from_value(result).map_err(|e| {
        CopilotError::Protocol(format!("Failed to parse {} response: {}", method, e))
    })
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else if let Some(message) = payload.downcast_ref::<&'static str>() {
        (*message).to_string()
    } else {
        "panic without message".to_string()
    }
}

fn tool_response_payload(
    session_id: &str,
    request_id: &str,
    result: ToolResultObject,
) -> Value {
    if result.result_type == "failure" {
        if let Some(error) = result.error.clone().filter(|error| !error.is_empty()) {
            return serde_json::json!({
                "sessionId": session_id,
                "requestId": request_id,
                "error": error,
            });
        }
    }

    serde_json::json!({
        "sessionId": session_id,
        "requestId": request_id,
        "result": result,
    })
}

async fn handle_broadcast_request_event(
    session_id: String,
    state: Arc<RwLock<SessionState>>,
    invoke_fn: Arc<InvokeFn>,
    event: SessionEvent,
) {
    match event.data {
        SessionEventData::ExternalToolRequested(data) => {
            let handler = {
                let state = state.read().await;
                state
                    .tools
                    .get(&data.tool_name)
                    .and_then(|registered| registered.handler.clone())
            };

            let Some(handler) = handler else {
                return;
            };

            let tool_name = data.tool_name;
            let request_id = data.request_id;
            let arguments = data.arguments.unwrap_or(Value::Null);
            let payload = match catch_unwind(AssertUnwindSafe(|| {
                handler(&tool_name, &arguments)
            })) {
                Ok(result) => tool_response_payload(&session_id, &request_id, result),
                Err(panic) => serde_json::json!({
                    "sessionId": session_id,
                    "requestId": request_id,
                    "error": format!("tool panic: {}", panic_message(panic)),
                }),
            };

            let _ =
                (invoke_fn)("session.tools.handlePendingToolCall", Some(payload)).await;
        }
        SessionEventData::PermissionRequested(data) => {
            let handler = {
                let state = state.read().await;
                state.permission_handler.clone()
            };

            let Some(handler) = handler else {
                return;
            };

            let result =
                catch_unwind(AssertUnwindSafe(|| handler(&data.permission_request)))
                    .unwrap_or_else(|_| PermissionRequestResult::denied());
            let payload = serde_json::json!({
                "sessionId": session_id,
                "requestId": data.request_id,
                "result": result,
            });

            let _ = (invoke_fn)(
                "session.permissions.handlePendingPermissionRequest",
                Some(payload),
            )
            .await;
        }
        _ => {}
    }
}

// =============================================================================
// Event Subscription
// =============================================================================

/// A subscription to session events.
///
/// Events are delivered via the broadcast channel receiver.
pub struct EventSubscription {
    pub receiver: broadcast::Receiver<SessionEvent>,
}

impl EventSubscription {
    /// Receive the next event.
    pub async fn recv(
        &mut self,
    ) -> std::result::Result<SessionEvent, broadcast::error::RecvError> {
        self.receiver.recv().await
    }
}

// =============================================================================
// Registered Tool
// =============================================================================

/// A tool registered with the session, including its handler.
#[derive(Clone)]
pub struct RegisteredTool {
    /// Tool definition.
    pub tool: Tool,
    /// Handler for tool invocations.
    pub handler: Option<ToolHandler>,
}

// =============================================================================
// Session
// =============================================================================

/// Shared session state.
struct SessionState {
    /// Registered tools.
    tools: HashMap<String, RegisteredTool>,
    /// Permission handler.
    permission_handler: Option<PermissionHandler>,
    /// User input handler.
    user_input_handler: Option<UserInputHandler>,
    /// Session hooks.
    hooks: Option<SessionHooks>,
    /// Callback-based event handlers.
    event_handlers: HashMap<u64, EventHandler>,
    /// Next handler ID.
    next_handler_id: AtomicU64,
}

/// A Copilot conversation session.
///
/// Sessions maintain conversation state, handle events, and manage tool execution.
///
/// # Example
///
/// ```no_run
/// use copilot_sdk::{Client, SessionConfig, SessionEventData};
///
/// #[tokio::main]
/// async fn main() -> copilot_sdk::Result<()> {
/// let client = Client::builder().build()?;
/// client.start().await?;
/// let session = client.create_session(SessionConfig::default()).await?;
///
/// // Subscribe to events
/// let mut events = session.subscribe();
///
/// // Send a message
/// session.send("Hello!").await?;
///
/// // Process events
/// while let Ok(event) = events.recv().await {
///     match &event.data {
///         SessionEventData::AssistantMessage(msg) => println!("{}", msg.content),
///         SessionEventData::SessionIdle(_) => break,
///         _ => {}
///     }
/// }
/// client.stop().await;
/// # Ok(())
/// # }
/// ```
pub struct Session {
    /// Session ID.
    session_id: String,
    /// Workspace path for infinite sessions.
    workspace_path: Option<String>,
    /// Event broadcaster.
    event_tx: broadcast::Sender<SessionEvent>,
    /// Session state.
    state: Arc<RwLock<SessionState>>,
    /// JSON-RPC invoke function (injected by Client).
    invoke_fn: Arc<InvokeFn>,
}

impl Session {
    /// Create a new session.
    ///
    /// This is typically called by the Client when creating a session.
    pub fn new<F>(
        session_id: String,
        workspace_path: Option<String>,
        invoke_fn: F,
    ) -> Self
    where
        F: Fn(&str, Option<Value>) -> InvokeFuture + Send + Sync + 'static,
    {
        let (event_tx, _) = broadcast::channel(1024);

        Self {
            session_id,
            workspace_path,
            event_tx,
            state: Arc::new(RwLock::new(SessionState {
                tools: HashMap::new(),
                permission_handler: None,
                user_input_handler: None,
                hooks: None,
                event_handlers: HashMap::new(),
                next_handler_id: AtomicU64::new(1),
            })),
            invoke_fn: Arc::new(invoke_fn),
        }
    }

    // =========================================================================
    // Session Properties
    // =========================================================================

    /// Get the session ID.
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Get the workspace path for infinite sessions.
    ///
    /// Contains checkpoints/, plan.md, and files/ subdirectories.
    /// Returns None if infinite sessions are disabled.
    pub fn workspace_path(&self) -> Option<&str> {
        self.workspace_path.as_deref()
    }

    // =========================================================================
    // Event Handling
    // =========================================================================

    /// Subscribe to session events.
    ///
    /// Returns a receiver that will receive all session events.
    pub fn subscribe(&self) -> EventSubscription {
        EventSubscription {
            receiver: self.event_tx.subscribe(),
        }
    }

    /// Register a callback-based event handler.
    ///
    /// Returns an unsubscribe closure. Call it to remove the handler.
    /// Alternatively, use [`Self::off`] with the internal handler ID.
    pub async fn on<F>(&self, handler: F) -> impl FnOnce()
    where
        F: Fn(&SessionEvent) + Send + Sync + 'static,
    {
        let mut state = self.state.write().await;
        let id = state.next_handler_id.fetch_add(1, Ordering::SeqCst);
        state.event_handlers.insert(id, Arc::new(handler));

        let state_ref = Arc::clone(&self.state);
        move || {
            tokio::spawn(async move {
                state_ref.write().await.event_handlers.remove(&id);
            });
        }
    }

    /// Unsubscribe a callback-based event handler.
    pub async fn off(&self, handler_id: u64) {
        let mut state = self.state.write().await;
        state.event_handlers.remove(&handler_id);
    }

    /// Dispatch an event to all subscribers.
    ///
    /// Broadcast request events are also adapted to the same tool and permission
    /// handlers used by legacy v2 RPC callbacks.
    pub async fn dispatch_event(&self, event: SessionEvent) {
        if matches!(
            &event.data,
            SessionEventData::ExternalToolRequested(_)
                | SessionEventData::PermissionRequested(_)
        ) {
            let state = Arc::clone(&self.state);
            let invoke_fn = Arc::clone(&self.invoke_fn);
            let session_id = self.session_id.clone();
            tokio::spawn(handle_broadcast_request_event(
                session_id,
                state,
                invoke_fn,
                event.clone(),
            ));
        }

        // Send to broadcast channel
        let _ = self.event_tx.send(event.clone());

        // Call registered handlers
        let state = self.state.read().await;
        for handler in state.event_handlers.values() {
            handler(&event);
        }
    }

    // =========================================================================
    // Messaging
    // =========================================================================

    /// Send a message to the session.
    ///
    /// Returns the message ID.
    pub async fn send(&self, options: impl Into<MessageOptions>) -> Result<String> {
        let options = options.into();
        let params = serde_json::json!({
            "sessionId": self.session_id,
            "prompt": options.prompt,
            "attachments": options.attachments,
            "mode": options.mode,
        });

        let result = (self.invoke_fn)("session.send", Some(params)).await?;

        result
            .get("messageId")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| CopilotError::Protocol("Missing messageId in response".into()))
    }

    /// Abort the current message processing.
    pub async fn abort(&self) -> Result<()> {
        let params = serde_json::json!({
            "sessionId": self.session_id,
        });

        (self.invoke_fn)("session.abort", Some(params)).await?;
        Ok(())
    }

    /// Get all messages in the session.
    pub async fn get_messages(&self) -> Result<Vec<SessionEvent>> {
        let params = serde_json::json!({
            "sessionId": self.session_id,
        });

        let result = (self.invoke_fn)("session.getMessages", Some(params)).await?;

        let events: Vec<SessionEvent> = result
            .get("events")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| SessionEvent::from_json(v).ok())
                    .collect()
            })
            .or_else(|| {
                result
                    .get("messages")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| SessionEvent::from_json(v).ok())
                            .collect()
                    })
            })
            .ok_or_else(|| {
                CopilotError::Protocol("Missing events in getMessages response".into())
            })?;

        Ok(events)
    }

    /// Get the current session mode.
    pub async fn mode_get(&self) -> Result<SessionModeGetResult> {
        let params = serde_json::json!({
            "sessionId": self.session_id,
        });
        let result = (self.invoke_fn)("session.mode.get", Some(params)).await?;
        parse_response("session.mode.get", result)
    }

    /// Convenience wrapper for the active session mode.
    pub async fn get_mode(&self) -> Result<AgentMode> {
        Ok(self.mode_get().await?.mode)
    }

    /// Set the current session mode.
    pub async fn mode_set(&self, mode: AgentMode) -> Result<SessionModeSetResult> {
        let params = serde_json::json!({
            "sessionId": self.session_id,
            "mode": mode,
        });
        let result = (self.invoke_fn)("session.mode.set", Some(params)).await?;
        parse_response("session.mode.set", result)
    }

    /// Convenience wrapper for switching the active session mode.
    pub async fn set_mode(&self, mode: AgentMode) -> Result<AgentMode> {
        Ok(self.mode_set(mode).await?.mode)
    }

    /// Get the current session model.
    pub async fn model_get_current(&self) -> Result<SessionModelGetCurrentResult> {
        let params = serde_json::json!({
            "sessionId": self.session_id,
        });
        let result = (self.invoke_fn)("session.model.getCurrent", Some(params)).await?;
        parse_response("session.model.getCurrent", result)
    }

    /// Convenience wrapper for the active model identifier.
    pub async fn get_current_model(&self) -> Result<Option<String>> {
        Ok(self.model_get_current().await?.model_id)
    }

    /// Switch the session to a different model.
    pub async fn model_switch_to(
        &self,
        model_id: &str,
        reasoning_effort: Option<&str>,
    ) -> Result<SessionModelSwitchToResult> {
        let params = serde_json::json!({
            "sessionId": self.session_id,
            "modelId": model_id,
            "reasoningEffort": reasoning_effort,
        });
        let result = (self.invoke_fn)("session.model.switchTo", Some(params)).await?;
        parse_response("session.model.switchTo", result)
    }

    /// Convenience wrapper for switching models and returning the active model.
    pub async fn switch_model(
        &self,
        model_id: &str,
        reasoning_effort: Option<&str>,
    ) -> Result<Option<String>> {
        Ok(self
            .model_switch_to(model_id, reasoning_effort)
            .await?
            .model_id)
    }

    /// Emit a session log entry into the timeline.
    pub async fn log_event(
        &self,
        message: &str,
        level: Option<SessionLogLevel>,
        ephemeral: Option<bool>,
    ) -> Result<SessionLogResult> {
        let params = serde_json::json!({
            "sessionId": self.session_id,
            "message": message,
            "level": level,
            "ephemeral": ephemeral,
        });
        let result = (self.invoke_fn)("session.log", Some(params)).await?;
        parse_response("session.log", result)
    }

    /// Convenience wrapper that returns the emitted session log event ID.
    pub async fn log(
        &self,
        message: &str,
        level: Option<SessionLogLevel>,
        ephemeral: Option<bool>,
    ) -> Result<String> {
        Ok(self.log_event(message, level, ephemeral).await?.event_id)
    }

    /// Read the workspace `plan.md` file for an infinite session.
    pub async fn plan_read(&self) -> Result<SessionPlanReadResult> {
        let params = serde_json::json!({
            "sessionId": self.session_id,
        });
        let result = (self.invoke_fn)("session.plan.read", Some(params)).await?;
        parse_response("session.plan.read", result)
    }

    /// Update the workspace `plan.md` file for an infinite session.
    pub async fn plan_update(&self, content: &str) -> Result<()> {
        let params = serde_json::json!({
            "sessionId": self.session_id,
            "content": content,
        });
        (self.invoke_fn)("session.plan.update", Some(params)).await?;
        Ok(())
    }

    /// Delete the workspace `plan.md` file for an infinite session.
    pub async fn plan_delete(&self) -> Result<()> {
        let params = serde_json::json!({
            "sessionId": self.session_id,
        });
        (self.invoke_fn)("session.plan.delete", Some(params)).await?;
        Ok(())
    }

    /// List relative file paths within the workspace `files/` directory.
    pub async fn workspace_list_files(&self) -> Result<Vec<String>> {
        let params = serde_json::json!({
            "sessionId": self.session_id,
        });
        let result =
            (self.invoke_fn)("session.workspace.listFiles", Some(params)).await?;
        Ok(parse_response::<SessionWorkspaceListFilesResult>(
            "session.workspace.listFiles",
            result,
        )?
        .files)
    }

    /// Read a file from the workspace `files/` directory.
    pub async fn workspace_read_file(&self, path: &str) -> Result<String> {
        let params = serde_json::json!({
            "sessionId": self.session_id,
            "path": path,
        });
        let result = (self.invoke_fn)("session.workspace.readFile", Some(params)).await?;
        Ok(parse_response::<SessionWorkspaceReadFileResult>(
            "session.workspace.readFile",
            result,
        )?
        .content)
    }

    /// Create or overwrite a file in the workspace `files/` directory.
    pub async fn workspace_create_file(&self, path: &str, content: &str) -> Result<()> {
        let params = serde_json::json!({
            "sessionId": self.session_id,
            "path": path,
            "content": content,
        });
        (self.invoke_fn)("session.workspace.createFile", Some(params)).await?;
        Ok(())
    }

    /// Start fleet mode for the session.
    pub async fn fleet_start(&self, prompt: Option<&str>) -> Result<bool> {
        let params = serde_json::json!({
            "sessionId": self.session_id,
            "prompt": prompt,
        });
        let result = (self.invoke_fn)("session.fleet.start", Some(params)).await?;
        Ok(
            parse_response::<SessionFleetStartResult>("session.fleet.start", result)?
                .started,
        )
    }

    /// List available custom agents for the session.
    pub async fn agent_list(&self) -> Result<Vec<SessionAgentInfo>> {
        let params = serde_json::json!({
            "sessionId": self.session_id,
        });
        let result = (self.invoke_fn)("session.agent.list", Some(params)).await?;
        Ok(
            parse_response::<SessionAgentListResult>("session.agent.list", result)?
                .agents,
        )
    }

    /// Get the currently selected custom agent, if any.
    pub async fn agent_get_current(&self) -> Result<Option<SessionAgentInfo>> {
        let params = serde_json::json!({
            "sessionId": self.session_id,
        });
        let result = (self.invoke_fn)("session.agent.getCurrent", Some(params)).await?;
        Ok(parse_response::<SessionAgentGetCurrentResult>(
            "session.agent.getCurrent",
            result,
        )?
        .agent)
    }

    /// Select a custom agent for the session.
    pub async fn agent_select(&self, name: &str) -> Result<SessionAgentInfo> {
        let params = serde_json::json!({
            "sessionId": self.session_id,
            "name": name,
        });
        let result = (self.invoke_fn)("session.agent.select", Some(params)).await?;
        Ok(
            parse_response::<SessionAgentSelectResult>("session.agent.select", result)?
                .agent,
        )
    }

    /// Deselect the current custom agent and return to the default agent.
    pub async fn agent_deselect(&self) -> Result<()> {
        let params = serde_json::json!({
            "sessionId": self.session_id,
        });
        (self.invoke_fn)("session.agent.deselect", Some(params)).await?;
        Ok(())
    }

    /// Trigger manual compaction for the session.
    pub async fn compaction_compact(&self) -> Result<SessionCompactionCompactResult> {
        let params = serde_json::json!({
            "sessionId": self.session_id,
        });
        let result = (self.invoke_fn)("session.compaction.compact", Some(params)).await?;
        parse_response("session.compaction.compact", result)
    }

    // =========================================================================
    // Tool Management
    // =========================================================================

    /// Register a tool with this session.
    pub async fn register_tool(&self, tool: Tool) {
        self.register_tool_with_handler(tool, None).await;
    }

    /// Register a tool with a handler.
    pub async fn register_tool_with_handler(
        &self,
        tool: Tool,
        handler: Option<ToolHandler>,
    ) {
        let mut state = self.state.write().await;
        let name = tool.name.clone();
        state.tools.insert(name, RegisteredTool { tool, handler });
    }

    /// Register multiple tools.
    pub async fn register_tools(&self, tools: Vec<Tool>) {
        let mut state = self.state.write().await;
        for tool in tools {
            let name = tool.name.clone();
            state.tools.insert(
                name,
                RegisteredTool {
                    tool,
                    handler: None,
                },
            );
        }
    }

    /// Get a registered tool by name.
    pub async fn get_tool(&self, name: &str) -> Option<Tool> {
        let state = self.state.read().await;
        state.tools.get(name).map(|rt| rt.tool.clone())
    }

    /// Get all registered tools.
    pub async fn get_tools(&self) -> Vec<Tool> {
        let state = self.state.read().await;
        state.tools.values().map(|rt| rt.tool.clone()).collect()
    }

    /// Invoke a tool handler.
    pub async fn invoke_tool(
        &self,
        name: &str,
        arguments: &Value,
    ) -> Result<ToolResultObject> {
        let state = self.state.read().await;
        let registered = state
            .tools
            .get(name)
            .ok_or_else(|| CopilotError::ToolNotFound(name.to_string()))?;

        let handler = registered.handler.as_ref().ok_or_else(|| {
            CopilotError::ToolError(format!("No handler for tool: {}", name))
        })?;

        Ok(handler(name, arguments))
    }

    // =========================================================================
    // Permission Handling
    // =========================================================================

    /// Register a permission handler.
    pub async fn register_permission_handler<F>(&self, handler: F)
    where
        F: Fn(&PermissionRequest) -> PermissionRequestResult + Send + Sync + 'static,
    {
        let mut state = self.state.write().await;
        state.permission_handler = Some(Arc::new(handler));
    }

    /// Handle a permission request.
    ///
    /// Delegates to the registered permission handler, or denies by default
    /// if no handler is set.
    pub async fn handle_permission_request(
        &self,
        request: &PermissionRequest,
    ) -> PermissionRequestResult {
        let state = self.state.read().await;

        if let Some(handler) = &state.permission_handler {
            handler(request)
        } else {
            // Default: deny all permissions
            PermissionRequestResult::denied()
        }
    }

    // =========================================================================
    // User Input Handling
    // =========================================================================

    /// Register a handler for user input requests from the server.
    pub async fn register_user_input_handler<F>(&self, handler: F)
    where
        F: Fn(&UserInputRequest, &UserInputInvocation) -> UserInputResponse
            + Send
            + Sync
            + 'static,
    {
        let mut state = self.state.write().await;
        state.user_input_handler = Some(Arc::new(handler));
    }

    /// Handle a user input request from the server.
    pub async fn handle_user_input_request(
        &self,
        request: &UserInputRequest,
    ) -> Result<UserInputResponse> {
        let state = self.state.read().await;
        if let Some(handler) = &state.user_input_handler {
            let invocation = UserInputInvocation {
                session_id: self.session_id.clone(),
            };
            Ok(handler(request, &invocation))
        } else {
            Err(CopilotError::Protocol(
                "No user input handler registered".into(),
            ))
        }
    }

    /// Check if a user input handler is registered.
    pub async fn has_user_input_handler(&self) -> bool {
        let state = self.state.read().await;
        state.user_input_handler.is_some()
    }

    // =========================================================================
    // Hooks
    // =========================================================================

    /// Register session hooks.
    pub async fn register_hooks(&self, hooks: SessionHooks) {
        let mut state = self.state.write().await;
        state.hooks = Some(hooks);
    }

    /// Check if any hooks are registered.
    pub async fn has_hooks(&self) -> bool {
        let state = self.state.read().await;
        state.hooks.as_ref().is_some_and(|h| h.has_any())
    }

    /// Handle a `hooks.invoke` callback from the server.
    ///
    /// Dispatches to the appropriate hook handler based on `hook_type` and returns
    /// the serialized output JSON.
    pub async fn handle_hooks_invoke(
        &self,
        hook_type: &str,
        input: &Value,
    ) -> Result<Value> {
        let state = self.state.read().await;
        let hooks = match &state.hooks {
            Some(h) => h,
            None => return Ok(Value::Null),
        };

        match hook_type {
            "preToolUse" => {
                if let Some(handler) = &hooks.on_pre_tool_use {
                    let hook_input: PreToolUseHookInput =
                        serde_json::from_value(input.clone()).map_err(|e| {
                            CopilotError::Protocol(format!(
                                "Invalid preToolUse input: {}",
                                e
                            ))
                        })?;
                    let output = handler(hook_input);
                    Ok(serde_json::to_value(output).unwrap_or(Value::Null))
                } else {
                    Ok(Value::Null)
                }
            }
            "postToolUse" => {
                if let Some(handler) = &hooks.on_post_tool_use {
                    let hook_input: PostToolUseHookInput =
                        serde_json::from_value(input.clone()).map_err(|e| {
                            CopilotError::Protocol(format!(
                                "Invalid postToolUse input: {}",
                                e
                            ))
                        })?;
                    let output = handler(hook_input);
                    Ok(serde_json::to_value(output).unwrap_or(Value::Null))
                } else {
                    Ok(Value::Null)
                }
            }
            "userPromptSubmitted" => {
                if let Some(handler) = &hooks.on_user_prompt_submitted {
                    let hook_input: UserPromptSubmittedHookInput =
                        serde_json::from_value(input.clone()).map_err(|e| {
                            CopilotError::Protocol(format!(
                                "Invalid userPromptSubmitted input: {}",
                                e
                            ))
                        })?;
                    let output = handler(hook_input);
                    Ok(serde_json::to_value(output).unwrap_or(Value::Null))
                } else {
                    Ok(Value::Null)
                }
            }
            "sessionStart" => {
                if let Some(handler) = &hooks.on_session_start {
                    let hook_input: SessionStartHookInput =
                        serde_json::from_value(input.clone()).map_err(|e| {
                            CopilotError::Protocol(format!(
                                "Invalid sessionStart input: {}",
                                e
                            ))
                        })?;
                    let output = handler(hook_input);
                    Ok(serde_json::to_value(output).unwrap_or(Value::Null))
                } else {
                    Ok(Value::Null)
                }
            }
            "sessionEnd" => {
                if let Some(handler) = &hooks.on_session_end {
                    let hook_input: SessionEndHookInput =
                        serde_json::from_value(input.clone()).map_err(|e| {
                            CopilotError::Protocol(format!(
                                "Invalid sessionEnd input: {}",
                                e
                            ))
                        })?;
                    let output = handler(hook_input);
                    Ok(serde_json::to_value(output).unwrap_or(Value::Null))
                } else {
                    Ok(Value::Null)
                }
            }
            "errorOccurred" => {
                if let Some(handler) = &hooks.on_error_occurred {
                    let hook_input: ErrorOccurredHookInput =
                        serde_json::from_value(input.clone()).map_err(|e| {
                            CopilotError::Protocol(format!(
                                "Invalid errorOccurred input: {}",
                                e
                            ))
                        })?;
                    let output = handler(hook_input);
                    Ok(serde_json::to_value(output).unwrap_or(Value::Null))
                } else {
                    Ok(Value::Null)
                }
            }
            _ => Ok(Value::Null),
        }
    }

    // =========================================================================
    // Lifecycle
    // =========================================================================

    async fn clear_local_state(&self) {
        let mut state = self.state.write().await;
        state.tools.clear();
        state.permission_handler = None;
        state.user_input_handler = None;
        state.hooks = None;
        state.event_handlers.clear();
    }

    /// Disconnect the session and release local in-memory handlers.
    ///
    /// Session state on disk is preserved and can be resumed later.
    pub async fn disconnect(&self) -> Result<()> {
        let params = serde_json::json!({
            "sessionId": self.session_id,
        });

        (self.invoke_fn)("session.destroy", Some(params)).await?;
        self.clear_local_state().await;
        Ok(())
    }

    /// Destroy the session and release local in-memory handlers.
    pub async fn destroy(&self) -> Result<()> {
        self.disconnect().await?;
        Ok(())
    }
}

// =============================================================================
// Convenience methods for waiting on events
// =============================================================================

impl Session {
    /// Default timeout for waiting on session events (60 seconds).
    const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);

    /// Wait for the session to become idle.
    ///
    /// Returns the last assistant message event, or None if no message was received.
    /// Uses the specified timeout, or 60 seconds if None.
    pub async fn wait_for_idle(
        &self,
        timeout: Option<Duration>,
    ) -> Result<Option<SessionEvent>> {
        let timeout = timeout.unwrap_or(Self::DEFAULT_TIMEOUT);
        let mut subscription = self.subscribe();
        let mut last_assistant_message: Option<SessionEvent> = None;

        let result = tokio::time::timeout(timeout, async {
            loop {
                match subscription.recv().await {
                    Ok(event) => match &event.data {
                        SessionEventData::AssistantMessage(_) => {
                            last_assistant_message = Some(event);
                        }
                        SessionEventData::AssistantMessageDelta(_) => {
                            // Deltas are intermediate; we track the full message
                        }
                        SessionEventData::SessionIdle(_) => {
                            break;
                        }
                        SessionEventData::SessionError(err) => {
                            return Err(CopilotError::Protocol(format!(
                                "Session error: {}",
                                err.message
                            )));
                        }
                        _ => {}
                    },
                    Err(broadcast::error::RecvError::Closed) => {
                        return Err(CopilotError::ConnectionClosed);
                    }
                    Err(broadcast::error::RecvError::Lagged(_)) => {
                        // Continue - we missed some events but can recover
                    }
                }
            }
            Ok(())
        })
        .await;

        match result {
            Ok(Ok(())) => Ok(last_assistant_message),
            Ok(Err(e)) => Err(e),
            Err(_) => Err(CopilotError::Timeout(timeout)),
        }
    }

    /// Send a message and wait for the complete response.
    ///
    /// Returns the last `AssistantMessage` event, or `None` if session
    /// became idle without producing an assistant message.
    /// Uses the specified timeout, or 60 seconds if None.
    pub async fn send_and_wait(
        &self,
        options: impl Into<MessageOptions>,
        timeout: Option<Duration>,
    ) -> Result<Option<SessionEvent>> {
        self.send(options).await?;
        self.wait_for_idle(timeout).await
    }

    /// Send a message and wait for the response content as a string.
    ///
    /// Convenience method that collects all assistant message/delta content.
    /// Uses the specified timeout, or 60 seconds if None.
    pub async fn send_and_collect(
        &self,
        options: impl Into<MessageOptions>,
        timeout: Option<Duration>,
    ) -> Result<String> {
        let timeout = timeout.unwrap_or(Self::DEFAULT_TIMEOUT);
        self.send(options).await?;

        let mut subscription = self.subscribe();
        let mut content = String::new();

        let result = tokio::time::timeout(timeout, async {
            loop {
                match subscription.recv().await {
                    Ok(event) => match &event.data {
                        SessionEventData::AssistantMessage(msg) => {
                            content.push_str(&msg.content);
                        }
                        SessionEventData::AssistantMessageDelta(delta) => {
                            content.push_str(&delta.delta_content);
                        }
                        SessionEventData::SessionIdle(_) => {
                            break;
                        }
                        SessionEventData::SessionError(err) => {
                            return Err(CopilotError::Protocol(format!(
                                "Session error: {}",
                                err.message
                            )));
                        }
                        _ => {}
                    },
                    Err(broadcast::error::RecvError::Closed) => {
                        return Err(CopilotError::ConnectionClosed);
                    }
                    Err(broadcast::error::RecvError::Lagged(_)) => {}
                }
            }
            Ok(())
        })
        .await;

        match result {
            Ok(Ok(())) => Ok(content),
            Ok(Err(e)) => Err(e),
            Err(_) => Err(CopilotError::Timeout(timeout)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use tokio::sync::mpsc;
    use tokio::time::timeout;

    fn mock_invoke(_method: &str, _params: Option<Value>) -> InvokeFuture {
        Box::pin(async { Ok(serde_json::json!({"messageId": "test-msg-123"})) })
    }

    fn mock_invoke_with_events(method: &str, _params: Option<Value>) -> InvokeFuture {
        let method = method.to_string();
        Box::pin(async move {
            if method == "session.getMessages" {
                return Ok(serde_json::json!({
                    "events": [{
                        "id": "evt-1",
                        "timestamp": "2024-01-01T00:00:00Z",
                        "type": "session.idle",
                        "data": {}
                    }]
                }));
            }
            Ok(serde_json::json!({"messageId": "test-msg-123"}))
        })
    }

    fn recording_invoke(
        tx: mpsc::UnboundedSender<(String, Option<Value>)>,
    ) -> impl Fn(&str, Option<Value>) -> InvokeFuture + Send + Sync + 'static {
        move |method: &str, params: Option<Value>| {
            let method = method.to_string();
            let tx = tx.clone();
            Box::pin(async move {
                let _ = tx.send((method, params));
                Ok(serde_json::json!({ "success": true }))
            })
        }
    }

    #[tokio::test]
    async fn test_session_id() {
        let session = Session::new("test-session-123".to_string(), None, mock_invoke);
        assert_eq!(session.session_id(), "test-session-123");
    }

    #[tokio::test]
    async fn test_workspace_path() {
        let session = Session::new(
            "test".to_string(),
            Some("/tmp/workspace".to_string()),
            mock_invoke,
        );
        assert_eq!(session.workspace_path(), Some("/tmp/workspace"));
    }

    #[tokio::test]
    async fn test_register_tool() {
        let session = Session::new("test".to_string(), None, mock_invoke);

        let tool = Tool::new("my_tool").description("A test tool");

        session.register_tool(tool.clone()).await;

        let retrieved = session.get_tool("my_tool").await;
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().name, "my_tool");
    }

    #[tokio::test]
    async fn test_register_tool_with_handler() {
        let session = Session::new("test".to_string(), None, mock_invoke);

        let tool = Tool::new("echo").description("Echo tool");
        let handler: ToolHandler = Arc::new(|_name, args| {
            let text = args.get("text").and_then(|v| v.as_str()).unwrap_or("empty");
            ToolResultObject::text(text)
        });

        session
            .register_tool_with_handler(tool, Some(handler))
            .await;

        let result = session
            .invoke_tool("echo", &serde_json::json!({"text": "hello"}))
            .await
            .unwrap();

        assert_eq!(result.text_result_for_llm, "hello");
    }

    #[tokio::test]
    async fn test_invoke_unknown_tool() {
        let session = Session::new("test".to_string(), None, mock_invoke);

        let result = session.invoke_tool("unknown", &serde_json::json!({})).await;

        assert!(matches!(result, Err(CopilotError::ToolNotFound(_))));
    }

    #[tokio::test]
    async fn test_event_subscription() {
        let session = Session::new("test".to_string(), None, mock_invoke);

        let mut sub1 = session.subscribe();
        let mut sub2 = session.subscribe();

        // Dispatch an event
        let event = SessionEvent::from_json(&serde_json::json!({
            "id": "evt-1",
            "timestamp": "2024-01-01T00:00:00Z",
            "type": "session.idle",
            "data": {}
        }))
        .unwrap();

        session.dispatch_event(event).await;

        // Both subscribers should receive it
        let received1 = sub1.recv().await.unwrap();
        let received2 = sub2.recv().await.unwrap();

        assert_eq!(received1.id, "evt-1");
        assert_eq!(received2.id, "evt-1");
    }

    #[tokio::test]
    async fn test_callback_handler() {
        let session = Session::new("test".to_string(), None, mock_invoke);
        let call_count = Arc::new(AtomicUsize::new(0));

        let count_clone = Arc::clone(&call_count);
        let unsubscribe = session
            .on(move |_event| {
                count_clone.fetch_add(1, Ordering::SeqCst);
            })
            .await;

        // Dispatch events
        let event = SessionEvent::from_json(&serde_json::json!({
            "id": "evt-callback-1",
            "timestamp": "2024-01-01T00:00:00Z",
            "type": "session.idle",
            "data": {}
        }))
        .unwrap();

        session.dispatch_event(event).await;

        assert_eq!(call_count.load(Ordering::SeqCst), 1);

        // Unsubscribe
        unsubscribe();
    }

    #[tokio::test]
    async fn test_dispatch_event_handles_external_tool_requested_v3() {
        let (rpc_tx, mut rpc_rx) = mpsc::unbounded_channel();
        let session = Session::new("test".to_string(), None, recording_invoke(rpc_tx));
        let mut subscription = session.subscribe();

        session
            .register_tool_with_handler(
                Tool::new("echo"),
                Some(Arc::new(|_, args| {
                    ToolResultObject::text(
                        args.get("text")
                            .and_then(|value| value.as_str())
                            .unwrap_or("missing"),
                    )
                })),
            )
            .await;

        let event = SessionEvent::from_json(&serde_json::json!({
            "id": "evt-external-tool",
            "timestamp": "2024-01-01T00:00:00Z",
            "type": "external_tool.requested",
            "data": {
                "requestId": "request-1",
                "sessionId": "test",
                "toolCallId": "tool-call-1",
                "toolName": "echo",
                "arguments": { "text": "hello" }
            }
        }))
        .unwrap();

        session.dispatch_event(event.clone()).await;

        let received_event = subscription.recv().await.unwrap();
        assert_eq!(received_event.id, event.id);

        let (method, params) = timeout(Duration::from_secs(1), rpc_rx.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(method, "session.tools.handlePendingToolCall");
        assert_eq!(
            params.unwrap(),
            serde_json::json!({
                "sessionId": "test",
                "requestId": "request-1",
                "result": {
                    "textResultForLlm": "hello",
                    "resultType": "success"
                }
            })
        );
    }

    #[tokio::test]
    async fn test_dispatch_event_ignores_external_tool_requested_without_handler() {
        let (rpc_tx, mut rpc_rx) = mpsc::unbounded_channel();
        let session = Session::new("test".to_string(), None, recording_invoke(rpc_tx));

        let event = SessionEvent::from_json(&serde_json::json!({
            "id": "evt-external-tool-miss",
            "timestamp": "2024-01-01T00:00:00Z",
            "type": "external_tool.requested",
            "data": {
                "requestId": "request-2",
                "sessionId": "test",
                "toolCallId": "tool-call-2",
                "toolName": "missing"
            }
        }))
        .unwrap();

        session.dispatch_event(event).await;
        tokio::task::yield_now().await;

        assert!(
            timeout(Duration::from_millis(50), rpc_rx.recv())
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn test_permission_handler() {
        let session = Session::new("test".to_string(), None, mock_invoke);

        // Default handler denies
        let request = PermissionRequest {
            kind: "tool_execution".to_string(),
            tool_call_id: Some("call-123".to_string()),
            extension_data: HashMap::new(),
        };
        let result = session.handle_permission_request(&request).await;
        assert!(result.kind.contains("denied"));

        // Register custom handler that approves
        session
            .register_permission_handler(|_req| PermissionRequestResult::approved())
            .await;

        let result = session.handle_permission_request(&request).await;
        assert_eq!(result.kind, "approved");
    }

    #[tokio::test]
    async fn test_dispatch_event_handles_permission_requested_v3() {
        let (rpc_tx, mut rpc_rx) = mpsc::unbounded_channel();
        let session = Session::new("test".to_string(), None, recording_invoke(rpc_tx));
        let mut subscription = session.subscribe();

        session
            .register_permission_handler(|request| {
                assert_eq!(request.kind, "shell");
                PermissionRequestResult::approved()
            })
            .await;

        let event = SessionEvent::from_json(&serde_json::json!({
            "id": "evt-permission",
            "timestamp": "2024-01-01T00:00:00Z",
            "type": "permission.requested",
            "data": {
                "requestId": "request-3",
                "permissionRequest": {
                    "kind": "shell",
                    "toolCallId": "tool-call-3",
                    "fullCommandText": "ls"
                }
            }
        }))
        .unwrap();

        session.dispatch_event(event.clone()).await;

        let received_event = subscription.recv().await.unwrap();
        assert_eq!(received_event.id, event.id);

        let (method, params) = timeout(Duration::from_secs(1), rpc_rx.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(method, "session.permissions.handlePendingPermissionRequest");
        assert_eq!(
            params.unwrap(),
            serde_json::json!({
                "sessionId": "test",
                "requestId": "request-3",
                "result": {
                    "kind": "approved"
                }
            })
        );
    }

    #[tokio::test]
    async fn test_dispatch_event_ignores_permission_requested_without_handler() {
        let (rpc_tx, mut rpc_rx) = mpsc::unbounded_channel();
        let session = Session::new("test".to_string(), None, recording_invoke(rpc_tx));

        let event = SessionEvent::from_json(&serde_json::json!({
            "id": "evt-permission-miss",
            "timestamp": "2024-01-01T00:00:00Z",
            "type": "permission.requested",
            "data": {
                "requestId": "request-4",
                "permissionRequest": {
                    "kind": "shell"
                }
            }
        }))
        .unwrap();

        session.dispatch_event(event).await;
        tokio::task::yield_now().await;

        assert!(
            timeout(Duration::from_millis(50), rpc_rx.recv())
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn test_get_messages_with_events_field() {
        let session = Session::new("test".to_string(), None, mock_invoke_with_events);
        let messages = session.get_messages().await.unwrap();
        assert_eq!(messages.len(), 1);
        assert!(matches!(
            messages[0].data,
            crate::events::SessionEventData::SessionIdle(_)
        ));
    }

    #[tokio::test]
    async fn test_mode_get_uses_session_rpc() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let session = Session::new("test".to_string(), None, move |method, params| {
            let method = method.to_string();
            let params_clone = params.clone();
            let tx = tx.clone();
            Box::pin(async move {
                let _ = tx.send((method, params_clone));
                Ok(serde_json::json!({ "mode": "plan" }))
            })
        });

        let result = session.mode_get().await.unwrap();
        assert_eq!(result.mode, AgentMode::Plan);

        let (method, params) = rx.recv().await.unwrap();
        assert_eq!(method, "session.mode.get");
        assert_eq!(params.unwrap(), serde_json::json!({ "sessionId": "test" }));
    }

    #[tokio::test]
    async fn test_model_switch_to_uses_session_rpc() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let session = Session::new("test".to_string(), None, move |method, params| {
            let method = method.to_string();
            let params_clone = params.clone();
            let tx = tx.clone();
            Box::pin(async move {
                let _ = tx.send((method, params_clone));
                Ok(serde_json::json!({ "modelId": "claude-sonnet-4.5" }))
            })
        });

        let result = session
            .model_switch_to("claude-sonnet-4.5", Some("high"))
            .await
            .unwrap();
        assert_eq!(result.model_id.as_deref(), Some("claude-sonnet-4.5"));

        let (method, params) = rx.recv().await.unwrap();
        assert_eq!(method, "session.model.switchTo");
        assert_eq!(
            params.unwrap(),
            serde_json::json!({
                "sessionId": "test",
                "modelId": "claude-sonnet-4.5",
                "reasoningEffort": "high"
            })
        );
    }

    #[tokio::test]
    async fn test_log_event_uses_session_rpc() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let session = Session::new("test".to_string(), None, move |method, params| {
            let method = method.to_string();
            let params_clone = params.clone();
            let tx = tx.clone();
            Box::pin(async move {
                let _ = tx.send((method, params_clone));
                Ok(serde_json::json!({ "eventId": "evt-log-1" }))
            })
        });

        let result = session
            .log_event(
                "switching to plan mode",
                Some(SessionLogLevel::Warning),
                Some(true),
            )
            .await
            .unwrap();
        assert_eq!(result.event_id, "evt-log-1");

        let (method, params) = rx.recv().await.unwrap();
        assert_eq!(method, "session.log");
        assert_eq!(
            params.unwrap(),
            serde_json::json!({
                "sessionId": "test",
                "message": "switching to plan mode",
                "level": "warning",
                "ephemeral": true
            })
        );
    }

    #[tokio::test]
    async fn test_plan_read_uses_session_rpc() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let session = Session::new("test".to_string(), None, move |method, params| {
            let method = method.to_string();
            let params_clone = params.clone();
            let tx = tx.clone();
            Box::pin(async move {
                let _ = tx.send((method, params_clone));
                Ok(serde_json::json!({
                    "exists": true,
                    "content": "phase 1",
                    "path": "/tmp/workspace/plan.md"
                }))
            })
        });

        let result = session.plan_read().await.unwrap();
        assert!(result.exists);
        assert_eq!(result.content.as_deref(), Some("phase 1"));

        let (method, params) = rx.recv().await.unwrap();
        assert_eq!(method, "session.plan.read");
        assert_eq!(params.unwrap(), serde_json::json!({ "sessionId": "test" }));
    }

    #[tokio::test]
    async fn test_workspace_create_file_uses_session_rpc() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let session = Session::new("test".to_string(), None, move |method, params| {
            let method = method.to_string();
            let params_clone = params.clone();
            let tx = tx.clone();
            Box::pin(async move {
                let _ = tx.send((method, params_clone));
                Ok(serde_json::json!({}))
            })
        });

        session
            .workspace_create_file("notes/todo.txt", "finish phase 3")
            .await
            .unwrap();

        let (method, params) = rx.recv().await.unwrap();
        assert_eq!(method, "session.workspace.createFile");
        assert_eq!(
            params.unwrap(),
            serde_json::json!({
                "sessionId": "test",
                "path": "notes/todo.txt",
                "content": "finish phase 3"
            })
        );
    }

    #[tokio::test]
    async fn test_disconnect_clears_local_handlers() {
        let session = Session::new("test".to_string(), None, move |_method, _params| {
            Box::pin(async move { Ok(serde_json::json!({})) })
        });

        session.register_tool(Tool::new("echo")).await;
        session
            .register_permission_handler(|_| PermissionRequestResult::approved())
            .await;
        session
            .register_user_input_handler(|_, _| UserInputResponse {
                answer: "ok".into(),
                was_freeform: Some(true),
            })
            .await;
        session
            .register_hooks(crate::types::SessionHooks {
                on_session_start: Some(Arc::new(|_| {
                    crate::types::SessionStartHookOutput::default()
                })),
                ..Default::default()
            })
            .await;

        session.disconnect().await.unwrap();

        assert!(session.get_tool("echo").await.is_none());
        assert!(!session.has_user_input_handler().await);
        assert!(!session.has_hooks().await);
        let denied = session
            .handle_permission_request(&PermissionRequest {
                kind: "shell".into(),
                tool_call_id: None,
                extension_data: HashMap::new(),
            })
            .await;
        assert!(denied.is_denied());
    }

    #[tokio::test]
    async fn test_agent_select_uses_session_rpc() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let session = Session::new("test".to_string(), None, move |method, params| {
            let method = method.to_string();
            let params_clone = params.clone();
            let tx = tx.clone();
            Box::pin(async move {
                let _ = tx.send((method, params_clone));
                Ok(serde_json::json!({
                    "agent": {
                        "name": "planner",
                        "displayName": "Planner",
                        "description": "Planning specialist"
                    }
                }))
            })
        });

        let result = session.agent_select("planner").await.unwrap();
        assert_eq!(result.name, "planner");

        let (method, params) = rx.recv().await.unwrap();
        assert_eq!(method, "session.agent.select");
        assert_eq!(
            params.unwrap(),
            serde_json::json!({
                "sessionId": "test",
                "name": "planner"
            })
        );
    }

    #[tokio::test]
    async fn test_compaction_compact_uses_session_rpc() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let session = Session::new("test".to_string(), None, move |method, params| {
            let method = method.to_string();
            let params_clone = params.clone();
            let tx = tx.clone();
            Box::pin(async move {
                let _ = tx.send((method, params_clone));
                Ok(serde_json::json!({
                    "success": true,
                    "tokensRemoved": 1200,
                    "messagesRemoved": 4
                }))
            })
        });

        let result = session.compaction_compact().await.unwrap();
        assert!(result.success);
        assert_eq!(result.tokens_removed, 1200);

        let (method, params) = rx.recv().await.unwrap();
        assert_eq!(method, "session.compaction.compact");
        assert_eq!(params.unwrap(), serde_json::json!({ "sessionId": "test" }));
    }

    #[tokio::test]
    async fn test_user_input_handler() {
        let session = Session::new("test".to_string(), None, mock_invoke);

        session
            .register_user_input_handler(|req, _inv| {
                assert_eq!(req.question, "What color?");
                UserInputResponse {
                    answer: "blue".into(),
                    was_freeform: Some(true),
                }
            })
            .await;

        let request = UserInputRequest {
            question: "What color?".into(),
            choices: Some(vec!["red".into(), "blue".into()]),
            allow_freeform: Some(true),
        };

        let response = session.handle_user_input_request(&request).await.unwrap();
        assert_eq!(response.answer, "blue");
        assert_eq!(response.was_freeform, Some(true));
    }

    #[tokio::test]
    async fn test_user_input_no_handler_errors() {
        let session = Session::new("test".to_string(), None, mock_invoke);

        let request = UserInputRequest {
            question: "?".into(),
            choices: None,
            allow_freeform: None,
        };

        let result = session.handle_user_input_request(&request).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_register_hooks() {
        let session = Session::new("test".to_string(), None, mock_invoke);

        assert!(!session.has_hooks().await);

        let hooks = crate::types::SessionHooks {
            on_pre_tool_use: Some(Arc::new(|input| {
                assert_eq!(input.tool_name, "my_tool");
                crate::types::PreToolUseHookOutput {
                    permission_decision: Some("allow".into()),
                    ..Default::default()
                }
            })),
            ..Default::default()
        };

        session.register_hooks(hooks).await;
        assert!(session.has_hooks().await);
    }

    #[tokio::test]
    async fn test_hooks_invoke_pre_tool_use() {
        let session = Session::new("test".to_string(), None, mock_invoke);

        let hooks = crate::types::SessionHooks {
            on_pre_tool_use: Some(Arc::new(|_input| {
                crate::types::PreToolUseHookOutput {
                    permission_decision: Some("allow".into()),
                    additional_context: Some("extra context".into()),
                    ..Default::default()
                }
            })),
            ..Default::default()
        };

        session.register_hooks(hooks).await;

        let input = serde_json::json!({
            "timestamp": 1234567890,
            "cwd": "/tmp",
            "toolName": "test_tool",
            "toolArgs": {"key": "value"}
        });

        let result = session
            .handle_hooks_invoke("preToolUse", &input)
            .await
            .unwrap();
        assert_eq!(
            result.get("permissionDecision").and_then(|v| v.as_str()),
            Some("allow")
        );
        assert_eq!(
            result.get("additionalContext").and_then(|v| v.as_str()),
            Some("extra context")
        );
    }

    #[tokio::test]
    async fn test_hooks_invoke_no_handler_returns_null() {
        let session = Session::new("test".to_string(), None, mock_invoke);

        // No hooks registered at all
        let result = session
            .handle_hooks_invoke("preToolUse", &serde_json::json!({}))
            .await
            .unwrap();
        assert!(result.is_null());

        // Hooks registered but not for this type
        let hooks = crate::types::SessionHooks {
            on_session_start: Some(Arc::new(|_input| {
                crate::types::SessionStartHookOutput::default()
            })),
            ..Default::default()
        };
        session.register_hooks(hooks).await;

        let input = serde_json::json!({
            "timestamp": 1234567890,
            "cwd": "/tmp",
            "toolName": "test_tool",
            "toolArgs": {}
        });
        let result = session
            .handle_hooks_invoke("preToolUse", &input)
            .await
            .unwrap();
        assert!(result.is_null());
    }

    #[tokio::test]
    async fn test_hooks_invoke_unknown_type_returns_null() {
        let session = Session::new("test".to_string(), None, mock_invoke);

        let hooks = crate::types::SessionHooks {
            on_pre_tool_use: Some(Arc::new(|_| {
                crate::types::PreToolUseHookOutput::default()
            })),
            ..Default::default()
        };
        session.register_hooks(hooks).await;

        let result = session
            .handle_hooks_invoke("unknownHookType", &serde_json::json!({}))
            .await
            .unwrap();
        assert!(result.is_null());
    }
}
