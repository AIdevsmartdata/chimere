//! OpenAI-compatible HTTP server for chimere-deltanet.
//!
//! Serves `/v1/chat/completions` with non-streaming and SSE streaming support.
//!
//! # Usage
//!
//! ```sh
//! CHIMERE_PORT=8090 cargo run --release --features server --bin chimere-server
//! ```
//!
//! # Curl examples
//!
//! Non-streaming:
//! ```sh
//! curl -s http://localhost:8090/v1/chat/completions \
//!   -H 'Content-Type: application/json' \
//!   -d '{"messages":[{"role":"user","content":"Say hello"}],"max_tokens":64}'
//! ```
//!
//! Streaming:
//! ```sh
//! curl -s http://localhost:8090/v1/chat/completions \
//!   -H 'Content-Type: application/json' \
//!   -d '{"messages":[{"role":"user","content":"Say hello"}],"max_tokens":64,"stream":true}'
//! ```

use std::convert::Infallible;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Json};
use axum::routing::post;
use axum::{http::StatusCode, Router};
use futures::stream;
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;
use tokio::sync::Mutex;

use crate::chimere_model::{ChimereModel, ModelArch};
use crate::generate::{generate_text, generate_text_generic};
use crate::generic_model::GenericModel;
use crate::mtp_scheduler::{SamplingParams, generate_with_mtp_streaming};
use crate::qwen35_model::Qwen35Model;
use crate::state::GdnRecurrentState;

// ---------------------------------------------------------------------------
// OpenAI request / response types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    pub messages: Vec<Message>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    #[serde(default = "default_top_p")]
    pub top_p: f64,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    #[serde(default = "default_presence_penalty")]
    pub presence_penalty: f64,
    #[serde(default)]
    pub repetition_penalty: Option<f64>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub logprobs: bool,
    /// Number of top logprobs to return per token (0-5, clamped to FFI max).
    /// Only used when `logprobs` is true. Mirrors OpenAI `top_logprobs` field.
    #[serde(default)]
    pub top_logprobs: Option<usize>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub tools: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    pub user: Option<String>,
    /// OpenAI-compatible: pass enable_thinking to Qwen3.5 Jinja template.
    #[serde(default)]
    pub chat_template_kwargs: Option<ChatTemplateKwargs>,
    /// Engram table path override (per-request, e.g. from ODO pipeline).
    #[serde(default)]
    pub engram_table: Option<String>,
    /// Engram alpha (logit bias strength, 0.0-1.0).
    #[serde(default)]
    pub engram_alpha: Option<f64>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ChatTemplateKwargs {
    #[serde(default = "default_enable_thinking")]
    pub enable_thinking: bool,
}

fn default_enable_thinking() -> bool {
    true
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Message {
    pub role: String,
    /// Accepts both String and Array<{type:"text",text:"..."}> (OpenAI multimodal format).
    #[serde(deserialize_with = "deserialize_content")]
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<serde_json::Value>>,
}

/// Accept content as string OR array of content parts (OpenAI multimodal format).
/// Array parts: [{"type":"text","text":"..."}, {"type":"image_url",...}] → concatenate text parts.
fn deserialize_content<'de, D>(deserializer: D) -> Result<String, D::Error>
where D: serde::Deserializer<'de> {
    use serde::de;
    struct ContentVisitor;
    impl<'de> de::Visitor<'de> for ContentVisitor {
        type Value = String;
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            write!(f, "a string or array of content parts")
        }
        fn visit_str<E: de::Error>(self, v: &str) -> Result<String, E> {
            Ok(v.to_string())
        }
        fn visit_string<E: de::Error>(self, v: String) -> Result<String, E> {
            Ok(v)
        }
        fn visit_seq<A: de::SeqAccess<'de>>(self, mut seq: A) -> Result<String, A::Error> {
            let mut text = String::new();
            while let Some(part) = seq.next_element::<serde_json::Value>()? {
                if let Some(t) = part.get("text").and_then(|t| t.as_str()) {
                    if !text.is_empty() { text.push('\n'); }
                    text.push_str(t);
                }
            }
            Ok(text)
        }
        fn visit_none<E: de::Error>(self) -> Result<String, E> {
            Ok(String::new())
        }
        fn visit_unit<E: de::Error>(self) -> Result<String, E> {
            Ok(String::new())
        }
    }
    deserializer.deserialize_any(ContentVisitor)
}

fn default_max_tokens() -> usize {
    2048
}
const MAX_TOKENS_LIMIT: usize = 32768;
fn default_temperature() -> f64 {
    0.7
}
fn default_top_p() -> f64 {
    0.95
}
fn default_top_k() -> usize {
    20
}
fn default_presence_penalty() -> f64 {
    0.0 // presence_penalty=0 for Qwen3.5 (1.5 was WRONG — kills code gen & long reasoning)
}

// --- Non-streaming response ---

#[derive(Debug, Serialize)]
pub struct ChatResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct Choice {
    pub index: usize,
    pub message: Message,
    pub finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<LogprobContent>,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

// --- Logprobs types (OpenAI format) ---

#[derive(Debug, Serialize, Clone)]
pub struct TopLogprobEntry {
    pub token: String,
    pub logprob: f32,
    /// UTF-8 bytes of the token (OpenAI spec). None if encoding fails.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<Vec<u8>>,
}

#[derive(Debug, Serialize, Clone)]
pub struct TokenLogprobEntry {
    pub token: String,
    pub logprob: f32,
    /// UTF-8 bytes of the token (OpenAI spec). None if encoding fails.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<Vec<u8>>,
    pub top_logprobs: Vec<TopLogprobEntry>,
}

#[derive(Debug, Serialize, Clone)]
pub struct LogprobContent {
    pub content: Vec<TokenLogprobEntry>,
}

// --- Streaming delta types (SSE) ---

#[derive(Debug, Serialize)]
pub struct ChatChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
}

#[derive(Debug, Serialize)]
pub struct ChunkChoice {
    pub index: usize,
    pub delta: Delta,
    pub finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<LogprobContent>,
}

#[derive(Debug, Serialize)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

// ---------------------------------------------------------------------------
// Multi-arch model wrapper (Step 7)
// ---------------------------------------------------------------------------
//
// Closed enum with 2 variants — NOT a trait object. Keeps arch-specific
// inherent methods (Qwen35Model::device, ::config, ::reset_cudarc_state,
// ::reset_llama_state) accessible without going through the trait, while
// letting call sites up-cast to `&dyn ChimereModel` for generation helpers.
//
// Both inner types are !Sync (RefCell for libllama / cudarc state). The
// outer Mutex in AppState serialises inference — one request at a time.
pub enum AppStateModel {
    /// Qwen3.5-35B-A3B — full production stack (MTP, cudarc, block diffusion,
    /// entropy routing, Candle path, Engram-aware sampling).
    Qwen35(Qwen35Model),
    /// libllama-native architectures (Mamba-1 / Mamba-2 / Nemotron-H MoE / ...).
    /// No MTP, no DART, no block diffusion — forward via LlamaForward only.
    Generic(GenericModel),
}

impl AppStateModel {
    /// Up-cast to `&dyn ChimereModel` for calls into `generate.rs` and
    /// `mtp_scheduler.rs` which accept the trait object.
    pub fn as_trait(&self) -> &dyn ChimereModel {
        match self {
            AppStateModel::Qwen35(m) => m,
            AppStateModel::Generic(m) => m,
        }
    }

    pub fn arch(&self) -> ModelArch {
        match self {
            AppStateModel::Qwen35(m) => ChimereModel::arch(m),
            AppStateModel::Generic(m) => ChimereModel::arch(m),
        }
    }
}

// !Sync inner types — manual Send is correct because the outer Mutex
// serialises access. Same pattern as the per-type unsafe impl already
// present on Qwen35Model and GenericModel.
unsafe impl Send for AppStateModel {}

// ---------------------------------------------------------------------------
// Shared application state
//
// Inner model types contain `RefCell<...>` which makes them !Sync.
// We wrap in `Arc<Mutex<...>>` so handler tasks can share the state.
// The `Mutex` also enforces single-request inference (model is stateful).
// ---------------------------------------------------------------------------

pub struct AppState {
    /// Model is behind a Mutex because the inner types are !Sync (RefCell).
    /// The mutex also serialises inference — one request at a time.
    pub model: Mutex<Qwen35Model>,
    pub tokenizer: Arc<Tokenizer>,
    pub model_name: String,
    /// Multi-agent context switching scheduler.
    pub agent_scheduler: Mutex<crate::agent_scheduler::AgentScheduler>,
    /// Map user names → agent IDs (auto-registered on first use).
    pub user_agent_map: Mutex<std::collections::HashMap<String, usize>>,
    /// Max agents (from env CHIMERE_MAX_AGENTS, default 4).
    pub max_agents: usize,
}

// ---------------------------------------------------------------------------
// Helper: build prompt from messages using Qwen3.5 chat template
// ---------------------------------------------------------------------------

fn messages_to_prompt(messages: &[Message], tools: Option<&Vec<serde_json::Value>>, enable_thinking: bool) -> String {
    // Separate system message (must be first per Qwen3.5 template)
    let (system_msg, rest) = if messages.first().map(|m| m.role == "system").unwrap_or(false) {
        (Some(&messages[0]), &messages[1..])
    } else {
        (None, &messages[..])
    };

    let mut prompt = String::new();

    // System + Tools block (matches Qwen3.5 Jinja template exactly)
    let has_tools = tools.as_ref().map(|t| !t.is_empty()).unwrap_or(false);
    if has_tools {
        let tool_list = tools.unwrap();
        prompt.push_str("<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>\n");
        for tool in tool_list {
            prompt.push_str(&serde_json::to_string(tool).unwrap_or_default());
            prompt.push('\n');
        }
        prompt.push_str("</tools>\n\n\
            If you choose to call a function ONLY reply in the following format with NO suffix:\n\n\
            <tool_call>\n\
            <function=example_function_name>\n\
            <parameter=example_parameter_1>\nvalue_1\n</parameter>\n\
            <parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n\
            </function>\n\
            </tool_call>\n\n\
            <IMPORTANT>\n\
            Reminder:\n\
            - Function calls MUST follow the specified format\n\
            - Required parameters MUST be specified\n\
            - Only call functions when the user's request requires it\n\
            - If there is no function call available, answer normally\n\
            </IMPORTANT>");
        if let Some(sys) = system_msg {
            if !sys.content.trim().is_empty() {
                prompt.push_str("\n\n");
                prompt.push_str(sys.content.trim());
            }
        }
        prompt.push_str("<|im_end|>\n");
    } else if let Some(sys) = system_msg {
        prompt.push_str(&format!("<|im_start|>system\n{}<|im_end|>\n", sys.content.trim()));
    }

    // Message loop with tool response grouping
    let mut i = 0;
    while i < rest.len() {
        let msg = &rest[i];
        match msg.role.as_str() {
            "user" => {
                prompt.push_str(&format!("<|im_start|>user\n{}<|im_end|>\n", msg.content));
            }
            "assistant" => {
                prompt.push_str(&format!("<|im_start|>assistant\n{}", msg.content));
                // Render tool_calls in history (for multi-turn tool conversations)
                if let Some(ref tcs) = msg.tool_calls {
                    for tc in tcs {
                        let func = tc.get("function").unwrap_or(tc);
                        let name = func.get("name").and_then(|n| n.as_str()).unwrap_or("unknown");
                        prompt.push_str(&format!("\n<tool_call>\n<function={}>\n", name));
                        if let Some(args_str) = func.get("arguments").and_then(|a| a.as_str()) {
                            if let Ok(args) = serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(args_str) {
                                for (k, v) in &args {
                                    let val = if v.is_string() { v.as_str().unwrap().to_string() } else { v.to_string() };
                                    prompt.push_str(&format!("<parameter={}>\n{}\n</parameter>\n", k, val));
                                }
                            }
                        }
                        prompt.push_str("</function>\n</tool_call>");
                    }
                }
                prompt.push_str("<|im_end|>\n");
            }
            "tool" => {
                // Group consecutive tool messages under a single user block
                prompt.push_str("<|im_start|>user");
                while i < rest.len() && rest[i].role == "tool" {
                    prompt.push_str(&format!("\n<tool_response>\n{}\n</tool_response>", rest[i].content));
                    i += 1;
                }
                prompt.push_str("<|im_end|>\n");
                continue; // skip i += 1 below
            }
            _ => {}
        }
        i += 1;
    }

    // Open the assistant turn
    if enable_thinking {
        prompt.push_str("<|im_start|>assistant\n<think>\n");
    } else {
        // Qwen3.5 Jinja outputs empty <think></think> even in no-think mode.
        prompt.push_str("<|im_start|>assistant\n<think>\n\n</think>\n\n");
    }
    prompt
}

// ---------------------------------------------------------------------------
// Extract tool calls from model output
// ---------------------------------------------------------------------------

/// Parse `<tool_call><function=NAME><parameter=KEY>VALUE</parameter></function></tool_call>`
/// Returns (remaining_content, Vec<tool_call_json>).
fn extract_tool_calls(text: &str) -> (String, Vec<serde_json::Value>) {
    let mut tool_calls = Vec::new();
    let mut remaining = String::new();
    let mut search_from = 0;
    let mut call_counter = 0u32;

    while let Some(tc_start) = text[search_from..].find("<tool_call>") {
        let abs_start = search_from + tc_start;
        remaining.push_str(text[search_from..abs_start].trim());

        if let Some(tc_end_rel) = text[abs_start..].find("</tool_call>") {
            let abs_end = abs_start + tc_end_rel + "</tool_call>".len();
            let block = &text[abs_start..abs_end];

            // Parse function name: <function=NAME>
            if let Some(fn_start) = block.find("<function=") {
                let name_start = fn_start + "<function=".len();
                if let Some(name_end) = block[name_start..].find('>') {
                    let func_name = &block[name_start..name_start + name_end];

                    // Parse parameters
                    let mut arguments = serde_json::Map::new();
                    let mut psearch = name_start + name_end;
                    while let Some(p_start) = block[psearch..].find("<parameter=") {
                        let p_abs = psearch + p_start + "<parameter=".len();
                        if let Some(p_name_end) = block[p_abs..].find('>') {
                            let param_name = block[p_abs..p_abs + p_name_end].to_string();
                            let val_start = p_abs + p_name_end + 1;
                            let val_start = if block.as_bytes().get(val_start) == Some(&b'\n') { val_start + 1 } else { val_start };
                            if let Some(p_end) = block[val_start..].find("\n</parameter>") {
                                let value = block[val_start..val_start + p_end].trim();
                                let json_val = serde_json::from_str(value)
                                    .unwrap_or_else(|_| serde_json::Value::String(value.to_string()));
                                arguments.insert(param_name, json_val);
                                psearch = val_start + p_end + "</parameter>".len();
                            } else { break; }
                        } else { break; }
                    }

                    call_counter += 1;
                    tool_calls.push(serde_json::json!({
                        "id": format!("call_{}_{}", unix_now(), call_counter),
                        "type": "function",
                        "function": {
                            "name": func_name,
                            "arguments": serde_json::to_string(&arguments).unwrap_or_default()
                        }
                    }));
                }
            }
            search_from = abs_end;
        } else {
            remaining.push_str(&text[abs_start..]);
            search_from = text.len();
            break;
        }
    }
    remaining.push_str(text[search_from..].trim());
    (remaining.trim().to_string(), tool_calls)
}

// ---------------------------------------------------------------------------
// Strip thinking block from model output
// ---------------------------------------------------------------------------

/// Extract reasoning and content from model output with `<think>` tags.
/// Returns (reasoning_content, content).
///
/// Handles multiple formats:
/// 1. Standard: text before `</think>` is reasoning, rest is content
/// 2. Full block: `<think>...</think>content`
/// 3. Fallback: if model generates "Thinking Process:" without tags,
///    split at the first double newline followed by a short answer line.
fn extract_thinking(text: &str) -> (Option<String>, String) {
    let mut reasoning = None;
    let mut result = text.to_string();

    // Handle leading </think> (when template already opened <think>)
    if let Some(pos) = result.find("</think>") {
        let thinking_text = result[..pos].trim().to_string();
        if !thinking_text.is_empty() {
            reasoning = Some(thinking_text);
        }
        result = result[pos + "</think>".len()..].to_string();
    }
    // Remove any remaining <think>...</think> blocks
    while let Some(start) = result.find("<think>") {
        if let Some(end) = result.find("</think>") {
            let end_tag = end + "</think>".len();
            result = format!("{}{}", &result[..start], &result[end_tag..]);
        } else {
            result = result[..start].to_string();
            break;
        }
    }

    // Fallback: if no </think> was found but content starts with known thinking
    // prefixes, try to separate reasoning from final answer.
    if reasoning.is_none() {
        let trimmed = result.trim();
        let thinking_prefixes = [
            "Thinking Process:",
            "Let me think",
            "I need to",
            "First, let me",
            "**Thinking",
            "Step 1:",
        ];
        let starts_with_thinking = thinking_prefixes
            .iter()
            .any(|p| trimmed.starts_with(p));

        if starts_with_thinking {
            // Heuristic: look for a clear boundary between thinking and answer.
            // Common pattern: thinking ends, then a short final answer follows
            // after a blank line + "The answer is" / "Therefore" / just a number.
            let answer_markers = [
                "\n\nThe answer is ",
                "\n\nTherefore, ",
                "\n\n**Answer:**",
                "\n\n**Final Answer:**",
                "\n\n**Result:**",
                "\n\n$$",
            ];
            for marker in &answer_markers {
                if let Some(pos) = trimmed.find(marker) {
                    reasoning = Some(trimmed[..pos].trim().to_string());
                    result = trimmed[pos..].trim().to_string();
                    return (reasoning, result);
                }
            }
            // If no clear marker, keep everything as content (don't guess wrong).
        }
    }

    (reasoning, result.trim().to_string())
}

// ---------------------------------------------------------------------------
// Unique ID / timestamp helpers
// ---------------------------------------------------------------------------

fn new_request_id() -> String {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("chatcmpl-{:x}", ts)
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Core inference helper: runs generation while holding the model lock.
// Returns (generated_text, prompt_tokens, completion_tokens, packed_logprobs).
// ---------------------------------------------------------------------------

async fn run_inference(
    state: &Arc<AppState>,
    prompt: &str,
    max_tokens: usize,
    params: &SamplingParams,
    user: Option<&str>,
) -> Result<(String, usize, usize, Vec<Vec<f32>>), String> {
    // Count prompt tokens (best-effort).
    let prompt_tokens = state
        .tokenizer
        .encode(prompt, false)
        .map(|enc| enc.len())
        .unwrap_or(0);

    // Lock the model for the duration of inference.
    let model = state.model.lock().await;

    // Multi-agent context switching: if user is specified, switch to that agent's context.
    if let Some(user_name) = user {
        let agent_id = {
            let mut map = state.user_agent_map.lock().await;
            if let Some(&id) = map.get(user_name) {
                id
            } else {
                // Auto-register new agent
                let mut sched = state.agent_scheduler.lock().await;
                let id = sched.register_agent(user_name);
                map.insert(user_name.to_string(), id);
                eprintln!("[AGENT] Registered '{}' as agent {}", user_name, id);
                id
            }
        };

        // Switch to this agent's context (save current, restore target)
        if model.llama_forward_active() {
            let mut sched = state.agent_scheduler.lock().await;
            let mut llama_ref = model.llama_forward_mut();
            if let Some(llama) = llama_ref.as_mut() {
                match sched.switch_to(agent_id, llama) {
                    Ok(ms) => {
                        if ms > 0.1 {
                            eprintln!("[AGENT] Switched to '{}' (agent {}) in {:.1}ms", user_name, agent_id, ms);
                        }
                    }
                    Err(e) => eprintln!("[AGENT] Context switch to '{}' failed: {}", user_name, e),
                }
            }
        }
    } else {
        // No user specified — reset state for stateless request.
        model.reset_llama_state();
    }

    model.reset_cudarc_state();

    // Build a fresh recurrent state for this request.
    let device = model.device.clone();
    let mut inf_state = GdnRecurrentState::new(&model.config, &device)
        .map_err(|e| format!("State init failed: {}", e))?;

    let gen = generate_text(
        &*model,
        &state.tokenizer,
        prompt,
        max_tokens,
        params,
        &mut inf_state,
    )?;

    let completion_tokens = gen.token_ids.len();
    Ok((gen.text, prompt_tokens, completion_tokens, gen.packed_logprobs))
}

// ---------------------------------------------------------------------------
// POST /v1/chat/completions — non-streaming path
// ---------------------------------------------------------------------------

async fn chat_completions_non_stream(
    state: Arc<AppState>,
    req: ChatRequest,
) -> impl IntoResponse {
    let enable_thinking = req.chat_template_kwargs
        .as_ref()
        .map(|k| k.enable_thinking)
        .unwrap_or(true);
    let prompt = messages_to_prompt(&req.messages, req.tools.as_ref(), enable_thinking);
    let params = SamplingParams {
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: req.top_k,
        min_p: 0.05, // matches ik_llama default — very effective anti-repetition
        repetition_penalty: req.repetition_penalty.unwrap_or(1.0) as f32,
        presence_penalty: req.presence_penalty as f32,
        dry_multiplier: 0.8,  // DRY penalty: prevents n-gram repetition loops in thinking
        dry_base: 1.75,
        dry_min_length: 2,
        dry_penalty_last_n: -1, // scan whole sequence
    };

    // Thinking budget is handled inside generate_with_mtp (detects </think> token).
    // max_tokens applies to the RESPONSE only — thinking has its own budget (8192 default).
    let total_budget = req.max_tokens.min(MAX_TOKENS_LIMIT);
    let want_logprobs = req.logprobs;
    let top_logprobs_n = req.top_logprobs.unwrap_or(5).min(5); // FFI max is 5

    match run_inference(&state, &prompt, total_budget, &params, req.user.as_deref()).await {
        Ok((raw_text, prompt_tokens, completion_tokens, packed_logprobs)) => {
            // Extract thinking/reasoning, then tool calls from response
            let (reasoning, text) = extract_thinking(&raw_text);
            let (content, tool_calls_parsed) = extract_tool_calls(&text);
            let has_tool_calls = !tool_calls_parsed.is_empty();

            // Build logprobs content from per-token packed logprobs (if requested).
            let choice_logprobs = if want_logprobs && !packed_logprobs.is_empty() {
                let entries: Vec<TokenLogprobEntry> = packed_logprobs.iter()
                    .map(|packed| {
                        let lpc = packed_to_logprob_content(packed, &state.tokenizer, top_logprobs_n);
                        // Each packed entry produces exactly one TokenLogprobEntry.
                        lpc.content.into_iter().next().unwrap()
                    })
                    .collect();
                Some(LogprobContent { content: entries })
            } else {
                None
            };

            let resp = ChatResponse {
                id: new_request_id(),
                object: "chat.completion".into(),
                created: unix_now(),
                model: state.model_name.clone(),
                choices: vec![Choice {
                    index: 0,
                    message: Message {
                        role: "assistant".into(),
                        content,
                        reasoning_content: reasoning,
                        tool_calls: if has_tool_calls { Some(tool_calls_parsed) } else { None },
                    },
                    finish_reason: Some(if has_tool_calls { "tool_calls" } else { "stop" }.into()),
                    logprobs: choice_logprobs,
                }],
                usage: Usage {
                    prompt_tokens,
                    completion_tokens,
                    total_tokens: prompt_tokens + completion_tokens,
                },
            };
            Json(serde_json::to_value(resp).unwrap()).into_response()
        }
        Err(e) => {
            let body = serde_json::json!({
                "error": { "message": e, "type": "server_error" }
            });
            (StatusCode::INTERNAL_SERVER_ERROR, Json(body)).into_response()
        }
    }
}

// ---------------------------------------------------------------------------
// POST /v1/chat/completions — streaming (SSE) path
//
// Real token-by-token streaming via generate_with_mtp_streaming callback.
// Thinking tokens (<think>...</think>) are suppressed — only response tokens
// are sent to the client, one SSE event per token as they are generated.
//
// Architecture: std::thread::spawn runs inference on a dedicated OS thread
// (outside the tokio runtime) using blocking_lock() on the model mutex.
// Each non-thinking token is sent through a tokio::sync::mpsc channel.
// The SSE stream reads from this channel via futures::stream::unfold.
// ---------------------------------------------------------------------------

/// Message sent through the streaming channel from the inference thread to the SSE stream.
enum StreamMsg {
    /// A content token to send to the client, with optional logprobs (pre-built).
    Token(String, Option<LogprobContent>),
    /// A thinking/reasoning token to send as reasoning_content delta.
    Thinking(String),
    /// Generation finished (with optional error message).
    Done(Option<String>),
}

/// Convert packed logprobs [token_id, n_top, t0, lp0, t1, lp1, ...] to OpenAI format.
/// Uses the tokenizer to decode token IDs to strings.
/// `max_top` limits the number of top logprob entries returned (0 = all available, max 5 from FFI).
fn packed_to_logprob_content(packed: &[f32], tokenizer: &Tokenizer, max_top: usize) -> LogprobContent {
    let n_top_available = packed[1] as usize;
    let n_top = if max_top > 0 { n_top_available.min(max_top) } else { n_top_available };
    let mut top_logprobs = Vec::with_capacity(n_top);
    for i in 0..n_top {
        let tok_id = packed[2 + i * 2] as u32;
        let lp = packed[2 + i * 2 + 1];
        let tok_text = tokenizer.decode(&[tok_id], false).unwrap_or_else(|_| format!("<{}>", tok_id));
        let tok_bytes = Some(tok_text.as_bytes().to_vec());
        top_logprobs.push(TopLogprobEntry {
            token: tok_text,
            logprob: lp,
            bytes: tok_bytes,
        });
    }

    // The selected token is packed[0]; find its logprob in the top entries.
    let selected_id = packed[0] as u32;
    let selected_text = tokenizer.decode(&[selected_id], false)
        .unwrap_or_else(|_| format!("<{}>", selected_id));
    let selected_bytes = Some(selected_text.as_bytes().to_vec());
    // Find the selected token's logprob in the top entries (it should be there).
    let selected_lp = top_logprobs.iter()
        .find(|e| e.token == selected_text)
        .map(|e| e.logprob)
        .unwrap_or_else(|| if !top_logprobs.is_empty() { top_logprobs[0].logprob } else { 0.0 });

    LogprobContent {
        content: vec![TokenLogprobEntry {
            token: selected_text,
            logprob: selected_lp,
            bytes: selected_bytes,
            top_logprobs,
        }],
    }
}

async fn chat_completions_stream(
    state: Arc<AppState>,
    req: ChatRequest,
) -> Sse<impl futures::Stream<Item = Result<Event, Infallible>>> {
    let model_name = state.model_name.clone();
    let req_id = new_request_id();
    let created = unix_now();

    let enable_thinking = req.chat_template_kwargs
        .as_ref()
        .map(|k| k.enable_thinking)
        .unwrap_or(true);
    let prompt = messages_to_prompt(&req.messages, req.tools.as_ref(), enable_thinking);
    let max_tokens = req.max_tokens.min(MAX_TOKENS_LIMIT);
    let params = SamplingParams {
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: req.top_k,
        min_p: 0.05,
        repetition_penalty: req.repetition_penalty.unwrap_or(1.0) as f32,
        presence_penalty: req.presence_penalty as f32,
        dry_multiplier: 0.8,
        dry_base: 1.75,
        dry_min_length: 2,
        dry_penalty_last_n: -1,
    };

    // Capture logprobs settings before moving req.
    let want_logprobs = req.logprobs;
    let top_logprobs_n = req.top_logprobs.unwrap_or(5).min(5); // FFI max is 5

    // Channel for streaming tokens from inference thread to SSE stream.
    // Buffer of 128 — inference produces ~57 tok/s, SSE consumes faster.
    let (tx, rx) = tokio::sync::mpsc::channel::<StreamMsg>(128);

    // Clone what we need for the spawned thread.
    let state_clone = Arc::clone(&state);
    let tokenizer_clone = Arc::clone(&state.tokenizer);

    // Spawn inference on a dedicated OS thread (NOT tokio::spawn) so we can
    // use blocking_lock() and blocking_send() without deadlocking the runtime.
    std::thread::spawn(move || {
        // Encode prompt tokens.
        let prompt_ids = match tokenizer_clone.encode(prompt.as_str(), false) {
            Ok(enc) => enc.get_ids().to_vec(),
            Err(e) => {
                let _ = tx.blocking_send(StreamMsg::Done(Some(format!("Tokenizer encode failed: {}", e))));
                return;
            }
        };

        if prompt_ids.is_empty() {
            let _ = tx.blocking_send(StreamMsg::Done(Some("Prompt encoded to zero tokens".into())));
            return;
        }

        // Lock the model (blocking — we're on a dedicated OS thread).
        let model = state_clone.model.blocking_lock();

        // Reset state for the active backend (llama or cudarc).
        model.reset_llama_state();
        model.reset_cudarc_state();

        // Build a fresh recurrent state for this request.
        let device = model.device.clone();
        let mut inf_state = match GdnRecurrentState::new(&model.config, &device) {
            Ok(s) => s,
            Err(e) => {
                let _ = tx.blocking_send(StreamMsg::Done(Some(format!("State init failed: {}", e))));
                return;
            }
        };

        // Prefill prompt tokens and sample first token.
        let prefill_logits = match model.forward_prefill(&prompt_ids, &mut inf_state) {
            Ok(l) => l,
            Err(e) => {
                let _ = tx.blocking_send(StreamMsg::Done(Some(format!("Prefill failed: {}", e))));
                return;
            }
        };

        // Extract first sampled token from prefill logits
        let dims = prefill_logits.dims();
        let last_tok = if dims.len() == 2 && dims[1] == 12 {
            // C++ fast path: token already sampled (packed format)
            prefill_logits.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0] as u32
        } else {
            // Slow path: argmax from full logits
            crate::mtp_scheduler::argmax(&prefill_logits).unwrap_or(0)
        };

        // Streaming generation with per-token callback.
        let tx_ref = &tx;
        let tok_ref = &*tokenizer_clone;
        let model_ref = &*model; // shared ref for logprobs access in callback

        // Signal thinking mode to the scheduler via env var (thread-local safe
        // because we're on a dedicated OS thread per request).
        if enable_thinking {
            std::env::set_var("CHIMERE_THINKING_ACTIVE", "1");
        } else {
            std::env::remove_var("CHIMERE_THINKING_ACTIVE");
        }

        let result = generate_with_mtp_streaming(
            model_ref,
            last_tok,
            max_tokens,
            &mut inf_state,
            &params,
            Some(tok_ref),
            &mut |_token_id: u32, decoded_text: &str, is_thinking: bool| -> bool {
                // Drain logprobs (avoids stale data). Only build content if requested.
                let logprob_content = if want_logprobs {
                    model_ref.take_last_packed_logprobs()
                        .map(|packed| packed_to_logprob_content(&packed, tok_ref, top_logprobs_n))
                } else {
                    // Still drain to avoid stale data, but discard.
                    let _ = model_ref.take_last_packed_logprobs();
                    None
                };

                // Skip empty decoded text (incomplete multi-byte sequence).
                if decoded_text.is_empty() {
                    return true;
                }

                // Route thinking tokens to reasoning_content, content tokens to content.
                let msg = if is_thinking {
                    StreamMsg::Thinking(decoded_text.to_string())
                } else {
                    StreamMsg::Token(decoded_text.to_string(), logprob_content)
                };

                match tx_ref.blocking_send(msg) {
                    Ok(()) => true,
                    Err(_) => false, // Channel closed — client disconnected.
                }
            },
        );

        // Send completion message.
        let err_msg = match result {
            Ok(_stats) => None,
            Err(e) => Some(format!("Generation error: {}", e)),
        };
        let _ = tx.blocking_send(StreamMsg::Done(err_msg));
        // tx is dropped here, closing the channel.
    });

    // Build the SSE stream from the channel receiver using futures::stream::unfold.
    // The unfold state tracks: receiver, request metadata, and phase flags.
    struct SseState {
        rx: tokio::sync::mpsc::Receiver<StreamMsg>,
        req_id: String,
        model_name: String,
        created: u64,
        sent_role: bool,
        finished: bool,
        want_logprobs: bool,
    }

    let initial_state = SseState {
        rx,
        req_id,
        model_name,
        created,
        sent_role: false,
        finished: false,
        want_logprobs,
    };

    let sse_stream = stream::unfold(initial_state, |mut st| async move {
        // Stream terminated — no more events.
        if st.finished {
            return None;
        }

        // First event: send role delta.
        if !st.sent_role {
            st.sent_role = true;
            let role_chunk = ChatChunk {
                id: st.req_id.clone(),
                object: "chat.completion.chunk".into(),
                created: st.created,
                model: st.model_name.clone(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: Delta {
                        role: Some("assistant".into()),
                        content: None,
                        reasoning_content: None,
                    },
                    finish_reason: None,
                    logprobs: None,
                }],
            };
            let data = serde_json::to_string(&role_chunk).unwrap_or_default();
            let evt: Result<Event, Infallible> = Ok(Event::default().data(data));
            return Some((evt, st));
        }

        // Read from channel — blocks until a token arrives or channel closes.
        match st.rx.recv().await {
            Some(StreamMsg::Token(text, lp)) => {
                let logprobs = if st.want_logprobs { lp } else { None };
                let chunk = ChatChunk {
                    id: st.req_id.clone(),
                    object: "chat.completion.chunk".into(),
                    created: st.created,
                    model: st.model_name.clone(),
                    choices: vec![ChunkChoice {
                        index: 0,
                        delta: Delta {
                            role: None,
                            content: Some(text),
                            reasoning_content: None,
                        },
                        finish_reason: None,
                        logprobs,
                    }],
                };
                let data = serde_json::to_string(&chunk).unwrap_or_default();
                let evt: Result<Event, Infallible> = Ok(Event::default().data(data));
                Some((evt, st))
            }
            Some(StreamMsg::Thinking(text)) => {
                // Route thinking tokens to reasoning_content delta.
                let chunk = ChatChunk {
                    id: st.req_id.clone(),
                    object: "chat.completion.chunk".into(),
                    created: st.created,
                    model: st.model_name.clone(),
                    choices: vec![ChunkChoice {
                        index: 0,
                        delta: Delta {
                            role: None,
                            content: None,
                            reasoning_content: Some(text),
                        },
                        finish_reason: None,
                        logprobs: None,
                    }],
                };
                let data = serde_json::to_string(&chunk).unwrap_or_default();
                let evt: Result<Event, Infallible> = Ok(Event::default().data(data));
                Some((evt, st))
            }
            Some(StreamMsg::Done(err)) => {
                // Build stop chunk (with optional error text).
                let (content, finish) = if let Some(err_msg) = err {
                    (Some(format!("\n\n[Error: {}]", err_msg)), Some("stop".into()))
                } else {
                    (None, Some("stop".into()))
                };
                let stop_chunk = ChatChunk {
                    id: st.req_id.clone(),
                    object: "chat.completion.chunk".into(),
                    created: st.created,
                    model: st.model_name.clone(),
                    choices: vec![ChunkChoice {
                        index: 0,
                        delta: Delta {
                            role: None,
                            content,
                            reasoning_content: None,
                        },
                        finish_reason: finish,
                        logprobs: None,
                    }],
                };
                let data = serde_json::to_string(&stop_chunk).unwrap_or_default();
                let evt: Result<Event, Infallible> = Ok(Event::default().data(data));
                // Next recv() will return None → emit [DONE] then terminate.
                Some((evt, st))
            }
            None => {
                // Channel closed — emit [DONE] and mark stream as finished.
                st.finished = true;
                let evt: Result<Event, Infallible> = Ok(Event::default().data("[DONE]"));
                Some((evt, st))
            }
        }
    });

    Sse::new(sse_stream)
}

// ---------------------------------------------------------------------------
// Unified handler: dispatch to streaming or non-streaming
// ---------------------------------------------------------------------------

async fn chat_completions_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatRequest>,
) -> axum::response::Response {
    if req.stream {
        chat_completions_stream(state, req).await.into_response()
    } else {
        chat_completions_non_stream(state, req).await.into_response()
    }
}

// ---------------------------------------------------------------------------
// Health check
// ---------------------------------------------------------------------------

async fn health() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "ok",
        "engine": "chimere-deltanet"
    }))
}

// ---------------------------------------------------------------------------
// Router factory
// ---------------------------------------------------------------------------

/// Build the Axum router. Call this from `main()` after loading the model.
pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions_handler))
        .route("/health", axum::routing::get(health))
        .with_state(state)
}
