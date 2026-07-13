use crate::markdown::StreamRenderer;
use futures::StreamExt;
use npcrs::r#gen::{Message, ToolCall, ToolCallFunction, Usage};
use serde_json::Value;

pub struct StreamRequest {
    pub model: String,
    pub provider: String,
    pub messages: Vec<Message>,
    pub commandstr: String,
    pub npc: Option<String>,
    pub registered_teams: Option<Vec<String>>,
    pub conversation_id: Option<String>,
    pub current_path: Option<String>,
    pub execution_mode: String,
}

pub struct StreamResponse {
    pub message: Message,
    pub tool_results: Vec<Message>,
    pub usage: Option<Usage>,
    pub streamed: bool,
}

pub async fn call_stream(
    client: &reqwest::Client,
    base_url: &str,
    request: &StreamRequest,
    permission_prompt: Option<&dyn Fn(&str) -> String>,
) -> Result<StreamResponse, String> {
    call_stream_with_interrupt(client, base_url, request, permission_prompt, None).await
}

pub async fn call_stream_with_interrupt(
    client: &reqwest::Client,
    base_url: &str,
    request: &StreamRequest,
    permission_prompt: Option<&dyn Fn(&str) -> String>,
    mut interrupt: Option<tokio::sync::mpsc::UnboundedReceiver<()>>,
) -> Result<StreamResponse, String> {
    let stream_url = format!("{}/api/stream", base_url);
    let body = serde_json::json!({
        "model": request.model,
        "provider": request.provider,
        "messages": request.messages,
        "commandstr": request.commandstr,
        "npc": request.npc,
        "registered_teams": request.registered_teams,
        "conversationId": request.conversation_id,
        "currentPath": request.current_path,
        "executionMode": request.execution_mode,
    });

    let resp = client
        .post(&stream_url)
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("HTTP stream request failed: {}", e))?;

    let status = resp.status();
    if !status.is_success() {
        let text = resp.text().await.unwrap_or_default();
        return Err(format!("HTTP stream returned {}: {}", status, text));
    }

    let mut content = String::new();
    let mut reasoning = String::new();
    let mut thinking = String::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();
    let mut tool_results: Vec<Message> = Vec::new();
    let mut usage: Option<Usage> = None;
    let mut saw_output = false;
    let mut renderer = StreamRenderer::new();

    let mut stream = resp.bytes_stream();
    let mut pending = String::new();

    loop {
        let chunk = if let Some(ref mut rx) = interrupt {
            tokio::select! {
                biased;
                _ = rx.recv() => break,
                chunk = stream.next() => chunk,
            }
        } else {
            stream.next().await
        };
        let chunk = match chunk {
            Some(Ok(bytes)) => bytes,
            Some(Err(e)) => return Err(format!("HTTP stream chunk: {}", e)),
            None => break,
        };
        let chunk_text = String::from_utf8_lossy(&chunk);
        pending.push_str(&chunk_text);

        while let Some(sep_pos) = pending.find("\n\n").or_else(|| pending.find("\r\n\r\n")) {
            let event_text = pending[..sep_pos].to_string();
            let newline_len = if pending[sep_pos..].starts_with("\r\n\r\n") {
                4
            } else {
                2
            };
            pending.replace_range(..sep_pos + newline_len, "");

            let data = match parse_sse_event_data(&event_text) {
                Some(d) => d,
                None => continue,
            };
            if data.trim() == "[DONE]" {
                break;
            }

            let json: Value = match serde_json::from_str(&data) {
                Ok(v) => v,
                Err(_) => continue,
            };

            let pause = apply_sse_event(
                client,
                base_url,
                json,
                &mut content,
                &mut reasoning,
                &mut thinking,
                &mut tool_calls,
                &mut tool_results,
                &mut usage,
                &mut saw_output,
                &mut renderer,
                permission_prompt,
            );
            if pause {
                continue;
            }
        }
    }

    if !pending.trim().is_empty() {
        if let Some(data) = parse_sse_event_data(&pending) {
            if data.trim() != "[DONE]" {
                if let Ok(json) = serde_json::from_str(&data) {
                    let _ = apply_sse_event(
                        client,
                        base_url,
                        json,
                        &mut content,
                        &mut reasoning,
                        &mut thinking,
                        &mut tool_calls,
                        &mut tool_results,
                        &mut usage,
                        &mut saw_output,
                        &mut renderer,
                        permission_prompt,
                    );
                }
            }
        }
    }

    renderer.flush();

    tool_calls.retain(|tc| !tc.function.name.is_empty());

    let message = Message {
        role: "assistant".to_string(),
        content: if content.is_empty() {
            None
        } else {
            Some(content)
        },
        tool_calls: if tool_calls.is_empty() {
            None
        } else {
            Some(tool_calls)
        },
        tool_call_id: None,
        name: None,
        thinking: if thinking.is_empty() {
            None
        } else {
            Some(thinking)
        },
        reasoning_content: if reasoning.is_empty() {
            None
        } else {
            Some(reasoning)
        },
    };

    Ok(StreamResponse {
        message,
        tool_results,
        usage,
        streamed: saw_output,
    })
}

fn parse_sse_event_data(event_text: &str) -> Option<String> {
    let mut data_lines: Vec<&str> = Vec::new();
    for line in event_text.lines() {
        let line = line.trim_start();
        if line.is_empty() || line.starts_with(':') {
            continue;
        }
        if let Some(rest) = line.strip_prefix("data:") {
            data_lines.push(rest.strip_prefix(' ').unwrap_or(rest));
        }
    }
    if data_lines.is_empty() {
        None
    } else {
        Some(data_lines.join("\n"))
    }
}

fn apply_sse_event(
    client: &reqwest::Client,
    base_url: &str,
    json: Value,
    content: &mut String,
    reasoning: &mut String,
    thinking: &mut String,
    tool_calls: &mut Vec<ToolCall>,
    tool_results: &mut Vec<Message>,
    usage: &mut Option<Usage>,
    saw_output: &mut bool,
    renderer: &mut StreamRenderer,
    permission_prompt: Option<&dyn Fn(&str) -> String>,
) -> bool {
    if let Some(typ) = json.get("type").and_then(|v| v.as_str()) {
        match typ {
            "usage" => {
                *usage = Some(Usage {
                    prompt_tokens: json
                        .get("input_tokens")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0),
                    completion_tokens: json
                        .get("output_tokens")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0),
                    total_tokens: json
                        .get("total_tokens")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0),
                });
            }
            "message_stop" | "stop" => {}
            "error" => {
                if let Some(msg) = json.get("message").and_then(|v| v.as_str()) {
                    eprintln!("\x1b[31mstream error: {}\x1b[0m", msg);
                }
            }
            "tool_call" | "tool_execution_start" => {
                if let Some(tc) = json.get("tool_call").or_else(|| json.get("tool_calls")) {
                    append_tool_call_json(tc, tool_calls, saw_output);
                }
            }
            "tool_start" => {
                let name = json.get("name").and_then(|v| v.as_str()).unwrap_or("tool");
                renderer.flush();
                eprintln!("\x1b[36m⚡ {}:\x1b[0m", name);
                renderer.clear();
                *saw_output = true;
            }
            "tool_result" => {
                renderer.flush();
                let name = json.get("name").and_then(|v| v.as_str()).unwrap_or("tool");
                let tool_id = json.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let result_text = json
                    .get("result")
                    .map(|v| {
                        if let Some(s) = v.as_str() {
                            s.to_string()
                        } else {
                            serde_json::to_string_pretty(v).unwrap_or_else(|_| v.to_string())
                        }
                    })
                    .unwrap_or_default();
                let content_text = json.get("content").and_then(|v| v.as_str()).unwrap_or("");
                let display = if !result_text.is_empty() {
                    result_text
                } else if !content_text.is_empty() {
                    content_text.to_string()
                } else {
                    String::new()
                };
                let preview = if display.len() > 500 {
                    format!("{}...\n[{} chars total]", &display[..500], display.len())
                } else {
                    display.clone()
                };
                if !preview.is_empty() {
                    eprintln!("\x1b[36m  {} result:\x1b[0m\n{}", name, preview);
                } else {
                    eprintln!("\x1b[36m  {} result: (empty)\x1b[0m", name);
                }
                renderer.clear();
                *saw_output = true;
                if !display.is_empty() {
                    tool_results.push(Message {
                        role: "tool".to_string(),
                        content: Some(display),
                        tool_calls: None,
                        tool_call_id: if tool_id.is_empty() { None } else { Some(tool_id) },
                        name: Some(name.to_string()),
                        thinking: None,
                        reasoning_content: None,
                    });
                }
            }
            "permission_request" => {
                let request_id = json
                    .get("request_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let command_key = json
                    .get("command_key")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let args_preview = json
                    .get("args_preview")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                renderer.flush();
                eprintln!("");
                let tool_name = json
                    .get("tool_name")
                    .and_then(|v| v.as_str())
                    .unwrap_or(command_key);
                let decision = permission_prompt
                    .map(|f| {
                        f(format!(
                            "Permission Required: {}\nCommand: {}\nArgs: {}",
                            tool_name, command_key, args_preview
                        )
                        .as_str())
                    })
                    .unwrap_or_else(|| "No".to_string());
                let resp_url = format!("{}/api/permission_response", base_url);
                let body = serde_json::json!({
                    "request_id": request_id,
                    "decision": decision,
                });
                let post_client = client.clone();
                tokio::spawn(async move {
                    let _ = post_client.post(&resp_url).json(&body).send().await;
                });
                renderer.clear();
                *saw_output = true;
                return true;
            }
            "tool_error" => {
                renderer.flush();
                let name = json.get("name").and_then(|v| v.as_str()).unwrap_or("tool");
                let err = json
                    .get("error")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown error");
                eprintln!("\x1b[31m  {} error: {}\x1b[0m", name, err);
                renderer.clear();
                *saw_output = true;
            }
            _ => {}
        }
        return false;
    }

    if let Some(choices) = json.get("choices").and_then(|v| v.as_array()) {
        for choice in choices {
            if let Some(delta) = choice.get("delta") {
                if let Some(text) = delta.get("content").and_then(|v| v.as_str()) {
                    let new_text =
                        if content.len() > text.len() || !text.starts_with(content.as_str()) {
                            text
                        } else {
                            &text[content.len()..]
                        };
                    if !new_text.is_empty() {
                        content.push_str(new_text);
                        *saw_output = true;
                        renderer.push(new_text);
                        let _ = std::io::Write::flush(&mut std::io::stderr());
                    }
                }
                if let Some(t) = delta.get("thinking").and_then(|v| v.as_str()) {
                    thinking.push_str(t);
                    *saw_output = true;
                    eprint!("\x1b[90m{}\x1b[0m", t);
                    let _ = std::io::Write::flush(&mut std::io::stderr());
                }
                if let Some(r) = delta.get("reasoning_content").and_then(|v| v.as_str()) {
                    reasoning.push_str(r);
                    *saw_output = true;
                    eprint!("\x1b[90m{}\x1b[0m", r);
                    let _ = std::io::Write::flush(&mut std::io::stderr());
                }
                if let Some(deltas) = delta.get("tool_calls").and_then(|v| v.as_array()) {
                    for (i, d) in deltas.iter().enumerate() {
                        let idx = d
                            .get("index")
                            .and_then(|v| v.as_u64())
                            .map(|n| n as usize)
                            .unwrap_or(i);
                        while tool_calls.len() <= idx {
                            tool_calls.push(ToolCall {
                                id: String::new(),
                                r#type: "function".to_string(),
                                function: ToolCallFunction {
                                    name: String::new(),
                                    arguments: String::new(),
                                },
                            });
                        }
                        *saw_output = true;
                        if let Some(id) = d.get("id").and_then(|v| v.as_str()) {
                            if !id.is_empty() {
                                tool_calls[idx].id = id.to_string();
                            }
                        }
                        if let Some(tc_type) = d.get("type").and_then(|v| v.as_str()) {
                            if !tc_type.is_empty() {
                                tool_calls[idx].r#type = tc_type.to_string();
                            }
                        }
                        if let Some(func) = d.get("function") {
                            if let Some(name) = func.get("name").and_then(|v| v.as_str()) {
                                if !name.is_empty() {
                                    tool_calls[idx].function.name = name.to_string();
                                }
                            }
                            if let Some(args) = func.get("arguments").and_then(|v| v.as_str()) {
                                tool_calls[idx].function.arguments.push_str(args);
                            }
                        }
                    }
                }
            }
            if let Some(finish) = choice.get("finish_reason").and_then(|v| v.as_str()) {
                if finish == "stop" || finish == "length" {}
            }
        }
    }
    false
}

fn append_tool_call_json(tc: &Value, tool_calls: &mut Vec<ToolCall>, saw_output: &mut bool) {
    let id = tc
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let name = tc
        .get("name")
        .or_else(|| tc.get("function_name"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let args = tc
        .get("arguments")
        .or_else(|| tc.get("function").and_then(|f| f.get("arguments")))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    if !name.is_empty() {
        tool_calls.push(ToolCall {
            id,
            r#type: "function".to_string(),
            function: ToolCallFunction {
                name,
                arguments: args,
            },
        });
        *saw_output = true;
    }
}
