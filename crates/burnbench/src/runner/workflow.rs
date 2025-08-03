use regex::Regex;
use reqwest::{
    blocking::{Client, Response},
    header::CONTENT_TYPE,
};
use serde_json::{Map as JsonMap, Value};
use std::{env, fs::File, io::BufReader};

use hmac_sha256::HMAC;
use uuid::Uuid;

use crate::{TRACEL_CI_SERVER_BASE_URL, ci_errorln};

fn get_webhook_url() -> String {
    format!("{TRACEL_CI_SERVER_BASE_URL}burn_bench/webhook/benchmark")
}

pub(crate) fn send_started_event(inputs_file: &str) {
    if let Some((json, pr_number)) = load_inputs(inputs_file) {
        let table = clean_output("no results", Some("no share link"));
        if let Some(payload) = serialize_result(json, pr_number, table.unwrap(), "started") {
            send_event("started", payload);
        }
    }
}

pub(crate) fn send_output_results(inputs_file: &str, table: &str, share_link: Option<&str>) {
    if let Some((json, pr_number)) = load_inputs(inputs_file) {
        if let Some(cleaned_table) = clean_output(table, share_link) {
            if let Some(payload) = serialize_result(json, pr_number, cleaned_table, "complete") {
                send_event("complete", payload);
            }
        }
    }
}

fn load_inputs(inputs_file: &str) -> Option<(Value, i64)> {
    let json = read_inputs(inputs_file).ok()?;
    let pr_number = get_pr_number(&json)?;
    Some((json, pr_number))
}

fn send_event(action: &str, payload: Vec<u8>) {
    let (signature, delivery_id) = match build_req_body(&payload) {
        Ok(t) => t,
        Err(e) => {
            ci_errorln!("❌ {e}");
            return;
        }
    };

    let post_url = get_webhook_url();
    match send_result(&post_url, payload, &signature, &delivery_id) {
        Ok(resp) => {
            let status = resp.status();
            if status.is_success() {
                println!("✅ Sent '{action}' webhook to server at '{post_url}'.");
            } else {
                let body = resp
                    .text()
                    .unwrap_or_else(|_| "<could not read body>".into());
                ci_errorln!("❌ Webhook failed with status: {status} ({body})");
            }
        }
        Err(e) => {
            ci_errorln!("❌ Error sending webhook: {e}");
        }
    }
}

fn get_pr_number(json: &Value) -> Option<i64> {
    match json["pr_number"].as_i64() {
        Some(n) => Some(n),
        None => {
            println!("ℹ️ No valid 'pr_number' found. Skipping webhook.");
            None
        }
    }
}

fn read_inputs(inputs_file: &str) -> Result<Value, ()> {
    let file =
        File::open(inputs_file).map_err(|e| ci_errorln!("❌ Cannot open inputs file: {e}"))?;
    let reader = BufReader::new(file);
    serde_json::from_reader(reader).map_err(|e| ci_errorln!("❌ Error reading JSON: {e}"))
}

fn clean_output(table: &str, share_link: Option<&str>) -> Option<JsonMap<String, Value>> {
    let ansi_re = Regex::new(r"\x1b\[[0-9;]*m")
        .map_err(|_| {
            ci_errorln!("❌ Failed to compile regex for ANSI codes");
        })
        .ok()?;
    let cleaned_table = ansi_re.replace_all(table, "").to_string();

    let mut map = JsonMap::new();
    map.insert("table".to_owned(), Value::String(cleaned_table));
    if let Some(link) = share_link {
        map.insert("share_link".to_owned(), Value::String(link.to_owned()));
    }

    Some(map)
}

fn serialize_result(
    mut json: Value,
    pr_number: i64,
    table: JsonMap<String, Value>,
    action: &str,
) -> Option<Vec<u8>> {
    json["results"] = Value::Object(table);
    json["action"] = Value::String(action.to_string());
    json["pr_number"] = Value::Number(serde_json::Number::from(pr_number));
    serde_json::to_vec(&json)
        .map_err(|e| {
            ci_errorln!("❌ Failed to serialize result: {e}");
        })
        .ok()
}

fn build_req_body(payload: &[u8]) -> Result<(String, String), String> {
    let secret = env::var("WEBHOOK_PAYLOAD_SECRET")
        .map_err(|_| "Missing WEBHOOK_PAYLOAD_SECRET".to_string())?;
    let mac = HMAC::mac(&payload, secret.as_bytes());
    let signature = format!("sha256={}", hex::encode(mac));
    let delivery_id = Uuid::new_v4().to_string();
    Ok((signature, delivery_id))
}

fn send_result(
    url: &str,
    payload: Vec<u8>,
    signature: &str,
    delivery_id: &str,
) -> Result<Response, reqwest::Error> {
    Client::new()
        .post(url)
        .header(CONTENT_TYPE, "application/json")
        .header("X-GitHub-Delivery", delivery_id)
        .header("X-Hub-Signature-256", signature)
        .body(payload)
        .send()
}
