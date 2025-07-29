use regex::Regex;
use reqwest::{blocking::{Client, Response}, header::CONTENT_TYPE};
use serde_json::{Map as JsonMap, Value};
use std::{env, fs::File, io::BufReader};

use hmac_sha256::HMAC;
use uuid::Uuid;

use crate::{ci_errorln, USER_BENCHMARK_SERVER_URL};

pub(crate) fn send_output_results(inputs_file: &str, table: &str, share_link: Option<&str>) {
    let file = match File::open(inputs_file) {
        Ok(f) => f,
        Err(e) => return ci_errorln!("❌ Cannot open inputs file: {e}"),
    };
    let reader = BufReader::new(file);
    let json: Value = match serde_json::from_reader(reader) {
        Ok(j) => j,
        Err(e) => return ci_errorln!("❌ Error reading JSON: {e}"),
    };

    let pr_number = match json["pr_number"].as_i64().ok_or("Missing 'pr_number'") {
        Ok(n) => n,
        Err(_) => {
            println!("ℹ️ No valid 'pr_number' found. Skipping webhook.");
            return;
        }
    };

    let cleaned = match clean_output(table, share_link) {
        Ok(c) => c,
        Err(e) => {
            ci_errorln!("❌ Failed to clean output: {e}");
            return;
        }
    };

    let payload = match serialize_result(json, pr_number, cleaned) {
        Ok(p) => p,
        Err(e) => {
            ci_errorln!("❌ Failed to serialize result: {e}");
            return;
        }
    };

    let (signature, delivery_id) = match build_req_body(&payload) {
        Ok(t) => t,
        Err(e) => {
            ci_errorln!("❌ Failed to build request: {e}");
            return;
        }
    };

    let post_url = format!("{USER_BENCHMARK_SERVER_URL}burn_bench/webhook/benchmark");
    match send_result(&post_url, payload, &signature, &delivery_id) {
        Ok(resp) => {
            if resp.status().is_success() {
                println!("✅ Sent 'complete' webhook to server at '{post_url}'.");
            } else {
                ci_errorln!("❌ Webhook failed with status: {}", resp.status());
            }
        }
        Err(e) => {
            ci_errorln!("❌ Error sending webhook: {e}");
        }
    }
}

fn clean_output(
    table: &str,
    share_link: Option<&str>,
) -> Result<JsonMap<String, Value>, &'static str> {
    let ansi_re = Regex::new(r"\x1b\[[0-9;]*m").map_err(|_| "Regex compile failed")?;
    let cleaned_table = ansi_re.replace_all(table, "").to_string();

    let mut map = JsonMap::new();
    map.insert("table".to_owned(), Value::String(cleaned_table));
    if let Some(link) = share_link {
        map.insert("share_link".to_owned(), Value::String(link.to_owned()));
    }

    Ok(map)
}

fn serialize_result(
    mut json: Value,
    pr_number: i64,
    cleaned_output: JsonMap<String, Value>,
) -> Result<Vec<u8>, &'static str> {
    json["results"] = Value::Object(cleaned_output);
    json["action"] = Value::String("complete".to_string());
    json["pr_number"] = Value::Number(serde_json::Number::from(pr_number));
    serde_json::to_vec(&json).map_err(|_| "Failed to serialize JSON")
}

fn build_req_body(payload: &[u8]) -> Result<(String, String), &'static str> {
    let secret =
        env::var("WEBHOOK_PAYLOAD_SECRET").map_err(|_| "Missing WEBHOOK_PAYLOAD_SECRET")?;
    let mac = HMAC::mac(&payload, secret.as_bytes());
    let signature = format!("sha256={}", hex::encode(mac));
    let delivery_id = Uuid::new_v4().to_string();
    Ok((signature, delivery_id))
}

fn send_result(url: &str, payload: Vec<u8>, signature: &str, delivery_id: &str) -> Result<Response, reqwest::Error> {
    let response = Client::new()
        .post(url)
        .header(CONTENT_TYPE, "application/json")
        .header("X-GitHub-Delivery", delivery_id)
        .header("X-Hub-Signature-256", signature)
        .body(payload)
        .send();
    response
}
