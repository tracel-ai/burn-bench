use arboard::Clipboard;
use serde::{Deserialize, Serialize};
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::{
    error::Error,
    fs::{self, File},
    path::{Path, PathBuf},
    thread, time,
};

pub(crate) const CLIENT_ID: &'static str = "Iv1.692f6a61b6086810";
const FIVE_SECONDS: time::Duration = time::Duration::new(5, 0);
const GITHUB_API_VERSION_HEADER: &'static str = "X-GitHub-Api-Version";
const GITHUB_API_VERSION: &'static str = "2022-11-28";
const GITHUB_BOT_TOKEN_ENV_VAR: &'static str = "GITHUB_BOT_TOKEN";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct Tokens {
    /// Token returned once the Burnbench Github app has been authorized by the user.
    /// This token is used to authenticate the user to the Burn benchmark server.
    /// This token is a short lived token (about 8 hours).
    pub access_token: String,
    /// Along with the access token, a refresh token is provided once the Burnbench
    /// GitHub app has been authorized by the user.
    /// This token can be presented to the Burn benchmark server in order to re-issue
    /// a new access token for the user.
    /// This token is longer lived (around 6 months).
    /// If None then this type of access_token cannot be refreshed.
    pub refresh_token: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct UserInfo {
    pub nickname: String,
}

pub(crate) fn get_bot_token() -> Option<String> {
    std::env::var(GITHUB_BOT_TOKEN_ENV_VAR).ok()
}

/// Retrieve an access token for GitHub API access.
///
/// If the `GITHUB_BOT_TOKEN` environment variable is set, it takes precedence and is
/// returned as the access token with no refresh token.
///
/// Otherwise, attempt to load cached tokens and refresh them if necessary. If no
/// cached token is found or the refresh fails, prompt the user to reauthorize
/// the Burnbench GitHub application.
pub(crate) fn get_tokens_with_verifier<F>(verifier: F) -> Option<Tokens>
where
    F: Fn(&Tokens) -> bool,
{
    // return the bot token if defined
    if let Some(bot_token) = get_bot_token() {
        return Some(Tokens {
            access_token: bot_token,
            refresh_token: None,
        });
    }
    // otherwise return the user tokens from the cache
    get_tokens_from_cache().map_or_else(
        auth,
        |tokens| {
            if verifier(&tokens) {
                Some(tokens)
            } else {
                refresh_tokens(&tokens).map_or_else(
                    || {
                        println!("⚠ Cannot refresh the access token. You need to reauthorize the Burnbench application.");
                        auth()
                    },
                    |new_tokens| {
                        save_tokens(&new_tokens);
                        Some(new_tokens)
                    },
                )
            }
        },
    )
}

pub(crate) fn get_tokens() -> Option<Tokens> {
    get_tokens_with_verifier(verify_user_tokens)
}

/// Returns the authenticated user name from access token
pub(crate) fn get_username(access_token: &str) -> Result<UserInfo, Box<dyn Error>> {
    let client = reqwest::blocking::Client::new();
    let response = client
        .get(format!("{USER_BENCHMARK_SERVER_URL}users/me"))
        .header(reqwest::header::USER_AGENT, "burnbench")
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .header(
            reqwest::header::AUTHORIZATION,
            get_auth_header_value(access_token),
        )
        .send()?;
    let status = response.status();
    if status != reqwest::StatusCode::OK {
        return Err(format!("error {status}").into());
    }
    let user_info = response.json::<UserInfo>()?;
    Ok(user_info)
}

pub(crate) fn get_auth_header_value(access_token: &str) -> String {
    if access_token.starts_with("ghu_") {
        format!("Bearer {}", access_token)
    } else if access_token.starts_with("github_pat_") {
        format!("token {}", access_token)
    } else {
        panic!("Unsupported token format. Only 'ghu_' and 'github_pat' format are supported.");
    }
}

fn auth() -> Option<Tokens> {
    let mut flow = match DeviceFlow::start(CLIENT_ID, None, None) {
        Ok(flow) => flow,
        Err(e) => {
            eprintln!("Error authenticating: {}", e);
            return None;
        }
    };
    println!(
        "🌐 Please visit for following URL in your browser (CTRL+click if your terminal supports it):"
    );
    println!("\n    {}\n", flow.verification_uri.clone().unwrap());
    let user_code = flow.user_code.clone().unwrap();
    println!("👉 And enter code: {}", &user_code);
    if let Ok(mut clipboard) = Clipboard::new() {
        if clipboard.set_text(user_code).is_ok() {
            println!("📋 Code has been successfully copied to clipboard.")
        };
    };
    // Wait for the minimum allowed interval to poll for authentication update
    // see: https://docs.github.com/en/apps/oauth-apps/building-oauth-apps/authorizing-oauth-apps#step-3-app-polls-github-to-check-if-the-user-authorized-the-device
    thread::sleep(FIVE_SECONDS);
    match flow.poll(20) {
        Ok(creds) => {
            let tokens = Tokens {
                access_token: creds.token.clone(),
                refresh_token: Some(creds.refresh_token.clone()),
            };
            save_tokens(&tokens);
            Some(tokens)
        }
        Err(e) => {
            eprint!("Authentication error: {}", e);
            None
        }
    }
}

/// Return the token saved in the cache file
#[inline]
fn get_tokens_from_cache() -> Option<Tokens> {
    let path = get_auth_cache_file_path();
    let file = File::open(path).ok()?;
    let tokens: Tokens = serde_json::from_reader(file).ok()?;
    Some(tokens)
}

/// Returns true if the token is still valid
fn verify_user_tokens(tokens: &Tokens) -> bool {
    let client = reqwest::blocking::Client::new();
    let response = client
        .get("https://api.github.com/user")
        .header(reqwest::header::USER_AGENT, "burnbench")
        .header(reqwest::header::ACCEPT, "application/vnd.github+json")
        .header(
            reqwest::header::AUTHORIZATION,
            format!("Bearer {}", tokens.access_token),
        )
        .header(GITHUB_API_VERSION_HEADER, GITHUB_API_VERSION)
        .send();
    response.is_ok_and(|resp| resp.status().is_success())
}

fn refresh_tokens(tokens: &Tokens) -> Option<Tokens> {
    if let Some(ref refresh_token) = tokens.refresh_token {
        println!("Access token must be refreshed.");
        println!("Refreshing token...");
        let client = reqwest::blocking::Client::new();
        let response = client
            .post(format!("{USER_BENCHMARK_SERVER_URL}auth/refresh-token"))
            .header(reqwest::header::USER_AGENT, "burnbench")
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .header(
                reqwest::header::AUTHORIZATION,
                format!("Bearer-Refresh {}", refresh_token),
            )
            // it is important to explicitly add an empty body otherwise
            // reqwest won't send the request in release build
            .body(reqwest::blocking::Body::from(""))
            .send();
        response.ok()?.json::<Tokens>().ok().inspect(|_new_tokens| {
            println!("✅ Token refreshed!");
        })
    } else {
        // PAT tokens does not need to be refreshed, we just return back the initial tokens
        println!("⚠️ PAT tokens don't not need to be refreshed.");
        Some(tokens.clone())
    }
}

/// Return the file path for the auth cache on disk
fn get_auth_cache_file_path() -> PathBuf {
    let home_dir = dirs::home_dir().expect("an home directory should exist");
    let path_dir = home_dir.join(".cache").join("burn").join("burnbench");
    #[cfg(test)]
    let path_dir = path_dir.join("test");
    let path = Path::new(&path_dir);
    path.join("token.txt")
}

/// Save token in Burn cache directory and adjust file permissions
fn save_tokens(tokens: &Tokens) {
    let path = get_auth_cache_file_path();
    fs::create_dir_all(path.parent().expect("path should have a parent directory"))
        .expect("directory should be created");
    let file = File::create(&path).expect("file should be created");
    serde_json::to_writer_pretty(file, &tokens).expect("Tokens should be saved to cache file.");
    // On unix systems we lower the permissions on the cache file to be readable
    // just by the current user
    #[cfg(unix)]
    fs::set_permissions(&path, fs::Permissions::from_mode(0o600))
        .expect("permissions should be set to 600");
    println!("✅ Token saved at location: {}", path.to_str().unwrap());
}

#[cfg(test)]
use serial_test::serial;

use crate::{USER_BENCHMARK_SERVER_URL, runner::auth::github_device_flow::DeviceFlow};

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn make_tokens(access: &str, refresh: &str) -> Tokens {
        Tokens {
            access_token: access.to_string(),
            refresh_token: Some(refresh.to_string()),
        }
    }

    fn cleanup_test_environment() {
        let path = get_auth_cache_file_path();
        if path.exists() {
            fs::remove_file(&path).expect("should be able to delete the token file");
        }
        let parent_dir = path
            .parent()
            .expect("token file should have a parent directory");
        if parent_dir.exists() {
            fs::remove_dir_all(parent_dir).expect("should be able to delete the cache directory");
        }
    }

    #[test]
    #[serial]
    fn test_save_token_when_file_does_not_exist() {
        cleanup_test_environment();
        let tokens = make_tokens("unique_test_token", "unique_refresh_token");
        save_tokens(&tokens);
        let retrieved_tokens = get_tokens_from_cache().unwrap();
        assert_eq!(retrieved_tokens.access_token, tokens.access_token);
        assert_eq!(retrieved_tokens.refresh_token, tokens.refresh_token);
        cleanup_test_environment();
    }

    #[test]
    #[serial]
    fn test_overwrite_saved_token_when_file_already_exists() {
        cleanup_test_environment();
        let old_tokens = make_tokens("old_token", "old_refresh");
        save_tokens(&old_tokens);
        let new_tokens = make_tokens("new_test_token", "new_refresh_token");
        save_tokens(&new_tokens);
        let retrieved_tokens = get_tokens_from_cache().unwrap();
        assert_eq!(retrieved_tokens.access_token, new_tokens.access_token);
        assert_eq!(retrieved_tokens.refresh_token, new_tokens.refresh_token);
        cleanup_test_environment();
    }

    #[test]
    #[serial]
    fn test_return_none_when_cache_file_does_not_exist() {
        cleanup_test_environment();
        let path = get_auth_cache_file_path();
        // Ensure the file does not exist
        if path.exists() {
            fs::remove_file(&path).unwrap();
        }
        assert!(get_tokens_from_cache().is_none());
        cleanup_test_environment();
    }

    #[test]
    #[serial]
    fn test_return_none_when_cache_file_exists_but_is_empty() {
        cleanup_test_environment();
        // Create an empty file
        let path = get_auth_cache_file_path();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("directory tree should be created");
        }
        File::create(&path).expect("empty file should be created");
        assert!(
            get_tokens_from_cache().is_none(),
            "Expected None for empty cache file, got Some"
        );
        cleanup_test_environment();
    }

    #[test]
    #[serial]
    fn test_return_pat_when_environment_variable_is_defined() {
        cleanup_test_environment();
        let bot_token = "github_pat_example_token_123";
        unsafe {
            std::env::set_var(GITHUB_BOT_TOKEN_ENV_VAR, bot_token);
        }
        let tokens = get_tokens().expect("Expected token from environment");
        assert_eq!(tokens.access_token, bot_token);
        assert_eq!(tokens.refresh_token, None);
        cleanup_test_environment();
        unsafe {
            std::env::remove_var(GITHUB_BOT_TOKEN_ENV_VAR);
        }
    }

    #[test]
    #[serial]
    fn test_return_user_tokens_when_environment_variable_is_not_defined() {
        cleanup_test_environment();
        unsafe {
            std::env::remove_var(GITHUB_BOT_TOKEN_ENV_VAR);
        }
        let cached_tokens = make_tokens("cached_access", "cached_refresh");
        save_tokens(&cached_tokens);
        let tokens = get_tokens_with_verifier(|_| true).expect("Expected tokens from cache");
        assert_eq!(tokens.access_token, cached_tokens.access_token);
        assert_eq!(tokens.refresh_token, cached_tokens.refresh_token);
        cleanup_test_environment();
        unsafe {
            std::env::remove_var(GITHUB_BOT_TOKEN_ENV_VAR);
        }
    }

    #[test]
    fn test_get_auth_header_value_with_ghu_token() {
        let token = "ghu_abc123";
        let header = get_auth_header_value(token);
        assert_eq!(header, "Bearer ghu_abc123");
    }

    #[test]
    fn test_get_auth_header_value_with_github_pat_token() {
        let token = "github_pat_abc123";
        let header = get_auth_header_value(token);
        assert_eq!(header, "token github_pat_abc123");
    }

    #[test]
    #[should_panic(expected = "Unsupported token format")]
    fn test_get_auth_header_value_with_unsupported_token() {
        let token = "invalid_token_format";
        let _ = get_auth_header_value(token);
    }
}
