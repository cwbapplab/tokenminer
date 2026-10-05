mod miners;
mod state;
// Only the Pearl engine drives a Stratum session, so the module is compiled with it.
#[cfg(feature = "pearl")]
mod stratum;

use state::AppState;
use tauri_plugin_log::{Target, TargetKind};

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // `RUST_LOG=trace npm run tauri dev` for full verbosity; defaults to Info.
    let level = match std::env::var("RUST_LOG")
        .unwrap_or_default()
        .to_ascii_lowercase()
        .as_str()
    {
        "trace" => log::LevelFilter::Trace,
        "debug" => log::LevelFilter::Debug,
        "warn" => log::LevelFilter::Warn,
        "error" => log::LevelFilter::Error,
        _ => log::LevelFilter::Info,
    };

    let mut builder = tauri::Builder::default();

    // Must be registered first, and before `deep-link`, so a second launch of the
    // app on an already-running instance routes through the deep-link handler.
    #[cfg(desktop)]
    {
        builder = builder.plugin(tauri_plugin_single_instance::init(|_app, _argv, _cwd| {}));
    }

    builder
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_http::init())
        .plugin(
            tauri_plugin_log::Builder::new()
                .level(level)
                // Stdout + a rotating file for `tauri dev` / tailing, and the
                // Webview target so the UI's "Backend log" shows Rust output.
                .target(Target::new(TargetKind::Stdout))
                .target(Target::new(TargetKind::LogDir {
                    file_name: Some("tokenminer".into()),
                }))
                .target(Target::new(TargetKind::Webview))
                .build(),
        )
        .plugin(tauri_plugin_process::init())
        .plugin(tauri_plugin_notification::init())
        .plugin(tauri_plugin_store::Builder::new().build())
        .plugin(tauri_plugin_deep_link::init())
        .manage(AppState::new())
        .invoke_handler(tauri::generate_handler![
            miners::start_miner,
            miners::start_session_miner,
            miners::stop_miner,
            miners::get_miner_status,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
