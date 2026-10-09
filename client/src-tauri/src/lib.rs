mod miners;
mod state;
// Only the Pearl engine drives a Stratum session, so the module is compiled with it.
#[cfg(feature = "pearl")]
mod stratum;
// The tray is the only route back to the UI once the window is hidden, so it is
// desktop-only along with the close-to-tray behaviour that hides the window.
#[cfg(desktop)]
mod tray;

use state::AppState;
// `Manager` supplies `try_state` / `app_handle` on the app and window types used
// by the desktop close-to-tray handlers below.
#[cfg(desktop)]
use tauri::Manager;
use tauri_plugin_log::{Target, TargetKind};

/// Flips the close-to-tray preference the window's close handler reads.
///
/// Persistence lives in the frontend (localStorage); this only mirrors the value
/// into the backend so the close can be intercepted without a round-trip to the
/// webview. Mobile has no tray, so the command is a no-op there.
#[cfg(desktop)]
#[tauri::command]
fn set_close_to_tray(state: tauri::State<'_, AppState>, enabled: bool) {
    state.set_close_to_tray(enabled);
}

#[cfg(not(desktop))]
#[tauri::command]
fn set_close_to_tray(_state: tauri::State<'_, AppState>, _enabled: bool) {}

/// Shows a system notification when a close-to-tray hide happens *while mining*.
///
/// The message is specifically "mining keeps running", so it is only sent when an
/// engine is actually active — otherwise the window just hides, quietly. The miner
/// lock is taken with `try_lock` (and never awaited): this runs on the main thread
/// inside the close handler, and an engine start can be holding that lock.
#[cfg(desktop)]
fn notify_hidden(app: &tauri::AppHandle, state: &AppState) {
    use crate::miners::MinerState;
    use tauri_plugin_notification::NotificationExt;

    let mining = state
        .miners
        .try_lock()
        .map(|manager| {
            manager
                .statuses()
                .iter()
                .any(|s| matches!(s.state, MinerState::Starting | MinerState::Running))
        })
        .unwrap_or(false);

    if !mining {
        return;
    }

    let _ = app
        .notification()
        .builder()
        .title("TokenMiner is still mining")
        .body("Closing the window sent TokenMiner to the tray. Mining keeps running in the background — open the tray icon to return.")
        .show();
}

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
        builder = builder.plugin(tauri_plugin_single_instance::init(|app, _argv, _cwd| {
            // Re-launching the app is the usual way back to a window that close-to-tray
            // hid, so surface it instead of running a second copy.
            tray::show_main(app);
        }));
    }

    builder = builder
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
        .manage(AppState::new());

    // Close-to-tray: build the tray icon, then intercept the window close. Both
    // live on the app/window level rather than the plugin level because the
    // window itself is what has to be caught before it is destroyed.
    #[cfg(desktop)]
    {
        builder = builder
            .setup(|app| {
                if let Err(error) = tray::setup(app) {
                    // A tray that fails to build must not take the app down with
                    // it; the close handler below still hides to a reachable state
                    // only if the user can get back, so fall back to a real close.
                    log::error!("could not create the tray icon: {error}");
                    if let Some(state) = app.try_state::<AppState>() {
                        state.set_close_to_tray(false);
                    }
                }
                Ok(())
            })
            .on_window_event(|window, event| {
                if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                    let app = window.app_handle();
                    let Some(state) = app.try_state::<AppState>() else {
                        return;
                    };
                    // With the setting off, let the close proceed: the process
                    // exits and the miners die with it, as before this feature.
                    if !state.close_to_tray() {
                        return;
                    }

                    api.prevent_close();
                    if let Err(error) = window.hide() {
                        // Hiding failed — closing anyway would be the only way out,
                        // so log it rather than trapping the user in a window.
                        log::error!("could not hide the window to the tray: {error}");
                        return;
                    }
                    notify_hidden(&app, &state);
                }
            });
    }

    builder
        .invoke_handler(tauri::generate_handler![
            miners::start_miner,
            miners::start_session_miner,
            miners::stop_miner,
            miners::get_miner_status,
            set_close_to_tray,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
