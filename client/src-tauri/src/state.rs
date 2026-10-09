use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use crate::miners::MinerManager;

/// Shared application state managed by Tauri.
pub struct AppState {
    pub miners: Mutex<MinerManager>,
    /// Whether closing the window hides to the tray instead of quitting.
    ///
    /// A plain atomic rather than a guard on `miners`: the close handler runs on
    /// the main thread and must read this *without* ever taking the miner lock,
    /// which a long-running engine start can hold.
    pub close_to_tray: AtomicBool,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            miners: Mutex::new(MinerManager::new()),
            // Matches the frontend default. The webview corrects it on first read
            // if the user has turned the setting off.
            close_to_tray: AtomicBool::new(true),
        }
    }

    pub fn set_close_to_tray(&self, enabled: bool) {
        self.close_to_tray.store(enabled, Ordering::Relaxed);
    }

    pub fn close_to_tray(&self) -> bool {
        self.close_to_tray.load(Ordering::Relaxed)
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}
