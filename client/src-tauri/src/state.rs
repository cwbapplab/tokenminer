use std::sync::Mutex;

use crate::miners::MinerManager;

/// Shared application state managed by Tauri.
pub struct AppState {
    pub miners: Mutex<MinerManager>,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            miners: Mutex::new(MinerManager::new()),
        }
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}
