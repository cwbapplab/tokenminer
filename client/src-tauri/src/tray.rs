//! The system-tray icon behind "close to tray".
//!
//! Mining runs on threads that belong to this process, not to the window, so
//! hiding the window keeps the miners alive with no extra work — but it also
//! removes the last way back to the UI unless one is put in the tray first.
//! This module adds that route: a menu to restore the window (or quit for real),
//! and a left-click on the icon that does the same as "Open".
//!
//! The close itself is intercepted in the crate root, not here; the tray only
//! offers the inverse operations.

use tauri::{
    menu::{Menu, MenuItem},
    tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent},
    App, AppHandle, Manager,
};

/// Menu-item ids, matched in the tray's menu-event handler.
const MENU_OPEN: &str = "tray-open";
const MENU_QUIT: &str = "tray-quit";

/// Builds the tray icon and its menu. Call once from `setup`, after the window
/// exists so the handlers can find it.
pub fn setup(app: &App) -> tauri::Result<()> {
    let handle = app.handle();

    let open = MenuItem::with_id(handle, MENU_OPEN, "Open TokenMiner", true, None::<&str>)?;
    let quit = MenuItem::with_id(handle, MENU_QUIT, "Quit", true, None::<&str>)?;
    let menu = Menu::with_items(handle, &[&open, &quit])?;

    let mut builder = TrayIconBuilder::with_id("main-tray")
        .menu(&menu)
        // Reuse the window icon so the tray matches the taskbar without shipping
        // a second asset.
        .show_menu_on_left_click(false)
        .on_menu_event(|app, event| match event.id().as_ref() {
            MENU_OPEN => show_main(app),
            MENU_QUIT => app.exit(0),
            _ => {}
        })
        .on_tray_icon_event(|tray, event| {
            // Left *click* (not the menu) restores the window, the convention on
            // Windows. The menu still opens on right-click.
            if let TrayIconEvent::Click {
                button: MouseButton::Left,
                button_state: MouseButtonState::Up,
                ..
            } = event
            {
                show_main(tray.app_handle());
            }
        });

    if let Some(icon) = app.default_window_icon() {
        builder = builder.icon(icon.clone());
    }

    builder.build(app)?;
    Ok(())
}

/// Shows and focuses the main window, undoing whatever a close-to-tray hide did.
pub fn show_main(app: &AppHandle) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.show();
        let _ = window.unminimize();
        let _ = window.set_focus();
    }
}
