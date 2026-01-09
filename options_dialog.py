"""
Options Dialog for Icon Generator
Handles configuration, workers, limits, source priority settings, API keys, and ROM directory settings
"""
import os
from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QPushButton, QLineEdit, QSpinBox, QFileDialog,
    QGroupBox, QDialogButtonBox, QMessageBox, QScrollArea, QWidget,
    QComboBox, QCheckBox
)

from source_priority_widget import SourcePriorityWidget
from api_key_manager import get_manager
from rom_parser import get_available_drives, find_iisu_directory


class OptionsDialog(QDialog):
    """Options dialog for configuring Icon Generator settings."""

    def __init__(self, parent=None, config_path="", workers=8, limit=0, source_priority_widget=None,
                 rom_directory_settings=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(650)
        self.setMinimumHeight(600)

        # Store initial values
        self.config_path_value = config_path
        self.workers_value = workers
        self.limit_value = limit
        self.source_priority_widget_ref = source_priority_widget

        # ROM directory settings
        self.rom_directory_settings = rom_directory_settings or {
            "mode": "manual",
            "rom_path": "",
            "auto_detect": False,
            "remember_last_path": True
        }

        # Hero image settings
        self.hero_settings = {
            "enabled": True,
            "count": 1,
            "save_with_icons": True
        }

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        # Create scroll area for all content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)

        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(16)

        # Config Group
        config_group = QGroupBox("Configuration")
        config_layout = QFormLayout()
        config_layout.setSpacing(12)

        # Config path
        config_row = QHBoxLayout()
        self.config_path = QLineEdit(self.config_path_value)
        btn_browse = QPushButton("Browse...")
        btn_browse.clicked.connect(self._browse_config)
        config_row.addWidget(self.config_path, 1)
        config_row.addWidget(btn_browse)
        config_layout.addRow("Config File:", config_row)

        config_group.setLayout(config_layout)
        scroll_layout.addWidget(config_group)

        # API Keys Group
        api_group = QGroupBox("API Keys")
        api_layout = QFormLayout()
        api_layout.setSpacing(12)

        # Get API key manager
        key_manager = get_manager()

        # SteamGridDB API Key
        self.sgdb_key = QLineEdit()
        self.sgdb_key.setPlaceholderText("Enter SteamGridDB API key (optional)")
        self.sgdb_key.setEchoMode(QLineEdit.Password)
        current_sgdb = key_manager.get_key("steamgriddb")
        if current_sgdb:
            self.sgdb_key.setText(current_sgdb)
        sgdb_row = QHBoxLayout()
        sgdb_row.addWidget(self.sgdb_key, 1)
        btn_sgdb_show = QPushButton("Show")
        btn_sgdb_show.setMaximumWidth(60)
        btn_sgdb_show.clicked.connect(lambda: self._toggle_password_visibility(self.sgdb_key))
        sgdb_row.addWidget(btn_sgdb_show)
        api_layout.addRow("SteamGridDB:", sgdb_row)

        sgdb_help = QLabel('<a href="https://www.steamgriddb.com/profile/preferences/api">Get SteamGridDB API key</a>')
        sgdb_help.setOpenExternalLinks(True)
        sgdb_help.setStyleSheet("color: #00DDFF; font-size: 10px;")
        api_layout.addRow("", sgdb_help)

        # IGDB API Keys
        self.igdb_client_id = QLineEdit()
        self.igdb_client_id.setPlaceholderText("Enter IGDB Client ID (optional)")
        self.igdb_client_id.setEchoMode(QLineEdit.Password)
        current_igdb_id = key_manager.get_key("igdb_client_id")
        if current_igdb_id:
            self.igdb_client_id.setText(current_igdb_id)
        igdb_id_row = QHBoxLayout()
        igdb_id_row.addWidget(self.igdb_client_id, 1)
        btn_igdb_id_show = QPushButton("Show")
        btn_igdb_id_show.setMaximumWidth(60)
        btn_igdb_id_show.clicked.connect(lambda: self._toggle_password_visibility(self.igdb_client_id))
        igdb_id_row.addWidget(btn_igdb_id_show)
        api_layout.addRow("IGDB Client ID:", igdb_id_row)

        self.igdb_client_secret = QLineEdit()
        self.igdb_client_secret.setPlaceholderText("Enter IGDB Client Secret (optional)")
        self.igdb_client_secret.setEchoMode(QLineEdit.Password)
        current_igdb_secret = key_manager.get_key("igdb_client_secret")
        if current_igdb_secret:
            self.igdb_client_secret.setText(current_igdb_secret)
        igdb_secret_row = QHBoxLayout()
        igdb_secret_row.addWidget(self.igdb_client_secret, 1)
        btn_igdb_secret_show = QPushButton("Show")
        btn_igdb_secret_show.setMaximumWidth(60)
        btn_igdb_secret_show.clicked.connect(lambda: self._toggle_password_visibility(self.igdb_client_secret))
        igdb_secret_row.addWidget(btn_igdb_secret_show)
        api_layout.addRow("IGDB Client Secret:", igdb_secret_row)

        igdb_help = QLabel('<a href="https://api-docs.igdb.com/#account-creation">Get IGDB API credentials</a>')
        igdb_help.setOpenExternalLinks(True)
        igdb_help.setStyleSheet("color: #00DDFF; font-size: 10px;")
        api_layout.addRow("", igdb_help)

        # TheGamesDB API Key - has built-in key, so this is optional override
        tgdb_note = QLabel("TheGamesDB: Using built-in API key (no configuration needed)")
        tgdb_note.setStyleSheet("color: #4CAF50; font-size: 11px;")
        api_layout.addRow(tgdb_note)

        api_note = QLabel("API keys are securely stored locally and will persist between sessions.")
        api_note.setWordWrap(True)
        api_note.setStyleSheet("color: #B0B0B0; font-size: 10px; margin-top: 8px;")
        api_layout.addRow(api_note)

        api_group.setLayout(api_layout)
        scroll_layout.addWidget(api_group)

        # ROM Directory Group
        rom_group = QGroupBox("ROM Directory")
        rom_layout = QFormLayout()
        rom_layout.setSpacing(12)

        # ROM path - manual selection by default
        rom_path_row = QHBoxLayout()
        self.rom_path = QLineEdit(self.rom_directory_settings.get("rom_path", ""))
        self.rom_path.setPlaceholderText("Browse to select your ROM folder...")
        rom_path_row.addWidget(self.rom_path, 1)

        btn_browse_rom = QPushButton("Browse...")
        btn_browse_rom.setToolTip("Select ROM directory (USB drive, external storage, etc.)")
        btn_browse_rom.clicked.connect(self._browse_rom_path)
        rom_path_row.addWidget(btn_browse_rom)

        rom_layout.addRow("ROM Folder:", rom_path_row)

        # Available drives display for convenience
        drives = get_available_drives()
        if drives:
            drives_text = ", ".join([d[1] for d in drives[:6]])
            if len(drives) > 6:
                drives_text += f" (+{len(drives) - 6} more)"
            drives_label = QLabel(f"<span style='color: #888; font-size: 10px;'>Connected drives: {drives_text}</span>")
            rom_layout.addRow("", drives_label)

        # Auto-detect option
        auto_detect_row = QHBoxLayout()
        btn_auto_detect = QPushButton("Search for ROM Folders")
        btn_auto_detect.setToolTip("Search connected USB drives and external storage for ROM directories")
        btn_auto_detect.clicked.connect(self._auto_detect_rom_folder)
        auto_detect_row.addWidget(btn_auto_detect)
        auto_detect_row.addStretch()
        rom_layout.addRow("", auto_detect_row)

        # Remember path
        self.remember_rom_path = QCheckBox("Remember selected folder for next session")
        self.remember_rom_path.setChecked(self.rom_directory_settings.get("remember_last_path", True))
        rom_layout.addRow("", self.remember_rom_path)

        # Help text
        rom_help = QLabel(
            "<span style='color: #888; font-size: 10px;'>"
            "Supports: USB drives, external hard drives, SD cards, Android device storage, network paths"
            "</span>"
        )
        rom_help.setWordWrap(True)
        rom_layout.addRow(rom_help)

        rom_group.setLayout(rom_layout)
        scroll_layout.addWidget(rom_group)

        # Hero Images Group
        hero_group = QGroupBox("Hero Images (SteamGridDB)")
        hero_layout = QFormLayout()
        hero_layout.setSpacing(12)

        # Enable hero downloads
        self.hero_enabled = QCheckBox("Download hero images when processing")
        self.hero_enabled.setChecked(self.hero_settings.get("enabled", True))
        self.hero_enabled.setToolTip("Download hero/banner images from SteamGridDB alongside icons")
        hero_layout.addRow("", self.hero_enabled)

        # Number of heroes to download
        self.hero_count = QSpinBox()
        self.hero_count.setRange(1, 5)
        self.hero_count.setValue(self.hero_settings.get("count", 1))
        self.hero_count.setToolTip("Number of hero images to download per game (top rated)")
        hero_layout.addRow("Heroes per game:", self.hero_count)

        # Save with icons
        self.hero_save_with_icons = QCheckBox("Save heroes in same folder as icons")
        self.hero_save_with_icons.setChecked(self.hero_settings.get("save_with_icons", True))
        hero_layout.addRow("", self.hero_save_with_icons)

        hero_note = QLabel("Hero images are wide banner images (1920x620) commonly used for Steam grid backgrounds.")
        hero_note.setWordWrap(True)
        hero_note.setStyleSheet("color: #888; font-size: 10px;")
        hero_layout.addRow(hero_note)

        hero_group.setLayout(hero_layout)
        scroll_layout.addWidget(hero_group)

        # Processing Group
        processing_group = QGroupBox("Processing Settings")
        processing_layout = QFormLayout()
        processing_layout.setSpacing(12)

        # Workers
        self.workers = QSpinBox()
        self.workers.setRange(1, 64)
        self.workers.setValue(self.workers_value)
        self.workers.setToolTip("Number of concurrent workers for parallel processing")
        processing_layout.addRow("Workers:", self.workers)

        # Limit
        self.limit = QSpinBox()
        self.limit.setRange(0, 2_000_000_000)
        self.limit.setValue(self.limit_value)
        self.limit.setToolTip("Maximum number of games to process per platform (0 = unlimited)")
        processing_layout.addRow("Per-Platform Limit:", self.limit)

        processing_group.setLayout(processing_layout)
        scroll_layout.addWidget(processing_group)

        # Source Priority Group
        source_group = QGroupBox("Artwork Source Priority")
        source_layout = QVBoxLayout()
        source_layout.setSpacing(8)

        # Add the source priority widget
        self.source_priority = SourcePriorityWidget()

        # Copy current order if we have a reference
        if self.source_priority_widget_ref:
            current_order = self.source_priority_widget_ref.get_source_order()
            self.source_priority.set_source_order(current_order)

        source_layout.addWidget(QLabel("Drag sources to reorder priority:"))
        source_layout.addWidget(self.source_priority)

        # Save config button
        btn_save_config = QPushButton("Save Priority to Config")
        btn_save_config.setToolTip("Save source priority to config.yaml")
        btn_save_config.clicked.connect(self._save_to_config)
        source_layout.addWidget(btn_save_config)

        source_group.setLayout(source_layout)
        scroll_layout.addWidget(source_group, 1)

        # Set scroll widget
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll, 1)

        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self._apply_api_keys)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _browse_config(self):
        """Browse for config file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Config File", str(Path.home()), "YAML (*.yaml *.yml)"
        )
        if path:
            self.config_path.setText(path)

    def _save_to_config(self):
        """Save source priority to config file."""
        if self.parent():
            self.parent().save_source_order_to_config(self.source_priority.get_source_order())

    def get_config_path(self):
        """Get the selected config path."""
        return self.config_path.text()

    def get_workers(self):
        """Get the number of workers."""
        return self.workers.value()

    def get_limit(self):
        """Get the per-platform limit."""
        return self.limit.value()

    def get_source_order(self):
        """Get the source priority order."""
        return self.source_priority.get_source_order()

    def _toggle_password_visibility(self, line_edit: QLineEdit):
        """Toggle password visibility for a line edit."""
        if line_edit.echoMode() == QLineEdit.Password:
            line_edit.setEchoMode(QLineEdit.Normal)
        else:
            line_edit.setEchoMode(QLineEdit.Password)

    def _browse_rom_path(self):
        """Browse for ROM directory."""
        current_path = self.rom_path.text()
        start_dir = current_path if current_path and Path(current_path).exists() else str(Path.home())

        path = QFileDialog.getExistingDirectory(
            self,
            "Select ROM Directory",
            start_dir,
            QFileDialog.ShowDirsOnly
        )
        if path:
            self.rom_path.setText(path)

    def _auto_detect_rom_folder(self):
        """Attempt to find ROM directories on connected drives."""
        QMessageBox.information(
            self,
            "Searching",
            "Searching connected drives for ROM folders...\n\n"
            "This may take a moment."
        )

        found_path = find_iisu_directory()
        if found_path:
            self.rom_path.setText(str(found_path))
            QMessageBox.information(
                self,
                "Found",
                f"Found ROM directory at:\n{found_path}"
            )
        else:
            QMessageBox.information(
                self,
                "Not Found",
                "No ROM directories were automatically detected.\n\n"
                "Use the Browse button to manually select your ROM folder\n"
                "from a USB drive or external storage device."
            )

    def get_rom_directory_settings(self):
        """Get the ROM directory settings."""
        rom_path = self.rom_path.text().strip()
        return {
            "mode": "manual" if not rom_path else "remembered",
            "rom_path": rom_path,
            "auto_detect": False,
            "remember_last_path": self.remember_rom_path.isChecked()
        }

    def get_hero_settings(self):
        """Get the hero image settings."""
        return {
            "enabled": self.hero_enabled.isChecked(),
            "count": self.hero_count.value(),
            "save_with_icons": self.hero_save_with_icons.isChecked()
        }

    def _apply_api_keys(self):
        """Save API keys to encrypted storage and accept dialog."""
        key_manager = get_manager()

        # Save all keys (also sets environment variables for current session)
        # Note: TheGamesDB uses a built-in key, so no user input needed
        key_manager.set_key("steamgriddb", self.sgdb_key.text().strip())
        key_manager.set_key("igdb_client_id", self.igdb_client_id.text().strip())
        key_manager.set_key("igdb_client_secret", self.igdb_client_secret.text().strip())

        QMessageBox.information(
            self,
            "Settings Saved",
            "API keys have been saved securely and will persist between sessions."
        )

        self.accept()
