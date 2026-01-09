"""
iiSU Icon Generator with integrated Border Generator
Main application with tabbed interface
"""
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QUrl, QSize
from PySide6.QtGui import QIcon, QFontDatabase, QDesktopServices, QPixmap, QPainter, QColor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton
)

from app_paths import get_app_dir, get_logo_path, get_theme_path, get_fonts_dir, get_src_dir


def create_colored_icon(icon_path: Path, color: QColor) -> QIcon:
    """Create a colored version of an icon by tinting it."""
    if not icon_path.exists():
        return QIcon()

    pixmap = QPixmap(str(icon_path))
    if pixmap.isNull():
        return QIcon()

    # Create a colored version - paint the color over the icon using composition
    colored = QPixmap(pixmap.size())
    colored.fill(Qt.transparent)

    painter = QPainter(colored)
    painter.setCompositionMode(QPainter.CompositionMode_Source)
    painter.drawPixmap(0, 0, pixmap)
    painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
    painter.fillRect(colored.rect(), color)
    painter.end()

    return QIcon(colored)

# Import UI components
from icon_generator_tab import IconGeneratorTab
from border_generator_tab import BorderGeneratorTab
from custom_image_tab import CustomImageTab
from cover_generator_tab import CoverGeneratorTab


class MainWindowWithTabs(QMainWindow):
    """Main window with tabbed interface for Icon Generator and Border Generator."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("iiSU Asset Tool")
        self.setMinimumSize(1200, 800)

        # Set window icon if logo exists
        logo_path = get_logo_path()
        if logo_path.exists():
            self.setWindowIcon(QIcon(str(logo_path)))

        # Load iiSU theme stylesheet
        self._load_theme()

        # Create central widget with tabs
        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header with logo and title
        header_widget = QWidget()
        header_widget.setObjectName("app_header")
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(20, 15, 20, 15)

        # Logo
        if logo_path.exists():
            from PySide6.QtGui import QPixmap
            logo_label = QLabel()
            logo_pixmap = QPixmap(str(logo_path))
            scaled_logo = logo_pixmap.scaledToHeight(48, Qt.SmoothTransformation)
            logo_label.setPixmap(scaled_logo)
            header_layout.addWidget(logo_label)

        # Title
        title = QLabel("iiSU Asset Tool")
        title.setObjectName("header")
        header_layout.addWidget(title)

        header_layout.addStretch(1)

        # Icon paths
        src_dir = get_src_dir()
        info_icon_path = src_dir / "InfoIcon.png"
        gear_icon_path = src_dir / "GearIcon.png"

        # Icon color for current theme (white for dark theme)
        icon_color = QColor("#FFFFFF")

        # Info button (about/credits)
        self.btn_info = QPushButton()
        self.btn_info.setMaximumWidth(40)
        self.btn_info.setMinimumWidth(40)
        self.btn_info.setMaximumHeight(40)
        self.btn_info.setToolTip("About & Credits")
        if info_icon_path.exists():
            self.btn_info.setIcon(create_colored_icon(info_icon_path, icon_color))
            self.btn_info.setIconSize(QSize(20, 20))
        else:
            self.btn_info.setText("ℹ")
            self.btn_info.setStyleSheet("QPushButton { font-size: 16px; }")
        self.btn_info.clicked.connect(self._show_info_dialog)
        header_layout.addWidget(self.btn_info)

        # Settings button
        self.btn_options = QPushButton()
        self.btn_options.setMaximumWidth(40)
        self.btn_options.setMinimumWidth(40)
        self.btn_options.setMaximumHeight(40)
        self.btn_options.setToolTip("Settings")
        if gear_icon_path.exists():
            self.btn_options.setIcon(create_colored_icon(gear_icon_path, icon_color))
            self.btn_options.setIconSize(QSize(20, 20))
        else:
            self.btn_options.setText("⚙")
            self.btn_options.setStyleSheet("QPushButton { font-size: 16px; }")
        self.btn_options.clicked.connect(self._open_settings)
        header_layout.addWidget(self.btn_options)

        layout.addWidget(header_widget)

        # Tab widget
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.North)
        self.tabs.setMovable(False)

        # Add tabs
        self.tabs.addTab(IconGeneratorTab(), "Icon Scraper")
        self.tabs.addTab(CustomImageTab(), "Custom Icons")
        self.tabs.addTab(BorderGeneratorTab(), "Custom Borders")
        self.tabs.addTab(CoverGeneratorTab(), "Custom Covers")

        layout.addWidget(self.tabs)

    def _load_theme(self):
        """Load iiSU theme stylesheet."""
        theme_path = get_theme_path()
        if theme_path.exists():
            try:
                with open(theme_path, 'r', encoding='utf-8') as f:
                    self.setStyleSheet(f.read())
            except Exception as e:
                print(f"Failed to load theme: {e}")
                self.setStyleSheet("QWidget { background-color: #212529; color: #E9E9E9; }")
        else:
            print("Theme file not found, using fallback")
            self.setStyleSheet("QWidget { background-color: #212529; color: #E9E9E9; }")

    def _show_info_dialog(self):
        """Show info dialog with sources and credits."""
        from PySide6.QtWidgets import QDialog, QTextBrowser, QDialogButtonBox

        dialog = QDialog(self)
        dialog.setWindowTitle("About iiSU Asset Tool")
        dialog.setMinimumSize(500, 450)

        layout = QVBoxLayout(dialog)

        text = QTextBrowser()
        text.setOpenExternalLinks(True)
        text.setHtml("""
        <h2>iiSU Asset Tool</h2>
        <p>Create custom icons, borders, and covers for your game library.</p>

        <h3>Artwork Sources</h3>
        <ul>
            <li><a href="https://www.steamgriddb.com/" style="color: #3d7eff;">SteamGridDB</a> - Community artwork database</li>
            <li><a href="https://thumbnails.libretro.com/" style="color: #3d7eff;">Libretro Thumbnails</a> - RetroArch thumbnails</li>
            <li><a href="https://www.igdb.com/" style="color: #3d7eff;">IGDB</a> - Internet Game Database</li>
            <li><a href="https://thegamesdb.net/" style="color: #3d7eff;">TheGamesDB</a> - Game information database</li>
        </ul>

        <h3>Credits</h3>
        <p>Built for the <a href="https://iisu.network/" style="color: #3d7eff;">iiSU Network</a> community.</p>
        <p>Special thanks to the iiSU team for the design aesthetic and inspiration.</p>

        <h3>License</h3>
        <p>This tool is provided as-is for creating custom game library assets.<br>
        Ensure compliance with artwork source terms of service.</p>
        """)
        layout.addWidget(text)

        button_box = QDialogButtonBox(QDialogButtonBox.Close)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        dialog.exec()

    def _open_settings(self):
        """Open the settings dialog."""
        from options_dialog import OptionsDialog

        # Get the current icon generator tab to access its settings
        icon_tab = self.tabs.widget(0)  # Icon Scraper is the first tab
        if hasattr(icon_tab, 'config_path'):
            dialog = OptionsDialog(
                parent=self,
                config_path=icon_tab.config_path,
                workers=icon_tab.workers_value,
                limit=icon_tab.limit_value,
                source_priority_widget=icon_tab.source_priority
            )

            if dialog.exec():
                # Update the icon tab's stored values
                icon_tab.config_path = dialog.get_config_path()
                icon_tab.workers_value = dialog.get_workers()
                icon_tab.limit_value = dialog.get_limit()

                # Update source priority
                source_order = dialog.get_source_order()
                icon_tab.source_priority.set_source_order(source_order)

                # Reload platforms if config changed
                icon_tab.load_platforms_from_config()


def main():
    # Windows: Set App User Model ID for taskbar icon
    import platform
    if platform.system() == "Windows":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("iiSU.IconGenerator.1.0")
        except:
            pass

    # Load saved API keys into environment on startup
    try:
        from api_key_manager import get_manager
        key_manager = get_manager()
        # Just accessing the keys will set environment variables if stored
        for service in ["steamgriddb", "igdb_client_id", "igdb_client_secret", "thegamesdb"]:
            key_manager.get_key(service)
    except Exception as e:
        print(f"Note: Could not load saved API keys: {e}")

    app = QApplication(sys.argv)

    # Load Continuum Bold font if available
    fonts_dir = get_fonts_dir()
    if fonts_dir.exists():
        for font_file in fonts_dir.glob("*.ttf"):
            font_id = QFontDatabase.addApplicationFont(str(font_file))
            if font_id != -1:
                font_families = QFontDatabase.applicationFontFamilies(font_id)
                print(f"Loaded font: {', '.join(font_families)}")
        for font_file in fonts_dir.glob("*.otf"):
            font_id = QFontDatabase.addApplicationFont(str(font_file))
            if font_id != -1:
                font_families = QFontDatabase.applicationFontFamilies(font_id)
                print(f"Loaded font: {', '.join(font_families)}")

    # Set application icon for taskbar
    logo_path = get_logo_path()
    if logo_path.exists():
        app.setWindowIcon(QIcon(str(logo_path)))

    w = MainWindowWithTabs()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
