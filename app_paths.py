"""Utility module for resolving application paths in both script and frozen modes."""
import sys
from pathlib import Path


def get_app_dir() -> Path:
    """Get the application's base directory.

    For frozen apps (PyInstaller), this is the directory containing the executable.
    For scripts, this is the script's directory.
    Assets should be placed alongside the executable.
    """
    if getattr(sys, 'frozen', False):
        # Running as compiled executable - assets are next to the exe
        return Path(sys.executable).parent
    else:
        # Running as script - use this file's location
        return Path(__file__).parent


def get_resource_path(relative_path: str) -> Path:
    """Get absolute path to a resource, works for dev and PyInstaller."""
    return get_app_dir() / relative_path


# All paths relative to app directory (next to executable)
def get_templates_dir() -> Path:
    return get_app_dir() / "templates"


def get_borders_dir() -> Path:
    return get_app_dir() / "borders"


def get_fonts_dir() -> Path:
    return get_app_dir() / "fonts"


def get_platform_icons_dir() -> Path:
    return get_app_dir() / "platform_icons"


def get_src_dir() -> Path:
    return get_app_dir() / "src"


def get_logo_path() -> Path:
    return get_app_dir() / "logo.png"


def get_theme_path() -> Path:
    return get_app_dir() / "iisu_theme.qss"


def get_config_path() -> Path:
    return get_app_dir() / "config.yaml"
