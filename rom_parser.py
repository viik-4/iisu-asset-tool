"""
ROM Parser Module for iiSU Asset Tool
Scans directories for ROM files and extracts game titles from folder/file names.
Supports iiSU ROM directory structure and manual folder selection.
"""
import os
import re
import string
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

# Common ROM file extensions by platform
ROM_EXTENSIONS = {
    # Nintendo
    "NES": {".nes", ".nez", ".unf", ".unif"},
    "SNES": {".smc", ".sfc", ".fig", ".swc"},
    "N64": {".n64", ".z64", ".v64"},
    "N64DD": {".ndd", ".n64"},
    "GAMECUBE": {".iso", ".gcm", ".gcz", ".rvz", ".wbfs", ".ciso"},
    "WII": {".iso", ".wbfs", ".rvz", ".wia", ".ciso"},
    "WII_U": {".wud", ".wux", ".rpx"},
    "SWITCH": {".nsp", ".xci", ".nsz", ".xcz"},
    "GAME_BOY": {".gb", ".gbc", ".sgb"},
    "GAME_BOY_COLOR": {".gbc", ".gb"},
    "GAME_BOY_ADVANCE": {".gba", ".agb"},
    "NINTENDO_DS": {".nds", ".dsi"},
    "NINTENDO_3DS": {".3ds", ".cia", ".cxi"},
    "VIRTUAL_BOY": {".vb", ".vboy"},
    # Sony
    "PS1": {".bin", ".cue", ".iso", ".img", ".pbp", ".chd"},
    "PS2": {".iso", ".bin", ".img", ".chd", ".cso"},
    "PS3": {".iso", ".pkg"},
    "PS4": {".pkg"},
    "PS5": {".pkg"},
    "PSP": {".iso", ".cso", ".pbp"},
    "PS_VITA": {".vpk", ".mai"},
    # Microsoft
    "XBOX": {".iso", ".xbe"},
    "XBOX_360": {".iso", ".xex", ".god"},
    # Sega
    "MASTER_SYSTEM": {".sms", ".sg"},
    "GENESIS": {".md", ".gen", ".bin", ".smd"},
    "SEGA_CD": {".iso", ".bin", ".cue", ".chd"},
    "SEGA_32X": {".32x", ".bin"},
    "SATURN": {".iso", ".bin", ".cue", ".chd"},
    "DREAMCAST": {".gdi", ".cdi", ".chd"},
    "GAME_GEAR": {".gg"},
    # SNK
    "NEO_GEO": {".zip", ".7z"},
    "NEO_GEO_CD": {".iso", ".bin", ".cue", ".chd"},
    "NEO_GEO_POCKET": {".ngp", ".ngc"},
    "NEO_GEO_POCKET_COLOR": {".ngc", ".ngp"},
    # Atari
    "ATARI_2600": {".a26", ".bin"},
    "ATARI_5200": {".a52", ".bin"},
    "ATARI_7800": {".a78", ".bin"},
    "ATARI_JAGUAR": {".j64", ".jag", ".rom", ".bin"},
    "ATARI_LYNX": {".lnx", ".lyx"},
    # Other classic
    "COLECOVISION": {".col", ".bin", ".rom"},
    "INTELLIVISION": {".int", ".bin", ".rom"},
    "TG16": {".pce", ".sgx"},
    "TG_CD": {".iso", ".bin", ".cue", ".chd"},
    "WONDERSWAN": {".ws"},
    "WONDERSWAN_COLOR": {".wsc", ".ws"},
    # Arcade/Other
    "MAME": {".zip", ".7z"},
    "FBA": {".zip", ".7z"},
    "SCUMMVM": {".scummvm"},
    "DOS": {".exe", ".com", ".bat"},
    "ANDROID": {".apk"},
}

# iiSU expected folder naming conventions (platform folder names in iiSU structure)
# Includes common naming variants: proper case, lowercase, no spaces, abbreviations
IISU_PLATFORM_FOLDERS = {
    "NES": ["NES", "Nintendo Entertainment System", "Famicom", "nes", "famicom", "fc"],
    "SNES": ["SNES", "Super Nintendo", "Super Famicom", "snes", "supernintendo", "superfamicom", "sfc"],
    "N64": ["N64", "Nintendo 64", "n64", "nintendo64"],
    "N64DD": ["N64DD", "Nintendo 64DD", "64DD", "n64dd"],
    "GAMECUBE": ["GameCube", "GC", "NGC", "gamecube", "gc", "ngc"],
    "WII": ["Wii", "wii"],
    "WII_U": ["Wii U", "WiiU", "wiiu"],
    "SWITCH": ["Switch", "Nintendo Switch", "NSW", "switch", "nsw"],
    "GAME_BOY": ["Game Boy", "GB", "gameboy", "gb"],
    "GAME_BOY_COLOR": ["Game Boy Color", "GBC", "gameboycolor", "gbc"],
    "GAME_BOY_ADVANCE": ["Game Boy Advance", "GBA", "gameboyadvance", "gba"],
    "NINTENDO_DS": ["Nintendo DS", "DS", "NDS", "nintendods", "ds", "nds"],
    "NINTENDO_3DS": ["Nintendo 3DS", "3DS", "3ds", "n3ds"],
    "PS1": ["PlayStation", "PS1", "PSX", "PS One", "playstation", "ps1", "psx", "psone"],
    "PS2": ["PlayStation 2", "PS2", "playstation2", "ps2"],
    "PS3": ["PlayStation 3", "PS3", "playstation3", "ps3"],
    "PS4": ["PlayStation 4", "PS4", "playstation4", "ps4"],
    "PS5": ["PlayStation 5", "PS5", "playstation5", "ps5"],
    "PSP": ["PSP", "PlayStation Portable", "psp", "playstationportable"],
    "PS_VITA": ["PS Vita", "PlayStation Vita", "Vita", "psvita", "playstationvita", "vita"],
    "XBOX": ["Xbox", "Original Xbox", "xbox", "originalxbox"],
    "XBOX_360": ["Xbox 360", "X360", "xbox360", "x360"],
    "MASTER_SYSTEM": ["Master System", "Sega Master System", "SMS", "mastersystem", "segamastersystem", "sms"],
    "GENESIS": ["Genesis", "Mega Drive", "Sega Genesis", "genesis", "megadrive", "segagenesis", "md"],
    "SEGA_CD": ["Sega CD", "Mega CD", "segacd", "megacd"],
    "SEGA_32X": ["32X", "Sega 32X", "32x", "sega32x"],
    "SATURN": ["Saturn", "Sega Saturn", "saturn", "segasaturn"],
    "DREAMCAST": ["Dreamcast", "Sega Dreamcast", "DC", "dreamcast", "segadreamcast", "dc"],
    "GAME_GEAR": ["Game Gear", "GG", "gamegear", "gg"],
    "NEO_GEO": ["Neo Geo", "NeoGeo", "neogeo", "ng"],
    "NEO_GEO_CD": ["Neo Geo CD", "NeoGeoCD", "neogeocd", "ngcd"],
    "NEO_GEO_POCKET": ["Neo Geo Pocket", "NGP", "neogeopocket", "ngp"],
    "NEO_GEO_POCKET_COLOR": ["Neo Geo Pocket Color", "NGPC", "neogeopocketcolor", "ngpc"],
    "ATARI_2600": ["Atari 2600", "atari2600", "2600"],
    "ATARI_5200": ["Atari 5200", "atari5200", "5200"],
    "ATARI_7800": ["Atari 7800", "atari7800", "7800"],
    "ATARI_JAGUAR": ["Atari Jaguar", "Jaguar", "atarijaguar", "jaguar"],
    "ATARI_LYNX": ["Atari Lynx", "Lynx", "atarilynx", "lynx"],
    "COLECOVISION": ["ColecoVision", "Coleco", "colecovision", "coleco"],
    "INTELLIVISION": ["Intellivision", "intellivision", "intv"],
    "TG16": ["TurboGrafx-16", "TG16", "PC Engine", "turbografx16", "tg16", "pcengine", "pce"],
    "TG_CD": ["TurboGrafx-CD", "TG-CD", "PC Engine CD", "turbografxcd", "tgcd", "pcecd"],
    "WONDERSWAN": ["WonderSwan", "wonderswan", "ws"],
    "WONDERSWAN_COLOR": ["WonderSwan Color", "wonderswancolor", "wsc"],
    "VIRTUAL_BOY": ["Virtual Boy", "virtualboy", "vb"],
    "MAME": ["MAME", "Arcade", "mame", "arcade"],
    "FBA": ["FBA", "Final Burn Alpha", "fba", "finalburnalpha", "fbneo"],
    "SCUMMVM": ["ScummVM", "scummvm"],
    "DOS": ["DOS", "DOSBox", "dos", "dosbox"],
    "ANDROID": ["Android", "android"],
}

# Build reverse lookup from folder name to platform key
def _build_folder_to_platform_map() -> Dict[str, str]:
    mapping = {}
    for platform_key, folder_names in IISU_PLATFORM_FOLDERS.items():
        for folder_name in folder_names:
            mapping[folder_name.lower()] = platform_key
    return mapping

FOLDER_TO_PLATFORM = _build_folder_to_platform_map()


def clean_game_title(name: str) -> str:
    """
    Clean a ROM filename/folder name to extract a clean game title.
    Removes region tags, version info, dump info, file extensions, etc.
    """
    # Remove file extension if present
    name = re.sub(r'\.(zip|7z|rar|' + '|'.join(ext.strip('.') for exts in ROM_EXTENSIONS.values() for ext in exts) + r')$', '', name, flags=re.IGNORECASE)

    # Remove common archive suffixes
    name = re.sub(r'\.(zip|7z|rar)$', '', name, flags=re.IGNORECASE)

    # Remove square bracket tags like [!], [U], [E], [J], [h], [b], etc.
    name = re.sub(r'\s*\[[^\]]*\]', '', name)

    # Remove parenthetical region/version tags
    # Match patterns like (USA), (Europe), (Rev A), (v1.0), (En,Fr,De), etc.
    name = re.sub(r'\s*\((USA|Europe|Japan|World|En|Fr|De|Es|It|Ja|Ko|Zh|Rev\s*[A-Z0-9]*|v\d+[.\d]*|Proto|Beta|Demo|Sample|Unl|Pirate|Virtual Console|[A-Za-z]{2}(,[A-Za-z]{2})*)\)', '', name, flags=re.IGNORECASE)

    # Remove parenthetical disc numbers
    name = re.sub(r'\s*\(Disc\s*\d+[^)]*\)', '', name, flags=re.IGNORECASE)

    # Remove trailing version numbers
    name = re.sub(r'\s+v\d+(\.\d+)*$', '', name, flags=re.IGNORECASE)

    # Remove leading/trailing whitespace and normalize spaces
    name = re.sub(r'\s+', ' ', name).strip()

    # Remove trailing dashes or underscores
    name = name.rstrip('-_ ')

    return name


def get_all_rom_extensions() -> Set[str]:
    """Get a set of all known ROM file extensions."""
    all_exts = set()
    for exts in ROM_EXTENSIONS.values():
        all_exts.update(exts)
    return all_exts


def is_rom_file(path: Path) -> bool:
    """Check if a file is likely a ROM based on extension."""
    return path.suffix.lower() in get_all_rom_extensions()


def is_archive_file(path: Path) -> bool:
    """Check if a file is an archive that might contain ROMs."""
    return path.suffix.lower() in {'.zip', '.7z', '.rar'}


def detect_platform_from_folder(folder_name: str) -> Optional[str]:
    """
    Attempt to detect the platform key from a folder name.
    Returns platform key (e.g., 'NES', 'PS2') or None if not recognized.
    """
    folder_lower = folder_name.lower().strip()

    # Direct match
    if folder_lower in FOLDER_TO_PLATFORM:
        return FOLDER_TO_PLATFORM[folder_lower]

    # Try partial matching
    for folder_variant, platform_key in FOLDER_TO_PLATFORM.items():
        if folder_variant in folder_lower or folder_lower in folder_variant:
            return platform_key

    return None


def scan_iisu_directory(root_path: Path) -> Dict[str, List[Tuple[str, Path]]]:
    """
    Scan an iiSU-style ROM directory structure.

    Expected structure:
    root_path/
    ├── NES/
    │   ├── Game 1/
    │   │   └── game1.nes
    │   ├── Game 2/
    │   │   └── game2.nes
    │   └── standalone_game.nes
    ├── SNES/
    │   └── ...
    └── ...

    Returns:
        Dict mapping platform_key -> List of (game_title, game_path) tuples
    """
    results: Dict[str, List[Tuple[str, Path]]] = {}

    if not root_path.exists() or not root_path.is_dir():
        return results

    # Scan top-level folders (platforms)
    for platform_folder in root_path.iterdir():
        if not platform_folder.is_dir():
            continue

        platform_key = detect_platform_from_folder(platform_folder.name)
        if not platform_key:
            continue

        if platform_key not in results:
            results[platform_key] = []

        # Scan the platform folder for games
        games = scan_platform_folder(platform_folder, platform_key)
        results[platform_key].extend(games)

    return results


def scan_platform_folder(platform_path: Path, platform_key: str) -> List[Tuple[str, Path]]:
    """
    Scan a single platform folder for games.

    Handles two common structures:
    1. One folder per game (folder name = game title)
    2. Loose ROM files (filename = game title)

    Returns:
        List of (game_title, game_path) tuples
    """
    games: List[Tuple[str, Path]] = []
    seen_titles: Set[str] = set()

    platform_exts = ROM_EXTENSIONS.get(platform_key, get_all_rom_extensions())

    for item in platform_path.iterdir():
        if item.is_dir():
            # Game folder - use folder name as title
            game_title = clean_game_title(item.name)
            if game_title and game_title.lower() not in seen_titles:
                seen_titles.add(game_title.lower())
                games.append((game_title, item))

        elif item.is_file():
            # Check if it's a ROM file or archive
            if item.suffix.lower() in platform_exts or is_archive_file(item):
                game_title = clean_game_title(item.stem)
                if game_title and game_title.lower() not in seen_titles:
                    seen_titles.add(game_title.lower())
                    games.append((game_title, item))

    # Sort by title
    games.sort(key=lambda x: x[0].lower())

    return games


def scan_generic_folder(folder_path: Path, platform_key: Optional[str] = None) -> List[Tuple[str, Path]]:
    """
    Scan a generic folder for ROM files/game folders.

    Args:
        folder_path: Path to scan
        platform_key: Optional platform key to filter by extension

    Returns:
        List of (game_title, game_path) tuples
    """
    games: List[Tuple[str, Path]] = []
    seen_titles: Set[str] = set()

    if platform_key:
        valid_exts = ROM_EXTENSIONS.get(platform_key, get_all_rom_extensions())
    else:
        valid_exts = get_all_rom_extensions()

    if not folder_path.exists() or not folder_path.is_dir():
        return games

    for item in folder_path.iterdir():
        if item.is_dir():
            # Check if folder contains ROMs (treat as game folder)
            has_roms = False
            for sub_item in item.iterdir():
                if sub_item.is_file() and (sub_item.suffix.lower() in valid_exts or is_archive_file(sub_item)):
                    has_roms = True
                    break

            if has_roms:
                game_title = clean_game_title(item.name)
                if game_title and game_title.lower() not in seen_titles:
                    seen_titles.add(game_title.lower())
                    games.append((game_title, item))

        elif item.is_file():
            if item.suffix.lower() in valid_exts or is_archive_file(item):
                game_title = clean_game_title(item.stem)
                if game_title and game_title.lower() not in seen_titles:
                    seen_titles.add(game_title.lower())
                    games.append((game_title, item))

    games.sort(key=lambda x: x[0].lower())
    return games


def check_adb_available() -> bool:
    """Check if ADB is available in the system PATH or common locations."""
    import subprocess
    import shutil

    # Check if adb is in PATH
    if shutil.which("adb"):
        return True

    # Check common installation locations on Windows
    common_paths = [
        r"C:\adb\adb.exe",
        r"C:\Android\platform-tools\adb.exe",
        r"C:\Program Files\Android\platform-tools\adb.exe",
        r"C:\Program Files (x86)\Android\platform-tools\adb.exe",
        os.path.expanduser(r"~\AppData\Local\Android\Sdk\platform-tools\adb.exe"),
    ]

    for path in common_paths:
        if os.path.exists(path):
            return True

    return False


def get_adb_path() -> Optional[str]:
    """Get the path to ADB executable."""
    import shutil

    # Check if adb is in PATH
    adb_in_path = shutil.which("adb")
    if adb_in_path:
        return adb_in_path

    # Check common installation locations on Windows
    common_paths = [
        r"C:\adb\adb.exe",
        r"C:\Android\platform-tools\adb.exe",
        r"C:\Program Files\Android\platform-tools\adb.exe",
        r"C:\Program Files (x86)\Android\platform-tools\adb.exe",
        os.path.expanduser(r"~\AppData\Local\Android\Sdk\platform-tools\adb.exe"),
    ]

    for path in common_paths:
        if os.path.exists(path):
            return path

    return None


def get_adb_devices() -> List[Tuple[str, str]]:
    """
    Get list of connected ADB devices.

    Returns:
        List of (device_id, device_status) tuples
    """
    adb_path = get_adb_path()
    if not adb_path:
        return []

    try:
        import subprocess
        result = subprocess.run(
            [adb_path, "devices"],
            capture_output=True, text=True, timeout=10
        )

        devices = []
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n')[1:]:  # Skip header line
                if '\t' in line:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        device_id = parts[0].strip()
                        status = parts[1].strip()
                        if status == "device":  # Only include ready devices
                            devices.append((device_id, status))

        return devices

    except Exception as e:
        print(f"Error getting ADB devices: {e}")
        return []


def scan_adb_device(device_id: str = "", rom_path: str = "/sdcard/roms") -> Dict[str, List[Tuple[str, Path]]]:
    """
    Scan an Android device for ROMs using ADB (MUCH faster than MTP).

    Args:
        device_id: ADB device ID (empty string for single device)
        rom_path: Path to ROMs on the device (default: /sdcard/roms)

    Returns:
        Dict mapping platform_key -> List of (game_title, game_path) tuples
    """
    results: Dict[str, List[Tuple[str, Path]]] = {}

    adb_path = get_adb_path()
    if not adb_path:
        print("ADB not found. Install Android SDK platform-tools or add ADB to PATH.")
        return results

    def run_adb_command(cmd: list, timeout: int = 30) -> Optional[str]:
        """Run ADB command and return stdout, handling encoding issues."""
        try:
            # Use bytes mode to handle encoding issues with special characters
            result = subprocess.run(cmd, capture_output=True, timeout=timeout)
            if result.returncode != 0:
                return None
            # Try UTF-8 first, fall back to latin-1 which accepts all bytes
            try:
                return result.stdout.decode('utf-8')
            except UnicodeDecodeError:
                return result.stdout.decode('latin-1', errors='replace')
        except subprocess.TimeoutExpired:
            print(f"ADB command timed out: {' '.join(cmd[:3])}...")
            return None
        except Exception as e:
            print(f"ADB command error: {e}")
            return None

    try:
        import subprocess

        # Build base ADB command
        base_cmd = [adb_path]
        if device_id:
            base_cmd.extend(["-s", device_id])

        # Normalize path (handle both /sdcard and /storage/emulated/0)
        if rom_path.startswith("/storage/emulated/0"):
            pass  # Already correct
        elif rom_path.startswith("/sdcard"):
            pass  # ADB handles this
        elif not rom_path.startswith("/"):
            rom_path = f"/sdcard/{rom_path}"

        print(f"ADB scan: Listing directories in {rom_path}...")

        # First, list platform folders (fast)
        cmd = base_cmd + ["shell", "ls", "-1", rom_path]
        output = run_adb_command(cmd, timeout=30)

        if output is None:
            # Try alternative path
            if "/sdcard/" in rom_path:
                alt_path = rom_path.replace("/sdcard/", "/storage/emulated/0/")
                cmd = base_cmd + ["shell", "ls", "-1", alt_path]
                output = run_adb_command(cmd, timeout=30)
                if output is not None:
                    rom_path = alt_path
                else:
                    print("ADB error: Could not list ROM directory")
                    return results
            else:
                print("ADB error: Could not list ROM directory")
                return results

        # Parse platform folders
        platform_folders = []
        for line in output.strip().split('\n'):
            folder_name = line.strip()
            if folder_name and not folder_name.startswith('.'):
                platform = detect_platform_from_folder(folder_name)
                if platform:
                    platform_folders.append((folder_name, platform))

        print(f"ADB scan: Found {len(platform_folders)} platform folders")

        # Now scan each platform folder for ROM files
        for folder_name, platform_key in platform_folders:
            folder_path = f"{rom_path}/{folder_name}"

            # List files in this platform folder
            cmd = base_cmd + ["shell", "ls", "-1", folder_path]
            output = run_adb_command(cmd, timeout=60)

            if output is None:
                continue

            games = []
            seen_titles: Set[str] = set()

            for line in output.strip().split('\n'):
                item_name = line.strip()
                if not item_name or item_name.startswith('.'):
                    continue

                # Clean the game title
                try:
                    game_title = clean_game_title(item_name.rsplit('.', 1)[0] if '.' in item_name else item_name)
                except Exception:
                    # Skip files with problematic names
                    continue

                if game_title and game_title.lower() not in seen_titles:
                    seen_titles.add(game_title.lower())
                    placeholder_path = Path(f"adb://{device_id}/{folder_path}/{item_name}")
                    games.append((game_title, placeholder_path))

            if games:
                # Merge with existing results for this platform (in case of duplicate folder names)
                if platform_key in results:
                    existing_titles = {g[0].lower() for g in results[platform_key]}
                    for game in games:
                        if game[0].lower() not in existing_titles:
                            results[platform_key].append(game)
                            existing_titles.add(game[0].lower())
                else:
                    results[platform_key] = games
                print(f"ADB scan: {platform_key} - {len(results[platform_key])} games")

        print(f"ADB scan: Complete - {len(results)} platforms, {sum(len(g) for g in results.values())} total games")

    except Exception as e:
        print(f"ADB scan error: {e}")

    return results


def scan_mtp_device(device_name: str, subfolder: str = "", max_items_per_folder: int = 50, max_platforms: int = 10) -> Dict[str, List[Tuple[str, Path]]]:
    """
    Scan an MTP/portable device for ROMs using Windows Shell COM.

    Args:
        device_name: Name of the device (e.g., "AYN Thor")
        subfolder: Optional subfolder path within the device (e.g., "Internal shared storage/ROMs")
        max_items_per_folder: Maximum items to scan per platform folder (for performance)
        max_platforms: Maximum number of platform folders to scan (for performance)

    Returns:
        Dict mapping platform_key -> List of (game_title, game_path) tuples
    """
    results: Dict[str, List[Tuple[str, Path]]] = {}

    if os.name != 'nt':
        return results

    try:
        import subprocess
        import tempfile

        # VERY optimized PowerShell script - only scans recognized platforms, limits everything
        # Key optimizations:
        # 1. Only scan folders that match known platform names
        # 2. Limit to first N platforms and M items per platform
        # 3. Use Select-Object -First for early termination
        ps_script = f'''
$ErrorActionPreference = "SilentlyContinue"
$s = New-Object -ComObject Shell.Application
$thispc = $s.NameSpace(17)
$device = $thispc.Items() | Where-Object {{ $_.Name -eq "{device_name}" }} | Select-Object -First 1

if (-not $device) {{
    Write-Error "Device not found: {device_name}"
    exit 1
}}

# Navigate to subfolder
$currentFolder = $device.GetFolder
$subfolderPath = "{subfolder}"

if ($subfolderPath) {{
    $parts = $subfolderPath -split '[/\\\\]'
    foreach ($part in $parts) {{
        if ($part -and $currentFolder) {{
            $found = $false
            foreach ($item in $currentFolder.Items()) {{
                if ($item.Name -eq $part -and $item.IsFolder) {{
                    $currentFolder = $item.GetFolder
                    $found = $true
                    break
                }}
            }}
            if (-not $found) {{
                Write-Error "Subfolder not found: $part"
                exit 1
            }}
        }}
    }}
}}

# Known platform folder names (lowercase for matching)
$knownPlatforms = @(
    'nes', 'snes', 'n64', 'n64dd', 'gamecube', 'gc', 'wii', 'wiiu', 'switch',
    'gb', 'gbc', 'gba', 'gameboy', 'gameboycolor', 'gameboyadvance',
    'nds', 'ds', '3ds', 'nintendo ds', 'nintendo 3ds',
    'ps1', 'ps2', 'ps3', 'ps4', 'psp', 'psvita', 'vita', 'playstation', 'psx',
    'xbox', 'xbox360',
    'genesis', 'megadrive', 'md', 'mastersystem', 'sms', 'saturn', 'dreamcast', 'dc', 'segacd', '32x', 'gamegear', 'gg',
    'neogeo', 'neogeocd', 'ngp', 'ngpc',
    'mame', 'arcade', 'fba', 'fbneo',
    'atari2600', '2600', 'atari5200', 'atari7800', 'lynx', 'jaguar',
    'colecovision', 'coleco', 'intellivision', 'intv',
    'tg16', 'pcengine', 'pce', 'turbografx',
    'wonderswan', 'wsc',
    'scummvm', 'dos'
)

# First pass: just get folder names (fast) - only recognized platforms
Write-Output "SCANNING_FOLDERS"
$platformFolders = @()
$platformCount = 0
$maxPlatforms = {max_platforms}

foreach ($item in $currentFolder.Items()) {{
    if ($item.IsFolder) {{
        $folderLower = $item.Name.ToLower()
        if ($knownPlatforms -contains $folderLower) {{
            Write-Output "PLATFORM:$($item.Name)"
            $platformFolders += $item
            $platformCount++
            if ($platformCount -ge $maxPlatforms) {{
                Write-Output "PLATFORM:... (more platforms, limit reached)"
                break
            }}
        }}
    }}
}}

# Second pass: get limited items from each recognized platform folder
Write-Output "SCANNING_ITEMS"
$maxItems = {max_items_per_folder}
foreach ($folder in $platformFolders) {{
    Write-Output "FOLDER:$($folder.Name)"
    $subFolder = $folder.GetFolder
    if ($subFolder) {{
        $count = 0
        foreach ($subItem in $subFolder.Items()) {{
            Write-Output "ITEM:$($subItem.Name)"
            $count++
            if ($count -ge $maxItems) {{
                Write-Output "ITEM:... (more items)"
                break
            }}
        }}
    }}
}}
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ps1', delete=False, encoding='utf-8') as f:
            f.write(ps_script)
            script_path = f.name

        try:
            # Use 180 second timeout - MTP can be very slow
            print(f"MTP scan: Starting scan of {device_name}/{subfolder}...")
            result = subprocess.run(
                ['powershell.exe', '-ExecutionPolicy', 'Bypass', '-File', script_path],
                capture_output=True, text=True, timeout=180
            )

            if result.returncode == 0:
                current_platform = None
                current_folder_name = None
                current_games = []
                unrecognized_folders = []
                scanning_items = False

                for line in result.stdout.split('\n'):
                    line = line.strip()

                    # Skip phase markers
                    if line == "SCANNING_FOLDERS":
                        continue
                    if line == "SCANNING_ITEMS":
                        scanning_items = True
                        continue

                    # Platform detection (first pass)
                    if line.startswith('PLATFORM:'):
                        folder_name = line[9:]
                        platform = detect_platform_from_folder(folder_name)
                        if not platform:
                            unrecognized_folders.append(folder_name)
                        continue

                    # Folder start (second pass with items)
                    if line.startswith('FOLDER:'):
                        # Save previous platform if any
                        if current_platform and current_games:
                            results[current_platform] = current_games

                        current_folder_name = line[7:]
                        current_platform = detect_platform_from_folder(current_folder_name)
                        current_games = []
                        continue

                    # Item within a folder
                    if line.startswith('ITEM:') and current_platform:
                        item_name = line[5:]
                        # Skip placeholder items
                        if item_name.startswith("... ("):
                            continue
                        game_title = clean_game_title(item_name.rsplit('.', 1)[0] if '.' in item_name else item_name)
                        if game_title:
                            # Use a placeholder path since we can't use real paths for MTP
                            placeholder_path = Path(f"mtp://{device_name}/{subfolder}/{current_folder_name}/{item_name}")
                            current_games.append((game_title, placeholder_path))

                # Save last platform
                if current_platform and current_games:
                    results[current_platform] = current_games

                # Log unrecognized folders for debugging
                if unrecognized_folders:
                    print(f"MTP scan: Unrecognized folders (not matched to platforms): {unrecognized_folders[:10]}")
                if results:
                    print(f"MTP scan: Found {len(results)} platforms with ROMs")
            else:
                # Log error for debugging
                print(f"MTP scan PowerShell error: {result.stderr[:500] if result.stderr else 'No error output'}")

        finally:
            try:
                os.unlink(script_path)
            except Exception:
                pass

    except subprocess.TimeoutExpired:
        print(f"MTP scan timed out after 180 seconds - device may be slow or have too many files")
        print("Tip: Try navigating to a more specific folder path, or use a folder with fewer files")
    except Exception as e:
        print(f"Error scanning MTP device: {e}")

    return results


def is_mtp_path(path: str) -> bool:
    """Check if a path looks like an MTP device path."""
    if not path:
        return False

    # Check for common MTP path patterns
    if (path.startswith("mtp://") or
        path.startswith("::{") or
        "This PC\\" in path or
        "\\\\?\\" in path):
        return True

    # Check if it matches a known portable device name
    # This handles paths like "AYN Thor/Internal shared storage/ROMs"
    try:
        devices = get_portable_devices()
        for _, label in devices:
            device_name = label.replace(" [Portable Device]", "")
            if path.startswith(device_name + "/") or path.startswith(device_name + "\\") or path == device_name:
                return True
    except Exception:
        pass

    return False


def get_portable_devices() -> List[Tuple[str, str]]:
    """
    Get list of MTP/portable devices (Android phones, gaming handhelds, etc.)
    These devices don't get drive letters but appear in Windows Explorer.

    Returns:
        List of (shell_path, device_name) tuples
    """
    devices = []

    if os.name != 'nt':
        return devices

    try:
        import subprocess
        import tempfile

        # Create a PowerShell script to get portable devices with their shell paths
        # Using raw string to avoid Python escape sequence issues
        ps_script = r'''
$s = New-Object -ComObject Shell.Application
$n = $s.NameSpace(17)
$n.Items() | ForEach-Object {
    $path = $_.Path
    # Check if it's a portable device (MTP) by looking for USB GUID paths
    if ($path -match '^\:\:\{.*\}\\\\') {
        Write-Output "$($_.Name)|$path"
    }
}
'''
        # Write script to temp file to avoid shell escaping issues
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ps1', delete=False) as f:
            f.write(ps_script)
            script_path = f.name

        try:
            result = subprocess.run([
                'powershell.exe', '-ExecutionPolicy', 'Bypass', '-File', script_path
            ], capture_output=True, text=True, timeout=10)

            if result.returncode == 0 and result.stdout.strip():
                for line in result.stdout.strip().split('\n'):
                    line = line.strip()
                    if '|' in line:
                        parts = line.split('|', 1)
                        if len(parts) == 2:
                            device_name, shell_path = parts
                            devices.append((shell_path, f"{device_name} [Portable Device]"))
        finally:
            # Clean up temp file
            try:
                os.unlink(script_path)
            except Exception:
                pass

    except Exception:
        pass

    return devices


def get_available_drives() -> List[Tuple[str, str]]:
    """
    Get list of available drives on the system.

    Returns:
        List of (drive_path, drive_label) tuples
    """
    drives = []

    if os.name == 'nt':  # Windows
        import ctypes
        kernel32 = ctypes.windll.kernel32

        # Drive type constants
        DRIVE_UNKNOWN = 0
        DRIVE_NO_ROOT_DIR = 1
        DRIVE_REMOVABLE = 2
        DRIVE_FIXED = 3
        DRIVE_REMOTE = 4
        DRIVE_CDROM = 5
        DRIVE_RAMDISK = 6

        drive_type_names = {
            DRIVE_REMOVABLE: "USB/Removable",
            DRIVE_FIXED: "Local Disk",
            DRIVE_REMOTE: "Network",
            DRIVE_CDROM: "CD/DVD",
            DRIVE_RAMDISK: "RAM Disk",
        }

        bitmask = kernel32.GetLogicalDrives()

        for letter in string.ascii_uppercase:
            if bitmask & 1:
                drive_path = f"{letter}:\\"

                # Get drive type
                drive_type = kernel32.GetDriveTypeW(drive_path)

                # Try to get volume label
                label = ""
                try:
                    buf = ctypes.create_unicode_buffer(256)
                    if kernel32.GetVolumeInformationW(
                        ctypes.c_wchar_p(drive_path),
                        buf, 256, None, None, None, None, 0
                    ):
                        label = buf.value
                except Exception:
                    pass

                # Check if drive is accessible
                if Path(drive_path).exists():
                    type_name = drive_type_names.get(drive_type, "")
                    if label and type_name:
                        display = f"{letter}: {label} [{type_name}]"
                    elif label:
                        display = f"{letter}: {label}"
                    elif type_name:
                        display = f"{letter}: [{type_name}]"
                    else:
                        display = f"{letter}:"
                    drives.append((drive_path, display))

            bitmask >>= 1

        # Also add portable/MTP devices
        portable = get_portable_devices()
        drives.extend(portable)

    else:  # Unix-like (macOS, Linux)
        # Check common mount points
        mount_points = ["/Volumes", "/media", "/mnt", "/run/media"]

        for mount_root in mount_points:
            mount_path = Path(mount_root)
            if mount_path.exists():
                for item in mount_path.iterdir():
                    if item.is_dir():
                        drives.append((str(item), item.name))

    return drives


def find_iisu_directory(search_paths: Optional[List[Path]] = None) -> Optional[Path]:
    """
    Attempt to find an iiSU ROM directory by searching common locations.

    Args:
        search_paths: Optional list of paths to search

    Returns:
        Path to iiSU directory or None if not found
    """
    if search_paths is None:
        search_paths = []

        # Add available drives
        for drive_path, _ in get_available_drives():
            search_paths.append(Path(drive_path))

    # Look for common iiSU folder names
    iisu_folder_names = ["iiSU", "iisu", "IISU", "Roms", "ROMs", "roms", "Games", "games"]

    for search_path in search_paths:
        if not search_path.exists():
            continue

        # Check top-level folders
        try:
            for item in search_path.iterdir():
                if item.is_dir() and item.name in iisu_folder_names:
                    # Verify it looks like a ROM directory
                    for sub_item in item.iterdir():
                        if sub_item.is_dir() and detect_platform_from_folder(sub_item.name):
                            return item
        except PermissionError:
            continue

    return None


class ROMScanner:
    """
    ROM Scanner class for managing ROM directory scanning with caching.
    """

    def __init__(self, iisu_path: Optional[Path] = None):
        self.iisu_path = iisu_path
        self._cache: Dict[str, List[Tuple[str, Path]]] = {}
        self._last_scan_time: Optional[float] = None

    def set_iisu_path(self, path: Optional[Path]):
        """Set the iiSU ROM directory path."""
        self.iisu_path = path
        self._cache.clear()
        self._last_scan_time = None

    def scan(self, force_refresh: bool = False) -> Dict[str, List[Tuple[str, Path]]]:
        """
        Scan the configured iiSU directory for ROMs.

        Args:
            force_refresh: Force re-scan even if cached

        Returns:
            Dict mapping platform_key -> List of (game_title, game_path) tuples
        """
        if not self.iisu_path:
            return {}

        if not force_refresh and self._cache:
            return self._cache

        import time
        self._cache = scan_iisu_directory(self.iisu_path)
        self._last_scan_time = time.time()

        return self._cache

    def get_platforms(self) -> List[str]:
        """Get list of available platforms from the scanned directory."""
        if not self._cache:
            self.scan()
        return sorted(self._cache.keys())

    def get_games(self, platform_key: str) -> List[Tuple[str, Path]]:
        """Get list of games for a specific platform."""
        if not self._cache:
            self.scan()
        return self._cache.get(platform_key, [])

    def get_total_game_count(self) -> int:
        """Get total number of games across all platforms."""
        if not self._cache:
            self.scan()
        return sum(len(games) for games in self._cache.values())

    def search_games(self, query: str, platform_key: Optional[str] = None) -> List[Tuple[str, str, Path]]:
        """
        Search for games matching a query.

        Args:
            query: Search query
            platform_key: Optional platform to filter by

        Returns:
            List of (game_title, platform_key, game_path) tuples
        """
        if not self._cache:
            self.scan()

        results: List[Tuple[str, str, Path]] = []
        query_lower = query.lower()

        platforms_to_search = [platform_key] if platform_key else self._cache.keys()

        for plat in platforms_to_search:
            games = self._cache.get(plat, [])
            for title, path in games:
                if query_lower in title.lower():
                    results.append((title, plat, path))

        return results
