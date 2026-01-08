# iiSU Asset Tool

Create custom icons, borders, and covers for your game library with the iiSU Asset Tool.

## Quick Start

> **Note:** Pre-built executables for Windows and macOS are available in the [Releases](https://github.com/viik-4/iisu-asset-tool/releases) section. If you want to run from source or build your own executable, follow the instructions below.

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Get API Keys

The tool uses two optional APIs to fetch game artwork:
- **SteamGridDB**: Get your key at [steamgriddb.com/profile/preferences/api](https://www.steamgriddb.com/profile/preferences/api)
- **IGDB**: Get your key at [api.igdb.com](https://api.igdb.com/)

You can configure these in the Settings menu once you run the app.

### 3. Run the App

```bash
python run_gui.py
```

## Features

### Icon Scraper
- Download and process game icons for 30+ platforms
- Automatically fetches artwork from SteamGridDB and Libretro
- Batch process hundreds of games at once
- Smart logo detection and cropping

### Custom Icons
- Upload your own artwork
- Drag to position within the icon frame
- Apply platform borders
- Adjust rotation and zoom
- Export at 1024×1024 resolution

### Custom Borders
- Create platform borders with custom gradients
- Upload platform icons (PNG, SVG, etc.)
- Drag to position the icon
- Real-time preview
- Export for use in other tabs

### Custom Covers
- Create game covers with custom artwork
- Apply gradient overlays
- Add platform icons
- Drag to position artwork
- Mouse wheel zoom support
- Export at 1024×1024 resolution

## Usage

### Icon Scraper Tab
1. Select platforms you want to process
2. Configure worker count and limits if needed
3. Click "Start Processing"
4. Icons are saved to the `output/` folder

### Custom Icons Tab
1. Upload your artwork
2. Select a platform border
3. Drag to position the artwork
4. Adjust rotation and zoom with sliders
5. Click "Export Icon"

### Custom Borders Tab
1. Choose gradient colors (or use presets)
2. Upload a platform icon
3. Drag to position the icon within the border
4. Click "Export Border"

### Custom Covers Tab
1. Upload game artwork
2. Choose gradient colors for overlay
3. Upload platform icon (optional)
4. Drag artwork to position
5. Use mouse wheel to zoom
6. Click "Export Cover"

## Settings

Access the Settings menu to configure:
- **API Keys**: Enter your SteamGridDB and IGDB keys
- **Source Priority**: Choose which artwork sources to use and in what order

## Output

All generated assets are saved to:
- **Icons**: `output/` folder (organized by platform)
- **Custom Icons**: Your chosen export location
- **Borders**: `borders/` folder
- **Covers**: Your chosen export location

## Supported Platforms

The tool supports 30+ gaming platforms including:
- Nintendo: NES, SNES, N64, GameCube, Wii, Game Boy, GBA, GBC, DS, 3DS
- PlayStation: PS1, PS2, PS3, PS4, PS5, PSP, PS Vita
- Sega: Genesis, Dreamcast, Game Gear, Master System
- And many more!

Only platforms with available borders are shown in the Icon Scraper.

## Troubleshooting

**No artwork found?**
- Make sure you've entered your API keys in Settings
- Try changing the source priority order
- Some games may not have artwork available

**Icons look wrong?**
- Use the Custom Icons tab to manually adjust positioning
- Try different artwork sources
- Check that the platform border file exists

**App won't start?**
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Check that Python 3.8+ is installed

## Building from Source

If you want to create standalone executables for distribution:

**Windows:**
```bash
build_windows.bat
```

**macOS:**
```bash
./build_macos.sh
```

See [BUILD.md](BUILD.md) for detailed build instructions.

## Credits

**Artwork Sources:**
- [SteamGridDB](https://www.steamgriddb.com/) - Community artwork database
- [Libretro Thumbnails](https://thumbnails.libretro.com/) - RetroArch thumbnails
- [IGDB](https://www.igdb.com/) - Internet Game Database

**Design:**
- Built for the [iiSU Network](https://iisu.network/) ecosystem

## License

This tool is provided as-is for creating custom game library assets.
Ensure compliance with artwork source terms of service.

---

**Built for the iiSU community** 🎮✨
