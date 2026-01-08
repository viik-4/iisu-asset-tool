
# iiSU Asset Tool

Create custom icons, borders, and covers for your game library. Built for the [iiSU Network](https://iisu.network/) community.

## Download

Download the latest release from the [Releases](https://github.com/viik-4/iisu-asset-tool/releases) page.

Extract and run `iiSU_Asset_Tool.exe`.

## Features

### Icon Scraper
Automatically fetch game artwork from multiple sources and apply platform-specific borders.
- Batch process hundreds of games at once
- Smart logo detection and cropping
- Multiple artwork sources with fallback

### Custom Icons
Upload your own images and apply borders with interactive positioning.
- Drag to position artwork
- Rotate and zoom controls
- Real-time preview

### Custom Borders
Create gradient borders with custom colors and platform icons.
- Color picker with gradient presets
- Upload custom platform icons (PNG, SVG)
- Adjustable icon positioning and scale

### Custom Covers
Generate cover artwork with gradients, overlays, and platform branding.
- Drag to position artwork
- Mouse wheel zoom
- Gradient color customization

## Artwork Sources

- [SteamGridDB](https://www.steamgriddb.com/) - Community-curated game artwork
- [IGDB](https://www.igdb.com/) - Internet Game Database
- [TheGamesDB](https://thegamesdb.net/) - Game information database
- [Libretro Thumbnails](https://thumbnails.libretro.com/) - RetroArch thumbnails

## API Keys

Some artwork sources require API keys:

| Source | Required | How to Get |
|--------|----------|------------|
| SteamGridDB | Optional | [steamgriddb.com/profile/preferences/api](https://www.steamgriddb.com/profile/preferences/api) |
| IGDB | Optional | [Twitch Developer Portal](https://dev.twitch.tv/console/apps) |
| TheGamesDB | Built-in | No configuration needed |
| Libretro | Built-in | No configuration needed |

Configure API keys in Settings (gear icon).

## Supported Platforms

**Nintendo:** NES, SNES, N64, GameCube, Wii, Wii U, Switch, Game Boy, GBC, GBA, DS, 3DS

**Sony:** PlayStation 1-5, PSP, PS Vita

**Microsoft:** Xbox, Xbox 360

**Sega:** Master System, Genesis, Saturn, Dreamcast, Game Gear

## Output

Generated assets are saved to:
- **Scraped Icons:** `output/` folder (organized by platform)
- **Custom exports:** Your chosen location

## License

MIT License - see [LICENSE](LICENSE) for details.

---

Built for the [iiSU Network](https://iisu.network/) community.
