# Fonts Directory

Place custom font files here for use in the iiSU Icon Generator GUI.

## Continuum Bold

The app is designed to use **Continuum Bold** font for the "visuals first" aesthetic.

### How to Add Continuum Bold

1. **Get the font file:**
   - Purchase or obtain `ContinuumBold.ttf` or `ContinuumBold.otf`
   - Common sources: Adobe Fonts, MyFonts, or licensed font providers

2. **Place in this directory:**
   ```
   fonts/
   └── ContinuumBold.ttf   (or .otf)
   ```

3. **Run the app:**
   ```bash
   python run_gui.py
   ```

The font will be automatically loaded and used throughout the interface.

## Supported Formats

- `.ttf` (TrueType Font)
- `.otf` (OpenType Font)

## Font Fallback

If Continuum Bold is not found, the app uses this fallback chain:

1. **Continuum Bold** (if in fonts/)
2. **Continuum** (system-installed)
3. **Segoe UI Semibold** (Windows)
4. **Segoe UI** (Windows fallback)
5. **system-ui** (OS default)
6. **sans-serif** (generic fallback)

## Custom Fonts

You can add any `.ttf` or `.otf` font files to this directory. They will be loaded automatically.

To use a different font in the theme:

1. Add your font file here
2. Edit `src/iisu_theme.qss`
3. Change the `font-family` property:

```css
QWidget {
    font-family: "Your Font Name", "Continuum Bold", ...;
}
```

## License Note

**Important:** Ensure you have proper licensing for any fonts you use, especially for distribution or commercial use. Continuum is a commercial font and requires a license.

## Free Alternatives

If you need a free alternative with similar bold aesthetic:

- **Montserrat Bold** (Google Fonts)
- **Roboto Bold** (Google Fonts)
- **Inter Bold** (GitHub)
- **Outfit Bold** (Google Fonts)

Download from [Google Fonts](https://fonts.google.com/) and place in this directory.
