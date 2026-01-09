# -*- mode: python ; coding: utf-8 -*-
import os

block_cipher = None

# Bundle internal resources that are extracted by PyInstaller
# User-modifiable files (config, borders, templates) should be copied alongside the app
a = Analysis(
    ['run_gui.py'],
    pathex=[],
    binaries=[],
    datas=[
        # Theme and styling
        ('iisu_theme.qss', '.'),
        # Logo
        ('logo.png', '.'),
        # Fonts directory
        ('fonts', 'fonts'),
        # Source assets (icons, grid pattern)
        ('src', 'src'),
    ],
    hiddenimports=[
        'PySide6.QtCore',
        'PySide6.QtGui',
        'PySide6.QtWidgets',
        'PySide6.QtSvg',
        'PIL',
        'PIL._imagingtk',
        'PIL._tkinter_finder',
        'PIL.ImageQt',
        'psd_tools',
        'psd_tools.psd',
        'psd_tools.psd.layer_and_mask',
        'yaml',
        'requests',
        'numpy',
        'cv2',
        'imagehash',
        'bs4',
        'tqdm',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='iiSU_Asset_Tool',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='iiSU_Asset_Tool',
)

app = BUNDLE(
    coll,
    name='iiSU Asset Tool.app',
    icon='logo.png' if os.path.exists('logo.png') else None,
    bundle_identifier='com.iisu.assettool',
    info_plist={
        'NSPrincipalClass': 'NSApplication',
        'NSAppleScriptEnabled': False,
        'CFBundleDocumentTypes': [],
        'CFBundleShortVersionString': '1.0.0',
        'CFBundleVersion': '1.0.0',
    },
)
