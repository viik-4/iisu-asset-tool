"""
Custom Image Tab for iiSU Icon Generator
Upload custom images and apply platform borders with manipulation controls.
Optimized for performance with debouncing and caching.
"""

from pathlib import Path
from typing import Optional, Tuple
import math

from PIL import Image, ImageOps, ImageQt, ImageChops
from PySide6.QtCore import Qt, QPointF, Signal, QTimer
from PySide6.QtGui import QPixmap, QImage, QPainter, QTransform, QWheelEvent, QMouseEvent, QKeyEvent
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog,
    QGroupBox, QComboBox, QSlider, QMessageBox, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QSizePolicy
)

from run_backend import compose_with_border, center_crop_to_square, load_yaml, corner_mask_from_border
from app_paths import get_config_path, get_borders_dir


class InteractiveImageView(QGraphicsView):
    """Interactive image view with pan and zoom (no rotation here for performance)."""

    # Signal emitted when user drags to reposition image
    position_dragged = Signal(float, float)  # delta_x, delta_y in 0-1 range
    # Signal emitted when user zooms the source image
    zoom_changed = Signal(float)  # zoom delta (positive = zoom in, negative = zoom out)
    # Signal emitted when user presses arrow keys
    arrow_key_pressed = Signal(float, float)  # delta_x, delta_y for fine positioning

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        # Setup view properties
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(Qt.darkGray)
        self.setFrameShape(QGraphicsView.NoFrame)

        # Enable focus for keyboard events
        self.setFocusPolicy(Qt.StrongFocus)

        # Image item
        self.image_item: Optional[QGraphicsPixmapItem] = None

        # Pan state (for repositioning the source image, not the view)
        self.panning = False
        self.pan_start = QPointF()

        # Zoom state (view zoom, not source image zoom)
        self.zoom_factor = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 5.0

        # Reference size for drag sensitivity
        self.drag_sensitivity = 0.002  # How much offset changes per pixel dragged
        self.arrow_key_step = 0.005  # Fine positioning step for arrow keys

    def set_image(self, pixmap: QPixmap):
        """Set the image to display."""
        self.scene.clear()
        self.image_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.image_item)
        self.fitInView(self.image_item, Qt.KeepAspectRatio)
        self.zoom_factor = 1.0

    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel for zooming the source image (not the view)."""
        if self.image_item is None:
            return

        # Get the zoom delta
        delta = event.angleDelta().y()

        # Emit zoom change signal for parent to handle source image zoom
        # Positive delta = zoom in, negative = zoom out
        # Use smaller increments for smoother control
        zoom_step = 0.05 if delta > 0 else -0.05
        self.zoom_changed.emit(zoom_step)
        event.accept()

    def mousePressEvent(self, event: QMouseEvent):
        """Start dragging to reposition the source image."""
        if event.button() == Qt.LeftButton:
            self.panning = True
            self.pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        """Drag to reposition the source image offset."""
        if self.panning:
            delta = event.pos() - self.pan_start
            self.pan_start = event.pos()

            # Convert pixel movement to offset delta (0-1 range)
            # Negative because dragging right should move image right (increase offset)
            delta_x = -delta.x() * self.drag_sensitivity
            delta_y = -delta.y() * self.drag_sensitivity

            # Emit signal to update parent's offset
            self.position_dragged.emit(delta_x, delta_y)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Stop dragging."""
        if event.button() == Qt.LeftButton:
            self.panning = False
            self.setCursor(Qt.ArrowCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def keyPressEvent(self, event: QKeyEvent):
        """Handle arrow keys for fine positioning."""
        if self.image_item is None:
            super().keyPressEvent(event)
            return

        delta_x = 0.0
        delta_y = 0.0

        if event.key() == Qt.Key_Left:
            delta_x = self.arrow_key_step
        elif event.key() == Qt.Key_Right:
            delta_x = -self.arrow_key_step
        elif event.key() == Qt.Key_Up:
            delta_y = self.arrow_key_step
        elif event.key() == Qt.Key_Down:
            delta_y = -self.arrow_key_step
        else:
            super().keyPressEvent(event)
            return

        self.arrow_key_pressed.emit(delta_x, delta_y)
        event.accept()

    def reset_view(self):
        """Reset zoom and pan to fit the image."""
        if self.image_item:
            self.resetTransform()
            self.fitInView(self.image_item, Qt.KeepAspectRatio)
            self.zoom_factor = 1.0


class CustomImageTab(QWidget):
    """Tab for uploading custom images and applying platform borders."""

    def __init__(self):
        super().__init__()

        # State
        self.original_image: Optional[Image.Image] = None
        self.current_platform: Optional[str] = None
        self.current_border: Optional[Path] = None
        self.rotation: int = 0
        self.zoom: float = 1.0
        self.offset_x: float = 0.5  # Center by default (0-1 range)
        self.offset_y: float = 0.5

        # Performance optimization: cache the preview size version
        self.preview_cache: Optional[Image.Image] = None
        self.preview_size = 512  # Lower resolution for interactive preview

        # Cache border images and masks to avoid reloading
        self.border_cache: Optional[Image.Image] = None  # Border at preview size
        self.border_mask_cache: Optional[Image.Image] = None  # Mask at preview size
        self.border_cache_full: Optional[Image.Image] = None  # Border at 1024x1024
        self.border_mask_cache_full: Optional[Image.Image] = None  # Mask at 1024x1024

        # Config
        self.config_path = get_config_path()
        self.platforms_config = {}
        self.borders_dir = get_borders_dir()

        # Debounce timer for slider updates
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self._do_update_preview)
        self.debounce_ms = 50  # Reduced to 50ms for better responsiveness

        self._load_config()
        self._setup_ui()

    def _load_config(self):
        """Load platform configuration."""
        if self.config_path.exists():
            cfg = load_yaml(self.config_path)
            self.platforms_config = cfg.get("platforms", {})
            paths = cfg.get("paths", {})
            borders_dir_str = paths.get("borders_dir", "./borders")
            self.borders_dir = (self.config_path.parent / borders_dir_str).resolve()

    def _setup_ui(self):
        """Setup the user interface."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Left panel - Controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(15)

        # Image upload section
        upload_group = QGroupBox("Image Upload")
        upload_layout = QVBoxLayout(upload_group)

        self.upload_btn = QPushButton("Upload Image")
        self.upload_btn.setMinimumHeight(40)
        self.upload_btn.clicked.connect(self._upload_image)
        upload_layout.addWidget(self.upload_btn)

        self.image_info = QLabel("No image loaded")
        self.image_info.setStyleSheet("color: #B0B0B0; font-size: 12px;")
        self.image_info.setWordWrap(True)
        upload_layout.addWidget(self.image_info)

        left_layout.addWidget(upload_group)

        # Platform selection section
        platform_group = QGroupBox("Platform & Border")
        platform_layout = QVBoxLayout(platform_group)

        platform_layout.addWidget(QLabel("Select Platform:"))
        self.platform_combo = QComboBox()
        self.platform_combo.addItem("Select Platform...", None)

        # Populate platforms
        for platform_key, platform_data in sorted(self.platforms_config.items()):
            border_file = platform_data.get("border_file")
            if border_file:
                display_name = platform_key.replace("_", " ").title()
                self.platform_combo.addItem(display_name, platform_key)

        self.platform_combo.currentIndexChanged.connect(self._on_platform_changed)
        platform_layout.addWidget(self.platform_combo)

        self.border_info = QLabel("No border selected")
        self.border_info.setStyleSheet("color: #B0B0B0; font-size: 11px;")
        self.border_info.setWordWrap(True)
        platform_layout.addWidget(self.border_info)

        # Custom border import button
        custom_border_btn = QPushButton("Import Custom Border")
        custom_border_btn.clicked.connect(self._import_custom_border)
        platform_layout.addWidget(custom_border_btn)

        left_layout.addWidget(platform_group)

        # Manipulation controls section
        manip_group = QGroupBox("Image Adjustments")
        manip_layout = QVBoxLayout(manip_group)

        # Rotation slider
        manip_layout.addWidget(QLabel("Rotation:"))
        rotation_row = QHBoxLayout()
        self.rotation_slider = QSlider(Qt.Horizontal)
        self.rotation_slider.setMinimum(-180)
        self.rotation_slider.setMaximum(180)
        self.rotation_slider.setValue(0)
        self.rotation_slider.setTickPosition(QSlider.TicksBelow)
        self.rotation_slider.setTickInterval(45)
        self.rotation_slider.valueChanged.connect(self._on_rotation_changed)
        rotation_row.addWidget(self.rotation_slider)
        self.rotation_label = QLabel("0°")
        self.rotation_label.setMinimumWidth(40)
        rotation_row.addWidget(self.rotation_label)
        manip_layout.addLayout(rotation_row)

        # Zoom slider
        manip_layout.addWidget(QLabel("Zoom:"))
        zoom_row = QHBoxLayout()
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setMinimum(50)  # 0.5x
        self.zoom_slider.setMaximum(200)  # 2.0x
        self.zoom_slider.setValue(100)  # 1.0x
        self.zoom_slider.setTickPosition(QSlider.TicksBelow)
        self.zoom_slider.setTickInterval(25)
        self.zoom_slider.valueChanged.connect(self._on_zoom_changed)
        zoom_row.addWidget(self.zoom_slider)
        self.zoom_label = QLabel("100%")
        self.zoom_label.setMinimumWidth(50)
        zoom_row.addWidget(self.zoom_label)
        manip_layout.addLayout(zoom_row)

        # Position info label
        manip_layout.addWidget(QLabel("Position:"))
        position_info = QLabel("Click and drag on preview to position\nUse arrow keys for fine adjustments")
        position_info.setStyleSheet("color: #B0B0B0; font-size: 11px;")
        position_info.setWordWrap(True)
        manip_layout.addWidget(position_info)

        # Reset button
        reset_btn = QPushButton("Reset Adjustments")
        reset_btn.clicked.connect(self._reset_adjustments)
        manip_layout.addWidget(reset_btn)

        left_layout.addWidget(manip_group)

        # Export section
        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout(export_group)

        self.export_btn = QPushButton("Export Image (1024x1024)")
        self.export_btn.setMinimumHeight(40)
        self.export_btn.clicked.connect(self._export_image)
        self.export_btn.setEnabled(False)
        export_layout.addWidget(self.export_btn)

        self.export_info = QLabel("Upload an image and select a platform to export")
        self.export_info.setStyleSheet("color: #B0B0B0; font-size: 11px;")
        self.export_info.setWordWrap(True)
        export_layout.addWidget(self.export_info)

        left_layout.addWidget(export_group)

        left_layout.addStretch()

        # Right panel - Preview
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        preview_label = QLabel("Preview (lower resolution for performance)")
        preview_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #E9E9E9;")
        right_layout.addWidget(preview_label)

        self.preview_view = InteractiveImageView()
        self.preview_view.setMinimumSize(600, 600)
        self.preview_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_layout.addWidget(self.preview_view)

        preview_help = QLabel("Left-click and drag to position | Mouse wheel to zoom view | Arrow keys for fine adjustments")
        preview_help.setStyleSheet("color: #00DDFF; font-size: 11px;")
        preview_help.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(preview_help)

        # Connect preview drag signal (use lambda to pass immediate=True)
        self.preview_view.position_dragged.connect(lambda dx, dy: self._on_position_changed(dx, dy, immediate=True))

        # Connect preview zoom signal
        self.preview_view.zoom_changed.connect(self._on_wheel_zoom)

        # Connect arrow key signal for fine positioning
        self.preview_view.arrow_key_pressed.connect(lambda dx, dy: self._on_position_changed(dx, dy, immediate=True))

        # Add panels to main layout
        layout.addWidget(left_panel, 0)
        layout.addWidget(right_panel, 1)

        # Enable keyboard focus for arrow keys
        self.setFocusPolicy(Qt.StrongFocus)

    def _upload_image(self):
        """Open file dialog to upload an image."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Upload Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.gif *.webp);;All Files (*)"
        )

        if not file_path:
            return

        try:
            # Load image
            loaded_image = Image.open(file_path).convert("RGBA")

            # Place image on a 1024x1024 transparent canvas, centered
            canvas = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))
            img_w, img_h = loaded_image.size

            # Center the loaded image on the canvas
            paste_x = (1024 - img_w) // 2
            paste_y = (1024 - img_h) // 2
            canvas.paste(loaded_image, (paste_x, paste_y), loaded_image)

            self.original_image = canvas

            # Clear cache
            self.preview_cache = None

            # Update info
            size_mb = Path(file_path).stat().st_size / (1024 * 1024)
            self.image_info.setText(
                f"Loaded: {Path(file_path).name}\n"
                f"Original: {img_w}x{img_h} → Canvas: 1024x1024 ({size_mb:.2f} MB)"
            )

            # Reset adjustments
            self._reset_adjustments()

            # Update preview
            self._schedule_update()

            # Enable export if border is also selected
            self._check_export_ready()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load image:\n{e}")

    def _on_platform_changed(self, index: int):
        """Handle platform selection change."""
        platform_key = self.platform_combo.itemData(index)

        if platform_key is None:
            self.current_platform = None
            self.current_border = None
            self.border_info.setText("No border selected")
            self._check_export_ready()
            return

        self.current_platform = platform_key
        platform_data = self.platforms_config.get(platform_key, {})
        border_file = platform_data.get("border_file")

        if border_file:
            self.current_border = self.borders_dir / border_file

            if self.current_border.exists():
                self.border_info.setText(f"Border: {border_file}")
                # Clear border caches when platform changes
                self.border_cache = None
                self.border_mask_cache = None
                self.border_cache_full = None
                self.border_mask_cache_full = None
                self._schedule_update()
            else:
                self.border_info.setText(f"Border file not found: {border_file}")
                self.current_border = None
        else:
            self.current_border = None
            self.border_info.setText("No border file configured")

        self._check_export_ready()

    def _import_custom_border(self):
        """Import a custom border image."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Custom Border",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)"
        )

        if not file_path:
            return

        try:
            # Load and validate border image
            border_img = Image.open(file_path).convert("RGBA")

            # Check if it's 1024x1024
            if border_img.size != (1024, 1024):
                reply = QMessageBox.question(
                    self,
                    "Resize Border?",
                    f"Border is {border_img.size[0]}x{border_img.size[1]}. Resize to 1024x1024?",
                    QMessageBox.Yes | QMessageBox.No
                )

                if reply == QMessageBox.Yes:
                    border_img = border_img.resize((1024, 1024), Image.LANCZOS)
                else:
                    return

            # Save to a temporary location or use directly
            self.current_border = Path(file_path)
            self.current_platform = "custom"

            # Clear platform combo selection
            self.platform_combo.setCurrentIndex(0)

            # Clear border caches
            self.border_cache = None
            self.border_mask_cache = None
            self.border_cache_full = None
            self.border_mask_cache_full = None

            self.border_info.setText(f"Custom border: {Path(file_path).name}")
            self._schedule_update()
            self._check_export_ready()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load border:\n{e}")

    def _on_rotation_changed(self, value: int):
        """Handle rotation slider change."""
        self.rotation = value
        self.rotation_label.setText(f"{value}°")
        self.preview_cache = None  # Clear cache on rotation change
        self._schedule_update()

    def _on_zoom_changed(self, value: int):
        """Handle zoom slider change."""
        self.zoom = value / 100.0
        self.zoom_label.setText(f"{value}%")
        self.preview_cache = None  # Clear cache on zoom change
        self._schedule_update()

    def _on_position_changed(self, delta_x: float, delta_y: float, immediate: bool = False):
        """Handle position change from drag or arrow keys."""
        self.offset_x = max(0.0, min(1.0, self.offset_x + delta_x))
        self.offset_y = max(0.0, min(1.0, self.offset_y + delta_y))

        if immediate:
            # Update immediately for dragging (no debounce)
            self._do_update_preview()
        else:
            # Use debouncing for arrow keys
            self._schedule_update()

    def _on_wheel_zoom(self, zoom_delta: float):
        """Handle mouse wheel zoom for source image."""
        # Update zoom value (0.5 to 2.0 range, which is 50% to 200%)
        new_zoom = self.zoom + zoom_delta
        new_zoom = max(0.5, min(2.0, new_zoom))  # Clamp to slider range

        # Update zoom slider (which will trigger preview update)
        slider_value = int(new_zoom * 100)
        self.zoom_slider.setValue(slider_value)

    def _reset_adjustments(self):
        """Reset all adjustment sliders to default."""
        self.rotation_slider.setValue(0)
        self.zoom_slider.setValue(100)
        self.offset_x = 0.5
        self.offset_y = 0.5
        self._schedule_update()

    def _schedule_update(self):
        """Schedule a preview update with debouncing."""
        # Restart the timer - only updates after user stops adjusting
        self.update_timer.stop()
        self.update_timer.start(self.debounce_ms)

    def _apply_transformations(self, img: Image.Image, use_high_quality: bool = False) -> Image.Image:
        """Apply rotation and zoom transformations to the image."""
        resample_method = Image.LANCZOS if use_high_quality else Image.BILINEAR

        original_size = img.size
        print(f"[Transform] Starting with image size: {original_size}")

        # Apply rotation
        if self.rotation != 0:
            print(f"[Transform] Applying rotation: {self.rotation}°")
            img = img.rotate(-self.rotation, expand=True, fillcolor=(0, 0, 0, 0), resample=resample_method)
            print(f"[Transform] After rotation: {img.size}")

        # Apply zoom by scaling
        if self.zoom != 1.0:
            w, h = img.size
            new_w = int(w * self.zoom)
            new_h = int(h * self.zoom)
            print(f"[Transform] Applying zoom {self.zoom}: {w}x{h} -> {new_w}x{new_h}")
            img = img.resize((new_w, new_h), resample_method)
            print(f"[Transform] After zoom: {img.size}")

        print(f"[Transform] Final transformed size: {img.size}")
        return img

    def _compose_preview_unconstrained(self, transformed_img: Image.Image, border_path: Path,
                                       out_size: int, centering: Tuple[float, float]) -> Image.Image:
        """
        Compose preview with border overlay WITHOUT cropping the transformed image.
        Allows image to expand beyond border boundaries.
        """
        # Create canvas at output size with transparency
        canvas = Image.new("RGBA", (out_size, out_size), (0, 0, 0, 0))

        # Scale the transformed image to match preview size vs export size (1024)
        # This ensures preview and export look the same
        scale_factor = out_size / 1024.0
        if scale_factor != 1.0:
            scaled_w = int(transformed_img.size[0] * scale_factor)
            scaled_h = int(transformed_img.size[1] * scale_factor)
            transformed_img = transformed_img.resize((scaled_w, scaled_h), Image.BILINEAR)

        # Get transformed image size (after scaling for preview)
        img_w, img_h = transformed_img.size

        # Calculate position based on centering (offset_x, offset_y are 0-1 range)
        # The offset controls which part of the image is visible through the border viewport
        # centering=(0.5, 0.5) centers the image
        # centering=(0, 0) shows the left/top of the image
        # centering=(1, 1) shows the right/bottom of the image
        cx, cy = centering

        # Invert the offset for viewport panning behavior:
        # - High horizontal % (0.9) should show the RIGHT side of the image (negative paste_x to shift image left)
        # - Low horizontal % (0.1) should show the LEFT side of the image (less negative or positive paste_x)
        # Formula: paste_x = -(img_w - out_size) * cx
        # Which simplifies to: paste_x = (out_size - img_w) * (1 - cx) when thinking about viewport
        # Actually, let's use: paste_x = -(img_w - out_size) * cx = out_size - img_w - (img_w - out_size) * cx

        # Simpler: invert cx and cy for viewport-style panning
        paste_x = -int((img_w - out_size) * cx)
        paste_y = -int((img_h - out_size) * cy)

        # Paste the transformed image onto the canvas
        canvas.paste(transformed_img, (paste_x, paste_y), transformed_img)

        # Load and prepare border (use cache for performance)
        if out_size == self.preview_size:
            # Preview size - use preview cache
            if self.border_cache is None:
                border = Image.open(border_path)
                border = ImageOps.exif_transpose(border).convert("RGBA")
                if border.size != (out_size, out_size):
                    border = border.resize((out_size, out_size), Image.BILINEAR)
                self.border_cache = border
                self.border_mask_cache = corner_mask_from_border(border, threshold=18, shrink_px=8, feather=0.8)

            border = self.border_cache
            mask = self.border_mask_cache
        else:
            # Full size - use full cache
            if self.border_cache_full is None:
                border = Image.open(border_path)
                border = ImageOps.exif_transpose(border).convert("RGBA")
                if border.size != (out_size, out_size):
                    border = border.resize((out_size, out_size), Image.LANCZOS)
                self.border_cache_full = border
                self.border_mask_cache_full = corner_mask_from_border(border, threshold=18, shrink_px=8, feather=0.8)

            border = self.border_cache_full
            mask = self.border_mask_cache_full

        # Apply border mask to canvas
        canvas.putalpha(ImageChops.multiply(canvas.split()[-1], mask))

        # Composite border on top
        result = Image.alpha_composite(canvas, border)

        return result

    def _do_update_preview(self):
        """Actually update the preview (called by debounce timer)."""
        if self.original_image is None:
            return

        try:
            # Use cached preview if transformations haven't changed
            if self.preview_cache is None:
                # Apply transformations to original
                transformed = self._apply_transformations(self.original_image.copy())
                self.preview_cache = transformed

            # Use centering based on offset sliders
            centering = (self.offset_x, self.offset_y)

            # Generate preview at lower resolution for performance
            # Use unconstrained composition - allows image to expand beyond border
            if self.current_border and self.current_border.exists():
                result = self._compose_preview_unconstrained(
                    self.preview_cache, self.current_border, self.preview_size, centering
                )
            else:
                # No border - just show the transformed image on a canvas
                canvas = Image.new("RGBA", (self.preview_size, self.preview_size), (0, 0, 0, 0))
                img_w, img_h = self.preview_cache.size
                paste_x = int((self.preview_size - img_w) * centering[0])
                paste_y = int((self.preview_size - img_h) * centering[1])
                canvas.paste(self.preview_cache, (paste_x, paste_y), self.preview_cache)
                result = canvas

            # Convert to QPixmap for display
            qimage = ImageQt.ImageQt(result)
            pixmap = QPixmap.fromImage(qimage)

            # Update preview
            self.preview_view.set_image(pixmap)

        except Exception as e:
            print(f"Preview update error: {e}")
            import traceback
            traceback.print_exc()

    def _check_export_ready(self):
        """Check if export is ready and update button state."""
        ready = (
            self.original_image is not None and
            self.current_border is not None and
            self.current_border.exists()
        )

        self.export_btn.setEnabled(ready)

        if ready:
            self.export_info.setText("Ready to export at full 1024x1024 resolution")
        elif self.original_image is None:
            self.export_info.setText("Upload an image to export")
        elif self.current_border is None:
            self.export_info.setText("Select a platform to export")
        else:
            self.export_info.setText("Border file not found")

    def _export_image(self):
        """Export the final image with border at full resolution."""
        if not self.export_btn.isEnabled():
            return

        # Get save path
        default_name = f"{self.current_platform}_custom.png"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Image",
            default_name,
            "PNG Image (*.png);;All Files (*)"
        )

        if not file_path:
            return

        try:
            # Debug output
            print(f"Exporting with: rotation={self.rotation}°, zoom={self.zoom*100}%, offset=({self.offset_x}, {self.offset_y})")

            # Apply transformations to ORIGINAL image (not preview cache) with HIGH QUALITY
            transformed = self._apply_transformations(self.original_image.copy(), use_high_quality=True)

            # DEBUG: Save the transformed image
            print(f"[Export] Transformed image: size={transformed.size}, mode={transformed.mode}")
            # Verify it's actually the transformed image
            debug_trans = transformed.copy()
            debug_trans.save("debug_transformed.png")
            print(f"[Export] Saved debug_transformed.png")

            # Compose with border at FULL RESOLUTION using unconstrained positioning
            centering = (self.offset_x, self.offset_y)

            # Create canvas at output size with transparency
            canvas = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))

            # Get transformed image size
            img_w, img_h = transformed.size
            print(f"[Export] Transformed image size: {img_w}x{img_h}")

            # Position the image on the canvas (viewport-style panning)
            # High offset % shows right/bottom of image, low offset % shows left/top
            cx, cy = centering
            paste_x = -int((img_w - 1024) * cx)
            paste_y = -int((img_h - 1024) * cy)
            print(f"[Export] Pasting at position: ({paste_x}, {paste_y})")

            # Paste the transformed image onto the canvas
            canvas.paste(transformed, (paste_x, paste_y), transformed)
            print(f"[Export] Canvas after paste: size={canvas.size}, mode={canvas.mode}")

            # Verify paste area
            bbox = canvas.getbbox()  # Get bounding box of non-transparent pixels
            print(f"[Export] Canvas content bounding box: {bbox}")
            if bbox:
                content_left, content_top, content_right, content_bottom = bbox
                content_w = content_right - content_left
                content_h = content_bottom - content_top
                print(f"[Export] Content spans: x={content_left} to {content_right} (width={content_w}), y={content_top} to {content_bottom} (height={content_h})")
                left_margin = content_left
                right_margin = 1024 - content_right
                top_margin = content_top
                bottom_margin = 1024 - content_bottom
                print(f"[Export] Margins: left={left_margin}, right={right_margin}, top={top_margin}, bottom={bottom_margin}")

            # DEBUG: Save canvas before masking
            canvas.save("debug_before_mask.png")
            print("[Export] Saved debug_before_mask.png")

            # Load and prepare border at full resolution
            border = Image.open(self.current_border)
            border = ImageOps.exif_transpose(border).convert("RGBA")
            if border.size != (1024, 1024):
                border = border.resize((1024, 1024), Image.LANCZOS)  # Use LANCZOS for export quality

            # Apply border mask to canvas
            mask = corner_mask_from_border(border, threshold=18, shrink_px=8, feather=0.8)
            canvas.putalpha(ImageChops.multiply(canvas.split()[-1], mask))

            # DEBUG: Save canvas after masking
            canvas.save("debug_after_mask.png")
            print("[Export] Saved debug_after_mask.png")

            # Composite border on top
            result = Image.alpha_composite(canvas, border)

            # Save
            result.save(file_path, "PNG")

            # Create summary message
            summary = f"Image exported successfully at 1024x1024 to:\n{file_path}\n\n"
            summary += f"Applied settings:\n"
            summary += f"  • Rotation: {self.rotation}°\n"
            summary += f"  • Zoom: {int(self.zoom * 100)}% ({img_w}x{img_h})\n"
            summary += f"  • Position: H={int(self.offset_x * 100)}%, V={int(self.offset_y * 100)}%\n"

            if bbox:
                summary += f"\nImage placement:\n"
                summary += f"  • Margins: L={left_margin}px, R={right_margin}px, T={top_margin}px, B={bottom_margin}px"

            QMessageBox.information(
                self,
                "Export Complete",
                summary
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Export Error",
                f"Failed to export image:\n{e}"
            )

    def keyPressEvent(self, event: QKeyEvent):
        """Handle arrow keys for fine position adjustments."""
        if self.original_image is None:
            super().keyPressEvent(event)
            return

        # Arrow key step size (0.01 = 1% adjustment)
        step = 0.01

        if event.key() == Qt.Key_Left:
            self._on_position_changed(-step, 0)
            event.accept()
        elif event.key() == Qt.Key_Right:
            self._on_position_changed(step, 0)
            event.accept()
        elif event.key() == Qt.Key_Up:
            self._on_position_changed(0, -step)
            event.accept()
        elif event.key() == Qt.Key_Down:
            self._on_position_changed(0, step)
            event.accept()
        else:
            super().keyPressEvent(event)
