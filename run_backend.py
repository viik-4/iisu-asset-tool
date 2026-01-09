import os
import re
import json
import time
import hashlib
import zipfile
import threading
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import html
from urllib.parse import unquote

import requests
import yaml
from PIL import Image, ImageOps, ImageChops, ImageFilter


# ==========================
# Cancel Token
# ==========================
class CancelToken:
    def __init__(self):
        self._evt = threading.Event()

    def cancel(self):
        self._evt.set()

    @property
    def is_cancelled(self) -> bool:
        return self._evt.is_set()


# ==========================
# Utilities
# ==========================
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def safe_slug(s: str, limit: int = 180) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^\w\- ]+", "", s, flags=re.UNICODE)
    s = re.sub(r"\s+", "_", s)
    return s[:limit] if len(s) > limit else s

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())

def _emit_log(callbacks, msg: str):
    if callbacks is None:
        return
    # Handle dict-style callbacks (from GUI)
    if isinstance(callbacks, dict):
        if "log" in callbacks and callable(callbacks["log"]):
            try:
                callbacks["log"](msg)
            except Exception:
                pass
    # Handle object-style callbacks
    elif hasattr(callbacks, "log"):
        try:
            callbacks.log.emit(msg)
        except Exception:
            pass

def _emit_progress(callbacks, done: int, total: int):
    if callbacks is None:
        return
    # Handle dict-style callbacks (from GUI)
    if isinstance(callbacks, dict):
        if "progress" in callbacks and callable(callbacks["progress"]):
            try:
                callbacks["progress"](done, total)
            except Exception:
                pass
    # Handle object-style callbacks
    elif hasattr(callbacks, "progress"):
        try:
            callbacks.progress.emit(done, total)
        except Exception:
            pass

def _emit_preview(callbacks, img_path: Path):
    if callbacks is None:
        return
    # Handle dict-style callbacks (from GUI)
    if isinstance(callbacks, dict):
        if "preview" in callbacks and callable(callbacks["preview"]):
            try:
                callbacks["preview"](str(img_path))
            except Exception:
                pass
    # Handle object-style callbacks
    elif hasattr(callbacks, "preview"):
        try:
            callbacks.preview.emit(str(img_path))
        except Exception:
            pass

def _request_user_selection(callbacks, title: str, platform: str, artwork_options: List[Dict[str, Any]]) -> Optional[int]:
    """
    Request user to select artwork from options.
    Returns selected index, None if skipped, -1 if cancelled all.
    """
    _emit_log(callbacks, f"[DEBUG] _request_user_selection called for {title} with {len(artwork_options)} options")

    if callbacks is None:
        _emit_log(callbacks, f"[DEBUG] No callbacks provided")
        return None

    # Handle dict-style callbacks (from GUI)
    if isinstance(callbacks, dict):
        _emit_log(callbacks, f"[DEBUG] Callbacks is dict, has request_selection: {'request_selection' in callbacks}")
        if "request_selection" in callbacks and callable(callbacks["request_selection"]):
            try:
                _emit_log(callbacks, f"[DEBUG] Calling request_selection callback...")
                result = callbacks["request_selection"](title, platform, artwork_options)
                _emit_log(callbacks, f"[DEBUG] Callback returned: {result}")
                return result
            except Exception as e:
                _emit_log(callbacks, f"[ERROR] Callback exception: {e}")
                import traceback
                _emit_log(callbacks, f"[ERROR] {traceback.format_exc()}")
                return None
    # Handle object-style callbacks
    elif hasattr(callbacks, "request_selection"):
        try:
            return callbacks.request_selection(title, platform, artwork_options)
        except Exception as e:
            _emit_log(callbacks, f"[ERROR] Object callback exception: {e}")
            return None

    _emit_log(callbacks, f"[DEBUG] No valid callback found")
    return None


# ==========================
# SteamGridDB (thread-local session)
# ==========================
_thread_local = threading.local()

def get_session(api_key: str) -> requests.Session:
    s = getattr(_thread_local, "session", None)
    if s is None:
        s = requests.Session()
        s.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "User-Agent": "iiSU-Icons/1.0",
        })
        _thread_local.session = s
    return s

def sgdb_get(api_key: str, base_url: str, path: str, params: Optional[dict], timeout_s: int) -> dict:
    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
    s = get_session(api_key)
    r = s.get(url, params=params, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    if not data.get("success", False):
        raise RuntimeError(data)
    return data

def search_autocomplete(api_key: str, base_url: str, term: str, timeout_s: int) -> List[dict]:
    term_q = requests.utils.quote(term)
    data = sgdb_get(api_key, base_url, f"search/autocomplete/{term_q}", None, timeout_s)
    return data.get("data", []) or []

def get_game_by_id(api_key: str, base_url: str, game_id: str, timeout_s: int) -> dict:
    data = sgdb_get(api_key, base_url, f"games/id/{game_id}", None, timeout_s)
    return data.get("data", {}) or []

def grids_by_game(
    api_key: str,
    base_url: str,
    game_id: str,
    dimensions: Optional[List[str]],
    styles: Optional[List[str]],
    timeout_s: int
) -> List[dict]:
    params = {}
    # SteamGridDB API expects comma-separated values, not multiple params
    if dimensions:
        params["dimensions"] = ",".join(dimensions) if isinstance(dimensions, list) else dimensions
    if styles:
        params["styles"] = ",".join(styles) if isinstance(styles, list) else styles
    data = sgdb_get(api_key, base_url, f"grids/game/{game_id}", params or None, timeout_s)
    return data.get("data", []) or []

def is_animated(url: str) -> bool:
    return url.lower().endswith(".webp")

def pick_best_grid(grids: List[dict], prefer_dim: str, allow_animated: bool, square_only: bool) -> Optional[dict]:
    if not grids:
        return None

    filtered = []
    for g in grids:
        url = (g.get("url") or "").strip()
        if not url:
            continue
        if not allow_animated and is_animated(url):
            continue
        filtered.append(g)
    if not filtered:
        return None

    if square_only:
        exact = [g for g in filtered if f"{g.get('width')}x{g.get('height')}" == prefer_dim]
        if not exact:
            return None
        exact.sort(key=lambda x: (x.get("score", 0), x.get("id", 0)), reverse=True)
        return exact[0]

    filtered.sort(key=lambda x: (x.get("score", 0), x.get("id", 0)), reverse=True)
    return filtered[0]

def download_bytes(url: str, timeout_s: int) -> bytes:
    r = requests.get(url, timeout=timeout_s)
    r.raise_for_status()
    return r.content


# ==========================
# Platform-aware candidate selection
# ==========================
def _flatten_strings(obj: Any) -> str:
    parts: List[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            parts.append(str(k))
            parts.append(_flatten_strings(v))
    elif isinstance(obj, list):
        for v in obj:
            parts.append(_flatten_strings(v))
    else:
        parts.append(str(obj))
    return " ".join(parts)

def score_candidate(title: str, candidate_name: str, game_meta: dict, platform_hints: List[str]) -> int:
    t = (title or "").lower().strip()
    n = (candidate_name or "").lower().strip()
    score = 0

    if n == t:
        score += 200
    elif n.startswith(t) or t.startswith(n):
        score += 140
    elif t in n or n in t:
        score += 90

    t_tokens = set(re.findall(r"[a-z0-9]+", t))
    n_tokens = set(re.findall(r"[a-z0-9]+", n))
    score += min(len(t_tokens & n_tokens) * 8, 80)

    meta_text = _flatten_strings(game_meta).lower()
    for h in platform_hints:
        hh = h.lower()
        if hh and hh in meta_text:
            score += 60

    return score

def choose_best_game_id(
    api_key: str,
    base_url: str,
    timeout_s: int,
    delay_s: float,
    title: str,
    platform_hints: List[str],
    autocomplete_results: List[dict],
    max_candidates: int = 8
) -> Optional[str]:
    if not autocomplete_results:
        return None

    candidates = autocomplete_results[:max_candidates]
    best_id = None
    best_score = -10**9

    for c in candidates:
        cid = c.get("id")
        if cid is None:
            continue
        cid = str(cid)

        meta = {}
        try:
            meta = get_game_by_id(api_key, base_url, cid, timeout_s)
            if delay_s > 0:
                time.sleep(delay_s)
        except Exception:
            meta = {}

        name = c.get("name") or meta.get("name") or ""
        s = score_candidate(title, name, meta, platform_hints)

        if s > best_score:
            best_score = s
            best_id = cid

    return best_id


# ==========================
# Libretro thumbnails provider
# ==========================
_LIBRETRO_BAD_CHARS = r'&\*/:<>?\|"'  # libretro recommends replacing certain characters with '_' in thumbnail filenames

def libretro_sanitize_filename(name: str) -> str:
    out = []
    for ch in (name or "").strip():
        if ch in _LIBRETRO_BAD_CHARS:
            out.append("_")
        else:
            out.append(ch)
    s = "".join(out).strip()
    s = re.sub(r"\s+", " ", s)
    return s

def libretro_candidate_names(title: str) -> List[str]:
    base = libretro_sanitize_filename(title)
    candidates = [base]

    regions = [
        "World", "USA", "Europe", "Japan",
        "USA, Europe", "USA, Australia", "Europe, Australia",
        "Japan, USA", "Japan, Europe",
    ]
    for r in regions:
        candidates.append(f"{base} ({r})")

    # tiny punctuation tweaks
    candidates.append(re.sub(r"\s*-\s*", " - ", base))
    candidates.append(re.sub(r"\s*:\s*", ": ", base))

    seen = set()
    out = []
    for c in candidates:
        k = c.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(c)
    return out

def _libretro_index_url(base_url: str, playlist_name: str, type_dir: str) -> str:
    # The index is browsable HTML
    return f"{base_url.rstrip('/')}/{requests.utils.quote(playlist_name)}/{requests.utils.quote(type_dir)}/"

def _parse_libretro_index_filenames(index_html: str) -> List[str]:
    """
    Parses the apache-style directory listing from thumbnails.libretro.com and returns .png filenames.
    """
    # Directory listings have links like: <a href="Super%20Mario%20Bros.%20(World).png">...
    # We'll extract href="...png"
    hrefs = re.findall(r'href="([^"]+\.png)"', index_html, flags=re.IGNORECASE)
    out = []
    for h in hrefs:
        # Convert %20 to spaces etc
        fname = unquote(html.unescape(h))
        # Some listings include absolute paths or weird prefixes; keep basename only
        fname = fname.split("/")[-1]
        if fname.lower().endswith(".png"):
            out.append(fname)
    return out

def _norm_for_match(s: str) -> str:
    # Aggressive normalization for matching titles to filenames
    s = (s or "").lower()
    s = re.sub(r"\.png$", "", s)
    # strip bracket tags like [h], [b], [iNES title], etc.
    s = re.sub(r"\[[^\]]+\]", "", s)
    # strip parenthetical chunks that are mostly region/lang/publisher/date noise,
    # but keep it gentle (we’ll still score tokens)
    s = re.sub(r"\(([^)]*)\)", r" \1 ", s)
    # punctuation -> spaces
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _score_match(title_norm: str, fname_norm: str) -> int:
    """
    Token overlap score with boosts for prefix/contains.
    """
    if not title_norm or not fname_norm:
        return -10**9

    if fname_norm == title_norm:
        return 500

    score = 0
    if fname_norm.startswith(title_norm) or title_norm.startswith(fname_norm):
        score += 200
    if title_norm in fname_norm or fname_norm in title_norm:
        score += 120

    t = set(title_norm.split())
    f = set(fname_norm.split())
    inter = len(t & f)
    union = max(1, len(t | f))
    score += int(300 * (inter / union))

    # small bonus for longer matches (discourages tiny collisions)
    score += min(len(fname_norm), 180) // 6
    return score

def _load_or_build_libretro_index(
    *,
    cache_dir: Path,
    base_url: str,
    playlist_name: str,
    type_dir: str,
    timeout_s: int,
    cache_hours: int = 168
) -> List[str]:
    ensure_dir(cache_dir)
    key = sha256_text(f"{base_url}|{playlist_name}|{type_dir}|index")
    cache_path = cache_dir / f"{key}.json"

    # Use cache if fresh
    if cache_path.exists():
        try:
            obj = json.loads(cache_path.read_text(encoding="utf-8"))
            ts = float(obj.get("ts", 0))
            if (time.time() - ts) < cache_hours * 3600 and isinstance(obj.get("files"), list):
                return obj["files"]
        except Exception:
            pass

    # Fetch index HTML
    url = _libretro_index_url(base_url, playlist_name, type_dir)
    r = requests.get(url, timeout=timeout_s)
    r.raise_for_status()
    files = _parse_libretro_index_filenames(r.text)

    cache_path.write_text(json.dumps({"ts": time.time(), "files": files}, indent=2), encoding="utf-8")
    return files

def libretro_try_download_boxart(
    base_url: str,
    playlist_name: str,
    type_dir: str,
    title: str,
    timeout_s: int,
    cache_dir: Optional[Path] = None,
    use_index_matching: bool = True,
    index_cache_hours: int = 168,
    debug_log=None
) -> Optional[bytes]:
    """
    1) Try direct candidate names (fast)
    2) If that fails and use_index_matching=True, build/load index for platform and fuzzy match
    """
    # ---- 1) Direct tries (fast path)
    for cand in libretro_candidate_names(title):
        path = "/".join([
            requests.utils.quote(playlist_name, safe=""),
            requests.utils.quote(type_dir, safe=""),
            requests.utils.quote(cand + ".png", safe=""),
        ])
        url = f"{base_url.rstrip('/')}/{path}"
        try:
            r = requests.get(url, timeout=timeout_s)
            if r.status_code == 200 and r.content:
                return r.content
        except Exception:
            continue

    # ---- 2) Index + fuzzy match
    if not use_index_matching or cache_dir is None:
        return None

    try:
        files = _load_or_build_libretro_index(
            cache_dir=cache_dir,
            base_url=base_url,
            playlist_name=playlist_name,
            type_dir=type_dir,
            timeout_s=timeout_s,
            cache_hours=index_cache_hours,
        )
    except Exception as e:
        if debug_log:
            debug_log(f"[LIBRETRO] Index fetch failed: {e}")
        return None

    title_norm = _norm_for_match(title)
    best = None
    best_score = -10**9

    for fname in files:
        s = _score_match(title_norm, _norm_for_match(fname))
        if s > best_score:
            best_score = s
            best = fname

    # Threshold to avoid nonsense matches
    if not best or best_score < 220:
        if debug_log:
            debug_log(f"[LIBRETRO] No good match for '{title}' (best={best} score={best_score})")
        return None

    url = f"{_libretro_index_url(base_url, playlist_name, type_dir)}{requests.utils.quote(best)}"
    try:
        r = requests.get(url, timeout=timeout_s)
        if r.status_code == 200 and r.content:
            if debug_log:
                debug_log(f"[LIBRETRO] Matched '{title}' -> '{best}' (score={best_score})")
            return r.content
    except Exception as e:
        if debug_log:
            debug_log(f"[LIBRETRO] Download failed for {best}: {e}")
        return None



# ==========================
# Image ops + border mask
# ==========================
try:
    import numpy as np
except ImportError:
    np = None

def center_crop_to_square(img: Image.Image, out_size: int, centering: Tuple[float, float] = (0.5, 0.5)) -> Image.Image:
    img = ImageOps.exif_transpose(img).convert("RGBA")
    cx, cy = centering
    cx = max(0.0, min(1.0, float(cx)))
    cy = max(0.0, min(1.0, float(cy)))
    return ImageOps.fit(img, (out_size, out_size), method=Image.LANCZOS, centering=(cx, cy))

def _content_centroid(img_rgba: Image.Image, alpha_threshold: int = 16, margin_pct: float = 0.06) -> Tuple[float, float, int]:
    """Returns (mx,my,count) centroid of non-transparent pixels, normalized to [0,1] in x/y."""
    img = ImageOps.exif_transpose(img_rgba).convert("RGBA")
    w, h = img.size
    if w <= 1 or h <= 1:
        return (0.5, 0.5, 0)

    mx = int(round(w * margin_pct))
    my = int(round(h * margin_pct))
    x1, y1 = mx, my
    x2, y2 = max(x1 + 1, w - mx), max(y1 + 1, h - my)
    region = img.crop((x1, y1, x2, y2))
    rw, rh = region.size

    if np is not None:
        a = np.array(region.split()[-1], dtype=np.uint8)
        mask = a > alpha_threshold
        cnt = int(mask.sum())
        if cnt <= 0:
            return (0.5, 0.5, 0)
        ys, xs = np.nonzero(mask)
        cx = float(xs.mean()) / max(1.0, (rw - 1))
        cy = float(ys.mean()) / max(1.0, (rh - 1))
        gx = (x1 + cx * (rw - 1)) / (w - 1)
        gy = (y1 + cy * (rh - 1)) / (h - 1)
        return (float(gx), float(gy), cnt)

    alpha = region.split()[-1]
    pix = alpha.load()
    total = 0
    sx = 0.0
    sy = 0.0
    for yy in range(rh):
        for xx in range(rw):
            if pix[xx, yy] > alpha_threshold:
                total += 1
                sx += xx
                sy += yy
    if total <= 0:
        return (0.5, 0.5, 0)
    cx = (sx / total) / max(1.0, (rw - 1))
    cy = (sy / total) / max(1.0, (rh - 1))
    gx = (x1 + cx * (rw - 1)) / (w - 1)
    gy = (y1 + cy * (rh - 1)) / (h - 1)
    return (float(gx), float(gy), int(total))

def _best_centering_for_img(img_rgba: Image.Image, out_size: int, steps: int = 5, span: float = 0.22,
                            alpha_threshold: int = 16, margin_pct: float = 0.06) -> Tuple[Tuple[float, float], Tuple[float, float, int]]:
    """Search a small grid of ImageOps.fit centering points and pick the one that best centers content."""
    steps = max(1, int(steps))
    span = max(0.0, min(0.49, float(span)))
    if steps == 1:
        best = (0.5, 0.5)
        fitted = center_crop_to_square(img_rgba, out_size, centering=best)
        mx, my, cnt = _content_centroid(fitted, alpha_threshold=alpha_threshold, margin_pct=margin_pct)
        return best, (mx, my, cnt)

    offsets = [(-span + (2 * span) * i / (steps - 1)) for i in range(steps)]
    best_c = (0.5, 0.5)
    best_metrics = (0.5, 0.5, 0)
    best_score = 1e9

    for oy in offsets:
        for ox in offsets:
            c = (0.5 + ox, 0.5 + oy)
            fitted = center_crop_to_square(img_rgba, out_size, centering=c)
            mx, my, cnt = _content_centroid(fitted, alpha_threshold=alpha_threshold, margin_pct=margin_pct)
            score = (mx - 0.5) ** 2 + (my - 0.5) ** 2
            if cnt <= 0:
                score += 10.0
            if score < best_score:
                best_score = score
                best_c = (max(0.0, min(1.0, c[0])), max(0.0, min(1.0, c[1])))
                best_metrics = (mx, my, cnt)

    return best_c, best_metrics


# ==========================
# Smart Logo/Art Detection
# ==========================

def _detect_content_bbox(img_rgba: Image.Image, alpha_threshold: int = 16, edge_padding: int = 5) -> Tuple[int, int, int, int]:
    """
    Detect tight bounding box around non-transparent content.
    Returns (x1, y1, x2, y2) in pixel coordinates.
    """
    img = ImageOps.exif_transpose(img_rgba).convert("RGBA")
    w, h = img.size

    if np is not None:
        alpha = np.array(img.split()[-1], dtype=np.uint8)
        mask = alpha > alpha_threshold

        if not mask.any():
            return (0, 0, w, h)

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        y_indices = np.where(rows)[0]
        x_indices = np.where(cols)[0]

        if len(y_indices) == 0 or len(x_indices) == 0:
            return (0, 0, w, h)

        x1 = max(0, int(x_indices[0]) - edge_padding)
        x2 = min(w, int(x_indices[-1]) + edge_padding + 1)
        y1 = max(0, int(y_indices[0]) - edge_padding)
        y2 = min(h, int(y_indices[-1]) + edge_padding + 1)

        return (x1, y1, x2, y2)

    # Fallback without NumPy
    alpha = img.split()[-1]
    pix = alpha.load()

    min_x, min_y = w, h
    max_x, max_y = 0, 0

    found_any = False
    for y in range(h):
        for x in range(w):
            if pix[x, y] > alpha_threshold:
                found_any = True
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)

    if not found_any:
        return (0, 0, w, h)

    x1 = max(0, min_x - edge_padding)
    x2 = min(w, max_x + edge_padding + 1)
    y1 = max(0, min_y - edge_padding)
    y2 = min(h, max_y + edge_padding + 1)

    return (x1, y1, x2, y2)


def _detect_logo_region_cv2(img_rgba: Image.Image, debug: bool = False) -> Optional[Tuple[int, int, int, int]]:
    """
    Advanced logo detection using OpenCV (if available).
    Uses edge detection + morphology to find the main logo region.
    Returns (x1, y1, x2, y2) or None if OpenCV unavailable.
    """
    try:
        import cv2
    except ImportError:
        return None

    img = ImageOps.exif_transpose(img_rgba).convert("RGBA")
    w, h = img.size

    # Convert to NumPy
    img_array = np.array(img, dtype=np.uint8)

    # Extract alpha channel
    alpha = img_array[:, :, 3]

    # Create mask of non-transparent regions
    mask = (alpha > 16).astype(np.uint8) * 255

    if mask.sum() == 0:
        return None

    # Convert RGB to grayscale for edge detection
    gray = cv2.cvtColor(img_array[:, :, :3], cv2.COLOR_RGB2GRAY)

    # Apply bilateral filter to reduce noise while keeping edges
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)

    # Canny edge detection
    edges = cv2.Canny(filtered, 50, 150)

    # Apply mask to edges (only consider edges in non-transparent areas)
    edges = cv2.bitwise_and(edges, edges, mask=mask)

    # Morphological operations to connect nearby edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Find the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)

    # Get bounding rectangle
    x, y, bw, bh = cv2.boundingRect(largest_contour)

    # Add padding
    padding = 10
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(w, x + bw + padding)
    y2 = min(h, y + bh + padding)

    return (x1, y1, x2, y2)


def detect_and_crop_logo(img_rgba: Image.Image,
                         method: str = "auto",
                         min_content_ratio: float = 0.15,
                         max_crop_ratio: float = 0.85,
                         debug_log=None) -> Image.Image:
    """
    Detect the main logo/artwork region and crop to it intelligently.

    Args:
        img_rgba: Input RGBA image
        method: "auto", "bbox", "cv2", or "none"
        min_content_ratio: Minimum ratio of content to keep (prevents over-cropping)
        max_crop_ratio: Maximum crop ratio (prevents tiny crops)
        debug_log: Optional logging function

    Returns:
        Cropped image (or original if detection fails)
    """
    img = ImageOps.exif_transpose(img_rgba).convert("RGBA")
    orig_w, orig_h = img.size

    if method == "none":
        return img

    bbox = None

    # Try CV2 method first if available and requested
    if method in ("auto", "cv2"):
        bbox = _detect_logo_region_cv2(img, debug=False)
        if bbox and debug_log:
            debug_log(f"[LOGO] CV2 detection: {bbox}")

    # Fallback to simple bbox if CV2 failed or not requested
    if bbox is None and method in ("auto", "bbox"):
        bbox = _detect_content_bbox(img, alpha_threshold=16, edge_padding=10)
        if debug_log:
            debug_log(f"[LOGO] BBox detection: {bbox}")

    if bbox is None:
        return img

    x1, y1, x2, y2 = bbox
    crop_w = x2 - x1
    crop_h = y2 - y1

    # Safety checks
    if crop_w <= 0 or crop_h <= 0:
        return img

    # Check if crop is too small
    content_ratio = (crop_w * crop_h) / (orig_w * orig_h)
    if content_ratio < min_content_ratio:
        if debug_log:
            debug_log(f"[LOGO] Crop too small ({content_ratio:.2%}), using original")
        return img

    # Check if crop is too similar to original (no point cropping)
    if crop_w > orig_w * max_crop_ratio and crop_h > orig_h * max_crop_ratio:
        if debug_log:
            debug_log(f"[LOGO] Crop too similar to original, using original")
        return img

    # Perform crop
    cropped = img.crop((x1, y1, x2, y2))

    if debug_log:
        debug_log(f"[LOGO] Cropped from {orig_w}x{orig_h} to {crop_w}x{crop_h} ({content_ratio:.2%})")

    return cropped


def fill_center_hole(alpha: Image.Image) -> Image.Image:
    a = alpha.convert("L")
    w, h = a.size
    px = a.load()
    cx, cy = w // 2, h // 2
    if px[cx, cy] != 0:
        return a
    q = deque([(cx, cy)])
    visited = {(cx, cy)}
    while q:
        x, y = q.popleft()
        px[x, y] = 255
        for nx, ny in ((x-1,y), (x+1,y), (x,y-1), (x,y+1)):
            if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in visited:
                if px[nx, ny] == 0:
                    visited.add((nx, ny))
                    q.append((nx, ny))
    return a

def corner_mask_from_border(border_rgba: Image.Image, threshold: int = 18, shrink_px: int = 8, feather: float = 0.8) -> Image.Image:
    border_alpha = border_rgba.split()[-1].convert("L")
    hard = border_alpha.point(lambda p: 255 if p >= threshold else 0, mode="L")
    hard = fill_center_hole(hard)
    if shrink_px > 0:
        hard = hard.filter(ImageFilter.MinFilter(2 * shrink_px + 1))
    if feather and feather > 0:
        hard = hard.filter(ImageFilter.GaussianBlur(radius=feather))
    return hard

def compose_with_border(base_img: Image.Image, border_path: Path, out_size: int, centering: Tuple[float, float] = (0.5, 0.5)) -> Image.Image:
    base = center_crop_to_square(base_img, out_size, centering=centering)

    border = Image.open(border_path)
    border = ImageOps.exif_transpose(border).convert("RGBA")
    if border.size != (out_size, out_size):
        border = border.resize((out_size, out_size), Image.LANCZOS)

    mask = corner_mask_from_border(border, threshold=18, shrink_px=8, feather=0.8)
    base.putalpha(ImageChops.multiply(base.split()[-1], mask))
    return Image.alpha_composite(base, border)


# ==========================
# Dataset import (EveryVideoGameEver)
# ==========================
def download_and_extract_zip(zip_url: str, cache_dir: Path, log_cb=None) -> Path:
    ensure_dir(cache_dir)
    zip_key = sha256_text(zip_url)
    zip_path = cache_dir / f"{zip_key}.zip"
    extract_root = cache_dir / f"{zip_key}_extracted"

    if extract_root.exists():
        return extract_root

    if not zip_path.exists():
        _emit_log(log_cb, f"[DATASET] Downloading zip: {zip_url}")
        data = download_bytes(zip_url, timeout_s=180)
        zip_path.write_bytes(data)

    _emit_log(log_cb, "[DATASET] Extracting zip…")
    ensure_dir(extract_root)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_root)

    return extract_root

def iter_json_files(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*.json") if p.is_file()])

def dedupe_preserve(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        k = x.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(x)
    return out

def extract_titles_from_json(obj: Any) -> List[str]:
    preferred_keys = ["name", "title", "game", "Game", "Title", "Name"]

    def extract_from_item(item: Any) -> Optional[str]:
        if isinstance(item, str):
            return item.strip()
        if isinstance(item, dict):
            for k in preferred_keys:
                v = item.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
            for v in item.values():
                if isinstance(v, str) and v.strip():
                    return v.strip()
        return None

    titles: List[str] = []
    if isinstance(obj, dict):
        for container_key in ["data", "games", "items", "list", "entries"]:
            v = obj.get(container_key)
            if isinstance(v, list):
                for it in v:
                    t = extract_from_item(it)
                    if t:
                        titles.append(t)
                return dedupe_preserve(titles)
        t = extract_from_item(obj)
        return [t] if t else []

    if isinstance(obj, list):
        for it in obj:
            t = extract_from_item(it)
            if t:
                titles.append(t)
        return dedupe_preserve(titles)

    return []

def load_dataset_platform_titles(dataset_root: Path, gamesdb_subdir: str) -> Dict[str, List[str]]:
    gamesdb = dataset_root / gamesdb_subdir
    if not gamesdb.exists():
        raise RuntimeError(f"[DATASET] Could not find GamesDB at: {gamesdb}")

    platform_map: Dict[str, List[str]] = {}
    for jf in iter_json_files(gamesdb):
        platform_name = jf.stem
        try:
            obj = json.loads(jf.read_text(encoding="utf-8", errors="replace"))
            titles = extract_titles_from_json(obj)
            if titles:
                platform_map[platform_name] = titles
        except Exception:
            continue

    if not platform_map:
        raise RuntimeError("[DATASET] No platform JSONs found / no titles extracted.")
    return platform_map

def resolve_platform_titles(
    dataset_platform_to_titles: Dict[str, List[str]],
    platform_aliases: Dict[str, List[str]],
    desired_platform_key: str,
    platform_config: Optional[Dict[str, Any]] = None,
    callbacks=None
) -> Tuple[str, List[str]]:
    """
    Strict-normalized resolver:
      - Matches aliases to dataset keys by normalized equality (case/punct insensitive).
      - NO substring/prefix fuzzy matching (prevents DS/3DS, GB/GBC/GBA collisions).
      - Falls back to Wikipedia scraping if platform has wikipedia_url in config.
    """
    desired = desired_platform_key.strip()
    aliases = (platform_aliases.get(desired_platform_key, []) or []) + [desired]

    # Build normalized lookup for dataset keys
    norm_map = {}
    for k in dataset_platform_to_titles.keys():
        nk = norm_key(k)
        # if collision, keep the first; collisions are rare and should be fixed via aliasing
        norm_map.setdefault(nk, k)

    for a in aliases:
        na = norm_key(a)
        if not na:
            continue
        if na in norm_map:
            real_key = norm_map[na]
            return real_key, dataset_platform_to_titles[real_key]

    # No match in dataset - check for Wikipedia fallback
    if platform_config:
        wikipedia_url = platform_config.get("wikipedia_url")
        if wikipedia_url:
            _emit_log(callbacks, f"[DATASET] No dataset match for {desired_platform_key}, trying Wikipedia fallback...")
            titles = fetch_wikipedia_game_list(wikipedia_url, callbacks=callbacks)
            if titles:
                _emit_log(callbacks, f"[DATASET] Wikipedia fallback loaded {len(titles)} titles for {desired_platform_key}")
                return desired_platform_key, titles
            else:
                _emit_log(callbacks, f"[DATASET] Wikipedia fallback failed for {desired_platform_key}")

    raise KeyError(f'No dataset platform match for {desired_platform_key}. Tried aliases: {aliases}')




# ==========================
# Public API for UI
# ==========================
def read_platform_keys(config_path: Path) -> List[str]:
    cfg = load_yaml(config_path)
    platforms_cfg = cfg.get("platforms", {}) or {}
    return sorted(platforms_cfg.keys())

def get_output_dir(config_path: Path) -> Path:
    cfg = load_yaml(config_path)
    root = Path(config_path).resolve().parent
    paths = cfg.get("paths", {}) or {}
    return (root / paths.get("output_dir", "./output")).resolve()

def get_review_dir(config_path: Path) -> Path:
    cfg = load_yaml(config_path)
    root = Path(config_path).resolve().parent
    paths = cfg.get("paths", {}) or {}
    return (root / paths.get("review_dir", "./review")).resolve()


# ==========================
# Wikipedia Game List Scraper
# ==========================
def fetch_wikipedia_game_list(url: str, callbacks=None) -> List[str]:
    """
    Scrape game titles from a Wikipedia "List of games" page.
    Returns list of game titles.
    """
    import re
    from html import unescape

    try:
        _emit_log(callbacks, f"[WIKIPEDIA] Fetching game list from {url}")

        # Use Wikipedia API for cleaner HTML
        api_url = 'https://en.wikipedia.org/w/api.php'

        # Extract page title from URL
        page_title = url.split('/wiki/')[-1]

        params = {
            'action': 'parse',
            'page': page_title,
            'format': 'json',
            'prop': 'text',
            'formatversion': '2'
        }

        headers = {
            'User-Agent': 'IconGenerator/1.0 (Educational project for game icon generation)'
        }

        response = requests.get(api_url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()

        if 'parse' not in data or 'text' not in data['parse']:
            _emit_log(callbacks, f"[WIKIPEDIA] No parse data returned for {page_title}")
            return []

        html_content = data['parse']['text']

        titles = []

        # Wikipedia game list format:
        # <tr><td><i>Game Title</i></td><td>Genre</td>...</tr>
        # Pattern to match <td><i>Title</i></td> at the start of table rows

        # Extract all <i> tags within <td> tags
        td_i_pattern = r'<td[^>]*><i>([^<]+)</i>'
        matches = re.findall(td_i_pattern, html_content)

        for match in matches:
            # Clean up the title
            title = unescape(match).strip()
            # Remove footnote references like [1], [a], etc.
            title = re.sub(r'\[[^\]]+\]', '', title).strip()

            # Skip empty, very short titles
            if len(title) < 2:
                continue

            # Filter out obvious non-game entries
            skip_terms = [
                'unreleased', 'cancelled', 'tba', 'tbd',
                'unknown', 'various', 'multiple', 'n/a',
                'yes', 'no', 'genre', 'developer', 'publisher'
            ]
            if any(skip in title.lower() for skip in skip_terms):
                continue

            # Only add unique titles
            if title not in titles:
                titles.append(title)

        _emit_log(callbacks, f"[WIKIPEDIA] Found {len(titles)} game titles")
        return titles

    except Exception as e:
        _emit_log(callbacks, f"[WIKIPEDIA] Error fetching {url}: {e}")
        return []

# ==========================
# Providers
# ==========================
def fetch_multiple_art_from_steamgriddb(
    *,
    api_key: str,
    base_url: str,
    timeout_s: int,
    delay_s: float,
    cache_dir: Path,
    allow_animated: bool,
    prefer_dim: str,
    square_styles: List[str],
    square_only: bool,
    platform_key: str,
    title: str,
    platform_hints: List[str],
    max_results: int = 5,
    callbacks=None
) -> List[Tuple[bytes, str]]:
    """
    Fetch multiple artwork options from SteamGridDB.
    Returns list of (bytes, source_tag) tuples.
    """
    results = []

    try:
        _emit_log(callbacks, f"[DEBUG] SteamGridDB: Searching for multiple results for '{title}'...")
        autocomplete_results = search_autocomplete(api_key, base_url, title, timeout_s)

        if not autocomplete_results:
            return results

        if delay_s > 0:
            time.sleep(delay_s)

        # Get best game ID
        game_id = choose_best_game_id(api_key, base_url, timeout_s, delay_s, title, platform_hints, autocomplete_results, 8)
        if not game_id:
            return results

        # Fetch all grids for this game
        grids = grids_by_game(api_key, base_url, game_id, [prefer_dim], square_styles, timeout_s)
        if not grids:
            return results

        if delay_s > 0:
            time.sleep(delay_s)

        # Filter grids based on preferences
        suitable_grids = []
        for grid in grids:
            # Check animation
            if not allow_animated and grid.get("mime", "").startswith("image/webp"):
                continue
            # Check if square only
            if square_only and grid.get("width") != grid.get("height"):
                continue
            suitable_grids.append(grid)

        # Take up to max_results grids
        for grid in suitable_grids[:max_results]:
            url = grid.get("url")
            if not url:
                continue

            try:
                cache_key = sha256_text(url)
                cache_path = cache_dir / f"{cache_key}.bin"

                if cache_path.exists():
                    img_bytes = cache_path.read_bytes()
                else:
                    img_bytes = download_bytes(url, timeout_s)
                    cache_path.write_bytes(img_bytes)

                # Add grid style info to source tag
                style = grid.get("style", "unknown")
                source_tag = f"SteamGridDB - {style}"
                results.append((img_bytes, source_tag))

            except Exception as e:
                _emit_log(callbacks, f"[DEBUG] SteamGridDB: Failed to download grid - {e}")
                continue

        _emit_log(callbacks, f"[DEBUG] SteamGridDB: Returning {len(results)} artwork options")
        return results

    except Exception as e:
        _emit_log(callbacks, f"[DEBUG] SteamGridDB: Error - {e}")
        return results

def fetch_art_from_steamgriddb_square(
    *,
    api_key: str,
    base_url: str,
    timeout_s: int,
    delay_s: float,
    cache_dir: Path,
    allow_animated: bool,
    prefer_dim: str,
    square_styles: List[str],
    square_only: bool,
    platform_key: str,
    title: str,
    platform_hints: List[str],
    callbacks=None
) -> Optional[Tuple[bytes, str]]:
    # returns (bytes, source_tag) or None

    # Write to debug file
    import datetime
    debug_file = Path("steamgriddb_debug.log")
    with debug_file.open("a", encoding="utf-8") as f:
        f.write(f"[{datetime.datetime.now()}] Function called for: {title}\n")
        f.flush()

    _emit_log(callbacks, f"[DEBUG] SteamGridDB: Function called for '{title}'")

    try:
        with debug_file.open("a", encoding="utf-8") as f:
            f.write(f"[{datetime.datetime.now()}] About to call search_autocomplete\n")
            f.flush()

        _emit_log(callbacks, f"[DEBUG] SteamGridDB: Searching autocomplete for '{title}'...")
        results = search_autocomplete(api_key, base_url, title, timeout_s)

        with debug_file.open("a", encoding="utf-8") as f:
            f.write(f"[{datetime.datetime.now()}] Autocomplete returned {len(results) if results else 0} results\n")
            f.flush()

        _emit_log(callbacks, f"[DEBUG] SteamGridDB: Autocomplete returned {len(results) if results else 0} results")
        if delay_s > 0:
            time.sleep(delay_s)
    except Exception as e:
        with debug_file.open("a", encoding="utf-8") as f:
            f.write(f"[{datetime.datetime.now()}] Exception: {type(e).__name__}: {e}\n")
            f.flush()

        _emit_log(callbacks, f"[DEBUG] SteamGridDB: Autocomplete failed - {type(e).__name__}: {e}")
        return None
    if not results:
        _emit_log(callbacks, f"[DEBUG] SteamGridDB: No autocomplete results for '{title}'")
        return None

    _emit_log(callbacks, f"[DEBUG] SteamGridDB: Choosing best game ID from {len(results)} results...")
    game_id = choose_best_game_id(api_key, base_url, timeout_s, delay_s, title, platform_hints, results, 8)
    if not game_id:
        _emit_log(callbacks, f"[DEBUG] SteamGridDB: No matching game ID found")
        return None
    _emit_log(callbacks, f"[DEBUG] SteamGridDB: Selected game ID: {game_id}")

    try:
        _emit_log(callbacks, f"[DEBUG] SteamGridDB: Fetching grids for game ID {game_id}...")
        grids = grids_by_game(api_key, base_url, game_id, [prefer_dim], square_styles, timeout_s)
        _emit_log(callbacks, f"[DEBUG] SteamGridDB: Found {len(grids) if grids else 0} grids")
        if delay_s > 0:
            time.sleep(delay_s)
    except Exception as e:
        _emit_log(callbacks, f"[DEBUG] SteamGridDB: Grid fetch failed - {type(e).__name__}: {e}")
        return None

    _emit_log(callbacks, f"[DEBUG] SteamGridDB: Picking best grid...")
    best = pick_best_grid(grids, prefer_dim=prefer_dim, allow_animated=allow_animated, square_only=square_only)
    if not best or not best.get("url"):
        _emit_log(callbacks, f"[DEBUG] SteamGridDB: No suitable grid found")
        return None
    _emit_log(callbacks, f"[DEBUG] SteamGridDB: Selected grid URL: {best.get('url')}")

    url = best["url"]
    cache_key = sha256_text(url)
    cache_path = cache_dir / f"{cache_key}.bin"
    try:
        if cache_path.exists():
            _emit_log(callbacks, f"[DEBUG] SteamGridDB: Using cached image")
            return cache_path.read_bytes(), "steamgriddb_square"
        _emit_log(callbacks, f"[DEBUG] SteamGridDB: Downloading image...")
        img_bytes = download_bytes(url, timeout_s)
        cache_path.write_bytes(img_bytes)
        _emit_log(callbacks, f"[DEBUG] SteamGridDB: Image downloaded and cached")
        return img_bytes, "steamgriddb_square"
    except Exception as e:
        _emit_log(callbacks, f"[DEBUG] SteamGridDB: Download failed - {type(e).__name__}: {e}")
        return None


def fetch_art_from_libretro(
    *,
    lr_base: str,
    lr_type_dir: str,
    lr_playlist_map: Dict[str, str],
    timeout_s: int,
    platform_key: str,
    title: str,
    cache_dir: Path,
    use_index_matching: bool,
    index_cache_hours: int,
    debug_log=None
) -> Optional[Tuple[bytes, str]]:
    playlist = lr_playlist_map.get(platform_key)
    if not playlist:
        return None

    b = libretro_try_download_boxart(
        base_url=lr_base,
        playlist_name=playlist,
        type_dir=lr_type_dir,
        title=title,
        timeout_s=timeout_s,
        cache_dir=cache_dir,
        use_index_matching=use_index_matching,
        index_cache_hours=index_cache_hours,
        debug_log=debug_log
    )
    if not b:
        return None
    return b, "libretro_boxart"


# ==========================
# IGDB Provider
# ==========================
_igdb_token_cache = {"token": None, "expires_at": 0}

def get_igdb_access_token(client_id: str, client_secret: str, timeout_s: int) -> Optional[str]:
    """Get IGDB access token using Twitch OAuth."""
    import time

    # Check cached token
    if _igdb_token_cache["token"] and time.time() < _igdb_token_cache["expires_at"]:
        return _igdb_token_cache["token"]

    # Request new token
    try:
        url = "https://id.twitch.tv/oauth2/token"
        params = {
            "client_id": client_id,
            "client_secret": client_secret,
            "grant_type": "client_credentials"
        }
        r = requests.post(url, params=params, timeout=timeout_s)
        r.raise_for_status()
        data = r.json()

        token = data.get("access_token")
        expires_in = data.get("expires_in", 3600)

        # Cache token with 5 minute buffer
        _igdb_token_cache["token"] = token
        _igdb_token_cache["expires_at"] = time.time() + expires_in - 300

        return token
    except Exception:
        return None


def fetch_art_from_igdb(
    *,
    client_id: str,
    client_secret: str,
    base_url: str,
    timeout_s: int,
    delay_s: float,
    platform_map: Dict[str, int],
    cover_size: str,
    platform_key: str,
    title: str,
    cache_dir: Path,
    debug_log=None
) -> Optional[Tuple[bytes, str]]:
    """Fetch artwork from IGDB."""
    def _log(msg):
        if debug_log and callable(debug_log):
            debug_log(msg)

    # Get platform ID
    platform_id = platform_map.get(platform_key)
    if not platform_id:
        _log(f"[DEBUG] IGDB: No platform mapping for {platform_key}")
        return None
    _log(f"[DEBUG] IGDB: Platform {platform_key} -> ID {platform_id}")

    # Get access token
    _log(f"[DEBUG] IGDB: Getting access token...")
    token = get_igdb_access_token(client_id, client_secret, timeout_s)
    if not token:
        _log(f"[DEBUG] IGDB: Failed to get access token")
        return None
    _log(f"[DEBUG] IGDB: Got access token: {token[:20]}...")

    try:
        # Search for game
        headers = {
            "Client-ID": client_id,
            "Authorization": f"Bearer {token}",
            "Accept": "application/json"
        }

        # IGDB uses POST with Apicalypse query language
        search_url = f"{base_url.rstrip('/')}/games"
        query = f'search "{title}"; fields name,cover.image_id,platforms; where platforms = ({platform_id}); limit 5;'

        _log(f"[DEBUG] IGDB: Searching for '{title}' on platform {platform_id}...")
        r = requests.post(search_url, headers=headers, data=query, timeout=timeout_s)
        r.raise_for_status()
        games = r.json()
        _log(f"[DEBUG] IGDB: Found {len(games)} games")

        if delay_s > 0:
            time.sleep(delay_s)

        if not games:
            _log(f"[DEBUG] IGDB: No games found for '{title}' on platform {platform_id}")
            return None

        # Get best match (first result, IGDB search is quite good)
        game = games[0]
        _log(f"[DEBUG] IGDB: Best match: '{game.get('name')}'")
        cover = game.get("cover")

        if not cover or "image_id" not in cover:
            _log(f"[DEBUG] IGDB: No cover found for '{game.get('name')}'")
            return None

        # Build cover URL
        image_id = cover["image_id"]
        # IGDB image URL format: https://images.igdb.com/igdb/image/upload/t_{size}/{image_id}.jpg
        # Sizes: cover_small (90x128), cover_big (264x374), 720p (1280x720), 1080p (1920x1080)
        cover_url = f"https://images.igdb.com/igdb/image/upload/t_{cover_size}/{image_id}.jpg"
        _log(f"[DEBUG] IGDB: Cover URL: {cover_url}")

        # Download and cache
        cache_key = sha256_text(cover_url)
        cache_path = cache_dir / f"{cache_key}.bin"

        if cache_path.exists():
            _log(f"[DEBUG] IGDB: Using cached image")
            return cache_path.read_bytes(), "igdb_cover"

        _log(f"[DEBUG] IGDB: Downloading cover...")
        img_bytes = download_bytes(cover_url, timeout_s)
        cache_path.write_bytes(img_bytes)
        _log(f"[DEBUG] IGDB: Cover downloaded and cached")

        return img_bytes, "igdb_cover"

    except Exception as e:
        _log(f"[DEBUG] IGDB: Error - {type(e).__name__}: {e}")
        return None


# ==========================
# TheGamesDB Provider
# ==========================
def fetch_art_from_thegamesdb(
    *,
    api_key: str,
    base_url: str,
    timeout_s: int,
    delay_s: float,
    platform_map: Dict[str, int],
    prefer_image_type: str,
    platform_key: str,
    title: str,
    cache_dir: Path,
    debug_log=None
) -> Optional[Tuple[bytes, str]]:
    """Fetch artwork from TheGamesDB."""
    def _log(msg):
        if debug_log and callable(debug_log):
            debug_log(msg)

    # Get platform ID
    platform_id = platform_map.get(platform_key)
    if not platform_id:
        _log(f"[DEBUG] TheGamesDB: No platform mapping for {platform_key}")
        return None
    _log(f"[DEBUG] TheGamesDB: Platform {platform_key} -> ID {platform_id}")

    try:
        # Search for game
        search_url = f"{base_url.rstrip('/')}/Games/ByGameName"
        params = {
            "apikey": api_key,
            "name": title,
            "filter[platform]": platform_id
        }

        _log(f"[DEBUG] TheGamesDB: Searching for '{title}' on platform {platform_id}...")
        r = requests.get(search_url, params=params, timeout=timeout_s)
        r.raise_for_status()
        data = r.json()

        if delay_s > 0:
            time.sleep(delay_s)

        games = data.get("data", {}).get("games", [])
        _log(f"[DEBUG] TheGamesDB: Found {len(games)} games")
        if not games:
            _log(f"[DEBUG] TheGamesDB: No games found for '{title}' on platform {platform_id}")
            return None

        # Get first match
        game = games[0]
        game_id = game.get("id")
        game_name = game.get("game_title", title)
        _log(f"[DEBUG] TheGamesDB: Best match: '{game_name}' (ID: {game_id})")

        # Fetch images for this game
        images_url = f"{base_url.rstrip('/')}/Games/Images"
        params = {
            "apikey": api_key,
            "games_id": game_id
        }

        _log(f"[DEBUG] TheGamesDB: Fetching images for game ID {game_id}...")
        r = requests.get(images_url, params=params, timeout=timeout_s)
        r.raise_for_status()
        img_data = r.json()

        if delay_s > 0:
            time.sleep(delay_s)

        # Get base image URL
        base_img_url = img_data.get("data", {}).get("base_url", {}).get("original")
        images_list = img_data.get("data", {}).get("images", {}).get(str(game_id), [])
        _log(f"[DEBUG] TheGamesDB: Found {len(images_list) if images_list else 0} images")

        if not images_list or not base_img_url:
            _log(f"[DEBUG] TheGamesDB: No images found for '{game_name}'")
            return None

        # Find preferred image type
        best_image = None
        for img in images_list:
            if img.get("type") == prefer_image_type:
                best_image = img
                break

        # Fallback to any boxart or first image
        if not best_image:
            for img in images_list:
                if img.get("type") == "boxart":
                    best_image = img
                    break

        if not best_image and images_list:
            best_image = images_list[0]

        if not best_image:
            _log(f"[DEBUG] TheGamesDB: No suitable images for '{game_name}'")
            return None

        # Build image URL
        filename = best_image.get("filename")
        image_url = f"{base_img_url}{filename}"
        _log(f"[DEBUG] TheGamesDB: Selected image: {image_url}")

        # Download and cache
        cache_key = sha256_text(image_url)
        cache_path = cache_dir / f"{cache_key}.bin"

        if cache_path.exists():
            _log(f"[DEBUG] TheGamesDB: Using cached image")
            return cache_path.read_bytes(), "thegamesdb_boxart"

        _log(f"[DEBUG] TheGamesDB: Downloading image...")
        img_bytes = download_bytes(image_url, timeout_s)
        cache_path.write_bytes(img_bytes)
        _log(f"[DEBUG] TheGamesDB: Image downloaded and cached")

        return img_bytes, "thegamesdb_boxart"

    except Exception as e:
        _log(f"[DEBUG] TheGamesDB: Error - {type(e).__name__}: {e}")
        return None


# Stub "elsewhere" provider you can implement later
def fetch_art_from_custom_http(*, timeout_s: int, platform_key: str, title: str) -> Optional[Tuple[bytes, str]]:
    return None


# ==========================
# Config Migration
# ==========================
def migrate_legacy_art_sources(art_sources: dict) -> dict:
    """Migrate old 'mode' string to new providers list structure."""
    if "providers" in art_sources:
        return art_sources  # Already migrated

    # Parse legacy mode string
    mode = art_sources.get("mode", "steamgriddb_then_libretro")
    sg_square = art_sources.get("steamgriddb_square_only", True)
    lr_crop = art_sources.get("libretro_crop_mode", "center_crop")

    # Build providers list from mode
    providers = []
    if mode == "steamgriddb":
        providers = [{"id": "steamgriddb", "enabled": True, "square_only": sg_square}]
    elif mode == "libretro":
        providers = [{"id": "libretro", "enabled": True, "crop_mode": lr_crop}]
    elif mode == "libretro_then_steamgriddb":
        providers = [
            {"id": "libretro", "enabled": True, "crop_mode": lr_crop},
            {"id": "steamgriddb", "enabled": True, "square_only": sg_square}
        ]
    else:  # steamgriddb_then_libretro (default)
        providers = [
            {"id": "steamgriddb", "enabled": True, "square_only": sg_square},
            {"id": "libretro", "enabled": True, "crop_mode": lr_crop}
        ]

    # Add new providers (disabled by default)
    providers.extend([
        {"id": "igdb", "enabled": False},
        {"id": "thegamesdb", "enabled": False}
    ])

    return {"providers": providers, "mode": mode}  # Keep mode for reference


def run_job(
    config_path: Path,
    platforms: List[str],
    workers: int,
    limit: int,
    cancel: CancelToken,
    callbacks=None,
    source_order: Optional[List[Dict[str, Any]]] = None,
    source_mode: Optional[str] = None,
    steamgriddb_square_only: Optional[bool] = None,
    search_term: Optional[str] = None,
    letter_filter: Optional[str] = None,
    interactive_mode: bool = False
) -> Tuple[bool, str]:

    config_path = Path(config_path)
    root = config_path.resolve().parent

    try:
        cfg = load_yaml(config_path)
    except Exception as e:
        return False, f"Failed to read config: {e}"

    out_size = int(cfg.get("output_size", 1024))
    export_format = str(cfg.get("export_format", "PNG")).upper()

    paths = cfg.get("paths", {}) or {}
    borders_dir = root / paths.get("borders_dir", "./borders")
    output_dir = root / paths.get("output_dir", "./output")
    review_dir = root / paths.get("review_dir", "./review")
    cache_dir = root / paths.get("cache_dir", "./cache")
    dataset_cache_dir = root / paths.get("dataset_cache_dir", "./dataset_cache")

    for d in [borders_dir, output_dir, review_dir, cache_dir, dataset_cache_dir]:
        ensure_dir(d)

    platforms_cfg = cfg.get("platforms", {}) or {}
    platform_aliases = cfg.get("platform_aliases", {}) or {}
    platform_hints_cfg = cfg.get("sgdb_platform_hints", {}) or {}

    # Dataset
    dataset_cfg = cfg.get("dataset", {}) or {}
    repo_zip_url = dataset_cfg.get("repo_zip_url")
    gamesdb_subdir = dataset_cfg.get("gamesdb_subdir", "EveryVideoGameEver-main/GamesDB")
    cfg_limit = int(dataset_cfg.get("per_platform_limit", 0))
    per_platform_limit = limit if limit > 0 else cfg_limit
    if not repo_zip_url:
        return False, "dataset.repo_zip_url is missing in config.yaml"

    # Art source configuration - migrate legacy format
    art_sources = migrate_legacy_art_sources(cfg.get("art_sources", {}) or {})

    # Build provider_order and settings
    provider_order = []
    provider_settings = {}

    # UI-provided source_order takes precedence
    if source_order is not None:
        providers_config = source_order
    else:
        providers_config = art_sources.get("providers", [])

    for prov_cfg in providers_config:
        prov_id = prov_cfg.get("id")
        if prov_cfg.get("enabled", False):
            provider_order.append(prov_id)
            provider_settings[prov_id] = prov_cfg

    # Legacy override from UI if source_mode provided (backward compat)
    if source_mode:
        mode = source_mode
        if mode == "steamgriddb":
            provider_order = ["steamgriddb"]
        elif mode == "libretro":
            provider_order = ["libretro"]
        elif mode == "libretro_then_steamgriddb":
            provider_order = ["libretro", "steamgriddb"]
        else:  # steamgriddb_then_libretro
            provider_order = ["steamgriddb", "libretro"]

        # Rebuild settings for legacy mode
        provider_settings = {}
        for pid in provider_order:
            provider_settings[pid] = {"id": pid, "enabled": True}
            if pid == "steamgriddb" and steamgriddb_square_only is not None:
                provider_settings[pid]["square_only"] = steamgriddb_square_only

    # Validate we have at least one provider
    if not provider_order:
        return False, "No artwork sources enabled. Enable at least one source in config or UI."

    # SteamGridDB config
    sg = cfg.get("steamgriddb", {}) or {}
    api_env = sg.get("api_key_env", "SGDB_API_KEY")
    api_key = os.environ.get(api_env, "").strip()
    base_url = sg.get("base_url", "https://www.steamgriddb.com/api/v2")
    timeout_s = int(sg.get("request_timeout_seconds", 40))
    delay_s = float(sg.get("delay_seconds", 0.25))
    allow_animated = bool(sg.get("allow_animated", False))
    prefer_dimensions = sg.get("prefer_dimensions", ["1024x1024"])
    prefer_dim = prefer_dimensions[0] if prefer_dimensions else "1024x1024"
    # Valid SteamGridDB styles: alternate, material, white_logo, blurred, no_logo
    # Invalid styles: official, black_logo (will cause 400 errors)
    square_styles = sg.get("square_styles", ["alternate", "material", "white_logo", "blurred", "no_logo"])
    # Get square_only from provider settings if steamgriddb is enabled
    sg_square_only = True  # Default value
    if "steamgriddb" in provider_settings:
        sg_square_only = bool(provider_settings["steamgriddb"].get("square_only", True))

    # Libretro config
    lr = cfg.get("libretro", {}) or {}
    lr_base = lr.get("base_url", "https://thumbnails.libretro.com")
    lr_type_dir = lr.get("type_dir", "Named_Boxarts")
    lr_playlist_map = lr.get("playlist_names", {}) or {}
    use_index_matching = bool(lr.get("use_index_matching", True))
    index_cache_hours = int(lr.get("index_cache_hours", 168))

    # IGDB config
    igdb_cfg = cfg.get("igdb", {}) or {}
    igdb_client_id_env = igdb_cfg.get("client_id_env", "IGDB_CLIENT_ID")
    igdb_client_secret_env = igdb_cfg.get("client_secret_env", "IGDB_CLIENT_SECRET")
    igdb_client_id = os.environ.get(igdb_client_id_env, "").strip()
    igdb_client_secret = os.environ.get(igdb_client_secret_env, "").strip()
    igdb_base_url = igdb_cfg.get("base_url", "https://api.igdb.com/v4")
    igdb_timeout = int(igdb_cfg.get("request_timeout_seconds", 30))
    igdb_delay = float(igdb_cfg.get("delay_seconds", 0.25))
    igdb_cover_size = igdb_cfg.get("cover_size", "cover_big")
    igdb_platform_map = igdb_cfg.get("platform_map", {}) or {}

    # TheGamesDB config
    tgdb_cfg = cfg.get("thegamesdb", {}) or {}
    tgdb_api_key_env = tgdb_cfg.get("api_key_env", "TGDB_API_KEY")
    tgdb_api_key = os.environ.get(tgdb_api_key_env, "").strip()
    tgdb_base_url = tgdb_cfg.get("base_url", "https://api.thegamesdb.net/v1")
    tgdb_timeout = int(tgdb_cfg.get("request_timeout_seconds", 30))
    tgdb_delay = float(tgdb_cfg.get("delay_seconds", 0.5))
    tgdb_image_type = tgdb_cfg.get("prefer_image_type", "boxart")
    tgdb_platform_map = tgdb_cfg.get("platform_map", {}) or {}

    # Auto-centering config
    ac = cfg.get("auto_centering", {}) or {}
    ac_enabled = bool(ac.get("enabled", True))
    ac_sources = set(ac.get("sources", ["libretro_boxart"]))
    ac_tolerance = float(ac.get("tolerance", 0.06))
    ac_steps = int(ac.get("search_steps", 5))
    ac_span = float(ac.get("search_span", 0.22))
    ac_alpha_threshold = int(ac.get("alpha_threshold", 16))
    ac_margin_pct = float(ac.get("margin_pct", 0.06))

    # Logo detection config
    ld = cfg.get("logo_detection", {}) or {}
    ld_enabled = bool(ld.get("enabled", False))
    ld_method = str(ld.get("method", "auto"))
    ld_sources = set(ld.get("sources", ["libretro_boxart", "steamgriddb_square"]))
    ld_min_content = float(ld.get("min_content_ratio", 0.15))
    ld_max_crop = float(ld.get("max_crop_ratio", 0.85))

    # Log provider order for debugging
    _emit_log(callbacks, f"[CONFIG] Provider order: {provider_order}")
    _emit_log(callbacks, f"[CONFIG] Provider settings: {provider_settings}")

    # Validate required API keys for enabled providers
    if "steamgriddb" in provider_order and not api_key:
        return False, f"Missing SteamGridDB API key env var: {api_env}"
    if "igdb" in provider_order and (not igdb_client_id or not igdb_client_secret):
        return False, f"Missing IGDB credentials: {igdb_client_id_env}, {igdb_client_secret_env}"
    if "thegamesdb" in provider_order and not tgdb_api_key:
        return False, f"Missing TheGamesDB API key env var: {tgdb_api_key_env}"

    # Log API key status
    _emit_log(callbacks, f"[CONFIG] SteamGridDB API key: {'SET' if api_key else 'NOT SET'}")
    _emit_log(callbacks, f"[CONFIG] IGDB Client ID: {'SET' if igdb_client_id else 'NOT SET'}")
    _emit_log(callbacks, f"[CONFIG] IGDB Client Secret: {'SET' if igdb_client_secret else 'NOT SET'}")
    _emit_log(callbacks, f"[CONFIG] TheGamesDB API key: {'SET' if tgdb_api_key else 'NOT SET'}")

    _emit_log(callbacks, f"[CONFIG] Using providers: {', '.join(provider_order)}")

    if cancel.is_cancelled:
        return False, "Cancelled."

    # Load dataset
    _emit_log(callbacks, "[DATASET] Loading game database...")
    dataset_root = download_and_extract_zip(repo_zip_url, dataset_cache_dir, log_cb=callbacks)
    dataset_platform_to_titles = load_dataset_platform_titles(dataset_root, gamesdb_subdir)
    _emit_log(callbacks, f"[DATASET] Found {len(dataset_platform_to_titles)} platform JSONs.")

    # Build task list
    tasks = []

    for platform_key in platforms:
        if cancel.is_cancelled:
            return False, "Cancelled."

        pconf = platforms_cfg.get(platform_key, {})
        border_file = pconf.get("border_file")
        border_path = borders_dir / border_file if border_file else None
        if not border_path or not border_path.exists():
            _emit_log(callbacks, f"[WARN] Missing border for {platform_key}: {border_path}")
            continue

        try:
            _, titles = resolve_platform_titles(
                dataset_platform_to_titles,
                platform_aliases,
                platform_key,
                platform_config=pconf,
                callbacks=callbacks
            )
        except Exception as e:
            _emit_log(callbacks, f"[WARN] {e}")
            continue

        # Apply search/filter before limit
        if search_term:
            # Search by name - case insensitive partial match
            search_lower = search_term.lower()
            titles = [t for t in titles if search_lower in t.lower()]
            _emit_log(callbacks, f"[FILTER] Search '{search_term}' on {platform_key}: {len(titles)} matches")
        elif letter_filter and letter_filter != "All":
            # Filter by starting letter
            if letter_filter == "0-9":
                titles = [t for t in titles if t[0].isdigit()]
            elif letter_filter == "#":
                titles = [t for t in titles if not t[0].isalnum()]
            else:
                titles = [t for t in titles if t[0].upper() == letter_filter.upper()]
            _emit_log(callbacks, f"[FILTER] Letter '{letter_filter}' on {platform_key}: {len(titles)} matches")

        if per_platform_limit > 0:
            titles = titles[:per_platform_limit]

        out_plat = output_dir / platform_key
        rev_plat = review_dir / platform_key
        ensure_dir(out_plat)
        ensure_dir(rev_plat)

        for title in titles:
            # Create folder per game with icon.png and title.png
            game_folder = out_plat / safe_slug(title)
            out_path = game_folder / f"icon.{export_format.lower()}"
            if out_path.exists():
                continue
            tasks.append((platform_key, title, border_path, out_path, rev_plat))

    total = len(tasks)
    if total == 0:
        return True, "Nothing to do (already generated / missing borders / no matches)."

    _emit_progress(callbacks, 0, total)
    _emit_log(callbacks, f"[PLAN] Queued {total} images. Workers={workers}")

    done = 0
    done_lock = threading.Lock()
    errors = 0

    def fetch_all_artwork_options(platform_key: str, title: str, hints: List[str]) -> List[Dict[str, Any]]:
        """
        Fetch ALL artwork options from ALL providers (doesn't stop at first match).
        Returns list of dicts with keys: 'image_data' (bytes), 'source' (str), 'provider' (str)
        """
        options = []

        for prov in provider_order:
            if cancel.is_cancelled:
                break

            _emit_log(callbacks, f"[INTERACTIVE] {platform_key}: {title} - Fetching from {prov}...")

            try:
                if prov == "steamgriddb":
                    # Fetch multiple results from SteamGridDB
                    results = fetch_multiple_art_from_steamgriddb(
                        api_key=api_key,
                        base_url=base_url,
                        timeout_s=timeout_s,
                        delay_s=delay_s,
                        cache_dir=cache_dir,
                        allow_animated=allow_animated,
                        prefer_dim=prefer_dim,
                        square_styles=square_styles,
                        square_only=sg_square_only,
                        platform_key=platform_key,
                        title=title,
                        platform_hints=hints,
                        max_results=5,
                        callbacks=callbacks,
                    )
                    # Add each result to options
                    for img_bytes, source_tag in results:
                        options.append({
                            'image_data': img_bytes,
                            'source': source_tag,
                            'provider': prov
                        })
                    _emit_log(callbacks, f"[INTERACTIVE] {platform_key}: {title} - Found {len(results)} from {prov}")
                elif prov == "libretro":
                    # Libretro returns single result
                    got = fetch_art_from_libretro(
                        lr_base=lr_base,
                        lr_type_dir=lr_type_dir,
                        lr_playlist_map=lr_playlist_map,
                        timeout_s=timeout_s,
                        platform_key=platform_key,
                        title=title,
                        cache_dir=cache_dir,
                        use_index_matching=use_index_matching,
                        index_cache_hours=index_cache_hours,
                        debug_log=lambda m: _emit_log(callbacks, m),
                    )
                    if got:
                        img_bytes, source_tag = got
                        options.append({
                            'image_data': img_bytes,
                            'source': source_tag,
                            'provider': prov
                        })
                        _emit_log(callbacks, f"[INTERACTIVE] {platform_key}: {title} - Found 1 from {prov}")

                elif prov == "igdb":
                    # IGDB returns single result
                    got = fetch_art_from_igdb(
                        client_id=igdb_client_id,
                        client_secret=igdb_client_secret,
                        base_url=igdb_base_url,
                        timeout_s=igdb_timeout,
                        delay_s=igdb_delay,
                        platform_map=igdb_platform_map,
                        cover_size=igdb_cover_size,
                        platform_key=platform_key,
                        title=title,
                        cache_dir=cache_dir,
                        debug_log=lambda m: _emit_log(callbacks, m),
                    )
                    if got:
                        img_bytes, source_tag = got
                        options.append({
                            'image_data': img_bytes,
                            'source': source_tag,
                            'provider': prov
                        })
                        _emit_log(callbacks, f"[INTERACTIVE] {platform_key}: {title} - Found 1 from {prov}")

                elif prov == "thegamesdb":
                    # TheGamesDB returns single result
                    got = fetch_art_from_thegamesdb(
                        api_key=tgdb_api_key,
                        base_url=tgdb_base_url,
                        timeout_s=tgdb_timeout,
                        delay_s=tgdb_delay,
                        platform_map=tgdb_platform_map,
                        prefer_image_type=tgdb_image_type,
                        platform_key=platform_key,
                        title=title,
                        cache_dir=cache_dir,
                        debug_log=lambda m: _emit_log(callbacks, m),
                    )
                    if got:
                        img_bytes, source_tag = got
                        options.append({
                            'image_data': img_bytes,
                            'source': source_tag,
                            'provider': prov
                        })
                        _emit_log(callbacks, f"[INTERACTIVE] {platform_key}: {title} - Found 1 from {prov}")

                elif prov == "custom_http":
                    # Custom HTTP returns single result
                    got = fetch_art_from_custom_http(
                        timeout_s=timeout_s,
                        platform_key=platform_key,
                        title=title
                    )
                    if got:
                        img_bytes, source_tag = got
                        options.append({
                            'image_data': img_bytes,
                            'source': source_tag,
                            'provider': prov
                        })
                        _emit_log(callbacks, f"[INTERACTIVE] {platform_key}: {title} - Found 1 from {prov}")

            except Exception as e:
                _emit_log(callbacks, f"[INTERACTIVE] {platform_key}: {title} - {prov} failed: {type(e).__name__}: {e}")
                continue

        return options

    def work_item(platform_key: str, title: str, border_path: Path, out_path: Path, rev_dir: Path) -> bool:
        nonlocal errors

        if cancel.is_cancelled:
            return False

        slug = safe_slug(title)
        hints = platform_hints_cfg.get(platform_key, []) or []

        img_bytes = None
        source_tag = None

        _emit_log(callbacks, f"[SEARCH] {platform_key}: {title} - Trying providers: {provider_order}")

        # Check if provider_order is empty
        if not provider_order:
            _emit_log(callbacks, f"[ERROR] {platform_key}: {title} - No providers configured!")
            (rev_dir / f"{slug}__no_providers.json").write_text(
                json.dumps({
                    "title": title,
                    "platform": platform_key,
                    "error": "no providers configured"
                }, indent=2),
                encoding="utf-8"
            )
            return False

        # Interactive mode: fetch from ALL providers and let user choose
        if interactive_mode:
            _emit_log(callbacks, f"[INTERACTIVE] {platform_key}: {title} - Fetching from all providers...")
            artwork_options = fetch_all_artwork_options(platform_key, title, hints)

            if not artwork_options:
                _emit_log(callbacks, f"[FAIL] {platform_key}: {title} - No artwork found from any provider")
                (rev_dir / f"{slug}__no_art.json").write_text(
                    json.dumps({
                        "title": title,
                        "platform": platform_key,
                        "provider_order": provider_order,
                        "error": "no art found from any provider"
                    }, indent=2),
                    encoding="utf-8"
                )
                return False

            # Request user selection from all options
            selected_index = _request_user_selection(callbacks, title, platform_key, artwork_options)

            if selected_index == -1:
                # User cancelled all - set cancel token
                _emit_log(callbacks, f"[STOP] User cancelled interactive mode")
                cancel.cancel()
                return False
            elif selected_index is None:
                # User skipped this title
                _emit_log(callbacks, f"[SKIP] {platform_key}: {title} - Skipped by user")
                return False
            elif 0 <= selected_index < len(artwork_options):
                # User selected an option
                selected = artwork_options[selected_index]
                img_bytes = selected['image_data']
                source_tag = selected['source']
                _emit_log(callbacks, f"[SELECTED] {platform_key}: {title} - User selected from {source_tag}")
            else:
                _emit_log(callbacks, f"[ERROR] {platform_key}: {title} - Invalid selection index")
                return False

        # Automatic mode: try each provider in order until one works
        else:
            for prov in provider_order:
                if cancel.is_cancelled:
                    return False

                if prov == "steamgriddb":
                    _emit_log(callbacks, f"[DB] {platform_key}: {title} - Searching SteamGridDB...")
                    try:
                        got = fetch_art_from_steamgriddb_square(
                            api_key=api_key,
                            base_url=base_url,
                            timeout_s=timeout_s,
                            delay_s=delay_s,
                            cache_dir=cache_dir,
                            allow_animated=allow_animated,
                            prefer_dim=prefer_dim,
                            square_styles=square_styles,
                            square_only=sg_square_only,
                            platform_key=platform_key,
                            title=title,
                            platform_hints=hints,
                            callbacks=callbacks,
                        )
                    except Exception as e:
                        _emit_log(callbacks, f"[ERROR] {platform_key}: {title} - SteamGridDB call failed: {type(e).__name__}: {e}")
                        got = None
                    if got:
                        img_bytes, source_tag = got
                        _emit_log(callbacks, f"[DB] {platform_key}: {title} - Found in SteamGridDB")
                        break
                    else:
                        _emit_log(callbacks, f"[DB] {platform_key}: {title} - Not found in SteamGridDB")

                elif prov == "libretro":
                    _emit_log(callbacks, f"[DB] {platform_key}: {title} - Searching Libretro...")
                    got = fetch_art_from_libretro(
                        lr_base=lr_base,
                        lr_type_dir=lr_type_dir,
                        lr_playlist_map=lr_playlist_map,
                        timeout_s=timeout_s,
                        platform_key=platform_key,
                        title=title,
                        cache_dir=cache_dir,
                        use_index_matching=use_index_matching,
                        index_cache_hours=index_cache_hours,
                        debug_log=lambda m: _emit_log(callbacks, m),
                    )
                    if got:
                        img_bytes, source_tag = got
                        _emit_log(callbacks, f"[DB] {platform_key}: {title} - Found in Libretro")
                        break
                    else:
                        _emit_log(callbacks, f"[DB] {platform_key}: {title} - Not found in Libretro")

                elif prov == "igdb":
                    _emit_log(callbacks, f"[DB] {platform_key}: {title} - Searching IGDB...")
                    got = fetch_art_from_igdb(
                        client_id=igdb_client_id,
                        client_secret=igdb_client_secret,
                        base_url=igdb_base_url,
                        timeout_s=igdb_timeout,
                        delay_s=igdb_delay,
                        platform_map=igdb_platform_map,
                        cover_size=igdb_cover_size,
                        platform_key=platform_key,
                        title=title,
                        cache_dir=cache_dir,
                        debug_log=lambda m: _emit_log(callbacks, m),
                    )
                    if got:
                        img_bytes, source_tag = got
                        _emit_log(callbacks, f"[DB] {platform_key}: {title} - Found in IGDB")
                        break
                    else:
                        _emit_log(callbacks, f"[DB] {platform_key}: {title} - Not found in IGDB")

                elif prov == "thegamesdb":
                    _emit_log(callbacks, f"[DB] {platform_key}: {title} - Searching TheGamesDB...")
                    got = fetch_art_from_thegamesdb(
                        api_key=tgdb_api_key,
                        base_url=tgdb_base_url,
                        timeout_s=tgdb_timeout,
                        delay_s=tgdb_delay,
                        platform_map=tgdb_platform_map,
                        prefer_image_type=tgdb_image_type,
                        platform_key=platform_key,
                        title=title,
                        cache_dir=cache_dir,
                        debug_log=lambda m: _emit_log(callbacks, m),
                    )
                    if got:
                        img_bytes, source_tag = got
                        _emit_log(callbacks, f"[DB] {platform_key}: {title} - Found in TheGamesDB")
                        break
                    else:
                        _emit_log(callbacks, f"[DB] {platform_key}: {title} - Not found in TheGamesDB")

                elif prov == "custom_http":
                    _emit_log(callbacks, f"[DB] {platform_key}: {title} - Searching Custom HTTP...")
                    got = fetch_art_from_custom_http(timeout_s=timeout_s, platform_key=platform_key, title=title)
                    if got:
                        img_bytes, source_tag = got
                        _emit_log(callbacks, f"[DB] {platform_key}: {title} - Found in Custom HTTP")
                        break
                    else:
                        _emit_log(callbacks, f"[DB] {platform_key}: {title} - Not found in Custom HTTP")

        if img_bytes is None:
            _emit_log(callbacks, f"[FAIL] {platform_key}: {title} - No art found from any provider")
            (rev_dir / f"{slug}__no_art.json").write_text(
                json.dumps({
                    "title": title,
                    "platform": platform_key,
                    "provider_order": provider_order,
                    "error": "no art found from any provider"
                }, indent=2),
                encoding="utf-8"
            )
            return False

        try:
            src_img = Image.open(BytesIO(img_bytes))

            # Logo detection and cropping if enabled and source matches
            if ld_enabled and source_tag in ld_sources:
                src_img = detect_and_crop_logo(
                    src_img,
                    method=ld_method,
                    min_content_ratio=ld_min_content,
                    max_crop_ratio=ld_max_crop,
                    debug_log=lambda m: _emit_log(callbacks, m)
                )

            # Auto-centering if enabled and source is in configured sources
            centering = (0.5, 0.5)
            if ac_enabled and source_tag in ac_sources:
                centering, (mx, my, cnt) = _best_centering_for_img(
                    src_img, out_size,
                    steps=ac_steps, span=ac_span,
                    alpha_threshold=ac_alpha_threshold, margin_pct=ac_margin_pct
                )
                dx, dy = abs(mx - 0.5), abs(my - 0.5)
                if dx > ac_tolerance or dy > ac_tolerance:
                    (rev_dir / f"{slug}__offcenter.json").write_text(
                        json.dumps({
                            "title": title,
                            "platform": platform_key,
                            "source": source_tag,
                            "centering": [centering[0], centering[1]],
                            "content_centroid": [mx, my],
                            "deviation": [dx, dy],
                            "count": cnt
                        }, indent=2),
                        encoding="utf-8"
                    )
                    _emit_log(callbacks, f"[ALIGN] Off-center: {platform_key}: {title} centroid=({mx:.3f},{my:.3f})")

            out_img = compose_with_border(src_img, border_path, out_size, centering=centering)
            # Ensure game folder exists
            ensure_dir(out_path.parent)
            # Save as icon.png
            out_img.save(out_path, export_format, optimize=True)
            # Save duplicate as title.png
            title_path = out_path.parent / f"title.{export_format.lower()}"
            out_img.save(title_path, export_format, optimize=True)
            _emit_preview(callbacks, out_path)
            if source_tag:
                _emit_log(callbacks, f"[OK] {platform_key}: {title} ({source_tag}) -> {out_path.parent.name}/")
            else:
                _emit_log(callbacks, f"[OK] {platform_key}: {title} -> {out_path.parent.name}/")
            return True
        except Exception as e:
            _emit_log(callbacks, f"[ERROR] {platform_key}: {title} - Compose error: {e}")
            (rev_dir / f"{slug}__compose_error.json").write_text(
                json.dumps({"title": title, "platform": platform_key, "source": source_tag, "error": str(e)}, indent=2),
                encoding="utf-8"
            )
            return False

    max_workers = max(1, int(workers))

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(work_item, p, t, b, o, r) for (p, t, b, o, r) in tasks]

        for fut in as_completed(futures):
            if cancel.is_cancelled:
                _emit_log(callbacks, "[STOP] Cancelled by user. Cancelling remaining tasks...")
                # Cancel all pending futures
                for f in futures:
                    f.cancel()
                # Shutdown executor immediately
                ex.shutdown(wait=False, cancel_futures=True)
                break

            ok = False
            try:
                ok = fut.result(timeout=1.0)  # Add timeout to prevent hanging
            except Exception:
                ok = False

            if not ok:
                errors += 1

            with done_lock:
                done += 1
                _emit_progress(callbacks, done, total)

    if cancel.is_cancelled:
        return False, f"Cancelled. Completed {done}/{total} (errors={errors})."

    return True, f"Finished. Completed {done}/{total} (errors={errors})."
