"""Reference implementation of the phone-photo cleanup applied before inference.

The Android port lives in
`android_app/app/src/main/java/com/kimhs/liverdx/ml/PhotoPreprocess.kt` and is unit-tested against a
fixture generated from this file, so both must stay in step.

Why the stages exist
--------------------
The model was trained on SMC-LUD B-mode frames, which are effectively grayscale (measured mean
channel spread over `example_images_hcc_case/`: 0.0-8.5 of 255). A phone photo of a screen or a print
is not: it carries blown-out specular reflections, an uneven backlight/vignette, a strong colour
cast, sensor noise and JPEG ringing. Each stage targets exactly one of those and each is switchable
from the app so the user can see what it does.

Fixed execution order (and why):

  1. remove_glare          first: illumination flattening rescales blown pixels back into range and
                           would hide the very pixels this stage has to find
  2. flatten_illumination  removes the large-scale brightness ramp (uses colour, so before subdivision)
  3. desaturate            collapse to luma: matches the grayscale training domain
  4. denoise               edge-preserving 5x5 bilateral (sensor noise + JPEG ringing)
  5. stretch_contrast      1-99 percentile stretch: undoes exposure differences
"""

from __future__ import annotations

import numpy as np

LUMA_WEIGHTS = (0.299, 0.587, 0.114)
FLATTEN_GAIN_MIN = 0.5
FLATTEN_GAIN_MAX = 2.0

GLARE_ABSOLUTE = 238.0
GLARE_EXCESS = 45.0
GLARE_FLOOR = 215.0
# A 1-pixel-thin bright structure is anatomy (vessel wall, diaphragm), not glare: erode then dilate.
GLARE_OPEN_RADIUS = 1
GLARE_HALO_RADIUS = 2
GLARE_FRACTION_WARN = 0.15

INPAINT_FILL_RADIUS_DIVISOR = 20
INPAINT_ITERATIONS = 48

BILATERAL_RADIUS = 2
BILATERAL_SIGMA_RANGE = 18.0

CONTRAST_LOW_PERCENTILE = 0.01
CONTRAST_HIGH_PERCENTILE = 0.99
CONTRAST_MIN_SPAN = 8.0
HISTOGRAM_BINS = 256

STAGE_GLARE = "remove_glare"
STAGE_FLATTEN = "flatten_illumination"
STAGE_DESATURATE = "desaturate"
STAGE_DENOISE = "denoise"
STAGE_CONTRAST = "stretch_contrast"
STAGE_ORDER = (STAGE_GLARE, STAGE_FLATTEN, STAGE_DESATURATE, STAGE_DENOISE, STAGE_CONTRAST)


def round_half_up(value: float) -> int:
    """Half-up rounding by truncation, so numpy and Kotlin agree on exact .5 ties."""
    return int(value + 0.5)


# Spatial weights of the 5x5 bilateral kernel, precomputed so both implementations share the table.
_SPATIAL_SIGMA = 1.6
_OFFSETS = tuple(
    (dy, dx)
    for dy in range(-BILATERAL_RADIUS, BILATERAL_RADIUS + 1)
    for dx in range(-BILATERAL_RADIUS, BILATERAL_RADIUS + 1)
)
SPATIAL_WEIGHTS = {
    (dy, dx): float(np.exp(-(dy * dy + dx * dx) / (2.0 * _SPATIAL_SIGMA ** 2)))
    for dy, dx in _OFFSETS
}


def luma(rgb: np.ndarray) -> np.ndarray:
    """Rec.601 luma of a float32 RGB array with shape (..., 3), kept in float32 like the port."""
    return (
        rgb[..., 0] * LUMA_WEIGHTS[0]
        + rgb[..., 1] * LUMA_WEIGHTS[1]
        + rgb[..., 2] * LUMA_WEIGHTS[2]
    )


def box_blur(field: np.ndarray, radius: int) -> np.ndarray:
    """Mean over a (2r+1)^2 window, computed through an integral image in float64."""
    height, width = field.shape
    padded = np.zeros((height + 1, width + 1), dtype=np.float64)
    padded[1:, 1:] = np.cumsum(np.cumsum(field.astype(np.float64), axis=0), axis=1)

    rows = np.arange(height)
    cols = np.arange(width)
    top = np.clip(rows - radius, 0, height)[:, None]
    bottom = np.clip(rows + radius + 1, 0, height)[:, None]
    left = np.clip(cols - radius, 0, width)[None, :]
    right = np.clip(cols + radius + 1, 0, width)[None, :]

    total = padded[bottom, right] - padded[top, right] - padded[bottom, left] + padded[top, left]
    count = (bottom - top) * (right - left)
    return total / count


def _morph(mask: np.ndarray, radius: int, mode: str) -> np.ndarray:
    """Square-window erode/dilate over a boolean mask (border pixels see clamped neighbours)."""
    height, width = mask.shape
    out = np.full_like(mask, mode == "erode")
    for y in range(height):
        for x in range(width):
            result = mode == "erode"
            for dy in range(-radius, radius + 1):
                ny = min(max(y + dy, 0), height - 1)
                for dx in range(-radius, radius + 1):
                    nx = min(max(x + dx, 0), width - 1)
                    if mode == "erode":
                        result = result and bool(mask[ny, nx])
                        if not result:
                            break
                    else:
                        result = result or bool(mask[ny, nx])
                        if result:
                            break
                if (mode == "erode" and not result) or (mode == "dilate" and result):
                    break
            out[y, x] = result
    return out


def glare_mask(rgb: np.ndarray) -> np.ndarray:
    height, width = rgb.shape[:2]
    radius = max(3, round_half_up(min(height, width) / 20))
    luminance = luma(rgb)
    background = box_blur(luminance, radius)

    raw = (luminance >= GLARE_ABSOLUTE) | (
        ((luminance - background) >= GLARE_EXCESS) & (luminance >= GLARE_FLOOR)
    )
    opened = _morph(raw, GLARE_OPEN_RADIUS, "erode")
    return _morph(opened, GLARE_HALO_RADIUS, "dilate")


def inpaint(rgb: np.ndarray, mask: np.ndarray, iterations: int = INPAINT_ITERATIONS) -> np.ndarray:
    """Fill every masked pixel from the local mean of the unmasked ones, then diffuse the seams.

    Diffusion alone needs O(radius^2) sweeps to reach the middle of a blob, so the interior is
    seeded first with a normalised box mean of the surviving pixels and only then smoothed.
    """
    height, width = mask.shape
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return rgb

    radius = max(2, round_half_up(min(height, width) / INPAINT_FILL_RADIUS_DIVISOR))
    valid = (~mask).astype(np.float64)
    neighbourhood = np.maximum(box_blur(valid, radius), 1e-6)

    values = rgb.astype(np.float32).copy()
    for channel in range(3):
        contribution = values[..., channel].astype(np.float64) * valid
        fill = box_blur(contribution, radius) / neighbourhood
        values[..., channel][rows, cols] = fill[rows, cols].astype(np.float32)

    neighbour_offsets = ((-1, 0), (1, 0), (0, -1), (0, 1))
    for _ in range(iterations):
        total = np.zeros((rows.size, 3), dtype=np.float64)
        for dy, dx in neighbour_offsets:
            neighbour_rows = np.clip(rows + dy, 0, height - 1)
            neighbour_cols = np.clip(cols + dx, 0, width - 1)
            total += values[neighbour_rows, neighbour_cols]
        values[rows, cols] = (total / len(neighbour_offsets)).astype(np.float32)
    return values


def flatten_illumination(rgb: np.ndarray) -> np.ndarray:
    height, width = rgb.shape[:2]
    radius = max(4, round_half_up(min(height, width) / 12))
    luminance = luma(rgb)
    background = box_blur(luminance, radius)
    gain = np.clip(np.mean(luminance) / np.maximum(background, 1e-3), FLATTEN_GAIN_MIN, FLATTEN_GAIN_MAX)
    return np.clip(rgb * gain[..., None], 0.0, 255.0)


def desaturate(rgb: np.ndarray) -> np.ndarray:
    return np.repeat(luma(rgb)[..., None], 3, axis=2)


def denoise(rgb: np.ndarray) -> np.ndarray:
    """5x5 bilateral filter; after desaturation the three channels are identical."""
    height, width = rgb.shape[:2]
    source = rgb.astype(np.float64)
    output = np.zeros_like(source)
    range_denominator = 2.0 * BILATERAL_SIGMA_RANGE ** 2

    for y in range(height):
        for x in range(width):
            centre = source[y, x]
            total = np.zeros(3, dtype=np.float64)
            weight_sum = 0.0
            for dy, dx in _OFFSETS:
                ny, nx = y + dy, x + dx
                if not (0 <= ny < height and 0 <= nx < width):
                    continue
                difference = source[ny, nx] - centre
                weight = SPATIAL_WEIGHTS[(dy, dx)] * np.exp(
                    -float(np.dot(difference, difference)) / range_denominator
                )
                total += source[ny, nx] * weight
                weight_sum += weight
            output[y, x] = total / weight_sum if weight_sum > 0.0 else centre
    return np.clip(output, 0.0, 255.0).astype(rgb.dtype)


def stretch_contrast(rgb: np.ndarray) -> np.ndarray:
    """1st-99th percentile stretch, 256 unit bins via truncation so both ports agree exactly."""
    bins = np.clip(np.floor(luma(rgb)), 0.0, HISTOGRAM_BINS - 1).astype(np.int64)
    histogram = np.bincount(bins.ravel(), minlength=HISTOGRAM_BINS)
    total = int(histogram.sum())
    if total <= 0:
        return rgb

    low = _first_bin_at(histogram, CONTRAST_LOW_PERCENTILE * total, default=0)
    high = _first_bin_at(histogram, CONTRAST_HIGH_PERCENTILE * total, default=HISTOGRAM_BINS - 1)
    span = float(high - low)
    if span < CONTRAST_MIN_SPAN:
        return rgb
    return np.clip((rgb - float(low)) * (255.0 / span), 0.0, 255.0)


def _first_bin_at(histogram: np.ndarray, target: float, default: int) -> int:
    cumulative = 0
    for index in range(HISTOGRAM_BINS):
        cumulative += int(histogram[index])
        if cumulative >= target:
            return index
    return default


def run(
    rgb: np.ndarray,
    remove_glare: bool = True,
    flatten: bool = True,
    desaturate_channels: bool = True,
    denoise_noise: bool = True,
    contrast: bool = True,
) -> tuple[np.ndarray, float]:
    """Applies the enabled stages in STAGE_ORDER. Returns (image, glare fraction)."""
    stages = {
        STAGE_GLARE: remove_glare,
        STAGE_FLATTEN: flatten,
        STAGE_DESATURATE: desaturate_channels,
        STAGE_DENOISE: denoise_noise,
        STAGE_CONTRAST: contrast,
    }
    result = rgb.astype(np.float32)
    glare_fraction = 0.0

    for stage in STAGE_ORDER:
        if not stages[stage]:
            continue
        if stage == STAGE_GLARE:
            mask = glare_mask(result)
            glare_fraction = float(mask.mean())
            result = inpaint(result, mask)
        elif stage == STAGE_FLATTEN:
            result = flatten_illumination(result)
        elif stage == STAGE_DESATURATE:
            result = desaturate(result)
        elif stage == STAGE_DENOISE:
            result = denoise(result)
        elif stage == STAGE_CONTRAST:
            result = stretch_contrast(result)

    return np.clip(result, 0.0, 255.0).astype(np.float32), glare_fraction


def bilinear_uint8(pixels: np.ndarray, dst_height: int, dst_width: int) -> np.ndarray:
    """`tf.image.resize(..., method='bilinear')` then `tf.cast(..., tf.uint8)`.

    That is what `dataloader.py::_decode_image` does, so it is the preprocessing the reported
    validation/test metrics were produced with. Half-pixel centres, no antialias, then truncation.
    """
    src_height, src_width = pixels.shape[:2]
    scale_y = src_height / dst_height
    scale_x = src_width / dst_width
    source = pixels.astype(np.float64)

    out = np.zeros((dst_height, dst_width, 3), dtype=np.float64)
    for dy in range(dst_height):
        source_y = np.clip((dy + 0.5) * scale_y - 0.5, 0.0, src_height - 1.0)
        y0 = int(np.floor(source_y))
        y1 = min(y0 + 1, src_height - 1)
        wy = source_y - y0
        for dx in range(dst_width):
            source_x = np.clip((dx + 0.5) * scale_x - 0.5, 0.0, src_width - 1.0)
            x0 = int(np.floor(source_x))
            x1 = min(x0 + 1, src_width - 1)
            wx = source_x - x0
            top = source[y0, x0] * (1.0 - wx) + source[y0, x1] * wx
            bottom = source[y1, x0] * (1.0 - wx) + source[y1, x1] * wx
            out[dy, dx] = top * (1.0 - wy) + bottom * wy

    return np.clip(np.trunc(out), 0.0, 255.0)
