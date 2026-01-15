import cv2
import numpy as np
import warnings
from scipy.signal import find_peaks, savgol_filter

from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from matplotlib.axes import Axes


# Image Loading from Disk, Defining its Properties and Processing Functions
def load_image_from_disk(image_path: str) -> np.ndarray:
    """Load an image from disk with cv2.imread(..., IMREAD_UNCHANGED)."""
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not load image at {image_path}.")
    return np.array(image)


def load_image(image_path, gray=False, autoCrop=True, debug=False) -> np.ndarray:
    image = load_image_from_disk(image_path)
    if autoCrop:
        image = auto_crop(image, debug=debug)
    if gray:
        return grayscale_image(image)
    else:
        return np.array(image)


def grayscale_image(image: np.ndarray) -> np.ndarray:
    # Expect image as H x W x C where C can be 3 (BGR) or 4 (BGRA)
    if image.ndim != 3 or image.shape[2] < 3:
        # Probably already grayscale 
        return image
    bgr = image[:, :, :3].astype(np.float32)
    gray = 0.299 * bgr[:, :, 2] + 0.587 * bgr[:, :, 1] + 0.114 * bgr[:, :, 0]

    # If an alpha channel exists, set grayscale to NaN where alpha == 0
    if image.shape[2] >= 4:
        alpha = image[:, :, 3].astype(np.float32)
        gray = gray.astype(np.float32)
        gray[alpha == 0] = np.nan

    return gray

# Compute mean grayscale differences along rows or columns
# Treat fully-transparent pixels (stored as NaN by `grayscale_image`) as missing data.
# This avoids large diff spikes at alpha boundaries.
def compute_mean_differences(gray, axis):
    g = np.asarray(gray, dtype=np.float32)
    if g.ndim != 2:
        raise ValueError("compute_mean_differences expects a 2D grayscale array")

    valid = np.isfinite(g)

    if axis == "rows":
        # Differences between row i and i+1, only where BOTH pixels are valid.
        d = np.abs(np.diff(g, axis=0))  # (H-1, W)
        valid_d = valid[:-1, :] & valid[1:, :]
        d = np.where(valid_d, d, np.nan)

        # Mean across columns, ignoring NaNs.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
            diffs = np.nanmean(d, axis=1)

    elif axis == "cols":
        # Differences between col j and j+1, only where BOTH pixels are valid.
        d = np.abs(np.diff(g, axis=1))  # (H, W-1)
        valid_d = valid[:, :-1] & valid[:, 1:]
        d = np.where(valid_d, d, np.nan)

        # Mean across rows, ignoring NaNs.
        # Dissable warning of empty slice
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
            diffs = np.nanmean(d, axis=0)

    else:
        raise ValueError("axis must be 'rows' or 'cols'")

    # If an entire line was invalid (all NaNs), nanmean returns NaN.
    # Treat those as 0 difference so they don't create artificial peaks.
    diffs = np.where(np.isfinite(diffs), diffs, 0.0)

    return diffs.astype(np.float32).tolist()


def find_local_maxima(arr, n=-1, min_dist=0):
    """Return the n largest local maxima of a 1D array.

    Returns (heights, indices) where:
      - heights: numpy array of peak values
      - indices: numpy array of peak indices

    If n == -1, returns all local maxima.

    Notes:
      - Endpoints are not considered peaks.
      - Flat/plateau peaks are supported (SciPy returns a representative peak
        position for plateaus).
      - When n != -1, peaks are selected by height (descending), then
        returned in the order they appear in the array.
      - min_dist is enforced via SciPy's `distance` parameter (in samples)
    """
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError("find_local_maxima expects a 1D array")
    if a.size < 3:
        return np.array([], dtype=a.dtype), np.array([], dtype=int)

    # Handle NaNs/infs gracefully by excluding them from peak detection.
    # We map non-finite values to -inf so they cannot become peaks.
    if not np.all(np.isfinite(a)):
        a = a.copy()
        a[~np.isfinite(a)] = -np.inf

    # SciPy peak finding (handles plateaus via plateau_size)
    # `distance` enforces a minimum spacing between detected peaks.
    # Treat non-positive/None as "no constraint".
    distance = None
    if min_dist is not None:
        md = int(min_dist)
        if md > 0:
            distance = md

    peaks, _props = find_peaks(a, plateau_size=True, distance=distance)
    if peaks.size == 0:
        print("Warning! No peaks found!")
        debug_plot(arr)
        return np.array([], dtype=a.dtype), np.array([], dtype=int)

    heights = a[peaks]

    if n is None or n == -1:
        # Keep original order
        return heights, peaks.astype(int)

    if n <= 0:
        return np.array([], dtype=a.dtype), np.array([], dtype=int)

    if peaks.size <= n:
        print("Warning! Fewer than expected maxima")
        debug_plot(arr)
        return heights, peaks.astype(int)

    # Select top-n by height (ties: earlier index wins deterministically)
    # Use lexsort to avoid ambiguity: primary key -height, secondary key index
    order = np.lexsort((peaks, -heights))

    
    top = order[:n]

    sel_peaks = peaks[top]
    sel_heights = heights[top]

    # Return selected peaks in left-to-right order
    in_order = np.argsort(sel_peaks)
    sel_peaks = sel_peaks[in_order]
    sel_heights = sel_heights[in_order]

    return sel_heights, sel_peaks.astype(int)


# --- New helper functions: gaussian_blur_ignore_nan and sample_bilinear_cv2 ---
def gaussian_blur_ignore_nan(
    gray: np.ndarray,
    *,
    radius: int = 3,
    sigma: float | None = None,
    eps: float = 1e-6,
) -> np.ndarray:
    """Gaussian blur that ignores NaNs (treats them as missing data).

    This implements normalized convolution:
      blur(gray * mask) / blur(mask)
    where mask = isfinite(gray).

    Pixels with effectively zero support remain NaN.
    """
    g = np.asarray(gray, dtype=np.float32)
    if g.ndim != 2:
        raise ValueError("gray must be a 2D array")

    if radius < 0:
        raise ValueError("radius must be >= 0")
    if radius == 0:
        return g

    if sigma is None:
        sigma = float(radius) / 2.0
    sigma = float(sigma)
    if sigma <= 0:
        raise ValueError("sigma must be > 0")

    k = int(2 * radius + 1)
    ksize = (k, k)

    valid = np.isfinite(g)
    v = np.where(valid, g, 0.0).astype(np.float32)
    m = valid.astype(np.float32)

    # Use a stable border type to avoid edge artifacts.
    blur_v = cv2.GaussianBlur(v, ksize, sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
    blur_m = cv2.GaussianBlur(m, ksize, sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)

    out = np.full_like(blur_v, np.nan, dtype=np.float32)
    good = blur_m > eps
    out[good] = blur_v[good] / blur_m[good]
    return out


def sample_bilinear_cv2(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Sample a 2D float image at floating-point coordinates using cv2.remap.

    x, y are 1D arrays in image coordinates (x=col, y=row).
    Returns a 1D float array.
    """
    im = np.asarray(img, dtype=np.float32)
    if im.ndim != 2:
        raise ValueError("img must be a 2D array")

    mapx = np.asarray(x, dtype=np.float32)[None, :]
    mapy = np.asarray(y, dtype=np.float32)[None, :]

    sampled = cv2.remap(
        im,
        mapx,
        mapy,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return sampled.reshape(-1)


def get_values_in_circle(
    gray: np.ndarray,
    point,
    radius: float,
    num_samples: int | None = None,
    width=3,
):
    if gray.ndim != 2:
        raise ValueError("gray must be a 2D array (grayscale).")

    cy, cx = float(point[0]), float(point[1])

    if num_samples is None:
        sample_rate = 2
        num_samples = max(8, sample_rate * int(np.ceil(2 * np.pi * max(radius, 1e-6))))

    # Validate width and build symmetric radial offsets.
    # Examples:
    #   width=1 -> [0]
    #   width=2 -> [-0.5, +0.5]
    #   width=3 -> [-1, 0, +1]
    #   width=4 -> [-1.5, -0.5, +0.5, +1.5]
    if width is None:
        width = 1
    if not isinstance(width, (int, np.integer)):
        # Allow things like 3.0 but reject 3.2
        if isinstance(width, float) and float(width).is_integer():
            width = int(width)
        else:
            raise TypeError("width must be an integer >= 1")
    width = int(width)
    if width < 1:
        raise ValueError("width must be >= 1")

    offsets = np.arange(width, dtype=np.float64) - (width - 1) / 2.0
    radii = float(radius) + offsets

    # Circle twice
    theta = np.linspace(0, 4 * np.pi, 2*num_samples, endpoint=False)

    ct = np.cos(theta)
    st = np.sin(theta)

    # Blur once (ignoring NaNs), then sample with bilinear interpolation.
    # This is much faster than per-sample Gaussian neighborhoods.
    blurred = gaussian_blur_ignore_nan(gray, radius=3)

    samples = []
    for r in radii:
        x = cx + r * ct
        y = cy - r * st  # Mathematical direction convention
        samples.append(sample_bilinear_cv2(blurred, x, y))

    vals = np.nanmean(np.stack(samples, axis=0), axis=0)
    
    # Now we choose where to cut the values
    # We don't want to cut off something interesting,
    # so we start in the first global minimum, and 
    # return 2pi from there
    # Find a robust start index (ignore NaNs if any).
    if np.all(~np.isfinite(vals)):
        raise ValueError("All sampled values are NaN; cannot choose a start index.")

    start = int(np.nanargmax(vals))
    # Return exactly one revolution (2π) starting at the first global minimum.
    # We sampled 0..4π with 2*num_samples points, so taking `num_samples` points
    # from `start` gives a contiguous 2π segment without needing wrap-around.
    end = start + num_samples
    vals = vals[start:end]
    theta = theta[start:end]

    return vals, theta


def smooth(y, window_length=11, polyorder=3):
    return savgol_filter(y, window_length=window_length, polyorder=polyorder)


def get_miniscus_width(theta1, theta2, radius):
    r = float(radius)
    if r < 0:
        raise ValueError("radius must be non-negative")

    t1 = np.asarray(theta1, dtype=np.float64)
    t2 = np.asarray(theta2, dtype=np.float64)

    # Points on the circle (center assumed at origin)
    x1 = r * np.cos(t1)
    y1 = r * np.sin(t1)
    x2 = r * np.cos(t2)
    y2 = r * np.sin(t2)

    # Euclidean distance between the two points
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# Horizontal Analysis Functions to identify the top and bottom boundaries of a channel.
def find_channel_edges(gray, show=False, debug=False):
    differences = compute_mean_differences(gray, "rows")
    mid = gray.shape[0] // 2

    # Find two candidate maxima for each half. Usually averaging is more stable,
    # but for bad images the two maxima can drift far apart. In that case,
    # fall back to using a single maximum ("second method").

    top_val, top_candidates = find_local_maxima(differences[:mid], 2)
    bottom_val, bottom_candidates = find_local_maxima(differences[mid:], 2)

    top_candidates = np.asarray(top_candidates, dtype=int)
    bottom_candidates = np.asarray(bottom_candidates, dtype=int)

    def pick_edge(candidates, values=None, offset=0, max_sep=10):

        candidates = np.asarray(candidates, dtype=int)
        if candidates.size == 0:
            return offset  # extremely defensive fallback
        if candidates.size == 1:
            return int(candidates[0]) + offset

        # Optional weights (peak heights). If provided, compute a weighted average.
        if values is not None:
            w = np.asarray(values, dtype=np.float64)
            # Guard against shape mismatch or non-finite weights.
            if w.shape != candidates.shape:
                w = None
            else:
                w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
                if np.sum(w) <= 0:
                    w = None
        else:
            w = None

        # Default: average ("first method")
        if w is None:
            avg_idx = int(np.round(np.mean(candidates))) + offset
        else:
            avg_idx = int(np.round(np.sum(candidates * w) / np.sum(w))) + offset

        # If the two maxima are far apart, use the stronger one ("second method")
        if abs(int(candidates[1]) - int(candidates[0])) > max_sep:
            if w is None:
                best = int(candidates[0])
            else:
                best = int(candidates[int(np.argmax(w))])
            return best + offset

        return avg_idx

    top_idx = pick_edge(top_candidates, top_val, offset=0, max_sep=10)
    bottom_idx = pick_edge(bottom_candidates, bottom_val, offset=mid, max_sep=10)

    if debug:
        debug_plot(differences, i_markers=[top_idx, bottom_idx])

    # Show the result
    if show:
        show_image_with_lines(
            gray,
            horizontal_lines=[int(top_idx), int(bottom_idx)],
            title="Image with Max Row Differences Marked",
            enable_plot=show,
        )

    return top_idx, bottom_idx


def auto_crop(image, extra=40, debug=False):
    gray = grayscale_image(image)
    top_idx, bottom_idx = find_channel_edges(gray, debug=debug)

    differences = compute_mean_differences(gray, "cols")
    val, index = find_local_maxima(differences, 1)
    middle_idx = int(np.mean(index)) + 1 

    # Show the result
    if debug:
        show_image_with_lines(
            gray,
            horizontal_lines=[top_idx, bottom_idx],
            vertical_lines=[middle_idx],
            title="Image with Max Row Differences Marked",
            enable_plot=True,
        )

    # Build a crop centered around the detected middle column, and vertically
    # centered between the detected top/bottom edges.
    H, W = gray.shape[:2]

    # Ensure indices are within image bounds.
    top_idx = int(np.clip(top_idx, 0, H - 1))
    bottom_idx = int(np.clip(bottom_idx, 0, H - 1))
    if bottom_idx < top_idx:
        top_idx, bottom_idx = bottom_idx, top_idx

    middle_idx = int(np.clip(middle_idx, 0, W - 1))

    # Target crop size: a square whose side is the detected height plus margin.
    # Clamp to both image dimensions.
    side = int(abs(bottom_idx - top_idx) + int(extra))
    side = int(np.clip(side, 1, min(H, W)))

    # Crop center (y, x).
    cy = 0.5 * (top_idx + bottom_idx)
    cx = float(middle_idx)

    half = side / 2.0

    # Proposed bounds.
    y0 = top_idx-extra
    y1 = bottom_idx+extra
    x0 = int(np.floor(cx - half))
    x1 = x0 + side

    # Shift the window back inside the image if it goes out of bounds.
    if y0 < 0:
        y0 = 0
    if y1 > H:
        y1 = H

    if x0 < 0:
        x0 = 0
    if x1 > W:
        x1 = W

    cropped_image = image[y0:y1, x0:x1]

    if debug:
        # Visualize the chosen crop rectangle.
        show_image_with_lines(
            gray,
            horizontal_lines=[ y0, y1 - 1],
            vertical_lines=[ x0, x1 - 1],
            title="Auto-crop: detected edges + crop bounds",
            enable_plot=True,
        )

    return cropped_image

def add_lines(
    ax,
    *,
    horizontal_lines=None,
    vertical_lines=None,
    horizontal_style=None,
    vertical_style=None,
):
    if horizontal_style is None:
        horizontal_style = {}
    if vertical_style is None:
        vertical_style = {}

    if horizontal_lines:
        for y in horizontal_lines:
            ax.axhline(y=y, **horizontal_style)

    if vertical_lines:
        for x in vertical_lines:
            ax.axvline(x=x, **vertical_style)


def add_radial_line(ax, p, theta, length, **kwargs):
    y, x = p
    t = float(theta)

    ct = np.cos(t)
    st = np.sin(t)
    x1 = x + ct * length
    y1 = y - st * length  # Mathematical direction convention

    # Provide a sensible default if the caller didn't specify a style.
    if "linewidth" not in kwargs and "lw" not in kwargs:
        kwargs["linewidth"] = 1.5

    (line,) = ax.plot([x, x1], [y, y1], **kwargs)
    return line


def add_marks(
    ax,
    points,
    *,
    radius=12,
    style=None,
):
    """Add circular marks at (y, x) points."""
    if style is None:
        style = {"edgecolor": "purple", "facecolor": "none", "linewidth": 2}

    if not points:
        return

    for y, x in points:
        ax.add_patch(Circle((x, y), radius=radius, **style))


# Display an image with optional line overlays using pyplot
def show_image_with_lines(
    image,
    horizontal_lines=None,
    vertical_lines=None,
    title=None,
    enable_plot=True,
):
    if not enable_plot:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(image, cmap="gray")

    add_lines(
        ax,
        horizontal_lines=horizontal_lines,
        vertical_lines=vertical_lines,
        horizontal_style={"color": "red", "linestyle": "--", "linewidth": 1},
        vertical_style={"color": "red", "linestyle": "-", "linewidth": 2},
    )

    if title:
        ax.set_title(title)

    ax.axis("off")
    fig.tight_layout()
    plt.show()
    plt.close(fig)


# Display an image with optional line overlays and circular marks
def show_image_with_marks(
    image,
    *,
    contact_points,
    horizontal_lines=None,
    vertical_lines=None,
    title=None,
    enable_plot=True,
    mark_radius=12,
    show=True,
) -> Axes:
    fig, ax = plt.subplots(figsize=(12, 6))
    if not enable_plot:
        return ax

    ax.imshow(image, cmap="gray")

    add_lines(
        ax,
        horizontal_lines=horizontal_lines,
        vertical_lines=vertical_lines,
        horizontal_style={"color": "red", "linestyle": "--", "linewidth": 1},
        vertical_style={"color": "red", "linestyle": "-", "linewidth": 2},
    )

    # Mark contact points (y, x)
    add_marks(ax, contact_points, radius=mark_radius)

    if title:
        ax.set_title(title)

    ax.axis("off")
    if show:
        fig.tight_layout()
        plt.show()
    return ax


# Visualization Functions to plot something with given labels and settings
def debug_plot(
    y: np.ndarray,
    x: np.ndarray = np.array([]),
    title=None,
    xlabel=None,
    ylabel=None,
    i_markers: np.ndarray = np.array([]),
    enable_plot=True,
    verticalLines=None,
    horizontalLines=None,
    show=True,
):
    if not enable_plot:
        return

    if len(x) == 0:
        x = np.arange(len(y))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, y)

    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    add_lines(
        ax,
        horizontal_lines=horizontalLines,
        vertical_lines=verticalLines,
        horizontal_style={"color": "red", "linestyle": "--", "linewidth": 1.5},
        vertical_style={"color": "green", "linestyle": "--", "linewidth": 1.5},
    )
    if len(i_markers) != 0:
        for i in i_markers:
            ax.scatter(x[i], y[i])

    ax.grid(visible=True)
    if show:
        plt.show()
    return ax
