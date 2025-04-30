import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg


# Image Loading and Processing Functions
def load_image(image_path):
    """Load an image and detect if it has an alpha channel."""
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not load image at {image_path}.")
    has_alpha = image.shape[2] == 4
    return image, has_alpha, image.shape[0], image.shape[1]


def pixels_to_grayscale(px, has_alpha):
    """Convert a BGR(A) pixel to grayscale intensity, ignoring transparent or black pixels."""
    if has_alpha and px[3] == 0:
        return None
    if np.all(px[:3] == 0):
        return None
    return 0.299 * px[2] + 0.587 * px[1] + 0.114 * px[0]


def compute_mean_differences(image, axis, has_alpha):
    """Compute mean grayscale differences along rows or columns."""
    diffs = []

    if axis == "rows":
        for i in range(1, image.shape[0]):
            values = []
            for px1, px2 in zip(image[i - 1], image[i]):
                g1 = pixels_to_grayscale(px1, has_alpha)
                g2 = pixels_to_grayscale(px2, has_alpha)
                if g1 is not None and g2 is not None:
                    values.append(abs(g2 - g1))
            diffs.append(np.mean(values) if values else 0)
    elif axis == "cols":
        for i in range(1, image.shape[1]):
            values = []
            for px1, px2 in zip(image[:, i - 1], image[:, i]):
                g1 = pixels_to_grayscale(px1, has_alpha)
                g2 = pixels_to_grayscale(px2, has_alpha)
                if g1 is not None and g2 is not None:
                    values.append(abs(g2 - g1))
            diffs.append(np.mean(values) if values else 0)
    else:
        print("Use rows or cols!")
    return diffs


def second_largest_arg(arr):
    """Find the index of the second largest value in an array."""
    arr = np.asarray(arr)
    if arr.size < 2:
        raise ValueError("Array must contain at least two elements")
    max_val = np.max(arr)
    arr = arr[arr != max_val]
    return np.argmax(arr)


# Visualization Functions
def debug_plot(
    y,
    x=None,
    title=None,
    xlabel=None,
    ylabel=None,
    enable_plot=True,
    verticalLines=None,
    horizontalLines=None,
):
    """Plot something with given labels and settings."""
    if not enable_plot:
        return
    if x is None:
        x = range(len(y))
    plt.figure(figsize=(10, 4))
    plt.plot(x, y)
    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if verticalLines:
        for x_val in verticalLines:
            plt.axvline(x=x_val, color="green", linestyle="--", linewidth=1.5)
    if horizontalLines:
        for y_val in horizontalLines:
            plt.axhline(y=y_val, color="red", linestyle="--", linewidth=1.5)
    plt.grid(visible=True)
    plt.show()
    plt.close()


def show_image_with_lines(image, lines, title=None, enable_plot=True):
    """Display an image with specified lines."""
    if not enable_plot:
        return

    plt.figure(figsize=(12, 6))
    plt.imshow(image)

    if title:
        plt.title(title)

    plt.axis("off")
    plt.tight_layout()
    plt.show()


# Horizontal Analysis Functions
def find_horizontal_boundaries(image_path, enable_plot=True):
    """Identify the top and bottom boundaries of a channel."""
    # Load image and compute row-wise differences
    image, has_alpha, height, width = load_image(image_path)
    differences = compute_mean_differences(image, "rows", has_alpha)

    # Plot differences if enabled
    debug_plot(
        differences,
        range(len(differences)),
        title="Row Differences",
        xlabel="Row Index",
        ylabel="Mean Gray Diff",
        enable_plot=enable_plot,
    )

    # Find max difference in top half and bottom half
    mid = height // 2
    top_index = (
        second_largest_arg(differences[:mid]) + 1
    )  # +1 to correct for starting at row 1
    bottom_index = second_largest_arg(differences[mid:]) + mid + 1

    # Draw red lines on a copy of the image (convert to BGR if it has alpha)
    image_display = image[:, :, :3].copy() if has_alpha else image.copy()
    cv2.line(image_display, (0, top_index), (width, top_index), (0, 0, 255), 2)
    cv2.line(image_display, (0, bottom_index), (width, bottom_index), (0, 0, 255), 2)

    # Show the result
    if enable_plot:
        image_rgb = cv2.cvtColor(image_display, cv2.COLOR_BGR2RGB)
        show_image_with_lines(
            image_rgb,
            [],
            title="Image with Max Row Differences Marked",
            enable_plot=enable_plot,
        )

    return top_index, bottom_index


# Vertical Analysis Functions
def smooth_data(data, window_size):
    """Apply convolution smoothing to data."""
    window = np.ones(window_size) / window_size
    return np.convolve(data, window, mode="same")


def find_vertical_boundaries(differences, start_idx, end_idx):
    """Find vertical boundaries based on local maxima in differences."""
    subregion_differences = np.array(differences[start_idx : end_idx + 1])

    # Find local maxima
    local_maxima_index = np.array(
        [
            i
            for i in range(1, len(subregion_differences) - 1)
            if subregion_differences[i] > subregion_differences[i - 1]
            and subregion_differences[i] > subregion_differences[i + 1]
        ]
    )
    local_maxima = subregion_differences[local_maxima_index]

    # We want to find thresholds for the local maxima in each region, and then
    # use these two thresholds to select the first and last local maximum in the
    # subregion_differences (with the interface)

    margin = 2.5
    threshold1 = np.mean(differences[:start_idx]) * margin
    threshold2 = np.mean(differences[end_idx:]) * margin

    first_local_max_index = local_maxima_index[local_maxima > threshold1][0]
    last_local_max_index = local_maxima_index[local_maxima > threshold2][-1]
    first_last_local_max_index = [first_local_max_index, last_local_max_index]
    debug_plot(
        local_maxima,
        local_maxima_index,
        horizontalLines=[threshold1, threshold2],
        verticalLines=first_last_local_max_index,
    )
    # Map back to full-image columns
    return [start_idx + i for i in first_last_local_max_index]


def analyze_vertical_differences(image_path, top_index, bottom_index, enable_plot=True):
    """Analyze vertical differences to find meniscus boundaries."""
    # Load image and compute column-wise differences
    image, has_alpha, height, width = load_image(image_path)
    # Crop image to only include channel
    CropAmount = 0.04
    image = image[
        top_index:bottom_index, int(CropAmount * width) : int((1 - CropAmount) * width)
    ]
    newWidth = int((1 - 2 * CropAmount) * width)
    plt.imshow(image)
    plt.show()

    differences = compute_mean_differences(image, "cols", has_alpha)

    # Plot raw column differences if enabled
    debug_plot(
        differences,
        title="Raw Column Differences",
        xlabel="Column Index",
        ylabel="Mean Gray Diff",
        enable_plot=enable_plot,
    )

    # Apply smoothing
    window_size = int(0.01 * width)  # 1% of image width
    smoothed_differences = smooth_data(differences, window_size)

    # Plot smoothed differences if enabled
    debug_plot(
        smoothed_differences,
        title="Smoothed Column Differences",
        xlabel="Column Index",
        ylabel="Mean Gray Diff",
        enable_plot=enable_plot,
    )

    # Calculate derivative and find extrema
    derivative = np.gradient(smoothed_differences)
    max_idx = np.argmax(derivative)
    min_idx = np.argmin(derivative)
    extraBuffer = 1.5
    interfaceWidth = abs(max_idx - min_idx) * extraBuffer
    # Make some extra room
    start_idx = int(min(max_idx, min_idx) - interfaceWidth / 2)
    end_idx = int(max(max_idx, min_idx) + interfaceWidth / 2)

    # Find vertical boundaries
    vertical_boundaries = find_vertical_boundaries(differences, start_idx, end_idx)

    debug_plot(derivative, range(len(derivative)), verticalLines=[start_idx, end_idx])

    return int(CropAmount * width) + np.array(vertical_boundaries)


# Geometry Calculations
def circle_from_3pts(a, b, c):
    """Compute circle parameters (center, radius) from three points."""
    (x1, y1), (x2, y2), (x3, y3) = a, b, c
    A = np.array([[x2 - x1, y2 - y1], [x3 - x1, y3 - y1]])
    B = np.array(
        [
            ((x2**2 - x1**2) + (y2**2 - y1**2)) / 2,
            ((x3**2 - x1**2) + (y3**2 - y1**2)) / 2,
        ]
    )
    cx, cy = np.linalg.solve(A, B)
    r = np.hypot(x1 - cx, y1 - cy)
    return cx, cy, r


def compute_contact_angle(p1, cx, cy):
    """Compute contact angle between meniscus and wall."""
    # Radius vector from circle center to the contact point p1
    rad_vec = np.array([p1[0] - cx, p1[1] - cy])

    # Tangent vector (90° rotation of radius)
    tangent = np.array([rad_vec[1], -rad_vec[0]])

    # Compute angle with vertical direction
    vertical = np.array([0, 1])
    cos_theta = np.dot(tangent, vertical) / (
        np.linalg.norm(tangent) * np.linalg.norm(vertical)
    )

    # Handle numerical errors
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def generate_arc_points(cx, cy, r, p1, p2):
    """Generate points along arc from p1 to p2."""
    theta1 = np.arctan2(p1[1] - cy, p1[0] - cx)
    theta2 = np.arctan2(p2[1] - cy, p2[0] - cx)

    # Ensure we go the shorter way around
    if theta2 < theta1:
        theta2 += 2 * np.pi

    thetas = np.linspace(theta1, theta2, 200)
    xs = cx + r * np.cos(thetas)
    ys = cy + r * np.sin(thetas)

    return xs, ys


# Visualization Functions
def plot_meniscus_with_annotations(
    photo,
    points,
    circle_params,
    boundaries,
    vertical_boundaries,
    angle_data,
    enable_plot=True,
):
    """Plot photo with meniscus annotations including contact angle and radius."""
    if not enable_plot:
        return

    p1, p2, p3 = points
    cx, cy, r = circle_params
    top_index, bottom_index = boundaries
    vertical_indices, left_index, right_index = vertical_boundaries
    contact_angle, beta = angle_data

    # Generate arc points
    xs, ys = generate_arc_points(cx, cy, r, p1, p2)

    # Draw everything
    plt.figure(figsize=(12, 6))
    plt.imshow(photo)

    # Green contact lines
    for idx in vertical_indices:
        plt.axvline(x=idx + 1, color="green", linestyle="-", linewidth=2)

    # Red channel-wall lines
    plt.hlines(
        [top_index, bottom_index],
        xmin=0,
        xmax=photo.shape[1],
        colors="red",
        linewidth=2,
    )

    # Blue meniscus arc
    plt.plot(xs, ys, "-", color="blue", linewidth=3)

    # Convert radius to micrometers (assuming 100 μm channel height)
    r_microns = r * (100 / abs(bottom_index - top_index))

    # Add annotations
    plt.text(
        photo.shape[1] * 0.05,
        photo.shape[0] * 0.05,
        r"$\mathrm{{Radius:\ {}}}\,\mu\mathrm{{m}}$".format(f"{r_microns:.1f}"),
        color="blue",
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.7),
    )

    plt.text(
        p1[0] + 10,
        bottom_index - 20,
        r"$\alpha = {}^\circ$".format(f"{contact_angle:.1f}"),
        color="purple",
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.7),
    )

    plt.text(
        p2[0] + 10,
        bottom_index + 20,
        r"$\beta = {}^\circ$".format(f"{beta:.1f}"),
        color="purple",
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.7),
    )

    # Draw radius
    plt.plot([cx, p1[0]], [cy, p1[1]], linestyle="--", color="blue", linewidth=2)

    plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_triangle_diagram(photo, points, circle_center, enable_plot=True):
    """Plot triangle diagram showing key geometric points."""
    if not enable_plot:
        return

    p1, p2, p3 = points
    cx, cy = circle_center

    plt.figure(figsize=(12, 6))
    plt.imshow(photo)

    # Plot points
    plt.plot(p1[0], p1[1], "o", color="black")  # Point A
    plt.plot(p1[0], p3[1], "o", color="black")  # Point B
    plt.plot(cx, cy, "o", color="black")  # Point O

    # Label points
    plt.text(p1[0] + 5, p1[1] - 10, "A", color="black", fontsize=12, weight="bold")
    plt.text(p1[0] + 5, p3[1] - 10, "B", color="black", fontsize=12, weight="bold")
    plt.text(cx + 5, cy - 10, "O", color="black", fontsize=12, weight="bold")

    # Draw triangle
    plt.plot(
        [p1[0], p1[0], cx, p1[0]],
        [p1[1], p3[1], cy, p1[1]],
        linestyle="-",
        color="black",
        linewidth=2,
    )

    plt.title("Annotated Photo with Points A, B, O and Triangle")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# Main analysis function
def analyze_meniscus(image_path, enable_plot=True):
    """Perform complete meniscus analysis on an image."""
    # Find horizontal boundaries
    top_index, bottom_index = find_horizontal_boundaries(image_path, False)

    # Find vertical boundaries
    vertical_boundaries = analyze_vertical_differences(
        image_path, top_index, bottom_index, enable_plot
    )

    # Compute key points
    left_index = min(vertical_boundaries)
    right_index = max(vertical_boundaries)

    p1 = [left_index, top_index]
    p2 = [left_index, bottom_index]
    middle_of_channel = top_index + (bottom_index - top_index) // 2
    p3 = [right_index, middle_of_channel]

    # Compute circle parameters
    cx, cy, r = circle_from_3pts(p1, p2, p3)

    # Compute contact angle
    contact_angle = compute_contact_angle(p1, cx, cy)
    beta = 180 - contact_angle

    print(f"Contact angle at wall: {contact_angle:.1f}°")

    # Load image for visualization
    photo = mpimg.imread(image_path)
    # Plot results
    plot_meniscus_with_annotations(
        photo=photo,
        points=[p1, p2, p3],
        circle_params=[cx, cy, r],
        boundaries=[top_index, bottom_index],
        vertical_boundaries=[vertical_boundaries, left_index, right_index],
        angle_data=[contact_angle, beta],
        enable_plot=enable_plot,
    )

    plot_triangle_diagram(photo, [p1, p2, p3], [cx, cy], enable_plot)

    return {
        "contact_angle": contact_angle,
        "beta": beta,
        "radius": r,
        "top_index": top_index,
        "bottom_index": bottom_index,
        "left_index": left_index,
        "right_index": right_index,
    }


# Execute analysis if run directly
if __name__ == "__main__":
    image_path = "test_angle_3.png"
    results = analyze_meniscus(image_path, enable_plot=True)
