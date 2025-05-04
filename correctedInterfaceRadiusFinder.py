#Written by Maja Pakula

import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg #Import Matplotlib's image module for RGB image loading and visualization capabilities


"""
    Load an image from disk and determine its properties.
    
    This function uses OpenCV to read an image file with all channels intact,
    including any alpha (transparency) channel if present. It performs basic
    validation to ensure the image was successfully loaded and extracts key
    image properties for easier downstream processing.
    
    Parameters:
    - image_path (str): Path to the image file to be loaded
    
    Returns:
    - image (ndarray): The loaded image as a NumPy array in BGR(A) format
    - has_alpha (bool): True if the image contains an alpha channel (4 channels total)
    - height (int): Height of the image in pixels
    - width (int): Width of the image in pixels
    
    Raises:
    - ValueError: If the image cannot be loaded from the specified path
    
    Note: Uses cv2.IMREAD_UNCHANGED flag to preserve all channels including alpha.
    The color order is BGR (or BGRA with alpha), not RGB, following OpenCV convention.
    """
# Image Loading from Disk, Defining its Properties and Processing Functions
def load_image(image_path):
    """Load an image and detect if it has an alpha channel."""
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not load image at {image_path}.")
    has_alpha = image.shape[2] == 4
    return image, has_alpha, image.shape[0], image.shape[1]


"""
    Convert a BGR(A) pixel to grayscale intensity, ignoring transparent or black pixels using standard luminance formula.
    
    This function processes individual pixels from an image and converts color values
    to a single grayscale intensity value. It handles special cases for transparency
    and entirely black pixels.
    
    Parameters:
    - px: numpy array containing a single pixel's BGR or BGRA values
    - has_alpha: boolean indicating whether the pixel has an alpha channel (BGRA format)
    
    Returns:
    - float: grayscale intensity value calculated using the standard ITU-R BT.601 formula
             (0.299R + 0.587G + 0.114B)
    - None: if the pixel is fully transparent (alpha=0) or completely black
    
    Note: Input pixel is expected in BGR(A) format (not RGB), as commonly used in OpenCV.
    """
# Convert a BGR(A) pixel to grayscale intensity, ignoring transparent or black pixels
def pixels_to_grayscale(px, has_alpha):
    if has_alpha and px[3] == 0:
        return None
    if np.all(px[:3] == 0):
        return None
    return 0.299 * px[2] + 0.587 * px[1] + 0.114 * px[0]


"""
   Calculate average grayscale intensity differences between adjacent pixels in an image.
   
   This function analyzes an image to detect intensity changes between neighboring pixels,
   either horizontally (between rows) or vertically (between columns). For each pair of
   adjacent pixels, it computes the absolute difference in grayscale values, ignoring
   transparent or black pixels. These differences are then averaged to quantify edge
   strength at each position, which can be useful for detecting content boundaries,
   text lines, or image segments.
   
   Parameters:
   - image: 2D array of pixels in BGR(A) format
   - axis: string ("rows" or "cols") specifying direction of analysis
   - has_alpha: boolean indicating whether the image has an alpha channel
   
   Returns:
   - diffs: list of mean intensity differences between adjacent rows or columns
   
   Note: Returns values of 0 for areas where no valid pixel comparisons could be made.
    """
#Compute mean grayscale differences along rows or columns
def compute_mean_differences(image, axis, has_alpha):
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


"""
   Find the index of the second largest value in the original array.
   
   This function identifies the second highest value in an array by first 
   masking out all occurrences of the maximum value, then finding the
   maximum of the remaining elements. It returns the index position of this
   second-highest value in the original array structure.
   
   Parameters:
   - arr: Array-like object containing numeric values
   
   Returns:
   - int: Index of the second largest value in the original array
   
   Raises:
   - ValueError: If the input array contains fewer than 2 elements
   
   Note: If multiple elements tie for second place, returns the index of the first occurrence.
    """
#Find the index of the second largest value in an array
def second_largest_arg(arr):
    arr = np.asarray(arr)
    if arr.size < 2:
        raise ValueError("Array must contain at least two elements")
    max_val = np.max(arr)
    arr = arr[arr != max_val]
    return np.argmax(arr)


"""
   Create a customizable line plot for data visualization and debugging.
   
   This utility function simplifies the creation of Matplotlib plots with 
   common visualization elements. It generates a line plot of data values with
   optional features like axis labels, title, and reference lines. The function
   is designed for quick debugging and analysis of numerical data, particularly
   useful for visualizing image processing metrics, signal data, or detection results.
   
   Parameters:
   - y: array-like, data values to plot on the y-axis
   - x: array-like, optional x-axis values (defaults to indices of y)
   - title: string, optional plot title
   - xlabel: string, optional x-axis label
   - ylabel: string, optional y-axis label
   - enable_plot: boolean, whether to display the plot (allows conditional plotting)
   - verticalLines: list, x-coordinates where to draw vertical reference lines
   - horizontalLines: list, y-coordinates where to draw horizontal reference lines
   
   Returns:
   - None: The function displays the plot but doesn't return any values
   
   Note: Vertical reference lines are green dashed lines, while horizontal 
   reference lines are red dashed lines.
    """
# Visualization Functions to plot something with given labels and settings
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


"""
   Display an image with overlay visualization of detected lines.
   
   This function creates a visualization of an image with the option to
   highlight specific line positions, which is useful for debugging and
   validating line detection algorithms. The function displays the original
   image and can be configured with a descriptive title.
   
   Parameters:
   - image: numpy array, the image to display
   - lines: list, line positions to visualize (currently unused in implementation)
   - title: string, optional title for the plot
   - enable_plot: boolean, whether to display the plot (for conditional visualization)
   
   Returns:
   - None: The function displays the plot but doesn't return any values
   
   Note: While the function accepts a 'lines' parameter, the current implementation
   doesn't actually draw these lines on the image. This may be intended for future
   enhancement or is leftover from previous development.
    """
# Display an image with specified lines
def show_image_with_lines(image, lines, title=None, enable_plot=True):
    if not enable_plot:
        return
    plt.figure(figsize=(12, 6))
    plt.imshow(image)

    if title:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


"""
    Identifies the top and bottom boundaries of a channel in an image by analyzing
    row-wise pixel differences.
    
    This function performs horizontal analysis on an image to detect significant changes
    in pixel values across rows, which typically indicate the boundaries of a channel or
    region of interest. The function:
    
    1. Loads the specified image file
    2. Computes the mean differences between adjacent rows of pixels
    3. These differences highlight transitions between different regions in the image
    4. Optionally displays a plot showing these differences for visual debugging
    
    The function is particularly useful for channel detection in technical images 
    like charts, diagrams, or scientific imagery where detecting horizontal borders 
    is important.
    
    Args:
        image_path (str): Path to the image file to be analyzed
        enable_plot (bool, optional): Whether to display a plot of the row differences.
                                     Defaults to True.
    
    Returns:
        None: The function currently only visualizes the differences but doesn't
              return the actual boundary values. Consider extending to return the
              detected boundary indices.
    
    Note:
        This function relies on helper functions:
        - load_image: To read and prepare the image
        - compute_mean_differences: To calculate row-wise differences
        - debug_plot: To visualize the differences
    """
# Horizontal Analysis Functions to identify the top and bottom boundaries of a channel.
def find_horizontal_boundaries(image_path, enable_plot=True):
    # Load image and compute row-wise differences
    image, has_alpha, height, width = load_image(image_path)
    differences = compute_mean_differences(image, "rows", has_alpha)


    """
    Creates and optionally displays a visualization of data for debugging purposes.
    
    This function generates a plot of the provided data against the given x-values,
    which is useful for visualizing patterns, trends, or anomalies in the data.
    The plot can be conditionally displayed based on the enable_plot parameter,
    making it convenient for toggling visualization during development or analysis.
    
    The function is particularly useful for debugging image processing operations
    by visualizing metrics like pixel differences across rows or columns, helping
    to identify boundaries, transitions, or regions of interest in images.
    
    Args:
        data (list/array): The y-values or data points to be plotted
        x_values (list/array): The x-values corresponding to each data point
        title (str, optional): The title of the plot. Defaults to "Plot".
        xlabel (str, optional): The label for the x-axis. Defaults to "X".
        ylabel (str, optional): The label for the y-axis. Defaults to "Y".
        enable_plot (bool, optional): Whether to actually display the plot.
                                     When False, the plotting operation is skipped.
                                     Defaults to True.
    
    Returns:
        None: This function doesn't return a value but displays a plot when enable_plot is True.
    
    Note:
        This function relies on matplotlib for visualization, which should be properly
        imported and configured in the environment.
        """
    # Plot differences if enabled for debugging
    debug_plot(
        differences,
        range(len(differences)),
        title="Row Differences",
        xlabel="Row Index",
        ylabel="Mean Gray Diff",
        enable_plot=enable_plot,
    )


    """
    Identifies the most significant horizontal boundaries in an image based on row differences.
    
    This function analyzes the calculated row-wise differences and determines the likely
    top and bottom boundaries of a channel or region of interest. It works by:
    
    1. Dividing the image into top and bottom halves
    2. Finding the second largest difference in each half (using second largest to avoid
       detecting extreme outliers that might be noise)
    3. Determining the row indices that represent significant transitions in the image
    
    The function particularly focuses on the second largest differences rather than the
    maximum differences to avoid potential noise or artifacts that could create false positives.
    
    Args:
        differences (array): Array of mean differences between adjacent rows of pixels
        height (int): Height of the original image in pixels
    
    Returns:
        tuple: (top_index, bottom_index) where:
               - top_index: Row index of the detected top boundary
               - bottom_index: Row index of the detected bottom boundary
"""
    # Find max difference in top half and bottom half
    mid = height // 2
    top_index = (
        second_largest_arg(differences[:mid]) + 1
    )  # +1 to correct for starting at row 1
    bottom_index = second_largest_arg(differences[mid:]) + mid + 1


    """
    Creates a visualization of the detected horizontal boundaries on the input image.
    
    This function takes the original image and draws horizontal red lines at the positions
    identified as the top and bottom boundaries of the channel or region of interest.
    The function handles images with or without alpha channels appropriately.
    
    Args:
        image (numpy.ndarray): The original image array
        top_index (int): Row index for the top boundary line
        bottom_index (int): Row index for the bottom boundary line
        width (int): Width of the image in pixels
        has_alpha (bool): Whether the image has an alpha channel
    
    Returns:
        numpy.ndarray: A copy of the original image with red horizontal lines drawn
                      at the specified boundary positions
    """
    # Draw red lines on a copy of the image (convert to BGR if it has alpha)
    image_display = image[:, :, :3].copy() if has_alpha else image.copy()
    cv2.line(image_display, (0, top_index), (width, top_index), (0, 0, 255), 2)
    cv2.line(image_display, (0, bottom_index), (width, bottom_index), (0, 0, 255), 2)


    """
    Displays the image with the detected horizontal boundaries marked with red lines.
    
    This function converts the image from BGR to RGB color space (as required by
    matplotlib) and displays it using the show_image_with_lines helper function.
    The display is conditional based on the enable_plot parameter.
    
    Args:
        image_display (numpy.ndarray): Image with boundary lines already drawn
        top_index (int): Row index of the detected top boundary
        bottom_index (int): Row index of the detected bottom boundary
        enable_plot (bool, optional): Whether to display the image.
                                     Defaults to True.
    
    Returns:
        tuple: (top_index, bottom_index) - The indices of the detected boundaries
    """
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


"""
    Apply convolution smoothing to data using a sliding window average.
    
    This function performs a one-dimensional convolution on the input data with a 
    uniform window filter to reduce noise and smooth out fluctuations. It creates
    a rectangular window of specified size where all elements have equal weight,
    and applies this window across the data sequence.
    
    Args:
        data (array-like): The input data array to be smoothed.
        window_size (int): The size of the smoothing window. Larger windows
                          produce stronger smoothing effects but may obscure
                          important details or shift features.
    
    Returns:
        numpy.ndarray: Smoothed version of the input data with the same length
                      as the original array.
    
    Note:
        This function uses 'same' mode convolution which means the output array
        has the same size as the input array, with boundary effects handled
        appropriately.
    """
# Vertical Analysis Functions
def smooth_data(data, window_size):
    #Apply convolution smoothing to data
    window = np.ones(window_size) / window_size
    return np.convolve(data, window, mode="same")
 
 
    """
    Find vertical boundaries based on local maxima in column-wise differences.
    
    This function identifies the left and right boundaries of a channel or region
    of interest by analyzing local maxima in the column difference data. It works by:
    
    1. Extracting the subregion of interest from the full differences array
    2. Identifying all local maxima points within this subregion
    3. Calculating appropriate thresholds for the left and right boundaries
       based on mean differences in the surrounding areas
    4. Selecting the first and last local maxima that exceed these thresholds
    5. Converting these indices back to the coordinate system of the full image
    
    The function uses adaptive thresholds that are proportional to the mean
    differences in the regions outside the potential channel, which helps
    accommodate varying contrast and noise levels across different images.
    
    Args:
        differences (array-like): Array of mean differences between adjacent
                                 columns of pixels across the entire image
        start_idx (int): Starting index of the subregion to analyze (usually
                        the approximate left edge of the potential channel)
        end_idx (int): Ending index of the subregion to analyze (usually
                      the approximate right edge of the potential channel)
    
    Returns:
        list: [left_boundary, right_boundary] indices in the coordinate system
              of the original image
    
    Note:
        This function includes debugging visualization that shows the local maxima,
        the thresholds used, and the selected boundary points.
    """
def find_vertical_boundaries(differences, start_idx, end_idx):
    #Find vertical boundaries based on local maxima in differences
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

# Calculate adaptive thresholds for boundary detection
# ----------------------------------------------------
# We determine two separate thresholds for left and right boundaries by:
# 1. Examining regions outside our area of interest (before start_idx and after end_idx)
# 2. Calculating the mean difference value in each region
# 3. Multiplying by a margin factor to set thresholds above background noise
#
# The margin factor (1.8) was determined empirically and provides good separation
# between significant edge transitions and background variations

    margin = 1.8
    threshold1 = np.mean(differences[:start_idx]) * margin # Left boundary threshold
    threshold2 = np.mean(differences[end_idx:]) * margin # Right boundary threshold
    # Find boundary positions by identifying the first and last local maxima
# that exceed their respective thresholds
# --------------------------------------------------------------------
# Note: The try-except block has been commented out, which could lead to IndexError
# if no peaks exceed the thresholds.
    #try:
    first_local_max_index = local_maxima_index[local_maxima > threshold1][0] # Leftmost significant peak
    last_local_max_index = local_maxima_index[local_maxima > threshold2][-1] # Rightmost significant peak
    first_last_local_max_index = [first_local_max_index, last_local_max_index]
    #except IndexError:
    # Fall back to default boundaries if no peaks exceed thresholds
    #first_last_local_max_index = [0, -1]  
    # Use full subregion width as fallback
    # Visualize the detected peaks, thresholds, and selected boundaries for debugging
    debug_plot(
        local_maxima, # Y-values: heights of local maxima
        local_maxima_index, # X-values: positions of local maxima
        horizontalLines=[threshold1, threshold2], # Show threshold levels
        verticalLines=first_last_local_max_index, # Show selected boundary positions
    )
    # Convert subregion coordinates back to full image coordinates
    # The indices were calculated relative to the subregion starting at start_idx,
    # so we add start_idx to map them back to the original coordinate system
    return [start_idx + i for i in first_last_local_max_index]










#Analyze vertical differences to find meniscus boundaries
def analyze_vertical_differences(image_path, top_index, bottom_index, enable_plot=True):
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
    window_size = int(0.03 * width)  # 1% of image width
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
    debug_plot(derivative, range(len(derivative)), verticalLines=[start_idx, end_idx])

    # Find vertical boundaries
    vertical_boundaries = find_vertical_boundaries(differences, start_idx, end_idx)


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
    image_path = "Figures/Wiktor_angle.png"
    results = analyze_meniscus(image_path, enable_plot=True)
