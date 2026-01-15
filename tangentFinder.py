import numpy as np
from matplotlib import pyplot as plt

from helperFunctions import (
    smooth,
    load_image,
    add_radial_line,
    compute_mean_differences,
    find_local_maxima,
    debug_plot,
    show_image_with_lines,
    show_image_with_marks,
    get_values_in_circle,
    get_miniscus_width,
)


# Horizontal Analysis Functions to identify the top and bottom boundaries of a channel.
def find_channel_edges(gray, enable_plot=True):
    differences = compute_mean_differences(gray, "rows")
    mid = gray.shape[0] // 2

    val, index = find_local_maxima(differences[:mid], 2)
    top_idx = int(np.mean(index)) + 1

    val, index = find_local_maxima(differences[mid:], 2)
    bottom_idx = int(np.mean(index)) + mid + 1

    # Show the result
    if enable_plot:
        show_image_with_lines(
            gray,
            horizontal_lines=[int(top_idx), int(bottom_idx)],
            title="Image with Max Row Differences Marked",
            enable_plot=enable_plot,
        )

    return top_idx, bottom_idx


def find_contact_point(gray, channel_edge):
    # We traverse horizontally along the channel edge
    # and look at the difference in intensity.
    # The place where the liquid connects with the edge
    # should appear as a darker point
    d = np.abs(np.diff(gray[channel_edge, :]))
    # We set the last 10% of the edges to zero
    d[-len(d) // 10 :] = 0
    d[: len(d) // 10] = 0
    # Oof, we don't have much to work with here...
    # The noise is almost as large as the signal
    # debug_plot(d)
    return np.array((channel_edge, np.argmax(d)+1))


def find_tangent_and_width(gray, point, search_radius=12, debug=False):
    circle, theta = get_values_in_circle(gray, point, search_radius, width=5)
    d = np.abs(np.diff(circle))
    d = smooth(d, window_length=4)

    # We should now have 4 local maxima, similar to what we found when looking
    # for horizontal boundaries.
    _, max_indexes = find_local_maxima(d, 4, min_dist=5)

    if debug:
        debug_plot(d, i_markers=max_indexes, title="Diff detected around contact point")

    ts = theta[max_indexes + 1]

    # Pair the 4 angles by choosing the two closest pairs on the circle.
    # This works across the 2π↔0 periodic border.
    def circ_dist(a, b):
        return np.abs(np.angle(np.exp(1j * (a - b))))

    def circ_mean(a, b):
        # Mean direction on the circle
        return np.arctan2(np.sin(a) + np.sin(b), np.cos(a) + np.cos(b))

    # For 4 points there are only 3 unique pairings.
    pairings = [
        ((0, 1), (2, 3)),
        ((0, 2), (1, 3)),
        ((0, 3), (1, 2)),
    ]

    costs = np.array([
        circ_dist(ts[i], ts[j]) + circ_dist(ts[k], ts[l])
        for (i, j), (k, l) in pairings
    ])
    (i1, j1), (i2, j2) = pairings[int(np.argmin(costs))]

    a1, b1 = ts[i1], ts[j1]
    a2, b2 = ts[i2], ts[j2]

    # Widths for each side of the meniscus
    w1 = get_miniscus_width(a1, b1, search_radius)
    w2 = get_miniscus_width(a2, b2, search_radius)
    w = (w1 + w2) / 2

    # Tangent directions for each side (mean angle of its two edge angles)
    t1 = circ_mean(a1, b1)
    t2 = circ_mean(a2, b2)

    if abs(w1 - w2) > w / 5:
        # If they differ too much, something has maybe gone wrong.
        # Choose the smaller width and show what happened.
        i = np.argmin((w1, w2))
        w = [w1, w2][i]
        print(f"Warning, the miniscus width was not consistent: {w1:.2f} vs {w2:.2f}")
        if debug:
            ax = show_image_with_marks(
                gray,
                contact_points=[point],
                horizontal_lines=[point[0]],
                title="Meniscus edge angles (debug)",
                show=False,
            )
            for _t in ts:
                add_radial_line(ax, point, _t, 20)
            plt.show()
            plt.close()


    # Now we have two angles, t1 and t2.
    # We only return the angle of the miniscus pointing into the channel,
    # not the one parallel to it
    def dist_to_vertical(a):
        a = a % np.pi
        return abs(a - np.pi / 2)

    return w, (t1 if dist_to_vertical(t1) < dist_to_vertical(t2) else t2)

def find_tangent(gray, point, search_radius=12, debug=False):
    # First we find the width of the miniscus
    w, t  = find_tangent_and_width(gray, point, search_radius)
    # Then we move the point to the center of the miniscus
    # We can find the direction to move by looking at t
    shift = w * -np.sign(np.cos(t))/2
    #                    (y, x)
    new_point = point + (0, shift)

    w, t  = find_tangent_and_width(gray, new_point, search_radius, debug=debug)
    return t, new_point

if __name__ == "__main__":
    image_path = "Figures/after_plasma.png"
    image_path = "Figures/test_angle_47.png"
    image_path = "Figures/test_angle_46.png"
    image_path = "Figures/test_angle_53.png"
    image_path = "Figures/test_angle_54.png"
    gray = load_image(image_path, gray=True)
    top, bottom = find_channel_edges(gray, enable_plot=False)
    top_point = find_contact_point(gray, top)
    bottom_point = find_contact_point(gray, bottom)
    top_tangent, top_point = find_tangent(gray, top_point, debug=False)
    bottom_tangent, bottom_point = find_tangent(gray, bottom_point, debug=False)
    ax = show_image_with_marks(
        gray,
        contact_points=[top_point, bottom_point],
        horizontal_lines=[int(top), int(bottom)],
        title="Contact Points on Channel Edges",
        enable_plot=True,
        mark_radius=12,
        show=False
    )
    add_radial_line(ax, top_point, top_tangent, 30)
    add_radial_line(ax, bottom_point, bottom_tangent, 30)
    plt.show()
