import numpy as np
from matplotlib import pyplot as plt

from helperFunctions import (
    smooth,
    load_image,
    add_radial_line,
    find_channel_edges,
    find_local_maxima,
    debug_plot,
    show_image_with_marks,
    get_values_in_circle,
    get_miniscus_width,
)



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
    shift = 2*w * -np.sign(np.cos(t))/2
    #                    (y, x)
    new_point = point + (0, shift)

    w, t  = find_tangent_and_width(gray, new_point, search_radius, debug=debug)
    return t, new_point

def find_tangents(image_path, show=True, debug=False, save=True):
    gray = load_image(image_path, gray=True, debug=debug)
    top, bottom = find_channel_edges(gray, show=debug, debug=debug)
    top_point = find_contact_point(gray, top)
    bottom_point = find_contact_point(gray, bottom)
    top_tangent, top_point = find_tangent(gray, top_point, debug=debug)
    bottom_tangent, bottom_point = find_tangent(gray, bottom_point, debug=debug)
    tt = 180+np.rad2deg(top_tangent)
    bt = 180-np.rad2deg(bottom_tangent)
    avg=round((tt+bt)/2)
    print(image_path)
    print(f"Top tangent: {tt:.0f}")
    print(f"Bottom tangent: {bt:.0f}")
    print(f"Average: {avg}")

    ax = show_image_with_marks(
        gray,
        contact_points=[top_point, bottom_point],
        horizontal_lines=[int(top), int(bottom)],
        #title="Contact Points on Channel Edges",
        enable_plot=True,
        mark_radius=12,
        show=False
    )
    add_radial_line(ax, top_point, top_tangent, 40)
    add_radial_line(ax, bottom_point, bottom_tangent, 40)
    if show:
        plt.show()
    if save:
        from pathlib import Path
        p = Path(image_path)
        o=Path("Figures/Analyzed")
        o.mkdir(exist_ok=True)
        newp = o/(p.stem+f"_{avg}"+p.suffix)
        ax.figure.savefig(newp)
    

def fitSpline(image_path, show=True, debug=True, save=True, smoothing=None,
             corner_exclusion_frac=0.0, track_window=12, edge_exclusion_frac=0.05):
    """Fit a spline to the *face-centered* meniscus profile (apex region) and exclude walls.

    This routine is intended to avoid using corner tangents/contact points (which are sensitive
    to local wedge filling and PDMS wall heterogeneities) for contact-angle inference.

    High-level steps:
      1) Detect top/bottom channel edges (existing helper).
      2) Extract the meniscus interface as x(y) by finding the strongest horizontal intensity
         gradient in each row, tracked from the apex outward.
      3) Exclude regions near the top/bottom walls by fitting only the central fraction of
         the channel height.
      4) Fit a cubic smoothing spline and compute apex curvature from derivatives.

    Returns
    -------
    result : dict with keys
      - 'top', 'bottom', 'left', 'right' (left/right are image bounds)
      - 'x_pts', 'y_pts' (all extracted points; x=col, y=row)
      - 'x_fit', 'y_fit' (points used for the fit; x=col, y=row)
      - 'pts_yx', 'fit_pts_yx' (points as (y, x) pairs, matching helper conventions)
      - 'spline' (UnivariateSpline)
      - 'apex_x', 'apex_y', 'apex_point' ((y, x) tuple)
      - 'kappa_apex', 'R_eff'
      - 'angle_top_deg', 'angle_bottom_deg' (angle vs channel wall, liquid on left)
      - 'contact_top', 'contact_bottom' ((y, x) tuples)
    """
    from scipy.interpolate import UnivariateSpline

    gray = load_image(image_path, gray=True, debug=debug)

    # 1) Detect horizontal channel edges
    top, bottom = find_channel_edges(gray, show=debug, debug=debug)
    top_i, bottom_i = int(top), int(bottom)

    H, W = gray.shape[:2]
    left_i, right_i = 0, W - 1

    # 2) Exclude regions near the top/bottom walls
    chan_h = bottom_i - top_i
    excl = int(max(0, corner_exclusion_frac * chan_h))
    y_min_fit = top_i + excl
    y_max_fit = bottom_i - excl

    if y_max_fit <= y_min_fit + 5:
        raise ValueError("Not enough vertical room after excluding wall regions.")

    def _smooth_1d(arr, win):
        if arr.size < 3:
            return arr
        win = int(win)
        if win % 2 == 0:
            win += 1
        win = min(win, arr.size if arr.size % 2 == 1 else arr.size - 1)
        if win < 3:
            return arr
        return smooth(arr, window_length=win)

    # 3) Extract interface points x(y) by locating the strongest horizontal gradient in each row.
    #    Track from the apex (middle of channel) outward to avoid snapping to the wrong edge.
    y_mid = (top_i + bottom_i) // 2

    def row_gradient_x(y, x_center=None, sign=None):
        row = gray[y, :]
        gx = np.diff(row)
        valid = np.isfinite(row[:-1]) & np.isfinite(row[1:])
        gx = np.where(valid, gx, 0.0)
        gx = _smooth_1d(gx, 5)
        if edge_exclusion_frac > 0:
            margin = int(max(1, edge_exclusion_frac * gx.size))
            gx[:margin] = 0.0
            gx[-margin:] = 0.0

        if x_center is None:
            j = int(np.argmax(np.abs(gx)))
        else:
            j0 = int(np.clip(x_center, 0, gx.size - 1))
            lo = max(0, j0 - track_window)
            hi = min(gx.size, j0 + track_window + 1)
            gwin = gx[lo:hi]
            if sign is None:
                j = int(lo + np.argmax(np.abs(gwin)))
            else:
                score = sign * gwin
                j_rel = int(np.argmax(score))
                if score[j_rel] <= 0:
                    j_rel = int(np.argmax(np.abs(gwin)))
                j = int(lo + j_rel)
        return j + 1, gx[j]

    # Initialize at apex row and lock the gradient sign to avoid flipping within the meniscus.
    x_mid, gx_mid = row_gradient_x(y_mid, x_center=None, sign=None)
    grad_sign = 1.0 if gx_mid >= 0 else -1.0

    xs = []
    ys = []

    # Track upward
    x_prev = x_mid
    for y in range(y_mid, top_i, -1):
        x_prev, _ = row_gradient_x(y, x_center=x_prev, sign=grad_sign)
        xs.append(x_prev)
        ys.append(y)

    # Track downward
    x_prev = x_mid
    for y in range(y_mid + 1, bottom_i):
        x_prev, _ = row_gradient_x(y, x_center=x_prev, sign=grad_sign)
        xs.append(x_prev)
        ys.append(y)

    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)

    # Sort by y
    order = np.argsort(ys)
    xs = xs[order]
    ys = ys[order]

    # Keep only points inside the channel (strictly)
    in_chan = (ys > top_i) & (ys < bottom_i)
    xs = xs[in_chan]
    ys = ys[in_chan]

    # 4) Select fit window excluding walls
    in_fit = (ys >= y_min_fit) & (ys <= y_max_fit)
    x_fit = xs[in_fit]
    y_fit = ys[in_fit]

    if x_fit.size < 20:
        raise ValueError("Not enough points in the face-centered region to fit a spline.")

    # Optional downweighting of points near the fit boundaries (helps stability)
    # We'll implement it by duplicating central points is overkill; instead keep it simple.

    # 5) Fit smoothing spline x(y)
    # If smoothing is not provided, choose a modest default based on point count.
    # Larger s => smoother.
    if smoothing is None:
        # Heuristic: scale with number of points and pixel-level noise
        smoothing = max(10.0, 0.5 * x_fit.size)

    spline = UnivariateSpline(y_fit, x_fit, k=3, s=smoothing)

    # Apex location: pick the extremum of x(y) with the strongest curvature.
    grid = np.linspace(y_min_fit, y_max_fit, 600)
    x_grid = spline(grid)

    d1 = spline.derivative(1)
    d2 = spline.derivative(2)
    d1_grid = d1(grid)

    # Find zero crossings of d1 on the grid (robust to non-cubic splines).
    s = np.sign(d1_grid)
    s[s == 0] = 1
    crossings = np.where(s[:-1] * s[1:] < 0)[0]
    roots = []
    for i in crossings:
        y0, y1 = grid[i], grid[i + 1]
        f0, f1 = d1_grid[i], d1_grid[i + 1]
        if f1 == f0:
            continue
        t = -f0 / (f1 - f0)
        roots.append(y0 + t * (y1 - y0))
    roots = np.array(roots, dtype=float)

    if roots.size > 0:
        curv = np.abs(d2(roots))
        apex_y = float(roots[int(np.argmax(curv))])
        apex_x = float(spline(apex_y))
    else:
        # Fallback: choose min/max x by larger curvature magnitude on the grid.
        i_min = int(np.argmin(x_grid))
        i_max = int(np.argmax(x_grid))
        cand_y = np.array([grid[i_min], grid[i_max]], dtype=float)
        cand_x = np.array([x_grid[i_min], x_grid[i_max]], dtype=float)
        curv = np.abs(d2(cand_y))
        pick = int(np.argmax(curv))
        apex_y = float(cand_y[pick])
        apex_x = float(cand_x[pick])

    # Curvature for x(y): kappa = |x''| / (1 + x'^2)^(3/2)
    dx = spline.derivative(1)(apex_y)
    d2x = spline.derivative(2)(apex_y)
    kappa_apex = float(np.abs(d2x) / (1.0 + dx**2) ** 1.5)
    R_eff = float(np.inf if kappa_apex == 0 else 1.0 / kappa_apex)

    def angle_vs_wall_from_dxdy(dxdy):
        # Angle between tangent and wall, measured on the liquid side (left).
        v = np.array([float(dxdy), 1.0], dtype=float)
        if v[0] > 0:
            v = -v
        return float(np.degrees(np.arctan2(abs(v[1]), abs(v[0]))))

    contact_top = (float(top_i), float(spline(top_i)))
    contact_bottom = (float(bottom_i), float(spline(bottom_i)))
    angle_top_deg = angle_vs_wall_from_dxdy(d1(top_i))
    angle_bottom_deg = angle_vs_wall_from_dxdy(d1(bottom_i))

    y_full = np.linspace(top_i, bottom_i, 600)
    x_full = spline(y_full)
    in_fit = (y_full >= y_min_fit) & (y_full <= y_max_fit)
    top_mask = y_full < y_min_fit
    bot_mask = y_full > y_max_fit

    def draw_overlay(ax):
        ax.imshow(gray, cmap="gray")
        ax.axhline(top_i, linestyle="--")
        ax.axhline(bottom_i, linestyle="--")
        if excl > 0:
            ax.axhline(y_min_fit, linestyle=":")
            ax.axhline(y_max_fit, linestyle=":")
        ax.plot(xs, ys, linewidth=1)
        (fit_line,) = ax.plot(x_full[in_fit], y_full[in_fit], linewidth=2)
        if excl > 0:
            color = fit_line.get_color()
            ax.plot(x_full[top_mask], y_full[top_mask], linewidth=2, linestyle="--", color=color)
            ax.plot(x_full[bot_mask], y_full[bot_mask], linewidth=2, linestyle="--", color=color)
        ax.scatter([apex_x], [apex_y], s=30)
        ax.set_title("Meniscus spline extraction (face-centered fit)")
        ax.set_xlim(0, gray.shape[1])
        ax.set_ylim(min(gray.shape[0], bottom_i + chan_h // 6), max(0, top_i - chan_h // 6))

    # ---- Debug plots ----
    if debug:
        # Plot 1: image with channel bounds and extracted interface points
        fig, ax = plt.subplots(figsize=(7, 5))
        draw_overlay(ax)

        # Plot 2: x(y) points + spline in fit region
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        ax2.scatter(x_fit, y_fit, s=8, label="points (fit window)")
        (fit_line2,) = ax2.plot(x_grid, grid, linewidth=2, label="spline (fit)")
        if excl > 0:
            color = fit_line2.get_color()
            ax2.plot(x_full[top_mask], y_full[top_mask], linewidth=2, linestyle="--", color=color, label="spline (extrap)")
            ax2.plot(x_full[bot_mask], y_full[bot_mask], linewidth=2, linestyle="--", color=color)
        ax2.scatter([apex_x], [apex_y], s=25, label="apex")
        ax2.set_title(f"Spline fit (s={smoothing:.1f}); kappa={kappa_apex:.4g} 1/px; R_eff={R_eff:.4g} px")
        ax2.set_xlabel("x [px]")
        ax2.set_ylabel("y [px]")
        ax2.invert_yaxis()  # image coordinates
        ax2.legend()

        if show:
            plt.show()
        plt.close(fig)
        plt.close(fig2)

    # Save a diagnostic overlay if requested
    if save:
        from pathlib import Path
        p = Path(image_path)
        outdir = Path("Figures/Analyzed")
        outdir.mkdir(exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 5))
        draw_overlay(ax)
        outpath = outdir / (p.stem + f"_spline_t{angle_top_deg:.0f}_b{angle_bottom_deg:.0f}" + p.suffix)
        fig.savefig(outpath, dpi=200, bbox_inches="tight")
        print(outpath)
        plt.close(fig)

    pts_yx = np.column_stack((ys, xs))
    fit_pts_yx = np.column_stack((y_fit, x_fit))
    apex_point = (apex_y, apex_x)

    return {
        "top": top_i,
        "bottom": bottom_i,
        "left": left_i,
        "right": right_i,
        "x_pts": xs,
        "y_pts": ys,
        "x_fit": x_fit,
        "y_fit": y_fit,
        "pts_yx": pts_yx,
        "fit_pts_yx": fit_pts_yx,
        "spline": spline,
        "apex_x": apex_x,
        "apex_y": apex_y,
        "apex_point": apex_point,
        "kappa_apex": kappa_apex,
        "R_eff": R_eff,
        "angle_top_deg": angle_top_deg,
        "angle_bottom_deg": angle_bottom_deg,
        "contact_top": contact_top,
        "contact_bottom": contact_bottom,
        "fit_window": (y_min_fit, y_max_fit),
        "smoothing": smoothing,
    }

if __name__ == "__main__":
    import os
    #find_tangents("Figures/test_angle_46.png", debug=True)
    #fitSpline("Figures/test_angle_46.png", debug=True, show=True, save=True)
    #fitSpline("Figures/article_figures/your_image.png", debug=True, show=True, save=True)

    folder = "Figures"
    folder = "Figures/article_figures"
    for filename in sorted(os.listdir(folder)):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
            continue
        image_path = os.path.join(folder, filename)
        try:
            #find_tangents(image_path, show=True)
            r=fitSpline(image_path=image_path, debug=False)
            print(r["angle_top_deg"])       
            print(r["angle_bottom_deg"])

        except ValueError as e:
            print(e)
            print(f"Analysis failed for {image_path}")
