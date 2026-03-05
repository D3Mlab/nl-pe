import matplotlib.pyplot as plt
import numpy as np


def make_unobserved_plot(**kwargs):

    # ---- figure settings ----
    fig_w = kwargs.get("fig_w", 3)
    fig_h = kwargs.get("fig_h", 4)
    dpi = kwargs.get("dpi", 100)

    x_lims = kwargs.get("x_lims", (0, 1))
    y_lims = kwargs.get("y_lims", (0, 1))

    inc_legend = kwargs.get("inc_legend", False)

    axis_labels = kwargs.get("axis_labels", [])

    show_fig = kwargs.get("show_fig", True)

    seed = kwargs.get("seed", 42)

    # ---- cluster parameters ----
    centroids = kwargs.get("unobserved_centroids", [])
    covs = kwargs.get("unobserved_cluster_covs", [])
    n_points = kwargs.get("unobserved_n_points_per_cluster", [])

    cov_scale = kwargs.get("cov_scale", 1.0)

    docs_name = kwargs.get("docs_name", "docs")

    point_overlap_allowed = kwargs.get("point_overlap_allowed", True)

    # ---- marker defaults ----
    default_marker_kwargs = dict(
        marker="o",
        facecolors="none",
        edgecolors="grey",
        s=30,
        label=docs_name
    )

    user_marker_kwargs = kwargs.get("unobserved_marker_kwargs", {})
    marker_kwargs = {**default_marker_kwargs, **user_marker_kwargs}

    s = marker_kwargs.get("s", 30)

    # ---- random seed ----
    np.random.seed(seed)

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    ax.set_xlim(*x_lims)
    ax.set_ylim(*y_lims)

    # remove ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # remove borders
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # axis labels
    if axis_labels:
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])

    # ---- compute marker radius in data units ----
    fig.canvas.draw()

    r_points = np.sqrt(s / np.pi)

    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width_in = bbox.width
    height_in = bbox.height

    x_range = x_lims[1] - x_lims[0]
    y_range = y_lims[1] - y_lims[0]

    x_per_in = x_range / width_in
    y_per_in = y_range / height_in

    r_data = r_points / 72 * max(x_per_in, y_per_in)
    min_dist = 2 * r_data

    # ---- sample points ----
    accepted_points = []

    for centroid, cov, n in zip(centroids, covs, n_points):

        cov = np.array(cov) * cov_scale

        count = 0
        attempts = 0

        while count < n and attempts < n * 200:

            candidate = np.random.multivariate_normal(
                mean=centroid,
                cov=cov
            )

            attempts += 1

            if point_overlap_allowed:
                accepted_points.append(candidate)
                count += 1
                continue

            if len(accepted_points) == 0:
                accepted_points.append(candidate)
                count += 1
                continue

            dists = np.linalg.norm(
                np.array(accepted_points) - candidate,
                axis=1
            )

            if np.all(dists >= min_dist):
                accepted_points.append(candidate)
                count += 1

    if len(accepted_points) > 0:
        pts = np.array(accepted_points)
        ax.scatter(pts[:, 0], pts[:, 1], **marker_kwargs)

    # legend
    if inc_legend:
        ax.legend()

    if show_fig:
        plt.show()

    return fig