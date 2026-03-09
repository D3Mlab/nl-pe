import matplotlib.pyplot as plt
import numpy as np
import torch
import gpytorch
from matplotlib.patches import Circle
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D


class PlotExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, lengthscale, signal_noise):
        super().__init__(train_x, train_y, likelihood)

        # Constant mean fixed at 0.0
        self.mean_module = gpytorch.means.ConstantMean()
        self.mean_module.initialize(constant=0.0)
        self.mean_module.raw_constant.requires_grad_(False)

        base_kernel = gpytorch.kernels.RBFKernel()
        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)
        self.covar_module.base_kernel.initialize(lengthscale=lengthscale)
        self.covar_module.initialize(outputscale=signal_noise)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def make_unobserved_plot(**kwargs):

    # ------------------------------------------------
    # BASIC SETTINGS
    # ------------------------------------------------
    fig_w = kwargs.get("fig_w",3)
    fig_h = kwargs.get("fig_h",4)
    dpi = kwargs.get("dpi",100)

    font = kwargs.get("font","Calibri")
    fontsize = kwargs.get("fontsize",12)

    x_lims = kwargs.get("x_lims",(0,1))
    y_lims = kwargs.get("y_lims",(0,1))

    axis_linewidth = kwargs.get("axis_linewidth",1.0)
    show_top_right_spines = kwargs.get("show_top_right_spines",False)

    inc_legend = kwargs.get("inc_legend",False)
    legend_coords = kwargs.get("legend_coords",None)
    legend_loc = kwargs.get("legend_loc","upper left")

    inc_scale = kwargs.get("inc_scale",False)
    scale_coords = kwargs.get("scale_coords",None)
    scale_loc = kwargs.get("scale_loc","right")
    scale_orientation = kwargs.get("scale_orientation","vertical")

    axis_labels = kwargs.get("axis_labels",[])

    show_fig = kwargs.get("show_fig",True)

    seed = kwargs.get("seed",42)

    cov_scale = kwargs.get("cov_scale",1.0)

    point_overlap_allowed = kwargs.get("point_overlap_allowed",True)

    color_style = kwargs.get("color_style",None)

    dense_cmap = kwargs.get("dense_cmap","viridis")
    dense_resolution = kwargs.get("dense_resolution",200)
    gp_contour_resolution = kwargs.get("gp_contour_resolution", dense_resolution)
    gp_contour_alpha = kwargs.get("gp_contour_alpha", 0.8)
    gp_contour_levels = kwargs.get("gp_contour_levels", 16)

    no_color_outside_circle = kwargs.get("no_color_outside_circle",False)

    show_circle_docs_in_col = kwargs.get("show_circle_docs_in_col", False)
    dot_col_v_space = kwargs.get("dot_col_v_space", 1.0)
    dot_col_size = kwargs.get("dot_col_size", 30)
    dot_col_k = kwargs.get("dot_col_k", None)
    color_points_inside_query_circle = kwargs.get("color_points_inside_query_circle", True)
    top_k_point_border = kwargs.get("top_k_point_border", {"edgecolors": "black", "linewidths": 1.0})
    color_query_with_heatmap = kwargs.get("color_query_with_heatmap", True)
    # backward-compat alias
    shade_points_in_query_circle = kwargs.get("shade_points_in_query_circle", None)
    if shade_points_in_query_circle is not None:
        color_points_inside_query_circle = bool(shade_points_in_query_circle)

    query_border_kwargs = kwargs.get("query_border_kwargs", {})

    print_points = kwargs.get("print_points",False)

    if color_style == "gp_cont":
        raise ValueError("color_style='gp_cont' is no longer supported. Use 'gp_contour' instead.")

    # ------------------------------------------------
    # DATA INPUTS
    # ------------------------------------------------
    centroids = kwargs.get("irel_unobserved_centroids",[])
    covs = kwargs.get("irel_unobserved_cluster_covs",[])
    n_points = kwargs.get("irel_unobserved_n_points_per_cluster",[])

    rel_unobserved_locs = kwargs.get("rel_unobserved_locs",[])

    query_loc = kwargs.get("query_loc",None)
    query_rel_circle_radius = kwargs.get("query_rel_circle_radius",None)
    extra_unobserved_irel_locs = kwargs.get("extra_unobserved_irel_locs", [])
    af_locs = kwargs.get("af_locs", [])

    observed_points = list(kwargs.get("observed_points",[]))
    irrel_observed_locs = kwargs.get("irrel_observed_locs", [])
    irrel_observed_value = kwargs.get("irrel_observed_value", 0.0)

    max_rel = kwargs.get("max_rel",1)

    if len(irrel_observed_locs) > 0:
        observed_points.extend([
            (tuple(pt), float(irrel_observed_value), "doc")
            for pt in irrel_observed_locs
        ])

    # ------------------------------------------------
    # GP PARAMETERS
    # ------------------------------------------------
    gp_ls = kwargs.get("gp_ls",0.2)
    gp_os = kwargs.get("gp_os",1.0)
    gp_noise = kwargs.get("gp_noise",1e-6)

    # ------------------------------------------------
    # MARKER DEFAULTS
    # ------------------------------------------------
    default_irel_marker=dict(marker="o",facecolors="none",edgecolors="grey",s=30,linewidths=1.2,label="Irel. docs")
    default_rel_marker=dict(marker="^",facecolors="none",edgecolors="grey",s=30,linewidths=1.2,label="Rel. docs")
    default_query_marker=dict(marker="x",color="black",s=30,linewidths=1.5,label="Query")

    default_obs_doc=dict(marker="D",s=40,linewidths=1.5,edgecolors="black",label="Observed Doc")
    default_obs_query=dict(marker="s",s=40,linewidths=1.2,edgecolors="black",label="Query")
    default_af_marker=dict(marker="*",s=60,color="red",edgecolors="black",linewidths=2,label="Acq. Func. Selection")

    default_circle=dict(edgecolor="grey",linestyle="--",linewidth=1.2,fill=False)

    irel_marker_kwargs={**default_irel_marker,**kwargs.get("irel_marker_kwargs",{})}
    rel_marker_kwargs={**default_rel_marker,**kwargs.get("rel_marker_kwargs",{})}
    query_marker_kwargs={**default_query_marker,**kwargs.get("query_marker_kwargs",{})}

    obs_doc_marker_kwargs={**default_obs_doc,**kwargs.get("obs_doc_marker_kwargs",{})}
    obs_query_marker_kwargs={**default_obs_query,**kwargs.get("obs_query_marker_kwargs",{})}
    af_marker_kwargs={**default_af_marker,**kwargs.get("af_marker_kwargs",{})}

    query_circle_kwargs={**default_circle,**kwargs.get("query_rel_circle_kwargs",{})}

    s = irel_marker_kwargs.get("s",30)

    np.random.seed(seed)

    # ------------------------------------------------
    # FIGURE
    # ------------------------------------------------
    fig,ax = plt.subplots(figsize=(fig_w,fig_h),dpi=dpi)

    ax.set_xlim(*x_lims)
    ax.set_ylim(*y_lims)

    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_linewidth(axis_linewidth)

    if not show_top_right_spines:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if axis_labels:
        ax.set_xlabel(axis_labels[0],fontsize=fontsize,fontname=font)
        ax.set_ylabel(axis_labels[1],fontsize=fontsize,fontname=font)

    cmap = plt.get_cmap(dense_cmap)
    scale_mappable=None
    scale_label=None

    # ------------------------------------------------
    # CONTINUOUS HEATMAP
    # ------------------------------------------------
    if color_style=="continuous_color" and query_loc is not None:

        xs=np.linspace(x_lims[0],x_lims[1],dense_resolution)
        ys=np.linspace(y_lims[0],y_lims[1],dense_resolution)

        X,Y=np.meshgrid(xs,ys)

        dist=np.sqrt((X-query_loc[0])**2+(Y-query_loc[1])**2)
        max_dist=np.max(dist)

        Z=1-(dist/max_dist)

        if no_color_outside_circle and query_rel_circle_radius is not None:
            mask=dist>query_rel_circle_radius
            Z[mask]=np.nan

        ax.imshow(Z,extent=[*x_lims,*y_lims],origin="lower",
                  cmap=dense_cmap,alpha=0.6,aspect="auto")

    # ------------------------------------------------
    # GP MODEL
    # ------------------------------------------------
    gp=None
    if color_style in ["gp_points", "gp_contour"]:

        if len(observed_points)==0:
            raise ValueError("GP coloring modes require observed_points")

        X_train=np.array([p[0] for p in observed_points])
        y_train=np.array([p[1] for p in observed_points])
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.initialize(noise=gp_noise)

        model = PlotExactGPModel(
            X_train_t,
            y_train_t,
            likelihood,
            lengthscale=gp_ls,
            signal_noise=gp_os,
        )

        model.eval()
        likelihood.eval()

        gp = (model, likelihood)

    # ------------------------------------------------
    # GP CONTOUR BACKGROUND
    # ------------------------------------------------
    if color_style=="gp_contour":
        xs=np.linspace(x_lims[0],x_lims[1],gp_contour_resolution)
        ys=np.linspace(y_lims[0],y_lims[1],gp_contour_resolution)

        Xg,Yg=np.meshgrid(xs,ys)
        grid_points = np.column_stack([Xg.ravel(), Yg.ravel()])

        model, likelihood = gp
        grid_points_t = torch.tensor(grid_points, dtype=torch.float32)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            gp_mean = likelihood(model(grid_points_t)).mean.detach().cpu().numpy()

        gp_mean = gp_mean.reshape(gp_contour_resolution, gp_contour_resolution)
        gp_mean = np.clip(gp_mean, 0, max_rel)
        gp_mean = gp_mean / max_rel

        contour_set = ax.contour(
            Xg,
            Yg,
            gp_mean,
            levels=gp_contour_levels,
            cmap=dense_cmap,
            linewidths=0.8,
            alpha=gp_contour_alpha,
        )

        scale_mappable=contour_set
        scale_label="GP Posterior Mean"

    # ------------------------------------------------
    # POINT GENERATION WITH OVERLAP CONTROL
    # ------------------------------------------------
    fig.canvas.draw()

    r_points=np.sqrt(s/np.pi)

    bbox=ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width_in=bbox.width
    height_in=bbox.height

    x_range=x_lims[1]-x_lims[0]
    y_range=y_lims[1]-y_lims[0]

    x_per_in=x_range/width_in
    y_per_in=y_range/height_in

    r_data=r_points/72*max(x_per_in,y_per_in)
    min_dist=2*r_data

    accepted_points=[]

    for centroid,cov,n in zip(centroids,covs,n_points):

        cov=np.array(cov)*cov_scale

        count=0
        attempts=0

        while count<n and attempts<n*300:

            candidate=np.random.multivariate_normal(centroid,cov)
            attempts+=1

            if point_overlap_allowed:
                accepted_points.append(candidate)
                count+=1
                continue

            if len(accepted_points)==0:
                accepted_points.append(candidate)
                count+=1
                continue

            dists=np.linalg.norm(np.array(accepted_points)-candidate,axis=1)

            if np.all(dists>=min_dist):
                accepted_points.append(candidate)
                count+=1

    if len(accepted_points) > 0:
        accepted_points=np.array(accepted_points, dtype=float).reshape(-1,2)
    else:
        accepted_points=np.empty((0,2), dtype=float)

    if len(extra_unobserved_irel_locs) > 0:
        accepted_points = np.vstack([
            accepted_points,
            np.array(extra_unobserved_irel_locs, dtype=float).reshape(-1,2)
        ])

    rel_pts=np.array(rel_unobserved_locs) if len(rel_unobserved_locs)>0 else np.empty((0,2))

    printable=[]
    plotted_docs=[]

    def _extract_marker_color(style_kwargs, default_color="grey"):
        if style_kwargs.get("c") is not None:
            c = style_kwargs.get("c")
            if isinstance(c, (list, tuple, np.ndarray)) and len(c) > 0:
                try:
                    return to_hex(c[0])
                except Exception:
                    return c[0]
            return c

        if style_kwargs.get("color") is not None:
            try:
                return to_hex(style_kwargs.get("color"))
            except Exception:
                return style_kwargs.get("color")

        if style_kwargs.get("edgecolors") is not None:
            edge = style_kwargs.get("edgecolors")
            if isinstance(edge, (list, tuple, np.ndarray)) and len(edge) > 0:
                try:
                    return to_hex(edge[0])
                except Exception:
                    return edge[0]
            try:
                return to_hex(edge)
            except Exception:
                return edge

        if style_kwargs.get("facecolors") is not None and style_kwargs.get("facecolors") != "none":
            face = style_kwargs.get("facecolors")
            if isinstance(face, (list, tuple, np.ndarray)) and len(face) > 0:
                try:
                    return to_hex(face[0])
                except Exception:
                    return face[0]
            try:
                return to_hex(face)
            except Exception:
                return face

        try:
            return to_hex(default_color)
        except Exception:
            return default_color

    def _is_hollow_marker_style(style_kwargs, default_hollow=False):
        face = style_kwargs.get("facecolors", style_kwargs.get("facecolor", None))
        if face is None:
            return default_hollow
        return str(face).lower() == "none"

    def _euclidean_distance(a, b):
        a = np.array(a, dtype=float)
        b = np.array(b, dtype=float)
        return float(np.linalg.norm(a - b))

    def _plot_colored_doc_points(points, colors, unobs_style, obs_style, default_marker):
        if len(points) == 0:
            return

        pts = np.array(points)
        inside_mask = np.zeros(len(pts), dtype=bool)

        if (
            color_points_inside_query_circle
            and query_loc is not None
            and query_rel_circle_radius is not None
        ):
            inside_mask = (
                np.linalg.norm(pts - np.array(query_loc), axis=1)
                <= query_rel_circle_radius
            )

        outside_mask = ~inside_mask

        if np.any(outside_mask):
            outside_pts = pts[outside_mask]
            outside_cols = colors[outside_mask]
            style = dict(unobs_style)
            style.pop("color", None)
            style.pop("c", None)
            style.pop("facecolors", None)
            style.pop("facecolor", None)
            style.pop("edgecolors", None)
            style.pop("edgecolor", None)

            ax.scatter(
                outside_pts[:, 0],
                outside_pts[:, 1],
                facecolors="none",
                edgecolors=outside_cols,
                **style,
            )

            marker = unobs_style.get("marker", default_marker)
            for p, c in zip(outside_pts, outside_cols):
                plotted_docs.append((p, marker, to_hex(c), True))

        if np.any(inside_mask):
            inside_pts = pts[inside_mask]
            inside_cols = colors[inside_mask]
            style = dict(unobs_style)
            style.pop("color", None)
            style.pop("c", None)
            style.pop("facecolors", None)
            style.pop("facecolor", None)
            style.pop("edgecolors", None)
            style.pop("edgecolor", None)
            style.pop("linewidths", None)
            style.pop("linewidth", None)

            border_style = dict(top_k_point_border) if isinstance(top_k_point_border, dict) else {}
            if top_k_point_border in [None, False, "none"]:
                border_style = {"edgecolors": "none", "linewidths": 0.0}
            edge_col = border_style.get("edgecolors", border_style.get("edgecolor", "black"))
            lw = border_style.get("linewidths", border_style.get("linewidth", 1.0))

            ax.scatter(
                inside_pts[:, 0],
                inside_pts[:, 1],
                facecolors=inside_cols,
                edgecolors=edge_col,
                linewidths=lw,
                **style,
            )

            marker = unobs_style.get("marker", default_marker)
            for p, c in zip(inside_pts, inside_cols):
                plotted_docs.append((p, marker, to_hex(c), False))

    # ------------------------------------------------
    # POINT COLOR MODES
    # ------------------------------------------------
    if color_style=="point_color" and query_loc is not None:

        all_points=np.vstack([accepted_points,rel_pts])

        dists=np.linalg.norm(all_points-query_loc,axis=1)
        max_dist=np.max(dists) if np.max(dists)!=0 else 1

        scores=1-(dists/max_dist)
        colors=cmap(scores)

        scale_mappable=plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        scale_label="query-passage sim."

        idx=0

        if len(accepted_points)>0:
            n=len(accepted_points)
            _plot_colored_doc_points(
                accepted_points,
                colors[idx:idx+n],
                irel_marker_kwargs,
                obs_doc_marker_kwargs,
                "o",
            )

            for p,c,scr in zip(accepted_points,colors[idx:idx+n],scores[idx:idx+n]):
                printable.append((scr,tuple(np.round(p,4)),to_hex(c)))

            idx+=n

        if len(rel_pts)>0:
            n=len(rel_pts)
            _plot_colored_doc_points(
                rel_pts,
                colors[idx:idx+n],
                rel_marker_kwargs,
                obs_doc_marker_kwargs,
                "^",
            )

            for p,c,scr in zip(rel_pts,colors[idx:idx+n],scores[idx:idx+n]):
                printable.append((scr,tuple(np.round(p,4)),to_hex(c)))

    elif color_style in ["gp_points", "gp_contour"]:

        all_points=np.vstack([accepted_points,rel_pts])
        model, likelihood = gp
        all_points_t = torch.tensor(all_points, dtype=torch.float32)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            preds = likelihood(model(all_points_t)).mean.detach().cpu().numpy()

        preds=np.clip(preds,0,max_rel)

        scores=preds/max_rel
        colors=cmap(scores)

        idx=0

        if len(accepted_points)>0:
            n=len(accepted_points)
            _plot_colored_doc_points(
                accepted_points,
                colors[idx:idx+n],
                irel_marker_kwargs,
                obs_doc_marker_kwargs,
                "o",
            )

            for p,c,scr in zip(accepted_points,colors[idx:idx+n],scores[idx:idx+n]):
                printable.append((scr,tuple(np.round(p,4)),to_hex(c)))

            idx+=n

        if len(rel_pts)>0:
            n=len(rel_pts)
            _plot_colored_doc_points(
                rel_pts,
                colors[idx:idx+n],
                rel_marker_kwargs,
                obs_doc_marker_kwargs,
                "^",
            )

            for p,c,scr in zip(rel_pts,colors[idx:idx+n],scores[idx:idx+n]):
                printable.append((scr,tuple(np.round(p,4)),to_hex(c)))

    else:

        if len(accepted_points)>0:
            ax.scatter(accepted_points[:,0],accepted_points[:,1],**irel_marker_kwargs)

            marker = irel_marker_kwargs.get("marker", "o")
            default_col = _extract_marker_color(irel_marker_kwargs, default_color="grey")
            is_hollow = _is_hollow_marker_style(irel_marker_kwargs, default_hollow=True)
            for p in accepted_points:
                plotted_docs.append((p, marker, default_col, is_hollow))

        if len(rel_pts)>0:
            ax.scatter(rel_pts[:,0],rel_pts[:,1],**rel_marker_kwargs)

            marker = rel_marker_kwargs.get("marker", "^")
            default_col = _extract_marker_color(rel_marker_kwargs, default_color="grey")
            is_hollow = _is_hollow_marker_style(rel_marker_kwargs, default_hollow=True)
            for p in rel_pts:
                plotted_docs.append((p, marker, default_col, is_hollow))

    # ------------------------------------------------
    # OBSERVED POINTS (STYLE BY TYPE, COLOR BY LABEL)
    # ------------------------------------------------
    for (pt,val,ptype) in observed_points:

        if ptype == "query" and color_style in ["continuous_color", "point_color", "gp_points", "gp_contour"]:
            score = 1.0
        else:
            score=np.clip(val,0,max_rel)/max_rel
        color=cmap(score)

        if ptype=="query":
            style=obs_query_marker_kwargs
        elif ptype=="doc":
            style=obs_doc_marker_kwargs
        else:
            raise ValueError("ptype must be 'query' or 'doc'")

        style=dict(style)
        style.pop("color",None)
        style.pop("c",None)

        ax.scatter(pt[0],pt[1],c=[color],**style)

        if ptype == "doc":
            marker = style.get("marker", "D")
            is_hollow = _is_hollow_marker_style(style, default_hollow=False)
            plotted_docs.append((np.array(pt), marker, to_hex(color), is_hollow))

        printable.append((score,tuple(np.round(pt,4)),to_hex(color)))

    # ------------------------------------------------
    # PRINT POINTS
    # ------------------------------------------------
    if print_points:
        printable.sort(key=lambda x:x[0],reverse=True)
        for score,point,color in printable:
            print(f"Point {point} score={score:.4f} color={color}")

    # ------------------------------------------------
    # QUERY
    # ------------------------------------------------
    if query_loc is not None:
        query_style = dict(query_marker_kwargs)
        q_border = dict(query_border_kwargs)
        q_edge = q_border.get("edgecolors", q_border.get("edgecolor", query_style.get("edgecolors", query_style.get("edgecolor", "black"))))
        q_lw = q_border.get("linewidths", q_border.get("linewidth", query_style.get("linewidths", query_style.get("linewidth", 1.5))))

        if color_style in ["continuous_color", "point_color", "gp_points", "gp_contour"] and color_query_with_heatmap:
            query_style.pop("color", None)
            query_style.pop("c", None)
            query_style.pop("facecolors", None)
            query_style.pop("facecolor", None)
            query_style.pop("edgecolors", None)
            query_style.pop("edgecolor", None)
            query_style.pop("linewidths", None)
            query_style.pop("linewidth", None)
            ax.scatter(
                [query_loc[0]],
                [query_loc[1]],
                facecolors=[cmap(1.0)],
                edgecolors=q_edge,
                linewidths=q_lw,
                **query_style,
            )
        else:
            if len(q_border) > 0:
                query_style.pop("edgecolors", None)
                query_style.pop("edgecolor", None)
                query_style.pop("linewidths", None)
                query_style.pop("linewidth", None)
                query_style["edgecolors"] = q_edge
                query_style["linewidths"] = q_lw
            ax.scatter([query_loc[0]],[query_loc[1]],**query_style)

    if query_loc is not None and query_rel_circle_radius is not None:
        circle=Circle(query_loc,query_rel_circle_radius,**query_circle_kwargs)
        ax.add_patch(circle)

        docs_inside_circle=[]
        for p, marker, color, is_hollow in plotted_docs:
            if np.linalg.norm(np.array(p)-np.array(query_loc)) <= query_rel_circle_radius:
                eucl_dist = _euclidean_distance(p, query_loc)
                docs_inside_circle.append((p, marker, color, is_hollow, eucl_dist))

        if show_circle_docs_in_col:
            docs_inside_circle.sort(key=lambda x: x[4])

            if dot_col_k is not None:
                docs_inside_circle = docs_inside_circle[:max(int(dot_col_k), 0)]

            n_col = len(docs_inside_circle)
            col_fig_w = 1.6
            col_fig_h = max(1.8, 0.9 + max(n_col - 1, 0) * dot_col_v_space * 0.55)
            fig_col, ax_col = plt.subplots(figsize=(col_fig_w, col_fig_h), dpi=dpi)
            ax_col.set_xticks([])
            ax_col.set_yticks([])
            for spine in ax_col.spines.values():
                spine.set_visible(False)

            if len(docs_inside_circle) == 0:
                ax_col.text(
                    0.5,
                    0.5,
                    "No docs in query circle",
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                    fontname=font,
                    transform=ax_col.transAxes,
                )
            else:
                ys = np.arange(len(docs_inside_circle))[::-1] * dot_col_v_space
                xs = np.zeros(len(docs_inside_circle))

                for x, y, (_, marker, color, is_hollow, _) in zip(xs, ys, docs_inside_circle):
                    if is_hollow:
                        ax_col.scatter(
                            [x],
                            [y],
                            marker=marker,
                            facecolors="none",
                            edgecolors=color,
                            s=dot_col_size,
                        )
                    else:
                        ax_col.scatter(
                            [x],
                            [y],
                            marker=marker,
                            c=[color],
                            s=dot_col_size,
                        )

                y_pad = max(0.3 * dot_col_v_space, 0.2)
                ax_col.set_xlim(-1, 1)
                ax_col.set_ylim(-y_pad, ys[0] + y_pad)

    # ------------------------------------------------
    # ACQUISITION FUNCTION POINTS (VISUAL ONLY)
    # ------------------------------------------------
    if len(af_locs) > 0:
        af_pts = np.array(af_locs, dtype=float).reshape(-1, 2)
        style = dict(af_marker_kwargs)
        for x, y in af_pts:
            ax.scatter(x, y, **style)

    # ------------------------------------------------
    # LEGEND
    # ------------------------------------------------
    if inc_legend:
        query_legend_label = query_marker_kwargs.get("label", "Query")
        query_legend_marker = query_marker_kwargs.get("marker", "s")
        query_legend_edge = query_border_kwargs.get(
            "edgecolors",
            query_border_kwargs.get(
                "edgecolor",
                query_marker_kwargs.get("edgecolors", query_marker_kwargs.get("edgecolor", "black")),
            ),
        )
        query_legend_lw = query_border_kwargs.get(
            "linewidths",
            query_border_kwargs.get(
                "linewidth",
                query_marker_kwargs.get("linewidths", query_marker_kwargs.get("linewidth", 1.5)),
            ),
        )
        
        if color_style == 'gp_contour':
            query_legend_face = "grey"
        elif color_style == 'point_color':    
            query_legend_face = "none"

        legend_handles=[]

        # 1) Query first
        legend_handles.append(
            Line2D([0],[0],marker=query_legend_marker,linestyle="None",markersize=7,
                   markerfacecolor=query_legend_face,markeredgecolor=query_legend_edge,
                   markeredgewidth=query_legend_lw,label=query_legend_label)
        )

        # 2) Acquisition function selection
        if len(af_locs) > 0:
            af_face = af_marker_kwargs.get(
                "facecolors",
                af_marker_kwargs.get(
                    "facecolor",
                    af_marker_kwargs.get(
                        "c",
                        af_marker_kwargs.get("color", "red"),
                    ),
                ),
            )
            if isinstance(af_face, (list, tuple, np.ndarray)) and len(af_face) > 0:
                af_face = af_face[0]
            legend_handles.append(
                Line2D(
                    [0],[0],
                    marker=af_marker_kwargs.get("marker","*"),
                    linestyle="None",
                    markersize=10,
                    markerfacecolor=af_face,
                    markeredgecolor=af_marker_kwargs.get("edgecolors","black"),
                    markeredgewidth=af_marker_kwargs.get("linewidths",2),
                    label=af_marker_kwargs.get("label","Acq. Func. Selection"),
                )
            )

        # 3) Observed docs
        if color_style in ["gp_points", "gp_contour"]:
            legend_handles.append(
                Line2D([0],[0],marker=obs_doc_marker_kwargs.get("marker","D"),linestyle="None",markersize=8,
                       markerfacecolor="grey",markeredgecolor=obs_doc_marker_kwargs.get("edgecolors","black"),
                       markeredgewidth=obs_doc_marker_kwargs.get("linewidths",1.5),label=obs_doc_marker_kwargs.get("label","Observed Doc"))
            )

        # 4) Unobserved relevant docs
        legend_handles.append(
            Line2D([0],[0],marker=rel_marker_kwargs.get("marker","^"),linestyle="None",markersize=7,
                   markerfacecolor="none",markeredgecolor="grey",label=rel_marker_kwargs.get("label","Unobserved, Rel."))
        )

        # 5) Unobserved irrelevant docs
        legend_handles.append(
            Line2D([0],[0],marker=irel_marker_kwargs.get("marker","o"),linestyle="None",markersize=7,
                   markerfacecolor="none",markeredgecolor="grey",label=irel_marker_kwargs.get("label","Unobserved, Irrel."))
        )

        if legend_coords is None:
            ax.legend(handles=legend_handles,
                      prop={"family":font,"size":fontsize},
                      loc=legend_loc)
        else:
            ax.legend(handles=legend_handles,
                      prop={"family":font,"size":fontsize},
                      bbox_to_anchor=legend_coords,
                      loc=legend_loc)

    # ------------------------------------------------
    # SCALE / COLORBAR
    # ------------------------------------------------
    if inc_scale and (scale_mappable is not None):
        if scale_coords is None:
            cbar=fig.colorbar(scale_mappable,
                              ax=ax,
                              orientation=scale_orientation,
                              location=scale_loc)
        else:
            cax = fig.add_axes(scale_coords)
            cbar=fig.colorbar(scale_mappable,
                              cax=cax,
                              orientation=scale_orientation)

        if scale_label is not None:
            cbar.set_label(scale_label, fontname=font, fontsize=fontsize)

    if show_fig:
        plt.show()

    return fig