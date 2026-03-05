import matplotlib.pyplot as plt
import numpy as np
import torch
import gpytorch
from matplotlib.patches import Circle
from matplotlib.colors import to_hex


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

    axis_labels = kwargs.get("axis_labels",[])

    show_fig = kwargs.get("show_fig",True)

    seed = kwargs.get("seed",42)

    cov_scale = kwargs.get("cov_scale",1.0)

    point_overlap_allowed = kwargs.get("point_overlap_allowed",True)

    color_style = kwargs.get("color_style",None)

    dense_cmap = kwargs.get("dense_cmap","viridis")
    dense_resolution = kwargs.get("dense_resolution",200)
    gp_cont_resolution = kwargs.get("gp_cont_resolution", dense_resolution)
    gp_cont_alpha = kwargs.get("gp_cont_alpha", 0.6)

    no_color_outside_circle = kwargs.get("no_color_outside_circle",False)

    print_points = kwargs.get("print_points",False)

    # ------------------------------------------------
    # DATA INPUTS
    # ------------------------------------------------
    centroids = kwargs.get("irel_unobserved_centroids",[])
    covs = kwargs.get("irel_unobserved_cluster_covs",[])
    n_points = kwargs.get("irel_unobserved_n_points_per_cluster",[])

    rel_unobserved_locs = kwargs.get("rel_unobserved_locs",[])

    query_loc = kwargs.get("query_loc",None)
    query_rel_circle_radius = kwargs.get("query_rel_circle_radius",None)

    observed_points = kwargs.get("observed_points",[])

    max_rel = kwargs.get("max_rel",1)

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

    default_obs_irel=dict(marker="o",s=40,linewidths=1.2,edgecolors="black",label="Obs Irel")
    default_obs_rel=dict(marker="^",s=40,linewidths=1.2,edgecolors="black",label="Obs Rel")
    default_obs_query=dict(marker="s",s=40,linewidths=1.2,edgecolors="black",label="Obs Query")

    default_circle=dict(edgecolor="grey",linestyle="--",linewidth=1.2,fill=False)

    irel_marker_kwargs={**default_irel_marker,**kwargs.get("irel_marker_kwargs",{})}
    rel_marker_kwargs={**default_rel_marker,**kwargs.get("rel_marker_kwargs",{})}
    query_marker_kwargs={**default_query_marker,**kwargs.get("query_marker_kwargs",{})}

    obs_irel_marker_kwargs={**default_obs_irel,**kwargs.get("obs_irel_marker_kwargs",{})}
    obs_rel_marker_kwargs={**default_obs_rel,**kwargs.get("obs_rel_marker_kwargs",{})}
    obs_query_marker_kwargs={**default_obs_query,**kwargs.get("obs_query_marker_kwargs",{})}

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
    if color_style in ["gp_points", "gp_cont"]:

        if len(observed_points)==0:
            raise ValueError("gp_points requires observed_points")

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
    # GP CONTINUOUS HEATMAP
    # ------------------------------------------------
    if color_style=="gp_cont":
        xs=np.linspace(x_lims[0],x_lims[1],gp_cont_resolution)
        ys=np.linspace(y_lims[0],y_lims[1],gp_cont_resolution)

        Xg,Yg=np.meshgrid(xs,ys)
        grid_points = np.column_stack([Xg.ravel(), Yg.ravel()])

        model, likelihood = gp
        grid_points_t = torch.tensor(grid_points, dtype=torch.float32)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            gp_mean = likelihood(model(grid_points_t)).mean.detach().cpu().numpy()

        gp_mean = gp_mean.reshape(gp_cont_resolution, gp_cont_resolution)
        gp_mean = np.clip(gp_mean, 0, max_rel)
        gp_mean = gp_mean / max_rel

        ax.imshow(
            gp_mean,
            extent=[*x_lims,*y_lims],
            origin="lower",
            cmap=dense_cmap,
            alpha=gp_cont_alpha,
            aspect="auto",
            interpolation="bicubic",
        )

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

    accepted_points=np.array(accepted_points)
    rel_pts=np.array(rel_unobserved_locs) if len(rel_unobserved_locs)>0 else np.empty((0,2))

    printable=[]

    # ------------------------------------------------
    # POINT COLOR MODES
    # ------------------------------------------------
    if color_style=="point_color" and query_loc is not None:

        all_points=np.vstack([accepted_points,rel_pts])

        dists=np.linalg.norm(all_points-query_loc,axis=1)
        max_dist=np.max(dists) if np.max(dists)!=0 else 1

        scores=1-(dists/max_dist)
        colors=cmap(scores)

        idx=0

        if len(accepted_points)>0:
            n=len(accepted_points)
            ax.scatter(accepted_points[:,0],accepted_points[:,1],
                       edgecolors=colors[idx:idx+n],facecolors="none",
                       **{k:v for k,v in irel_marker_kwargs.items()
                          if k not in ["edgecolors","facecolors"]})

            for p,c,scr in zip(accepted_points,colors[idx:idx+n],scores[idx:idx+n]):
                printable.append((scr,tuple(np.round(p,4)),to_hex(c)))

            idx+=n

        if len(rel_pts)>0:
            n=len(rel_pts)
            ax.scatter(rel_pts[:,0],rel_pts[:,1],
                       edgecolors=colors[idx:idx+n],facecolors="none",
                       **{k:v for k,v in rel_marker_kwargs.items()
                          if k not in ["edgecolors","facecolors"]})

            for p,c,scr in zip(rel_pts,colors[idx:idx+n],scores[idx:idx+n]):
                printable.append((scr,tuple(np.round(p,4)),to_hex(c)))

    elif color_style=="gp_points":

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
            ax.scatter(accepted_points[:,0],accepted_points[:,1],
                       edgecolors=colors[idx:idx+n],facecolors="none",
                       **{k:v for k,v in irel_marker_kwargs.items()
                          if k not in ["edgecolors","facecolors"]})

            for p,c,scr in zip(accepted_points,colors[idx:idx+n],scores[idx:idx+n]):
                printable.append((scr,tuple(np.round(p,4)),to_hex(c)))

            idx+=n

        if len(rel_pts)>0:
            n=len(rel_pts)
            ax.scatter(rel_pts[:,0],rel_pts[:,1],
                       edgecolors=colors[idx:idx+n],facecolors="none",
                       **{k:v for k,v in rel_marker_kwargs.items()
                          if k not in ["edgecolors","facecolors"]})

            for p,c,scr in zip(rel_pts,colors[idx:idx+n],scores[idx:idx+n]):
                printable.append((scr,tuple(np.round(p,4)),to_hex(c)))

    elif color_style=="gp_cont":

        if len(accepted_points)>0:
            ax.scatter(
                accepted_points[:,0],
                accepted_points[:,1],
                edgecolors="grey",
                facecolors="none",
                **{k:v for k,v in irel_marker_kwargs.items()
                   if k not in ["edgecolors","facecolors"]}
            )

        if len(rel_pts)>0:
            ax.scatter(
                rel_pts[:,0],
                rel_pts[:,1],
                edgecolors="grey",
                facecolors="none",
                **{k:v for k,v in rel_marker_kwargs.items()
                   if k not in ["edgecolors","facecolors"]}
            )

    else:

        if len(accepted_points)>0:
            ax.scatter(accepted_points[:,0],accepted_points[:,1],**irel_marker_kwargs)

        if len(rel_pts)>0:
            ax.scatter(rel_pts[:,0],rel_pts[:,1],**rel_marker_kwargs)

    # ------------------------------------------------
    # OBSERVED POINTS (STYLE BY TYPE, COLOR BY LABEL)
    # ------------------------------------------------
    for (pt,val,ptype) in observed_points:

        score=np.clip(val,0,max_rel)/max_rel
        color=cmap(score)

        if ptype=="irel":
            style=obs_irel_marker_kwargs
        elif ptype=="rel":
            style=obs_rel_marker_kwargs
        elif ptype=="query":
            style=obs_query_marker_kwargs
        else:
            raise ValueError("ptype must be 'irel','rel','query'")

        style=dict(style)
        style.pop("color",None)
        style.pop("c",None)

        ax.scatter(pt[0],pt[1],c=[color],**style)

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
        ax.scatter([query_loc[0]],[query_loc[1]],**query_marker_kwargs)

    if query_loc is not None and query_rel_circle_radius is not None:
        circle=Circle(query_loc,query_rel_circle_radius,**query_circle_kwargs)
        ax.add_patch(circle)

    # ------------------------------------------------
    # LEGEND
    # ------------------------------------------------
    if inc_legend:

        if legend_coords is None:
            ax.legend(prop={"family":font,"size":fontsize})
        else:
            ax.legend(prop={"family":font,"size":fontsize},
                      bbox_to_anchor=legend_coords,
                      loc="upper left")

    if show_fig:
        plt.show()

    return fig