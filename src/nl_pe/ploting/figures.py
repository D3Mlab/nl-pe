import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from matplotlib.colors import to_hex
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel


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

    obs_points = kwargs.get("obs_points",[])
    obs_vals = kwargs.get("obs_vals",[])
    max_rel = kwargs.get("max_rel",1)

    # ------------------------------------------------
    # GP PARAMS
    # ------------------------------------------------
    gp_ls = kwargs.get("gp_ls",0.2)
    gp_os = kwargs.get("gp_os",1.0)
    gp_noise = kwargs.get("gp_noise",1e-6)

    # ------------------------------------------------
    # MARKER DEFAULTS
    # ------------------------------------------------
    default_irel_marker_kwargs=dict(
        marker="o",facecolors="none",edgecolors="grey",s=30,linewidths=1.2,label="Irel. docs"
    )

    default_rel_marker_kwargs=dict(
        marker="^",facecolors="none",edgecolors="grey",s=30,linewidths=1.2,label="Rel. docs"
    )

    default_query_marker_kwargs=dict(
        marker="x",color="black",s=30,linewidths=1.5,label="Query"
    )

    default_query_circle_kwargs=dict(
        edgecolor="grey",linestyle="--",linewidth=1.2,fill=False
    )

    default_obs_marker_kwargs=dict(
        marker='o',s=40,linewidths=1.2,edgecolors='black',label="Obs"
    )

    irel_marker_kwargs={**default_irel_marker_kwargs,**kwargs.get("irel_marker_kwargs",{})}
    rel_marker_kwargs={**default_rel_marker_kwargs,**kwargs.get("rel_marker_kwargs",{})}
    query_marker_kwargs={**default_query_marker_kwargs,**kwargs.get("query_marker_kwargs",{})}
    query_circle_kwargs={**default_query_circle_kwargs,**kwargs.get("query_rel_circle_kwargs",{})}
    obs_marker_kwargs={**default_obs_marker_kwargs,**kwargs.get("obs_marker_kwargs",{})}

    s=irel_marker_kwargs.get("s",30)

    np.random.seed(seed)

    # ------------------------------------------------
    # FIGURE
    # ------------------------------------------------
    fig,ax=plt.subplots(figsize=(fig_w,fig_h),dpi=dpi)

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

    cmap=plt.get_cmap(dense_cmap)

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

        ax.imshow(
            Z,
            extent=[*x_lims,*y_lims],
            origin="lower",
            cmap=dense_cmap,
            alpha=0.6,
            aspect="auto"
        )

    # ------------------------------------------------
    # GP MODEL
    # ------------------------------------------------
    gp=None
    if color_style=="gp_points":

        if len(obs_points)==0:
            raise ValueError("gp_points requires obs_points and obs_vals")

        kernel=ConstantKernel(gp_os)*RBF(length_scale=gp_ls)
        gp=GaussianProcessRegressor(kernel=kernel,alpha=gp_noise)

        gp.fit(np.array(obs_points),np.array(obs_vals))

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
    # POINT COLOR MODE
    # ------------------------------------------------
    if color_style=="point_color" and query_loc is not None:

        all_points=np.vstack([accepted_points,rel_pts])

        if len(all_points)>0:

            dists=np.linalg.norm(all_points-query_loc,axis=1)
            max_dist=np.max(dists) if np.max(dists)!=0 else 1

            scores=1-(dists/max_dist)
            colors=cmap(scores)

            idx=0

            if len(accepted_points)>0:

                n=len(accepted_points)

                ax.scatter(
                    accepted_points[:,0],
                    accepted_points[:,1],
                    edgecolors=colors[idx:idx+n],
                    facecolors="none",
                    **{k:v for k,v in irel_marker_kwargs.items()
                       if k not in ["edgecolors","facecolors"]}
                )

                for p,c,s in zip(accepted_points,colors[idx:idx+n],scores[idx:idx+n]):
                    printable.append(("irel",s,tuple(np.round(p,4)),to_hex(c)))

                idx+=n

            if len(rel_pts)>0:

                n=len(rel_pts)

                ax.scatter(
                    rel_pts[:,0],
                    rel_pts[:,1],
                    edgecolors=colors[idx:idx+n],
                    facecolors="none",
                    **{k:v for k,v in rel_marker_kwargs.items()
                       if k not in ["edgecolors","facecolors"]}
                )

                for p,c,s in zip(rel_pts,colors[idx:idx+n],scores[idx:idx+n]):
                    printable.append(("rel",s,tuple(np.round(p,4)),to_hex(c)))

    # ------------------------------------------------
    # GP COLOR MODE
    # ------------------------------------------------
    elif color_style=="gp_points":

        all_points=np.vstack([accepted_points,rel_pts])

        if len(all_points)>0:

            preds=gp.predict(all_points)
            preds=np.clip(preds,0,max_rel)

            scores=preds/max_rel
            colors=cmap(scores)

            idx=0

            if len(accepted_points)>0:

                n=len(accepted_points)

                ax.scatter(
                    accepted_points[:,0],
                    accepted_points[:,1],
                    edgecolors=colors[idx:idx+n],
                    facecolors="none",
                    **{k:v for k,v in irel_marker_kwargs.items()
                       if k not in ["edgecolors","facecolors"]}
                )

                for p,c,s in zip(accepted_points,colors[idx:idx+n],scores[idx:idx+n]):
                    printable.append(("irel_gp",s,tuple(np.round(p,4)),to_hex(c)))

                idx+=n

            if len(rel_pts)>0:

                n=len(rel_pts)

                ax.scatter(
                    rel_pts[:,0],
                    rel_pts[:,1],
                    edgecolors=colors[idx:idx+n],
                    facecolors="none",
                    **{k:v for k,v in rel_marker_kwargs.items()
                       if k not in ["edgecolors","facecolors"]}
                )

                for p,c,s in zip(rel_pts,colors[idx:idx+n],scores[idx:idx+n]):
                    printable.append(("rel_gp",s,tuple(np.round(p,4)),to_hex(c)))

    else:

        if len(accepted_points)>0:
            ax.scatter(accepted_points[:,0],accepted_points[:,1],**irel_marker_kwargs)

        if len(rel_pts)>0:
            ax.scatter(rel_pts[:,0],rel_pts[:,1],**rel_marker_kwargs)

    # ------------------------------------------------
    # OBSERVED POINTS
    # ------------------------------------------------
    if len(obs_points)>0:

        obs_pts=np.array(obs_points)

        vals=np.clip(np.array(obs_vals),0,max_rel)
        scores=vals/max_rel

        colors=cmap(scores)

        obs_kwargs=dict(obs_marker_kwargs)
        obs_kwargs.pop("color",None)
        obs_kwargs.pop("c",None)

        ax.scatter(obs_pts[:,0],obs_pts[:,1],c=colors,**obs_kwargs)

        for p,v,c in zip(obs_pts,vals,colors):
            printable.append(("obs",v,tuple(np.round(p,4)),to_hex(c)))

    # ------------------------------------------------
    # PRINT
    # ------------------------------------------------
    if print_points:

        printable.sort(key=lambda x:x[1] if x[1] is not None else -np.inf,reverse=True)

        for typ,score,pt,col in printable:
            print(f"{typ:8s} {pt} score={score:.4f} color={col}")

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