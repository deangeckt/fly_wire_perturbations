import numpy as np
from matplotlib.patches import RegularPolygon

##############################
###### Data utils
##############################

def add_node_info_to_edges(edges_df, nodes_coords):
    """
    edges_df: columns [pre_root_id, post_root_id, syn_count, ...]
    nodes_coords: columns [fafb_783_id, cell_type, column_id, p, q, side(optional)]
    """
    node_cols = ["fafb_783_id", "cell_type", "column_id", "p", "q"]
    if "side" in nodes_coords.columns:
        node_cols.append("side")

    nodes = nodes_coords[node_cols].drop_duplicates("fafb_783_id").copy()

    out = edges_df.merge(
        nodes.add_prefix("pre_"),
        left_on="pre_root_id", right_on="pre_fafb_783_id",
        how="inner"
    ).merge(
        nodes.add_prefix("post_"),
        left_on="post_root_id", right_on="post_fafb_783_id",
        how="inner"
    )

    # intracolumnar only
    out = out[out["pre_column_id"] == out["post_column_id"]].copy()

    # keep one column_id coordinate set (they're the same by construction)
    out["column_id"] = out["pre_column_id"]
    out["p"] = out["pre_p"]
    out["q"] = out["pre_q"]

    return out

def build_nodes_coords_for_side(side, fafb_meta, codex_meta,root_ids):
    cols = ["root_id", "column_id", "x", "y", "p", "q"] 
    df_col_nodes = fafb_meta[fafb_meta["fafb_783_id"].isin(root_ids)].copy()
    df_col_nodes = df_col_nodes[df_col_nodes["side"] == side].reset_index(drop=True)

    nodes_coords = df_col_nodes.merge(
        codex_meta[cols],
        left_on="fafb_783_id",
        right_on="root_id",
        how="inner"
    ).drop(columns=["root_id", "super_class", "cell_class", "cell_sub_class"])
    return nodes_coords


def build_edges_for_nodes(nodes_coords, edge_df):
    keep = nodes_coords["fafb_783_id"]
    edges_df = edge_df[edge_df["pre_root_id"].isin(keep) & edge_df["post_root_id"].isin(keep)].reset_index(drop=True)
    edges_annot = add_node_info_to_edges(edges_df, nodes_coords).reset_index(drop=True)
    return edges_annot


##############################
###### Viz utils
##############################

def pq_to_xy(p, q, hexsize=1.0, rotate=-np.pi/6):
    """
    Map Zhao (p,q) coords to 2D xy for plotting.
    Matches your Julia convention: (p,q)->(q,-p) then rotate by -30°.
    """
    a = q
    b = -p

    # pointy-top axial -> pixel
    x = hexsize * np.sqrt(3) * (a + b/2)
    y = hexsize * 1.5 * b

    if rotate is not None:
        c, s = np.cos(rotate), np.sin(rotate)
        xr = c*x - s*y
        yr = s*x + c*y
        return xr, yr

    return x, y

def plot_eye(ax, nodes_coords, edges_annot, ctype1, ctype2, norm, cmap, *,
             metric= "syn_count", mirror_x=False, mirror_y=True, x_offset=0.0, y_offset=0.0, hexsize=1.0, plot_col_ids=False):
    type_pair = edges_annot[(edges_annot["pre_cell_type"]==ctype1) &
                            (edges_annot["post_cell_type"]==ctype2)].copy()

    # per-column syn count (enforce 0/1 edge per column; change later if needed)
    syn_by_col = type_pair.groupby("column_id")[metric].agg(list).to_dict()

    def nsyns_for_col(cid):
        vals = syn_by_col.get(cid, [])
        if len(vals) == 0:
            return 0
        if len(vals) == 1:
            return int(vals[0])
        if len(vals) > 1:
            print(f"Column {cid} has {len(vals)} edges; using mean.")
            return int(np.mean(vals))

    colpos = nodes_coords[["column_id", "p", "q"]].drop_duplicates("column_id").copy()
    x, y = pq_to_xy(colpos["p"].to_numpy(), colpos["q"].to_numpy(), hexsize=hexsize)

    if mirror_x:
        x = -x
    if mirror_y:
        y = -y
    x = x + x_offset
    y = y + y_offset

    for xi, yi, cid in zip(x, y, colpos["column_id"].to_numpy()):
    # for xi, yi, cid in zip(x, y, type_pair['column_id'].unique()):
        nsyns = nsyns_for_col(cid)
        color = cmap(norm(nsyns))
        ax.add_patch(
            RegularPolygon(
                (xi, yi), numVertices=6, radius=hexsize, orientation=np.pi/6,
                facecolor=color, edgecolor="black", linewidth=0.4
            )
        )
        if plot_col_ids:
            ax.text(xi, yi, str(cid), fontsize=6, ha="center", va="center", color="black")

    return x, y

