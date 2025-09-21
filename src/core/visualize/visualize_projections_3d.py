import matplotlib.pyplot as plt
import mplcursors
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - required for 3D projection


def get_annotation_text(tokenizer, token):
    return f"{tokenizer.decode(token)}"


def visualize_projections_3d(
    hidden_space_by_language,
    token_map_by_language,
    projections,
    tokenizer,
    target_layers=None,
):
    languages = list(hidden_space_by_language.keys())

    n_layers = len(projections[languages[0]])

    # Note: tokens_by_artist and handles are re-initialized per figure (per layer)

    if target_layers is None:
        # choose first, middle, last layer (guard small n_layers)
        mid = max(0, min(n_layers - 1, n_layers // 2))
        target_layers = sorted(set([0, mid, n_layers - 1]))

    def _update_alpha(ax, scatter_handles, near_opacity=0.9, far_opacity=0.2):
        """Update per-point alpha based on distance to the viewer direction.

        We approximate depth using the current view angles (elev, azim) and
        compute the dot product with the view direction after centering points.
        """
        if not scatter_handles:
            return

        # Compute view direction vector from current elevation/azimuth
        elev = np.deg2rad(ax.elev)
        azim = np.deg2rad(ax.azim)
        # In matplotlib, azim is the rotation around z, elev is tilt from xy plane
        # View direction (unit vector) pointing from origin toward the camera
        v = np.array(
            [
                np.cos(elev) * np.cos(azim),
                np.cos(elev) * np.sin(azim),
                np.sin(elev),
            ]
        )

        # Gather all points across all artists to compute a common center and min/max
        all_pts = []
        per_artist_pts = []
        for h in scatter_handles:
            try:
                xs, ys, zs = h._offsets3d  # type: ignore[attr-defined]
            except Exception:
                # If unavailable, skip alpha update for this handle
                per_artist_pts.append(None)
                continue
            pts = np.column_stack([np.asarray(xs), np.asarray(ys), np.asarray(zs)])
            per_artist_pts.append(pts)
            all_pts.append(pts)

        if not all_pts:
            return

        all_pts = np.vstack(all_pts)
        center = all_pts.mean(axis=0, keepdims=True)

        # Compute depth scores (larger => closer to viewer along v)
        dots = []
        for pts in per_artist_pts:
            if pts is None:
                dots.append(None)
            else:
                dots.append(((pts - center) @ v))

        # Normalize across all points
        concat = np.concatenate([d for d in dots if d is not None])
        d_min, d_max = float(concat.min()), float(concat.max())
        denom = (d_max - d_min) if (d_max - d_min) > 1e-12 else 1.0

        for h, d in zip(scatter_handles, dots):
            if d is None:
                continue
            norm = (d - d_min) / denom  # 0 = farthest, 1 = nearest (by our convention)
            alphas = far_opacity + norm * (near_opacity - far_opacity)

            # Update facecolors while preserving original RGB
            fc = h.get_facecolors()
            n = len(d)
            if fc is None or len(fc) == 0:
                # default to matplotlib cycle color
                color = np.array([[0.12156863, 0.46666667, 0.70588235, 1.0]])
                fc = np.tile(color, (n, 1))
            elif len(fc) == 1:
                fc = np.tile(fc, (n, 1))
            else:
                # ensure we have one color per point
                if len(fc) != n:
                    base = fc[0]
                    fc = np.tile(base, (n, 1))

            fc[:, 3] = np.clip(alphas, 0.0, 1.0)
            h.set_facecolors(fc)
            ec = h.get_edgecolors()
            if ec is not None and len(ec) > 0:
                if len(ec) == 1:
                    ec = np.tile(ec, (n, 1))
                elif len(ec) != n:
                    ec = np.tile(ec[0], (n, 1))
                ec[:, 3] = np.clip(alphas, 0.0, 1.0)
                h.set_edgecolors(ec)

    for layer in target_layers:  # Plot only a few layers for clarity
        tokens_by_artist = {}
        handles = []
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")
        for lang in languages:
            # Get projections for this language and layer
            proj = projections[lang][layer]

            if proj.shape[1] < 3:
                raise ValueError(
                    f"visualize_projections_3d requires >= 3 PCA components, got {proj.shape[1]} for language {lang}, layer {layer}"
                )

            # Plot the first three components in 3D
            h = ax.scatter(
                proj[:, 0],
                proj[:, 1],
                proj[:, 2],
                label=f"{lang}",
                s=60,
                depthshade=False,  # we control alpha ourselves
            )
            ax.set_xlabel("PCA Component 1", labelpad=20)
            ax.set_ylabel("PCA Component 2", labelpad=20)
            ax.set_zlabel("PCA Component 3", labelpad=20)
            ax.grid(True, alpha=0.3)

            tokens_by_artist[h] = token_map_by_language[lang]
            handles.append(h)

        fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
        plt.subplots_adjust(top=0.9, bottom=0.1)

        # Initialize alpha based on current view and update on draw (e.g., rotate)
        _update_alpha(ax, handles)

        def _on_draw(event):
            if event.canvas is fig.canvas:
                _update_alpha(ax, handles)

        fig.canvas.mpl_connect("draw_event", _on_draw)

        cursor = mplcursors.cursor(handles, hover=True)

        @cursor.connect("add")
        def on_add(sel):
            token = tokens_by_artist[sel.artist][sel.index]
            sel.annotation.set_text(get_annotation_text(tokenizer, token))
            sel.annotation.get_bbox_patch().set(facecolor="lightblue", alpha=0.7)
            sel.annotation.arrow_patch.set(arrowstyle="->", facecolor="black", alpha=0.5)

        plt.show()
