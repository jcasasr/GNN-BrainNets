The script is plot_glass_brain_top_rois.py. It reads your weights CSV, ranks the ROIs by mean |W|, and draws the top 10 as markers on a nilearn glass brain, with color and size proportional to importance.

## bash
> pip install nilearn
> python plot_glass_brain_top_rois.py Step-7-1_MM_FPGCN-0-0_lr0_0001_wd0_0005_noDA_weights.csv \
    --top-k 10 --out top_rois_glass_brain.png

## Options you may want:

`--modality FA|GM|RS` ranks the ROIs using only that modality’s columns.
`--mode lyrz` shows left and right lateral views plus the top view, which is useful for seeing the bilateral precuneus and hippocampus. The default is ortho.
`--coords my_centroids.csv` overrides the built-in coordinates (columns ROI,x,y,z).

The coordinates are approximate. Your CSV has no spatial information, so I hard-coded representative MNI centroids for the Desikan-Killiany and subcortical ROIs that are likely to rank high. They are fine for a quick visualization, but for a paper you should replace them with the centroids of the atlas you used. The script includes a helper, centroids_from_atlas(labels_img, label_to_roi), that computes them from a labeled NIfTI in MNI space using find_parcellation_cut_coords. Save the result as a CSV and pass it with --coords.

## Some practical notes:

No labels on the plot. plot_markers doesn’t write ROI names, so the figure shows locations and importance only. Name the regions in the caption or refer to your ROI table.
Ranking uses all 24 metrics. The resting-state local efficiency column is nearly uniform across ROIs, so it affects all ROIs similarly and shouldn’t change the ranking much. If you want to check, run with --modality FA or GM too.
Version differences. The script uses plot_markers, which exists in recent nilearn releases (0.10 and later). If your version complains about a keyword, tell me the error and I’ll adjust it.