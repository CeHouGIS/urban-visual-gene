# Ten image graphs and 512D activations

This visualization uses ten centroid-near images from the compact graph
experiment: the first representative of A01–A08, plus the second
representative of A01 and A02. All ten panorama IDs are unique.

Each row in the main atlas contains four linked views of the same directional
image:

1. **Photo + activation overlay.** The 14×14 F-winner map is enlarged with
   nearest-neighbour interpolation and blended over the original photograph.
2. **14×14 F map.** Each patch-level Feature-MAE winning dimension is mapped
   through the frozen D→F hierarchy.
3. **Image graph.** Node size is the F-category patch area. Edge width is the
   number of horizontal/vertical patch boundaries shared by two different F
   categories. For readability the panel shows the 12 largest nodes, the 18
   strongest nonzero edges, and any extra edge endpoints.
4. **512D spatial support.** This is the number of the 196 patches for which
   each dimension is the top-1 winner. These counts always sum to 196.

The continuous 512D activation-strength view is kept in the separate 10×512
heatmap and in each high-resolution individual sample figure. For every
Feature-MAE dimension, that score is the mean of its 20 strongest patch
responses and is shown as a cached global standardized score.

The activation-strength and winner-count panels answer different questions.
A dimension may respond strongly to several patches without becoming their
top-1 winner, while a broadly dominant dimension can have many winner patches
without an unusually high standardized activation.

Generated outputs:

- `paper/figures/supplementary/compact_graph_examples/Fig_Compact_Graph_10_Image_Examples.png`
- `paper/figures/supplementary/compact_graph_examples/Fig_Compact_Graph_10_Image_Activation_Overlays.png`
- `paper/figures/supplementary/compact_graph_examples/Fig_Compact_Graph_10x512_Activations.png`
- Ten high-resolution individual sample figures under the `individual/`
  subdirectory.
- Numeric source tables under
  `paper/data/image_graph_compact_archetypes/examples/`.
