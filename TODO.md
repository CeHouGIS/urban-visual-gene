# TODO

## Rectangular-panorama Feature-MAE adaptation (deferred)

- [ ] Fine-tune or retrain Feature-MAE for DINOv3 tokens produced from a full
  `896x224` panorama rather than relying only on the frozen model trained from
  square `224x224` views.
- Keep DINOv3 frozen and retain the current eight circular `14x14` windows
  extracted from each `14x56` panorama token map.
- Mix original single-view windows and panorama-derived windows during
  training so the learned 512D space remains comparable with the existing
  experiment.
- Include circular horizontal shifts during training so the arbitrary 0/360
  cut does not become a privileged boundary.
- Validate against the current frozen-MAE baseline using reconstruction loss,
  seam continuity, winner-map stability, small-object retention, and semantic
  interpretability before replacing any published result.

The current full-city inference intentionally keeps Feature-MAE frozen. It is
a baseline prediction run, not the deferred adaptation experiment above.

## Rectangular-panorama inference cleanup

- [ ] Accept four direction images when they share the same square resolution,
  rather than requiring every source image to be exactly `640x640`; construct
  the `4:1` panorama at its native size and continue resizing it to `896x224`
  before DINOv3 inference.
- [ ] Retry the 53 Vienna panoramas currently marked as failed because their
  four source directions are `400x400`, updating the existing city shard in
  place without recomputing successful panoramas.
- [ ] Re-run global QA after the retry and require all status arrays to contain
  only successful entries before declaring the 30-city prediction complete.
