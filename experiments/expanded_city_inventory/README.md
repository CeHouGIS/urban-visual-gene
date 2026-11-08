# Google street-view coverage for 300 cities

`render_google_coverage.py` maps the 50 m footprints of panorama IDs with valid coordinates and an image-availability flag. `render_google_road_coverage.py` downloads Overture Maps transportation segments (release `2026-10-21.0`), keeps driveable road classes, clips them to the panorama extent, and subdivides them into straight sections no longer than 50 m. A section is green when its midpoint is within 50 m of a geolocated panorama ID; otherwise it is red. Road percentages are weighted by section length.

The red class means no **geolocated, metadata-available** panorama was found within the threshold. It does not prove that no JPG exists on disk. The metadata IDs have not been matched to disk JPGs one by one, and city extents are the observed panorama bounding boxes rather than municipal boundaries. Cities with fewer than 1% geolocated disk panoids receive no road classification.

The runners checkpoint each city under `results/`. Re-running resumes completed cities. To build the webpage from an existing road checkpoint without recomputing panorama footprints:

```bash
python experiments/expanded_city_inventory/render_google_coverage.py \
  --output experiments/expanded_city_inventory/results/google_50m_coverage_recovered \
  --road-report experiments/expanded_city_inventory/results/google_road_coverage/progress.csv \
  --refresh-site-only
```

The resulting `index.html`, `road-maps/`, and `city_road_coverage.csv` form the web atlas. A completed run also writes `results/google_road_coverage/city_road_coverage.csv`.
