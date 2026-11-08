# Google street-view coverage for 300 cities

`match_ghsl_boundaries.py` downloads the [GHS-UCDB GLOBE R2024A V1.2 package](https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_UCDB_GLOBE_R2024A/GHS_UCDB_GLOBE_R2024A/V1-2/GHS_UCDB_GLOBE_R2024A_V1_2.zip) when needed, then matches the 300 study cities to its urban-centre polygons. The primary match requires the configured city centre to fall inside exactly one GHSL polygon. An exact normalized city-name match within 10 km is allowed for small centre-point offsets and recorded in the audit. Unmatched cities receive no substitute boundary. Both image footprints and Overture roads are clipped to the matched GHSL polygon. `render_google_coverage.py` maps 50 m footprints of panorama IDs with valid coordinates and an image-availability flag. `render_google_road_coverage.py` downloads Overture Maps transportation segments (release `2026-10-21.0`), keeps driveable road classes, clips them to the same GHSL boundary, and subdivides them into straight sections no longer than 50 m. A section is green when its midpoint is within 50 m of a geolocated panorama ID; otherwise it is red. Road percentages are weighted by section length.

The red class means no **geolocated, metadata-available** panorama was found within the threshold. It does not prove that no JPG exists on disk. The metadata IDs have not been matched to disk JPGs one by one. Cities without geolocated panoramas still show roads inside their matched GHSL boundary, classified as uncovered.

The runners checkpoint each city under `results/`. Re-running resumes completed cities. To build the webpage from an existing road checkpoint without recomputing panorama footprints:

```bash
python experiments/expanded_city_inventory/render_google_coverage.py \
  --output experiments/expanded_city_inventory/results/google_50m_coverage_ghsl \
  --road-report experiments/expanded_city_inventory/results/google_road_coverage_ghsl/city_road_coverage.csv \
  --refresh-site-only
```

The resulting `index.html`, `maps/`, `road-maps/`, and `city_road_coverage.csv` form the web atlas. A completed run also writes `results/google_road_coverage_ghsl/city_road_coverage.csv`.
