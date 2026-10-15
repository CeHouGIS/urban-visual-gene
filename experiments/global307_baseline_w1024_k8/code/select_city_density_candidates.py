"""Rank downloaded Google Street View city folders by GHSL urban-centre density."""
from __future__ import annotations

import argparse
import unicodedata
from pathlib import Path

import geopandas as gpd
import pandas as pd


COUNTRY_ALIASES = {
    "CzechRepublic": "Czechia",
    "Korea": "Republic of Korea",
    "Russia": "Russian Federation",
    "SouthAfrica": "South Africa",
    "SouthKorea": "Republic of Korea",
    "SriLanka": "Sri Lanka",
    "UAE": "United Arab Emirates",
    "US": "United States of America",
    "UnitedKingdom": "United Kingdom",
    "Swaziland": "Eswatini",
    "Bolivia": "Bolivia (Plurinational State of)",
    "Curaçao": "Curacao",
    "Palestina": "State of Palestine",
}
GHSL_LAYER = "GHSL_UCDB_THEME_GENERAL_CHARACTERISTICS_GLOBE_R2024A"


def normalize(value: object) -> str:
    return "".join(
        char
        for char in unicodedata.normalize("NFKD", str(value)).casefold()
        if char.isalnum()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata-summary",
        type=Path,
        default=Path("/host/root/mnt/nas/huangyj/GoogleSV/metadata_summary.csv"),
    )
    parser.add_argument("--ghsl", type=Path, required=True, help="GHSL UCDB R2024A GeoPackage")
    parser.add_argument(
        "--output", type=Path, default=Path("configs/formal_city_density_candidates.csv")
    )
    parser.add_argument("--limit", type=int, default=350)
    parser.add_argument("--nearest-km", type=float, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = pd.read_csv(args.metadata_summary)
    source = source[source["#images"].fillna(0) > 0].copy().reset_index(drop=True)
    ghsl = gpd.read_file(
        args.ghsl,
        layer=GHSL_LAYER,
        columns=[
            "ID_UC_G0",
            "GC_UCN_MAI_2025",
            "GC_UCN_LIS_2025",
            "GC_CNT_UNN_2025",
            "GC_UCA_KM2_2025",
            "geometry",
        ],
    ).reset_index(drop=True)
    ghsl["_country"] = ghsl["GC_CNT_UNN_2025"].map(normalize)
    ghsl["_name_main"] = ghsl["GC_UCN_MAI_2025"].map(normalize)
    ghsl["_name_local"] = ghsl["GC_UCN_LIS_2025"].map(normalize)
    source["_country"] = source["country"].map(
        lambda value: normalize(COUNTRY_ALIASES.get(value, value))
    )
    source["_name"] = source["city"].map(normalize)
    ghsl_names: dict[tuple[str, str], list[int]] = {}
    for index, row in ghsl.iterrows():
        for name in {row["_name_main"], row["_name_local"]}:
            ghsl_names.setdefault((row["_country"], name), []).append(index)

    result = source.copy()
    result["ghsl_city"] = None
    result["ghsl_uc_id"] = None
    result["ghsl_country"] = None
    result["ghsl_area_km2"] = None
    result["ghsl_match"] = None
    result["ghsl_match_distance_km"] = None

    # Exact names are preferred. A source centre disambiguates repeated names.
    for index, row in source.iterrows():
        matches = ghsl_names.get((row["_country"], row["_name"]), [])
        if not matches:
            continue
        if len(matches) > 1 and pd.notna(row["center_lat"]) and pd.notna(row["center_lon"]):
            point = gpd.GeoSeries(
                [gpd.points_from_xy([row["center_lon"]], [row["center_lat"]])[0]], crs=4326
            ).to_crs(ghsl.crs).iloc[0]
            chosen = min(matches, key=lambda item: point.distance(ghsl.geometry.iloc[item].centroid))
        elif len(matches) == 1:
            chosen = matches[0]
        else:
            continue
        result.loc[index, "ghsl_city"] = ghsl.loc[chosen, "GC_UCN_MAI_2025"]
        result.loc[index, "ghsl_uc_id"] = ghsl.loc[chosen, "ID_UC_G0"]
        result.loc[index, "ghsl_country"] = ghsl.loc[chosen, "GC_CNT_UNN_2025"]
        result.loc[index, "ghsl_area_km2"] = ghsl.loc[chosen, "GC_UCA_KM2_2025"]
        result.loc[index, "ghsl_match"] = "exact_name"

    # Where names differ, use the GHSL polygon containing the source city centre.
    has_point = source["center_lat"].notna() & source["center_lon"].notna()
    pending = result["ghsl_area_km2"].isna() & has_point
    points = gpd.GeoDataFrame(
        source.loc[pending, ["_country", "center_lon", "center_lat"]].copy(),
        geometry=gpd.points_from_xy(
            source.loc[pending, "center_lon"], source.loc[pending, "center_lat"]
        ),
        crs="EPSG:4326",
    ).to_crs(ghsl.crs)
    inside = gpd.sjoin(
        points,
        ghsl[["ID_UC_G0", "GC_UCN_MAI_2025", "GC_CNT_UNN_2025", "GC_UCA_KM2_2025", "_country", "geometry"]],
        how="left",
        predicate="within",
    )
    for index, row in inside.iterrows():
        if pd.isna(row["GC_UCA_KM2_2025"]) or row["_country_left"] != row["_country_right"]:
            continue
        result.loc[index, "ghsl_city"] = row["GC_UCN_MAI_2025"]
        result.loc[index, "ghsl_uc_id"] = row["ID_UC_G0"]
        result.loc[index, "ghsl_country"] = row["GC_CNT_UNN_2025"]
        result.loc[index, "ghsl_area_km2"] = row["GC_UCA_KM2_2025"]
        result.loc[index, "ghsl_match"] = "centre_inside_boundary"
        result.loc[index, "ghsl_match_distance_km"] = 0.0

    # A limited nearest-centre fallback recovers border and centroid offsets.
    pending = result["ghsl_area_km2"].isna() & has_point
    points = gpd.GeoDataFrame(
        source.loc[pending, ["_country", "center_lon", "center_lat"]].copy(),
        geometry=gpd.points_from_xy(
            source.loc[pending, "center_lon"], source.loc[pending, "center_lat"]
        ),
        crs="EPSG:4326",
    ).to_crs(ghsl.crs)
    nearest = gpd.sjoin_nearest(
        points,
        ghsl[["ID_UC_G0", "GC_UCN_MAI_2025", "GC_CNT_UNN_2025", "GC_UCA_KM2_2025", "_country", "geometry"]],
        how="left",
        distance_col="_distance_m",
    )
    for index, row in nearest.iterrows():
        if (
            pd.isna(row["_distance_m"])
            or row["_distance_m"] > args.nearest_km * 1000
            or row["_country_left"] != row["_country_right"]
        ):
            continue
        result.loc[index, "ghsl_city"] = row["GC_UCN_MAI_2025"]
        result.loc[index, "ghsl_uc_id"] = row["ID_UC_G0"]
        result.loc[index, "ghsl_country"] = row["GC_CNT_UNN_2025"]
        result.loc[index, "ghsl_area_km2"] = row["GC_UCA_KM2_2025"]
        result.loc[index, "ghsl_match"] = "nearest_same_country"
        result.loc[index, "ghsl_match_distance_km"] = row["_distance_m"] / 1000

    result = result[result["ghsl_area_km2"].notna()].copy()
    result["image_count"] = result["#images"].astype("int64")
    result["estimated_four_view_panos"] = (result["image_count"] // 4).astype("int64")
    result["images_per_km2"] = result["image_count"] / result["ghsl_area_km2"]
    result["image_count_sufficient_for_12800"] = result["image_count"] >= 4 * 12800
    result = result.sort_values(["images_per_km2", "image_count"], ascending=False)
    result = result.drop_duplicates("ghsl_uc_id", keep="first").head(args.limit)

    columns = [
        "country",
        "city",
        "image_count",
        "estimated_four_view_panos",
        "ghsl_country",
        "ghsl_city",
        "ghsl_uc_id",
        "ghsl_area_km2",
        "images_per_km2",
        "image_count_sufficient_for_12800",
        "ghsl_match",
        "ghsl_match_distance_km",
        "center_lat",
        "center_lon",
        "folder",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result[columns].to_csv(args.output, index=False)
    print(
        f"wrote {len(result)} unique GHSL urban centres to {args.output}; "
        f"{int(result['image_count_sufficient_for_12800'].sum())} have enough image files for "
        "12,800 four-view panos before completeness checks; "
        f"countries={result['country'].nunique()}"
    )
    print(result.groupby("country").size().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
