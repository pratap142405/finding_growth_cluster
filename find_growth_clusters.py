#!/usr/bin/env python3
"""
Find small clusters of buildings (default 15-20) that have GROWN recently,
i.e. places where a settlement is starting to form.

Pipeline
  1. Region name -> boundary polygon        (OpenStreetMap Nominatim)
  2. Building footprints in that boundary   (Google Open Buildings v3 via Earth Engine)
  3. DBSCAN clustering of building centroids
  4. Keep clusters with MIN_SIZE..MAX_SIZE buildings
  5. Building counts per year for each cluster
     (Google Open Buildings Temporal, via Google Earth Engine)
  6. Rank by growth, export CSV / GeoJSON / KML

Usage
  python find_growth_clusters.py "Shamshabad mandal, Telangana" --gee-project my-gcp-project
"""
import argparse
import re
from pathlib import Path

import time
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

import ee
import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import simplekml
from scipy.spatial import cKDTree
from shapely.geometry import MultiPoint, box, mapping, shape
from sklearn.cluster import DBSCAN

TEMPORAL_ASSET = "GOOGLE/Research/open-buildings-temporal/v1"


# --------------------------------------------------------------------------- #
# 1. Region boundary
# --------------------------------------------------------------------------- #
def get_region(name):
    r = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": name, "format": "jsonv2", "polygon_geojson": 1, "limit": 1},
        headers={"User-Agent": "growth-cluster-finder/1.0"},
        timeout=60,
    )
    r.raise_for_status()
    res = r.json()
    if not res:
        raise SystemExit(f"Nominatim found nothing for '{name}'. Try a more specific name.")
    geom = shape(res[0]["geojson"])
    if geom.geom_type not in ("Polygon", "MultiPolygon"):
        raise SystemExit(
            f"'{name}' resolved to a {geom.geom_type}, not an area. "
            "Add 'mandal', 'district', 'taluk', etc. to the name."
        )
    print(f"Region: {res[0]['display_name']}")
    return geom


# --------------------------------------------------------------------------- #
# 2. Buildings
# --------------------------------------------------------------------------- #
def get_buildings(region, min_area, max_area, tile_km=1.0, min_conf=0.65):
    """Fetch building centroids from Google Open Buildings v3 via Earth Engine, tile by tile."""
    fc = ee.FeatureCollection("GOOGLE/Research/open-buildings/v3/polygons")
    minx, miny, maxx, maxy = region.bounds
    step = tile_km / 111.0
    tiles = []
    for x in np.arange(minx, maxx, step):
        for y in np.arange(miny, maxy, step):
            t = box(x, y, min(x + step, maxx), min(y + step, maxy))
            if t.intersects(region):
                tiles.append(t.bounds)
    print(f"Fetching buildings in {len(tiles)} tiles (Google Open Buildings)...")

    def to_point(feat):
        c = feat.geometry().centroid(1).coordinates()
        return ee.Feature(None, {"lon": c.get(0), "lat": c.get(1), "area_m2": feat.get("area_in_meters")})

    def fetch(b):
        x0, y0, x1, y1 = b
        sub = (
            fc.filterBounds(ee.Geometry.Rectangle([x0, y0, x1, y1]))
            .filter(ee.Filter.gte("confidence", min_conf))
            .filter(ee.Filter.gte("area_in_meters", min_area))
            .filter(ee.Filter.lte("area_in_meters", max_area))
        )
        for _ in range(3):
            try:
                feats = sub.map(to_point).getInfo()["features"]
                break
            except Exception:
                time.sleep(2)
        else:
            return []
        if len(feats) >= 5000:  # hit the getInfo cap: split the tile in four and retry
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            return [r for q in [(x0, y0, mx, my), (mx, y0, x1, my), (x0, my, mx, y1), (mx, my, x1, y1)] for r in fetch(q)]
        rows = [f["properties"] for f in feats]
        # keep only centroids inside this tile so tile edges don't double count
        return [r for r in rows if x0 <= r["lon"] < x1 and y0 <= r["lat"] < y1]

    rows, done = [], 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        for res in ex.map(fetch, tiles):
            rows += res
            done += 1
            if done % 10 == 0 or done == len(tiles):
                print(f"  {done}/{len(tiles)} tiles, {len(rows)} buildings so far")

    if not rows:
        raise SystemExit("No buildings returned. Check the region name / Earth Engine access.")
    gdf = gpd.GeoDataFrame(
        rows, geometry=gpd.points_from_xy([r["lon"] for r in rows], [r["lat"] for r in rows]), crs=4326
    )
    gdf = gdf[gdf.intersects(region)].copy()
    utm = gdf.estimate_utm_crs()
    gdf = gdf.to_crs(utm)
    print(f"Buildings in region after size/confidence filter: {len(gdf)}")
    return gdf[["area_m2", "geometry"]].reset_index(drop=True), utm


# --------------------------------------------------------------------------- #
# 3 + 4. Clustering
# --------------------------------------------------------------------------- #
def find_clusters(b, utm, eps, min_size, max_size, isolation):
    xy = np.c_[b.geometry.x.values, b.geometry.y.values]
    labels = DBSCAN(eps=eps, min_samples=3).fit_predict(xy)
    b = b.assign(label=labels)

    sizes = pd.Series(labels[labels >= 0]).value_counts()
    keep = sizes[(sizes >= min_size) & (sizes <= max_size)].index
    print(f"Clusters total: {len(sizes)}; with {min_size}-{max_size} buildings: {len(keep)}")

    tree = cKDTree(xy)
    rows = []
    for lab in keep:
        idx = np.where(labels == lab)[0]
        pts = xy[idx]
        if isolation > 0:
            neigh = tree.query_ball_point(pts, r=isolation)
            if any(labels[j] != lab for n in neigh for j in n):
                continue
        hull = MultiPoint(pts).convex_hull.buffer(40)  # 40 m margin around houses
        c = MultiPoint(pts).centroid
        rows.append(
            {
                "cluster_id": int(lab),
                "n_buildings_now": len(idx),
                "median_area_m2": float(np.median(b.area_m2.values[idx])),
                "geometry": hull,
                "cx": c.x,
                "cy": c.y,
            }
        )
    if not rows:
        raise SystemExit("No clusters matched. Try a larger --eps, or widen --min-size/--max-size.")
    g = gpd.GeoDataFrame(rows, crs=utm).to_crs(4326)
    cen = gpd.GeoSeries(gpd.points_from_xy(g["cx"], g["cy"]), crs=utm).to_crs(4326)
    g["lat"], g["lon"] = cen.y.values, cen.x.values
    return g.drop(columns=["cx", "cy"])


# --------------------------------------------------------------------------- #
# 5. Growth over time (Google Earth Engine)
# --------------------------------------------------------------------------- #
def init_ee(project):
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)


def available_years():
    ts = ee.ImageCollection(TEMPORAL_ASSET).aggregate_array("system:time_start").getInfo()
    return sorted({datetime.fromtimestamp(t / 1000, timezone.utc).year for t in ts})


def growth_counts(clusters, years, batch=300):
    ic = ee.ImageCollection(TEMPORAL_ASSET)
    bands = [
        ic.filterDate(f"{y}-01-01", f"{y + 1}-01-01")
        .select("building_fractional_count")
        .mosaic()
        .rename(f"y{y}")
        for y in years
    ]
    stack = ee.Image.cat(bands)

    out = []
    for i in range(0, len(clusters), batch):
        chunk = clusters.iloc[i : i + batch]
        feats = [
            ee.Feature(ee.Geometry(mapping(g)), {"cluster_id": int(cid)})
            for cid, g in zip(chunk.cluster_id, chunk.geometry)
        ]
        res = stack.reduceRegions(
            collection=ee.FeatureCollection(feats),
            reducer=ee.Reducer.sum(),
            scale=4,
            tileScale=4,
        ).getInfo()
        out += [f["properties"] for f in res["features"]]
        print(f"  growth computed for {min(i + batch, len(clusters))}/{len(clusters)} clusters")
    df = pd.DataFrame(out).fillna(0)
    for y in years:
        df[f"y{y}"] = df[f"y{y}"].round(1)
    return df


# --------------------------------------------------------------------------- #
# 6. Output
# --------------------------------------------------------------------------- #
def write_kml(df, path, y0, y1):
    kml = simplekml.Kml()
    for _, r in df.iterrows():
        p = kml.newpoint(
            name=f"#{int(r['rank'])}  +{r['new_buildings']:.0f} houses",
            coords=[(r["lon"], r["lat"])],
        )
        p.description = (
            f"Buildings now (Overture): {r['n_buildings_now']}\n"
            f"{y0}: {r[f'y{y0}']}  ->  {y1}: {r[f'y{y1}']}\n"
            f"Growth: {r['new_buildings']:.1f} ({r['growth_pct']:.0f}%)\n"
            f"https://www.google.com/maps/@{r['lat']},{r['lon']},18z/data=!3m1!1e3"
        )
    kml.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("region", help='e.g. "Shamshabad mandal, Telangana"')
    ap.add_argument("--gee-project", required=True, help="Your Google Cloud project ID (registered for Earth Engine)")
    ap.add_argument("--min-size", type=int, default=15)
    ap.add_argument("--max-size", type=int, default=20)
    ap.add_argument("--eps", type=float, default=50, help="Max gap in metres between neighbouring houses (default 50)")
    ap.add_argument("--isolation", type=float, default=0,
                    help="If >0, drop clusters with other buildings within this many metres (default off)")
    ap.add_argument("--min-footprint", type=float, default=2, help="Ignore buildings smaller than this (m2)")
    ap.add_argument("--max-footprint", type=float, default=100, help="Ignore buildings larger than this (m2)")
    ap.add_argument("--start-year", type=int, default=2018)
    ap.add_argument("--end-year", type=int, default=2023)
    ap.add_argument("--min-growth", type=float, default=2, help="Minimum new buildings between start and end year")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "_", a.region.lower()).strip("_")[:40]

    region = get_region(a.region)
    init_ee(a.gee_project)
    b, utm = get_buildings(region, a.min_footprint, a.max_footprint)
    clusters = find_clusters(b, utm, a.eps, a.min_size, a.max_size, a.isolation)

    avail = available_years()
    print(f"Years available in Open Buildings Temporal: {avail[0]}-{avail[-1]}")
    # years = [y for y in range(a.start_year+2, a.end_year + 1) if y in avail]
    years = [y for y in range(a.start_year, a.end_year + 1) if y in avail]
    if len(years) < 2:
        raise SystemExit(f"Need at least two years within {avail[0]}-{avail[-1]}.")
    if years[0] != a.start_year or years[-1] != a.end_year:
        print(f"Adjusting year range to {years[0]}-{years[-1]}")
    a.start_year, a.end_year = years[0], years[-1]
    growth = growth_counts(clusters, years)
    df = clusters.merge(growth, on="cluster_id", how="left")

    y0, y1 = f"y{a.start_year}", f"y{a.end_year}"
    df["new_buildings"] = df[y1] - df[y0]
    df["growth_pct"] = np.where(df[y0] > 0, 100 * df["new_buildings"] / df[y0], np.nan)
    df["growth_pct"] = df["growth_pct"].fillna(999)  # grew from ~nothing

    ycols = [f"y{y}" for y in years]
    print("\nUnfiltered growth diagnostics:")
    print(df[["n_buildings_now"] + ycols + ["new_buildings"]].describe().round(2).to_string())
    df.drop(columns="geometry").to_csv(out / f"{slug}_all_clusters_unfiltered.csv", index=False)

    df = df[df.new_buildings >= a.min_growth].sort_values("new_buildings", ascending=False)
    df["rank"] = range(1, len(df) + 1)
    if df.empty:
        raise SystemExit("Clusters found, but none met --min-growth. Lower it or widen the year range.")

    df["maps_link"] = df.apply(
        lambda r: f"https://www.google.com/maps/@{r.lat},{r.lon},18z/data=!3m1!1e3", axis=1
    )
    df.drop(columns="geometry").to_csv(out / f"{slug}_clusters.csv", index=False)
    df.to_file(out / f"{slug}_clusters.geojson", driver="GeoJSON")
    write_kml(df, str(out / f"{slug}_clusters.kml"), a.start_year, a.end_year)

    print(f"\n{len(df)} growing clusters. Top 10:")
    cols = ["rank", "lat", "lon", "n_buildings_now", y0, y1, "new_buildings"]
    print(df[cols].head(10).to_string(index=False))
    print(f"\nFiles written to {out.resolve()}")


if __name__ == "__main__":
    main()