"""
Run once to geocode Bangalore location names → lat/lon.
Output: data/location_coords.csv
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from train import preprocess
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

df = preprocess(pd.read_csv("data/bengaluru_house_prices.csv"))

loc_stats = (
    df.groupby("location")
    .agg(median_price=("price", "median"), count=("price", "count"))
    .reset_index()
)
# Top 60 locations by transaction count (excludes 'other')
top_locs = loc_stats[loc_stats["location"] != "other"].nlargest(60, "count")

geolocator = Nominatim(user_agent="blr_house_price_app_v1")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1.1)

rows = []
for _, row in top_locs.iterrows():
    query = f"{row['location']}, Bangalore, Karnataka, India"
    result = geocode(query)
    if result:
        rows.append({
            "location":     row["location"],
            "lat":          result.latitude,
            "lon":          result.longitude,
            "median_price": row["median_price"],
            "count":        row["count"],
        })
        print(f"OK  {row['location']}")
    else:
        print(f"MISS {row['location']}")

out = pd.DataFrame(rows)
out.to_csv("data/location_coords.csv", index=False)
print(f"\nSaved {len(out)} locations → data/location_coords.csv")
