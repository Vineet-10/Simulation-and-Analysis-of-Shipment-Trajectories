import pandas as pd
import math
from sqlalchemy import create_engine
import pymysql

# ---------------------------------------
# Database Configuration
# ---------------------------------------
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "AMS_ship_traj03",
    "database": "ship_data",
}
engine = create_engine(
    f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}/{db_config['database']}"
)

# ---------------------------------------
# Parameters
# ---------------------------------------
TIME_WINDOW_SEC = 20
LAT_TOL = 0.005  # Approx 500m latitude
LON_TOL = 0.01  # Approx 500m longitude near Europe


# ---------------------------------------
# Haversine and Classification
# ---------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # meters
    phi1, phi2 = map(math.radians, [lat1, lat2])
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def classify_encounter(lat1, lon1, heading2, lat2, lon2):
    dy = lat1 - lat2
    dx = lon1 - lon2
    brg = (math.degrees(math.atan2(dx, dy)) + 360) % 360
    rel_brg = (brg - heading2 + 360) % 360
    if rel_brg <= 22.5 or rel_brg >= 337.5:
        return "head-on"
    elif 22.5 < rel_brg <= 112.5:
        return "crossing_starboard"
    elif 247.5 <= rel_brg < 337.5:
        return "crossing_port"
    elif 112.5 < rel_brg < 247.5:
        return "overtaking"
    return "undefined"


# ---------------------------------------
# Setup: Get all dates and create output table
# ---------------------------------------
print("Fetching distinct dates...")
dates = pd.read_sql(
    "SELECT DISTINCT DATE(datetime_utc) as d FROM ais_open_water_polygon_env ORDER BY d",
    engine,
)["d"].tolist()

# Create output table with correct structure
pd.DataFrame(
    columns=[
        "id_1",
        "id_2",
        "mmsi_1",
        "mmsi_2",
        "timestamp",
        "latitude",
        "longitude",
        "distance_m",
        "encounter_type",
        "datetime_utc_1",
        "datetime_utc_2",
    ]
).to_sql("ais_open_water_polygon_enc", engine, if_exists="replace", index=False)


# ---------------------------------------
# Process Each Date
# ---------------------------------------
for d in dates:
    print(f"\n🕓 Processing {d}...")
    df = pd.read_sql(
        f"""
        SELECT id, mmsi, datetime_utc, latitude, longitude, sog, true_heading
        FROM ais_open_water_polygon_env
        WHERE DATE(datetime_utc) = '{d}'
        AND sog BETWEEN 2 AND 50
        AND true_heading IS NOT NULL
    """,
        engine,
    )

    if df.empty:
        print("No data.")
        continue

    if 247389200 not in df["mmsi"].values:
        print(f"⚠️  MMSI 247389200 not present in filtered data for {d}.")

    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
    df = df.sort_values("datetime_utc")

    seen_encounters = set()
    encounter_ids_today = set()
    daily_encounters = []

    for i, row in df.iterrows():
        for j in range(i + 1, len(df)):
            other = df.iloc[j]

            # Skip same vessel
            if row["mmsi"] == other["mmsi"]:
                continue

            # Time window filter
            time_diff = abs(
                (row["datetime_utc"] - other["datetime_utc"]).total_seconds()
            )
            if time_diff > TIME_WINDOW_SEC:
                break

            # Lat/Lon rough filter
            if (
                abs(row["latitude"] - other["latitude"]) > LAT_TOL
                or abs(row["longitude"] - other["longitude"]) > LON_TOL
            ):
                continue

            # Haversine distance check
            dist = haversine(
                row["latitude"], row["longitude"], other["latitude"], other["longitude"]
            )
            if dist > 500:
                continue

            # Deduplicate
            mmsi1, mmsi2 = sorted([row["mmsi"], other["mmsi"]])
            timestamp = min(row["datetime_utc"], other["datetime_utc"])
            key = (mmsi1, mmsi2, timestamp)

            if key in seen_encounters:
                continue

            seen_encounters.add(key)
            encounter_ids_today.update([row["id"], other["id"]])

            encounter_type = classify_encounter(
                lat1=row["latitude"],
                lon1=row["longitude"],
                heading2=other["true_heading"],
                lat2=other["latitude"],
                lon2=other["longitude"],
            )

            daily_encounters.append(
                {
                    "id_1": row["id"],
                    "id_2": other["id"],
                    "mmsi_1": mmsi1,
                    "mmsi_2": mmsi2,
                    "timestamp": timestamp,
                    "latitude": (row["latitude"] + other["latitude"]) / 2,
                    "longitude": (row["longitude"] + other["longitude"]) / 2,
                    "distance_m": dist,
                    "encounter_type": encounter_type,
                    "datetime_utc_1": row["datetime_utc"],
                    "datetime_utc_2": other["datetime_utc"],
                }
            )

    # Save to SQL
    # Save encounters for this date
    if daily_encounters:
        pd.DataFrame(daily_encounters).to_sql(
            "ais_open_water_polygon_enc", engine, if_exists="append", index=False
        )
        print(f"✓ {len(daily_encounters)} encounters written.")
    else:
        print("✓ No encounters found.")

    # Save non-encounter rows for this date
    if encounter_ids_today:
        ids_tuple = tuple(map(int, encounter_ids_today))
        non_encounter_df = pd.read_sql(
            f"""
            SELECT * FROM ais_open_water_polygon_env
            WHERE DATE(datetime_utc) = '{d}'
            AND id NOT IN {ids_tuple}
            """,
            engine,
        )
    else:
        non_encounter_df = pd.read_sql(
            f"""
            SELECT * FROM ais_open_water_polygon_env
            WHERE DATE(datetime_utc) = '{d}'
            """,
            engine,
        )

    if not non_encounter_df.empty:
        non_encounter_df.to_sql(
            "ais_open_water_polygon_enc_no", engine, if_exists="append", index=False
        )
        print(f"✓ {len(non_encounter_df)} non-encounter lines saved.")
    else:
        print("✓ No non-encounter lines for this date.")
