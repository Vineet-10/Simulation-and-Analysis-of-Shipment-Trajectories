import csv
from collections import defaultdict
from datetime import datetime


# This script processes AIS navigation data from a CSV file, analyzes vessel statistics,
# and writes a summary to a new CSV file.
def safe_timestamp_conversion(timestamp, is_ms=True):
    try:
        ts = float(timestamp)
        if is_ms:
            ts = ts / 1000.0
        if ts > 0 and ts < 2147483647:
            return datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError, OSError):
        pass
    return ""


def analyze_navigation_data(input_file, output_csv):
    vessel_stats = defaultdict(
        lambda: {
            "positions": [],
            "speeds": [],
            "headings": [],
            "cogs": [],
            "nav_statuses": set(),
            "first_seen_recv_time": None,
            "last_seen_recv_time": None,
            "raim_flags": set(),
            "message_types": set(),
            "recv_times": [],
        }
    )

    with open(input_file, "r", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        for row_num, row in enumerate(reader, 1):
            if not row:
                continue

            try:
                if "ERROR" in [row.get("Latitude"), row.get("Longitude")]:
                    continue

                user_id = row.get("UserID")
                if not user_id:
                    continue

                try:
                    lat = float(row["Latitude"]) if row["Latitude"] else None
                    lon = float(row["Longitude"]) if row["Longitude"] else None
                except ValueError:
                    continue

                sog = (
                    float(row["SOG"]) if row["SOG"] and row["SOG"] != "ERROR" else None
                )
                cog = (
                    float(row["COG"]) if row["COG"] and row["COG"] != "ERROR" else None
                )
                heading = (
                    float(row["Trueheading"])
                    if row["Trueheading"] and row["Trueheading"] != "ERROR"
                    else None
                )

                try:
                    recv_time = (
                        float(row["recv_time"])
                        if row["recv_time"] and row["recv_time"] != "0"
                        else None
                    )
                except ValueError:
                    recv_time = None

                if lat is not None and lon is not None:
                    vessel_stats[user_id]["positions"].append((lat, lon))
                if sog is not None:
                    vessel_stats[user_id]["speeds"].append(sog)
                if cog is not None:
                    vessel_stats[user_id]["cogs"].append(cog)
                if heading is not None:
                    vessel_stats[user_id]["headings"].append(heading)
                if recv_time is not None:
                    vessel_stats[user_id]["recv_times"].append(recv_time)

                if row.get("Navigationalstatus"):
                    vessel_stats[user_id]["nav_statuses"].add(row["Navigationalstatus"])
                if row.get("RAIM-flag"):
                    vessel_stats[user_id]["raim_flags"].add(row["RAIM-flag"])
                if row.get("MessageID"):
                    vessel_stats[user_id]["message_types"].add(row["MessageID"])

                if recv_time:
                    if (
                        vessel_stats[user_id]["first_seen_recv_time"] is None
                        or recv_time < vessel_stats[user_id]["first_seen_recv_time"]
                    ):
                        vessel_stats[user_id]["first_seen_recv_time"] = recv_time
                    if (
                        vessel_stats[user_id]["last_seen_recv_time"] is None
                        or recv_time > vessel_stats[user_id]["last_seen_recv_time"]
                    ):
                        vessel_stats[user_id]["last_seen_recv_time"] = recv_time

            except Exception as e:
                print(f"⚠️ Error processing row {row_num}: {e}")
                continue

    results = []
    for user_id, stats in vessel_stats.items():
        positions = stats["positions"]
        speeds = stats["speeds"]
        headings = stats["headings"]
        cogs = stats["cogs"]

        if not positions:
            continue

        lats = [p[0] for p in positions]
        lons = [p[1] for p in positions]
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)

        avg_speed = sum(speeds) / len(speeds) if speeds else None
        max_speed = max(speeds) if speeds else None
        heading_range = max(headings) - min(headings) if headings else None
        cog_range = max(cogs) - min(cogs) if cogs else None

        first_ts = stats["first_seen_recv_time"]
        last_ts = stats["last_seen_recv_time"]

        first_utc = safe_timestamp_conversion(first_ts, is_ms=True) if first_ts else ""
        if not first_utc:
            first_utc = safe_timestamp_conversion(first_ts, is_ms=False)

        last_utc = safe_timestamp_conversion(last_ts, is_ms=True) if last_ts else ""
        if not last_utc:
            last_utc = safe_timestamp_conversion(last_ts, is_ms=False)

        if first_ts and last_ts:
            first_sec = first_ts / 1000 if first_utc else first_ts
            last_sec = last_ts / 1000 if last_utc else last_ts
            duration_min = (last_sec - first_sec) / 60
            duration_hhmm = (
                f"{int(duration_min // 60):02d}:{int(duration_min % 60):02d}"
            )
        else:
            duration_min = None
            duration_hhmm = ""

        results.append(
            {
                "user_id": user_id,
                "first_seen_ts": int(first_ts) if first_ts else None,
                "first_seen_utc": first_utc,
                "last_seen_ts": int(last_ts) if last_ts else None,
                "last_seen_utc": last_utc,
                "duration_min": (
                    round(duration_min, 1) if duration_min is not None else None
                ),
                "duration_hhmm": duration_hhmm,
                "min_latitude": round(min_lat, 6),
                "max_latitude": round(max_lat, 6),
                "min_longitude": round(min_lon, 6),
                "max_longitude": round(max_lon, 6),
                "avg_speed_knots": round(avg_speed, 1) if avg_speed else None,
                "max_speed_knots": round(max_speed, 1) if max_speed else None,
                "heading_range": round(heading_range, 1) if heading_range else None,
                "cog_range": round(cog_range, 1) if cog_range else None,
                "navigation_statuses": ", ".join(sorted(stats["nav_statuses"])),
                "raim_flags": ", ".join(sorted(stats["raim_flags"])),
                "message_types": ", ".join(sorted(stats["message_types"])),
            }
        )

    fieldnames = list(results[0].keys()) if results else []

    with open(output_csv, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ Summary written to: {output_csv}")
    print(f"Total valid vessels processed: {len(results)}")


if __name__ == "__main__":
    analyze_navigation_data(
        input_file="valid_ais_data.csv",  # Replace with your input file
        output_csv="ais_summary_data_ready.csv",  # Output ready for DB
    )
