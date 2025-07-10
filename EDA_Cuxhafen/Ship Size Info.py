import csv


def extract_vessel_dimensions_with_details(input_file, output_file):
    vessel_info = {}  # user_id -> entry

    with open(input_file, "r", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        for row_num, row in enumerate(reader, 1):
            if not row:
                continue

            try:
                user_id = row.get("UserID")
                if not user_id:
                    continue

                # Only keep the first valid entry for each vessel
                if user_id not in vessel_info:
                    # Convert dimensions to integers (handle empty strings)
                    dim_a = (
                        int(row.get("DimensionA", 0)) if row.get("DimensionA") else 0
                    )
                    dim_b = (
                        int(row.get("DimensionB", 0)) if row.get("DimensionB") else 0
                    )
                    dim_c = (
                        int(row.get("DimensionC", 0)) if row.get("DimensionC") else 0
                    )
                    dim_d = (
                        int(row.get("DimensionD", 0)) if row.get("DimensionD") else 0
                    )

                    # Only store if we have valid dimensions
                    if any([dim_a, dim_b, dim_c, dim_d]):
                        vessel_info[user_id] = {
                            "Name": row.get("Name", "Unknown"),
                            "Destination": row.get("Destination", "Unknown"),
                            "ETA": row.get("ETA", "Unknown"),
                            "Max Draught": row.get(
                                "Maximumpresentstaticdraught", "Unknown"
                            ),
                            "Ship Type": row.get("Typeofshipandcargotype", "Unknown"),
                            "IMO Number": row.get("IMOnumber", "Unknown"),
                            "Call Sign": row.get("Callsign", "Unknown"),
                            "Dimension A": dim_a,
                            "Dimension B": dim_b,
                            "Dimension C": dim_c,
                            "Dimension D": dim_d,
                        }

            except Exception as e:
                print(f"⚠️ Error processing row {row_num}: {e}")
                continue

    # Prepare summary data
    summary_rows = []
    for user_id, entry in vessel_info.items():
        # Calculate dimensions
        length = entry.get("Dimension A", 0) + entry.get("Dimension B", 0)
        width = entry.get("Dimension C", 0) + entry.get("Dimension D", 0)

        # Beam category
        if width < 10:
            beam_category = "Beam < 10m"
        elif 10 <= width < 23:
            beam_category = "10m ≤ Beam < 23m"
        elif 23 <= width < 33:
            beam_category = "23m ≤ Beam < 33m"
        else:
            beam_category = "Beam ≥ 33m"

        summary_rows.append(
            {
                "User ID": user_id,
                "Name": entry.get("Name", "Unknown"),
                "Destination": entry.get("Destination", "Unknown"),
                "ETA": entry.get("ETA", "Unknown"),
                "Max Draught": entry.get("Max Draught", "Unknown"),
                "Ship Type": entry.get("Ship Type", "Unknown"),
                "IMO Number": entry.get("IMO Number", "Unknown"),
                "Call Sign": entry.get("Call Sign", "Unknown"),
                "Length (m)": length,
                "Beam (m)": width,
                "Beam Category": beam_category,
                "Dimension A": entry.get("Dimension A", 0),
                "Dimension B": entry.get("Dimension B", 0),
                "Dimension C": entry.get("Dimension C", 0),
                "Dimension D": entry.get("Dimension D", 0),
            }
        )

    # Write to CSV
    fieldnames = [
        "User ID",
        "Name",
        "Destination",
        "ETA",
        "Max Draught",
        "Ship Type",
        "IMO Number",
        "Call Sign",
        "Length (m)",
        "Beam (m)",
        "Beam Category",
        "Dimension A",
        "Dimension B",
        "Dimension C",
        "Dimension D",
    ]

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved vessel summary for {len(summary_rows)} vessels to {output_file}")


if __name__ == "__main__":
    input_file = "D:/ovgu/sem3/AMS/DataCleaner/message_5.csv"
    output_file = "vessel_details_summary.csv"
    extract_vessel_dimensions_with_details(input_file, output_file)
