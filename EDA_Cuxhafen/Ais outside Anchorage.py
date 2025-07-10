import mysql.connector
import pandas as pd
from shapely.geometry import Polygon
from shapely.ops import unary_union
from mysql.connector import errorcode
import numpy as np

# This script retrieves AIS data from a MySQL database, filters it based on specified polygons, and saves the results to a new table.


def get_and_save_ais_data_to_table():
    # Database configuration
    db_config = {
        "host": "localhost",
        "user": "root",
        "password": "AMS_ship_traj03",
        "database": "ship_data",
    }

    # New target table
    target_table = "ais_data_outside_anchorages"

    # Replace these with your two polygons
    polygon1_coords = [
        [8.713, 53.8855],
        [8.7176383, 53.8870833],
        [8.7251, 53.8798],
        [8.733333, 53.871795],
        [8.7377167, 53.8666433],
        [8.713, 53.8855],
    ]

    polygon2_coords = [
        [9.132117, 53.877083],
        [9.170717, 53.87765],
        [9.209067, 53.87765],
        [9.20905, 53.875133],
        [9.170717, 53.8743],
        [9.1321, 53.87375],
        [9.132117, 53.877083],
    ]

    polygon3_coords = [
        [9.076833, 53.877917],
        [9.11545, 53.88265],
        [9.125566, 53.8874447],
    ]

    polygon4_coords = [
        [9.1429746, 53.8877745],
        [9.1430444, 53.8878694],
        [9.1431382, 53.8879516],
        [9.1438776, 53.888425],
        [9.144246, 53.888459],
        [9.1444418, 53.888508],
        [9.1446643, 53.8885647],
        [9.1582302, 53.8888053],
        [9.1646742, 53.8891689],
        [9.1667556, 53.8891903],
        [9.1667337, 53.8868607],
        [9.1845372, 53.8868833],
        [9.1845202, 53.8895925],
        [9.1922224, 53.8897115],
        [9.1984239, 53.8897521],
        [9.1987136, 53.8895498],
        [9.1994136, 53.889387],
        [9.2046352, 53.8890607],
        [9.2082918, 53.888919],
        [9.211328, 53.8889886],
        [9.2127397, 53.8889422],
        [9.21274, 53.88721],
        [9.21752, 53.8844567],
        [9.2207333, 53.8832],
        [9.1543333, 53.8834833],
        [9.1429746, 53.8877745],
    ]

    polygon5_coords = [
        [9.0217667, 53.8679667],
        [9.045933, 53.8743],
        [9.08021, 53.88459],
        [9.076833, 53.877917],
        [9.051, 53.8706667],
        [9.02545, 53.8634833],
        [9.0217667, 53.8679667],
    ]

    polygon6_coords = [
        [8.936107, 53.857141],
        [8.9654333, 53.8588],
        [8.99515, 53.8631833],
        [9.0217667, 53.8679667],
        [9.02545, 53.8634833],
        [8.999617, 53.856483],
        [8.968217, 53.852367],
        [8.935167, 53.8493],
        [8.936107, 53.857141],
    ]

    polygon1 = Polygon(polygon1_coords)
    polygon2 = Polygon(polygon2_coords)
    polygon3 = Polygon(polygon3_coords)
    polygon4 = Polygon(polygon4_coords)
    polygon5 = Polygon(polygon5_coords)
    polygon6 = Polygon(polygon6_coords)

    # Create a union polygon to exclude all areas inside both
    combined_polygon = unary_union(
        [polygon1, polygon2, polygon3, polygon4, polygon5, polygon6]
    )
    combined_polygon_wkt = combined_polygon.wkt
    print(f"Combined polygon WKT: {combined_polygon_wkt}")
    print("Combined polygon created successfully.")
    # Hardcoded column definitions based on your schema
    COLUMN_DEFINITIONS = {
        "mmsi": {"type": "BIGINT", "nullable": False},
        "datetime_utc": {"type": "TIMESTAMP", "nullable": False},
        "datetime_interpolated": {"type": "TIMESTAMP", "nullable": False},
        "recv_time": {"type": "BIGINT", "nullable": False},
        "latitude": {"type": "DOUBLE PRECISION", "nullable": False},
        "longitude": {"type": "DOUBLE PRECISION", "nullable": False},
        "sog": {"type": "DOUBLE PRECISION", "nullable": True},
        "cog": {"type": "DOUBLE PRECISION", "nullable": True},
        "true_heading": {"type": "INT", "nullable": True},
        "navigational_status": {"type": "INT", "nullable": False},
        "message_id": {"type": "INT", "nullable": False},
        "raim_flag": {"type": "INT", "nullable": True},
        "special_manoeuvre_indicator": {"type": "INT", "nullable": True},
        "rate_of_turn_rotais": {"type": "INT", "nullable": True},
        "timestamp_ais": {"type": "INT", "nullable": False},
        "position_accuracy": {"type": "INT", "nullable": True},
    }

    # Columns to select (excluding spatial columns)
    SELECT_COLUMNS = [col for col in COLUMN_DEFINITIONS.keys() if col != "coordinates"]

    conn = None
    cursor = None

    try:
        # Connect to the database
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)

        # Step 1: Retrieve AIS data within the polygon
        print(
            "\nStep 1: Retrieving AIS data from 'ais_data_polygon' within the polygon..."
        )

        # SQL query to select points *outside* the polygons
        query = f"""
        SELECT {', '.join(SELECT_COLUMNS)}
        FROM ais_data_polygon
        WHERE NOT ST_Within(POINT(Longitude, Latitude), ST_GeomFromText('{combined_polygon_wkt}'))
        """

        cursor.execute(query)
        result = cursor.fetchall()

        if not result:
            print("No data found within the polygon.")
            return

        df = pd.DataFrame(result)
        print(f"Retrieved {len(df)} records from the database.")

        # Step 2: Create the target table if it doesn't exist
        print(f"\nStep 2: Creating/verifying table '{target_table}'...")

        # Generate CREATE TABLE statement
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS `{target_table}` (
            `id` INT AUTO_INCREMENT PRIMARY KEY,
            {', '.join([f"`{col}` {defn['type']} {'NOT NULL' if not defn['nullable'] else ''}" 
                       for col, defn in COLUMN_DEFINITIONS.items()])}
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """

        cursor.execute(create_table_sql)
        conn.commit()
        print(f"Table '{target_table}' is ready.")

        # Step 3: Prepare data for insertion
        print("\nStep 3: Preparing data for insertion...")

        # Convert NaN/NaT to None
        df = df.replace({np.nan: None, pd.NaT: None})

        # Convert datetime columns to strings
        datetime_cols = ["datetime_utc", "datetime_interpolated"]
        for col in datetime_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col]).dt.strftime("%Y-%m-%d %H:%M:%S")

        # Prepare the INSERT statement
        insert_sql = f"""
        INSERT INTO `{target_table}` ({', '.join([f'`{col}`' for col in SELECT_COLUMNS])})
        VALUES ({', '.join(['%s'] * len(SELECT_COLUMNS))})
        """

        # Convert DataFrame to list of tuples
        data_to_insert = [tuple(row) for row in df.to_records(index=False)]

        # --- Step 4: Insert data in batches ---
        print("\nStep 4: Inserting data in batches...")

        def convert_numpy_types(value):
            """Convert numpy types to native Python types for MySQL compatibility"""
            if pd.isna(value):
                return None
            if isinstance(value, (np.int64, np.int32)):
                return int(value)
            if isinstance(value, (np.float64, np.float32)):
                return float(value)
            if isinstance(value, np.bool_):
                return bool(value)
            if isinstance(value, (np.datetime64, pd.Timestamp)):
                return str(pd.to_datetime(value))
            return value

        batch_size = 1000
        total_rows = len(data_to_insert)
        inserted_rows = 0

        for i in range(0, total_rows, batch_size):
            batch = data_to_insert[i : i + batch_size]

            # Convert numpy types in the entire batch
            converted_batch = []
            for row in batch:
                converted_row = tuple(convert_numpy_types(x) for x in row)
                converted_batch.append(converted_row)

            try:
                cursor.executemany(insert_sql, converted_batch)
                conn.commit()
                inserted_rows += len(converted_batch)
                print(
                    f"Inserted {min(i + batch_size, total_rows)} of {total_rows} rows..."
                )
            except mysql.connector.Error as batch_err:
                print(f"Error in batch {i}-{i+batch_size}: {batch_err}")
                conn.rollback()

                # Detailed error handling for problematic rows
                error_count = 0
                for row_idx, row in enumerate(converted_batch):
                    try:
                        cursor.execute(insert_sql, row)
                        conn.commit()
                        inserted_rows += 1
                    except mysql.connector.Error as row_err:
                        error_count += 1
                        # Print first few errors to avoid flooding output
                        if error_count <= 5:
                            print(f"Row {i + row_idx} failed: {row_err}")
                            print(f"Problematic row values: {row}")
                        conn.rollback()

                if error_count > 5:
                    print(f"... plus {error_count - 5} more errors in this batch")

        print(
            f"\nSuccessfully inserted {inserted_rows} of {total_rows} rows into '{target_table}'."
        )

    except mysql.connector.Error as err:
        print(f"Database error (MySQL Error Code: {err.errno}): {err.msg}")
        if conn:
            conn.rollback()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        if conn:
            conn.rollback()
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()
            print("\nMySQL connection closed.")


if __name__ == "__main__":
    get_and_save_ais_data_to_table()
    print("\nProcess completed.")
