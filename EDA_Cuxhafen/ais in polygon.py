import mysql.connector
import pandas as pd
from shapely.geometry import Polygon
from mysql.connector import errorcode
import numpy as np


def get_and_save_ais_data_to_table():
    # Database configuration
    db_config = {
        "host": "localhost",
        "user": "root",
        "password": "AMS_ship_traj03",
        "database": "ship_data",
    }

    # Define the target table name
    target_table = "ais_east_polygon"

    # Define your polygon coordinates
    """polygon_coords = [
        [8.866357682634487, 53.968612375580875],
        [8.70718895159439, 53.87971928320604],
        [8.786041488954766, 53.83598032041533],
        [8.926740437497955, 53.83373276761864],
        [9.08063395615909, 53.86295496041569],
        [9.178025184987149, 53.86931121229844],
        [9.277017566770382, 53.85952029450908],
        [9.286156067737664, 53.872302463013824],
        [9.222209324237411, 53.886474104090524],
        [9.125855419471634, 53.886042579316324],
        [9.057819286039745, 53.890359630983966],
        [8.954068535144273, 53.88776829990357],
        [8.933196796551897, 53.90664476911101],
        [8.866357682634487, 53.968612375580875],
    ]"""

    """polygon_coords = [
        [8.803834765061907, 53.8482602897254],
        [8.800842600599083, 53.83568167180323],
        [8.93212381643238, 53.83501953459975],
        [8.93249783699028, 53.85024604186299],
        [8.803086723946222, 53.85355541942201],
        [8.803834765061907, 53.8482602897254],
    ]  # polygon in open water near cuxhaven
    """
    polygon_coords = [
        [9.243618357622069, 53.88211074450771],
        [9.233519601557845, 53.866012243854186],
        [9.275035885228732, 53.86005694872969],
        [9.316926189474486, 53.83688940853915],
        [9.34086350618631, 53.84902638587283],
        [9.299724737588434, 53.869100596442564],
        [9.262320278676157, 53.87659833733136],
        [9.243618357622069, 53.88211074450771],
    ]
    polygon = Polygon(polygon_coords)
    polygon_wkt = polygon.wkt

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
        print("\nStep 1: Retrieving AIS data from 'vessel_data' within the polygon...")

        query = f"""
        SELECT {', '.join(SELECT_COLUMNS)}
        FROM ais_data_polygon
        WHERE ST_Within(POINT(Longitude, Latitude), ST_GeomFromText('{polygon_wkt}'))
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
