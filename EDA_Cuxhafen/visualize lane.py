import mysql.connector
import pandas as pd
import folium

# DB config (update as needed)
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "AMS_ship_traj03",
    "database": "ship_data",
}


def read_fairway_points():
    cnx = mysql.connector.connect(**db_config)
    query = """
        SELECT id,latitude, longitude, lane, mmsi, cluster_id, trip_bearing
        FROM fairway_lane_points_2
        WHERE lane IS NOT NULL
        LIMIT 10000
    """
    df = pd.read_sql(query, cnx)
    cnx.close()
    return df


def visualize_lanes_on_map(df):
    center_lat = df["latitude"].mean()
    center_lon = df["longitude"].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

    lane_colors = {"inbound": "blue", "outbound": "red"}

    for lane, group in df.groupby("lane"):
        color = lane_colors.get(lane, "gray")
        for _, row in group.iterrows():
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=2,
                color=color,
                fill=True,
                fill_opacity=0.6,
                popup=f"MMSI: {row['mmsi']}\n id: {row['id']}\nlongitude: {row['longitude']:.1f}\nLane: {lane}",
            ).add_to(m)
    return m


if __name__ == "__main__":
    df = read_fairway_points()
    folium_map = visualize_lanes_on_map(df)
    folium_map.save("fairway_lanes_map.html")
    print("Map saved as 'fairway_lanes_map.html'")
