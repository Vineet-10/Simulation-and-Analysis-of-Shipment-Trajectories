import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import json

def load_ais_data(file_path):
    df = pd.read_csv(file_path, header=None, names=['id', 'timestamp', 'data'])

    # Parse the JSON data in the 'data' column
    df['data'] = df['data'].apply(json.loads)

    # Expand the JSON data into separate columns
    data_df = pd.json_normalize(df['data'])

    # Combine with the original DataFrame
    df = pd.concat([df.drop('data', axis=1), data_df], axis=1)

    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    return df

def calculate_total_duration(df):
    # Convert the timestamp column to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Group by mmsi_id to calculate duration per vessel
    df['total_duration'] = df.groupby('mmsi')['timestamp'].transform(lambda x: x.max() - x.min())

    return df

# Function to simulate ship movement
def simulate_trajectory(df, ais_data, mmsi_id):
    # Filter the dataset for the given mmsi_id
    vessel_data = df[df['mmsi'] == mmsi_id]
    total_duration = calculate_total_duration(df)
    duration = total_duration[total_duration['mmsi'] == mmsi_id]

    # Get the trajectory coordinates (lat, lon)
    trajectory_coords = list(zip(vessel_data['lat'], vessel_data['lon']))

    # Get the timestamps for the trajectory
    timestamps = pd.to_datetime(vessel_data['timestamp'])

    return trajectory_coords, timestamps, vessel_data

# Function to create a Plotly animated plot
def create_ship_animation(trajectory_coords, timestamps, vessel_data, ais_data, mmsi_id):
    # Getting destination of Vessel
    vessel_destination = ais_data[ais_data['mmsi'] == mmsi_id]
    destination = vessel_destination["destination"].unique()
    for i in destination:
        if i is not None and i != '':
            dest = i

    # Create a Plotly figure
    fig = go.Figure()

    # Add the ship's trajectory as a line (path) with animation
    fig.add_trace(go.Scattermap(
        lon=[coord[1] for coord in trajectory_coords],
        lat=[coord[0] for coord in trajectory_coords],
        mode='lines',
        line=dict(width=2, color='blue'),
        name="Ship's Path"
    ))

    # Add an animated marker that moves along the trajectory (Ship icon)
    fig.add_trace(go.Scattermap(
        lon=[trajectory_coords[0][1]],  # Start at the first point
        lat=[trajectory_coords[0][0]],
        mode='markers+text',
        marker=dict(size=8, color='red'),
        text=['Start'],
        textfont=dict(color='black'),
        name="Ship Marker",
        showlegend=False
    ))

    # Define the animation frames (moving the marker and the polyline)
    frames = [go.Frame(
        data=[
            go.Scattermap(
                lon=[trajectory_coords[i][1]],  # Update marker position
                lat=[trajectory_coords[i][0]],
                mode='markers+text',
                textfont=dict(color='black'),
                marker=dict(size=8, color='red'),
                text=[f'MMSI: {vessel_data["mmsi"].iloc[0]}<br>Total Duration: {0}<br>Destination: {dest}'],
            ),
            # Update polyline as the marker moves (the path follows the marker)
            go.Scattermap(
                lon=[coord[1] for coord in trajectory_coords[:i+1]],  # Update polyline (path)
                lat=[coord[0] for coord in trajectory_coords[:i+1]],
                mode='lines',
                line=dict(width=2, color='blue'),
                name="Ship's Path"
            )
        ],
        name=f'frame{i}'  # Frame name for each marker position
    ) for i in range(1, len(trajectory_coords))]

    # Set up the layout of the map and animation
    fig.update_layout(
        map=dict(
            style="open-street-map",
            center=dict(lat=trajectory_coords[0][0], lon=trajectory_coords[0][1]),
            zoom=10
        ),
        updatemenus=[dict(
            type="buttons", showactive=False,
            buttons=[dict(label="Play",
                          method="animate", args=[None, dict(frame=dict(duration=500, redraw=True), fromcurrent=True)])]
        )],
        title="Ship Movement Simulation"
    )

    # Add frames for the animation
    fig.frames = frames

    return fig

# Streamlit App
def run_app():

    file_path = 'rhein.csv'
    ais_data = load_ais_data(file_path)
    drop_mmsi = [ 2268402,211668670, 211478300, 244700399,244890469]
    idx = ais_data[ais_data['mmsi'].isin(drop_mmsi)].index

# Drop them in‐place (or assign back to df)
    ais_data.drop(idx, inplace=True)
    ais_cleaned = ais_data.dropna(subset=['lat', 'lon'])

    # Let the user select a vessel (based on MMSI ID) to simulate its trajectory
    mmsi_id = st.selectbox("Select Vessel (MMSI)", ais_data['mmsi'].unique())

    # Simulate the ship's movement and get the trajectory
    trajectory_coords, timestamps, vessel_data = simulate_trajectory(ais_cleaned, ais_data, mmsi_id)

    # Create the Plotly animated plot
    fig = create_ship_animation(trajectory_coords, timestamps, vessel_data, ais_data, mmsi_id)

    # Display the animated plot in the Streamlit app
    st.plotly_chart(fig)


if __name__ == "__main__":
    run_app()
