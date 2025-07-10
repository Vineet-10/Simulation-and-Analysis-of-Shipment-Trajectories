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

# Function to simulate ship movement
def simulate_trajectory(df, mmsi_id):
    # Filter the dataset for the given mmsi_id
    vessel_data = df[df['mmsi'] == mmsi_id]

    # Get the trajectory coordinates (lat, lon)
    trajectory_coords = list(zip(vessel_data['lat'], vessel_data['lon']))

    # Get the timestamps for the trajectory (this could be time-based intervals if desired)
    timestamps = pd.to_datetime(vessel_data['timestamp'])

    return trajectory_coords, timestamps

# Function to create a Plotly animated plot
def create_ship_animation(trajectory_coords, timestamps):
    # Create a Plotly figure
    fig = go.Figure()

    # Add the ship's trajectory as a line (path)
    fig.add_trace(go.Scattermapbox(
        lon=[coord[1] for coord in trajectory_coords],
        lat=[coord[0] for coord in trajectory_coords],
        mode='lines',
        fillcolor= 'blue',
        line=dict(width=2, color='blue'),
        name="Ship's Path"
    ))

    # Add an animated marker that moves along the trajectory
    fig.add_trace(go.Scattermapbox(
        lon=[trajectory_coords[0][1]],  # Start at the first point
        lat=[trajectory_coords[0][0]],
        mode='markers',
        marker=dict(size=8, color='red'),
        name="Ship Marker",
        showlegend=False
    ))

    # Define the animation frames (moving the marker)
    frames = [go.Frame(
        data=[go.Scattermapbox(
            lon=[trajectory_coords[i][1]],
            lat=[trajectory_coords[i][0]],
            mode='markers',
            marker=dict(size=8, color='red'),
        )],
        name=f'frame{i}'  # Frame name for each marker position
    ) for i in range(1, len(trajectory_coords))]

    # Set up the layout of the map and animation
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",  # You can choose other map styles
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
    # Load your AIS data (replace this with your actual data loading logic)
    # Load the data
    file_path = 'rhein.csv'
    ais_data = load_ais_data(file_path)
    ais_cleaned = ais_data.dropna(subset=['lat', 'lon'])

    # Let the user select a vessel (based on MMSI ID) to simulate its trajectory
    mmsi_id = st.selectbox("Select Vessel (MMSI)", ais_data['mmsi'].unique())

    # Simulate the ship's movement and get the trajectory
    trajectory_coords, timestamps = simulate_trajectory(ais_cleaned, mmsi_id)

    # Create the Plotly animated plot
    fig = create_ship_animation(trajectory_coords, timestamps)

    # Display the animated plot in the Streamlit app
    st.plotly_chart(fig)

# Run the Streamlit app
if __name__ == "__main__":
    run_app()
