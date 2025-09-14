import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from physics_basedv2 import simulate_motion
#import physic_based
import physics_basedv2
import json
import numpy as np
import folium
from shapely.geometry import Point, Polygon
from streamlit_folium import st_folium

# Initialize session state
if 'click_coords' not in st.session_state:
    st.session_state.click_coords = {'start': None, 'end': None}
if 'simulation_data' not in st.session_state:
    st.session_state.simulation_data = None


def load_data(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

#with open('boundary.geojson') as f:
#    geojson_data = json.load(f)
boundary_coords_rhein = 'boundary.json'
polygons_data_rhein = load_data(boundary_coords_rhein)
boundary_coords_cuxhaven = 'Cuxhaven_bounding_box.json'
polygons_data_cuxhaven = load_data(boundary_coords_cuxhaven)
# Create polygon objects for all polygons in the JSON file
polygons_rhein = []
polygons_cuxhaven = []
for polygon_name, coords in polygons_data_rhein.items():
    polygons_rhein.append(Polygon(coords))

for polygon_name, coords in polygons_data_cuxhaven.items():
    polygons_cuxhaven.append(Polygon(coords))

#BOUNDARY_COORDS = geojson_data["features"][0]["geometry"]["coordinates"][0]

# Create boundary polygon
#boundary_polygon = Polygon(BOUNDARY_COORDS)
st.set_page_config(layout="wide", page_title="Ship Trajectory Simulation")

def is_within_bounds(lat, lon, location_style):
    point = Point(lon, lat)
    if location_style == "Rheinhafen":
        return any(polygon.contains(point) for polygon in polygons_rhein)
    elif location_style == "Cuxhaven":
        return any(polygon.contains(point) for polygon in polygons_cuxhaven)

    #return boundary_polygon.contains(point)



def create_proper_map(location_style):
    """Create a Folium map with all polygons displayed"""
    # Generate grid points for all polygons
    #grid_points_lat_lon = generate_grid_points_in_polygons(polygons_data, spacing_degrees=0.1)

    # Calculate center of all polygons for map initialization
    all_coords = []
    if location_style == "Rheinhafen":
        polygons_data = polygons_data_rhein
        for polygon_coords in polygons_data_rhein.values():
            all_coords.extend(polygon_coords)
    elif location_style == "Cuxhaven":
        polygons_data = polygons_data_cuxhaven
        for polygon_coords in polygons_data_cuxhaven.values():
            all_coords.extend(polygon_coords)

    avg_lat = np.mean([p[1] for p in all_coords])
    avg_lon = np.mean([p[0] for p in all_coords])

    folium_map = folium.Map(
        location=[avg_lat, avg_lon],
        zoom_start=13,
        tiles="OpenStreetMap"
    )



    # Add each polygon to the map with different styling
    for i, (polygon_name, polygon_coords) in enumerate(polygons_data.items()):

        # Create GeoJSON feature for this polygon
        polygon_geojson = {
            "type": "Feature",
            "properties": {"name": polygon_name},
            "geometry": {
                "type": "Polygon",
                "coordinates": [polygon_coords]
            }
        }

        folium.GeoJson(
            polygon_geojson,
            name="Monitoring Polygon",
            popup="Working Area Boundary",
            style_function=lambda x: {
                "fillColor": "#fdae61",
                "color": "#d7191c",
                "weight": 2,
                "fillOpacity": 0.3,
            },
        ).add_to(folium_map)

    # Add layer control to toggle polygons
    #folium.LayerControl().add_to(folium_map)

    return folium_map




# Prepare path for the simulation based on user selection
def prepare_path(los_coords, user_inputs):
    line_of_sight_reverse = los_coords[::-1]
    initial_x, initial_y = user_inputs[0]
    final_x, final_y = user_inputs[-1]

    use_reverse = (final_x - initial_x) < 0
    source_path = line_of_sight_reverse if use_reverse else los_coords

    x_min, x_max = min(initial_x, final_x), max(initial_x, final_x)
    return [pt for pt in source_path if x_min <= pt[0] <= x_max]

def simulate_ship_path(reduced_path, user_inputs):
    initial_x, initial_y = user_inputs[0]
    final_x, final_y = user_inputs[-1]
    initial_heading = (90 - physics_basedv2.calculate_bearing([initial_x, initial_y], reduced_path[1])) % 360
    params = physics_basedv2.ShipParameters(initial_x, initial_y, initial_heading)
    x_traj, y_traj, _, _ = physics_basedv2.simulate_motion(params, reduced_path, final_x, final_y)
    return x_traj[::10], y_traj[::10]  # Downsampling for smoother animation

def create_trajectory_df(x_traj, y_traj):
    return pd.DataFrame({'lat': x_traj, 'lon': y_traj})

# Create an animation figure
def create_animation_figure(trajectory_coords, los_coords, map_style="open-street-map"):
    fig = go.Figure()

    # Add static Line of Sight reference (orange line)
    fig.add_trace(go.Scattermap(
        lon=[c[1] for c in los_coords],
        lat=[c[0] for c in los_coords],
        mode='markers+lines',
        line=dict(width=3, color='orange'),
        name="Line of Sight",
        marker = {'size': 10}
    ))

    # Add the ship's path (blue line), generated by the physics model
    fig.add_trace(go.Scattermap(
        lon=[c[1] for c in trajectory_coords],
        lat=[c[0] for c in trajectory_coords],
        mode='lines',
        line=dict(width=2, color='blue'),
        name="Ship Path"
    ))

    # Add the ship marker (red) on the first point
    fig.add_trace(go.Scattermap(
        lon=[trajectory_coords[0][1]],
        lat=[trajectory_coords[0][0]],
        mode='markers',
        marker=dict(size=10, color='red'),
        name="Ship Position"
    ))

    # Create frames for animation
    frames = [
        go.Frame(
            data=[
                go.Scattermap(
                    lon=[c[1] for c in los_coords],
                    lat=[c[0] for c in los_coords],
                    mode='markers+lines',
                    marker=dict(size=6, color='green')
                ),
                go.Scattermap(
                    lon=[c[1] for c in trajectory_coords[:k+1]],
                    lat=[c[0] for c in trajectory_coords[:k+1]],
                    mode='markers+lines',
                    marker=dict(size=6, color='red')
                ),
                go.Scattermap(
                    lon=[trajectory_coords[k][1]],
                    lat=[trajectory_coords[k][0]],
                    mode='markers+lines',
                    marker=dict(size=10, color='red')
                )
            ],
            name=f'frame_{k}'
        )
        for k in range(len(trajectory_coords))
    ]
    fig.frames = frames


    fig.update_layout(
        width = 800,
        height = 500,
        map=dict(
            style=map_style,
            center=dict(
                lat=(trajectory_coords[0][0] + los_coords[0][0])/2,
                lon=(trajectory_coords[0][1] + los_coords[0][1])/2
            ),
            zoom=14
        ),
        updatemenus=[
            dict(
                type="buttons",
                showactive=True,
                buttons=[
                    dict(
                        label="▶️ Play",
                        method="animate",
                        args=[None, {"frame": {"duration": 200, "redraw": True},
                                    "fromcurrent": True, "mode": "immediate"}]
                    ),
                    dict(
                        label="⏹ Stop",
                        method="animate",
                        args=[[None], {"frame": {"duration": 0, "redraw": False},
                                     "mode": "immediate", "transition": {"duration": 0}}]
                    ),
                    dict(
                        label="⏮ Reset",
                        method="animate",
                        args=[None, {
                            "frame": {"duration": 0, "redraw": True},
                            "mode": "immediate",
                            "transition": {"duration": 0},
                            "frame": {"redraw": True}
                        }],
                        args2=[{"frame": {"duration": 0, "redraw": True},
                              "mode": "immediate",
                              "transition": {"duration": 0}}]
                    )
                ],
                bgcolor="rgba(255,255,255,0.1)",
                bordercolor="#DDDDDD",
                borderwidth=1,
                direction="left",
                pad={"r": 10, "t": 10, "b": 10},
                x=0.1,
                xanchor="right",
                y=1.1,
                yanchor="top"
            )
        ],
        margin={"r":0,"t":60,"l":0,"b":0},
        legend=dict(
            traceorder="normal",
            font=dict(size=12)
        )
    )

    return fig



# Main function
def main():
    st.markdown("""
        <style>
        /* Card styling */
        .css-1aumxhk {
            background-color: #ffffff;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            padding: 2rem;
            margin-bottom: 1.5rem;
        }

        /* Section headers */
        .stSubheader {
            color: #2c3e50;
            border-bottom: 2px solid #f0f2f6;
            padding-bottom: 0.5rem;
            margin-bottom: 1rem;
        }
        </style>
        """, unsafe_allow_html=True)

    with st.sidebar:
        st.title("Simulation Controls")
        location_style = st.selectbox("Choose Location", ["Rheinhafen", "Cuxhaven"])
        if st.button("Reset Points"):
            st.session_state.click_coords = {'start': None, 'end': None}
            st.session_state.simulation_data = None
            st.rerun()

        st.markdown("---")
        animation_speed = st.slider("Animation Speed (ms)", 50, 500, 200, 50)
        show_stats = st.checkbox("Show Statistics", True)
        map_style = st.selectbox("Map Style", ["open-street-map", "carto-voyager", "basic", "carto-darkmatter"])


    selection_map = create_proper_map(location_style)

    # Add markers if they exist
    if st.session_state.click_coords['start']:
        folium.Marker(
            location=st.session_state.click_coords['start'],
            popup=folium.Popup(f"Start Point<br>{st.session_state.click_coords['start'][0]:.4f}, {st.session_state.click_coords['start'][1]:.4f}", max_width=200),
            icon=folium.Icon(color='green', icon='ship', prefix='fa')
        ).add_to(selection_map)

    if st.session_state.click_coords['end']:
        folium.Marker(
            location=st.session_state.click_coords['end'],
            popup=folium.Popup(f"End Point<br>{st.session_state.click_coords['end'][0]:.4f}, {st.session_state.click_coords['end'][1]:.4f}", max_width=200),
            icon=folium.Icon(color='red', icon='flag-checkered', prefix='fa')
        ).add_to(selection_map)



    col1, col2 = st.columns([4, 1])

    with col1:

        st.subheader("1. Select Points on Map")
        map_data = st_folium(selection_map, width=800, height=500, returned_objects=["last_clicked"])

        # Handle map clicks
        if map_data and map_data.get("last_clicked"):
            click_lat = map_data["last_clicked"]["lat"]
            click_lon = map_data["last_clicked"]["lng"]

            if not is_within_bounds(click_lat, click_lon, location_style):
                st.warning("Please click inside the blue boundary area")
            else:
                if not st.session_state.click_coords['start']:
                    st.session_state.click_coords['start'] = [click_lat, click_lon]
                    st.rerun()
                elif not st.session_state.click_coords['end']:
                    st.session_state.click_coords['end'] = [click_lat, click_lon]
                    #st.rerun()


                if all(st.session_state.click_coords.values()):

                    user_inputs = [
                        st.session_state.click_coords['start'],
                        st.session_state.click_coords['end']
                    ]



                    polys_lonlat   = load_data(physics_basedv2.POLYGONS_JSON)
                    ports_latlon  = load_data(physics_basedv2.PORTLINES_JSON)
                    Rheinhafen_centerline_latlon  = load_data(physics_basedv2.RHEINHAFEN_CENTERLINE_JSON)
                    Cuxhaven_centerline_latlon  = load_data(physics_basedv2.CUXHAVEN_CENTERLINE_JSON)
                    start_coord = (st.session_state.click_coords['start'][0], st.session_state.click_coords['start'][1])
                    end_coord =(st.session_state.click_coords['end'][0], st.session_state.click_coords['end'][1])
                    if location_style == "Rheinhafen":
                        LoS_Line = physics_basedv2.build_route_port_aware_Rhein(start_coord,
                                                                      end_coord,
                                                                      polys_lonlat, ports_latlon, Rheinhafen_centerline_latlon)
                    elif location_style == "Cuxhaven":
                        LoS_Line =  physics_basedv2.lineofsight_Cuxhaven(start_coord,
                                                                          end_coord,
                                                                          Cuxhaven_centerline_latlon)
                    cte0 = physics_basedv2.first_cte(start_coord, LoS_Line)

                    LoS_Line.remove(LoS_Line[0])
                    LoS_Line.remove(LoS_Line[-2])
                    LoS_Line.remove(LoS_Line[-1])

                    if cte0 > 0:
                        side = "left"
                    else:
                        side = "right"

                    if location_style == "Rheinhafen":
                        line_offset = [start_coord] + physics_basedv2.offset_polyline_latlon(LoS_Line, offset_m=abs(20), side="right") + [end_coord]
                    elif location_style == "Cuxhaven":
                        line_offset = [start_coord] + physics_basedv2.offset_polyline_latlon(LoS_Line, offset_m=abs(cte0), side=side) + [end_coord]

                    simulation_path = line_offset
                    reduced_path = simulation_path[::1]
                    # Generate a trajectory using the physics-based model
                    #reduced_path = prepare_path(los_coords,user_inputs)  # Use user input to generate reduced path
                    x_traj, y_traj = simulate_ship_path(reduced_path, user_inputs)  # Simulated trajectory

                    trajectory_df = create_trajectory_df(x_traj, y_traj)


                    # Save simulation data in session for later use (e.g., for animation)
                    st.session_state.simulation_data = {
                        'trajectory': list(zip(trajectory_df['lat'], trajectory_df['lon'])),
                        'line_of_sight': reduced_path  # Static Line of Sight data for reference line
                    }
                    st.rerun()

        # Show simulation results if available
        if st.session_state.simulation_data:
            st.subheader("2. Simulation Results")
            fig = create_animation_figure(
                st.session_state.simulation_data['trajectory'],
                st.session_state.simulation_data['line_of_sight'],
                map_style
            )
            st.plotly_chart(fig, use_container_width=False)

    with col2:
        if show_stats:
            st.subheader("Selection Status")
            # Start point indicator
            if st.session_state.click_coords['start']:
                st.success(f"✅ Start Point Set: {st.session_state.click_coords['start'][0]:.4f}, {st.session_state.click_coords['start'][1]:.4f}")
            else:
                st.warning("⏳ Start Point: Not set")

        # End point indicator
            if st.session_state.click_coords['end']:
                st.error(f"⛔ End Point Set: {st.session_state.click_coords['end'][0]:.4f}, {st.session_state.click_coords['end'][1]:.4f}")
            else:
                st.info("ℹ️ End Point: Not set")

        # Progress bar
            progress_value = 0
            if st.session_state.click_coords['start']:
                progress_value += 0.5
            if st.session_state.click_coords['end']:
                progress_value += 0.5
            st.progress(progress_value)
        st.markdown("""
                <style>
                .small-font {
                    font-size:16px !important;
                }
                </style>
                """, unsafe_allow_html=True)

        st.markdown('<p class="small-font">Start: {:.4f}, {:.4f}</p>'.format(*st.session_state.click_coords['start']) if st.session_state.click_coords['start'] else 'Start Point: Not set', unsafe_allow_html=True)
        st.markdown('<p class="small-font">End: {:.4f}, {:.4f}</p>'.format(*st.session_state.click_coords['end']) if st.session_state.click_coords['end'] else 'End Point: Not set', unsafe_allow_html=True)
            # CSS Styling for Buttons (Global Style for all buttons)
        st.markdown("""
                    <style> .stButton>button {
                    background-color: #2ca02c;  /* Green background */
                    color: white;  /* White text */
                    border-radius: 12px;  /* Rounded corners */
                    padding: 10px;  /* Padding inside the button */
                    border: none;  /* No border */
                    font-size: 16px;  /* Font size */
                }
                .stButton>button:hover {
                    background-color: #206c1a;  /* Darker green when hovered */
                    }
                </style>""", unsafe_allow_html=True)
        if st.session_state.simulation_data:
            st.markdown("---")
            st.subheader("Simulation Stats")
            trajectory = st.session_state.simulation_data['trajectory']
            st.markdown('<p class="small-font">Trajectory Points: {}</p>'.format(len(trajectory)), unsafe_allow_html=True)
            st.markdown('<p class="small-font">Start: {:.4f}, {:.4f}</p>'.format(*trajectory[0]), unsafe_allow_html=True)
            st.markdown('<p class="small-font">End: {:.4f}, {:.4f}</p>'.format(*trajectory[-1]), unsafe_allow_html=True)

            trajectory_df = pd.DataFrame(trajectory, columns=['lat', 'lon'])
            st.download_button(
                label="Download Trajectory Data",
                data=trajectory_df.to_csv(index=False).encode('utf-8'),
                file_name='trajectory.csv',
                mime='text/csv'
            )

if __name__ == "__main__":
    main()
