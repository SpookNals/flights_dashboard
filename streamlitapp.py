import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import zipfile

st.set_page_config(
    page_title="Flights Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

df_count = pd.read_csv("airport_count_CLEAN.csv")

with zipfile.ZipFile('airport_merge_CLEAN.zip', 'r') as z:
        # Open the CSV file within the ZIP file
        with z.open('airport_merge_CLEAN.csv') as f:
            # Read the CSV file into a pandas DataFrame
            df_merge = pd.read_csv(f)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Airports", "Vertragingen", "Vlucht Data", "Data Cleaning", "Model"])

with tab1:

   # Selectbox for region
    region = st.selectbox("Select a region", df_count['Region'].unique())
    region_df = df_count[df_count['Region'] == region]
    grouped_df = region_df.groupby(['Name', 'City', 'Country', 'Latitude', 'Longitude'], as_index=False).first()

    # Center coordinates for the map
    center_lat = grouped_df['Latitude'].mean()
    center_lon = grouped_df['Longitude'].mean()

    # Create the map
    fig = go.Figure()

    # Add markers for each airport
    fig.add_trace(go.Scattermapbox(
        lat=grouped_df['Latitude'],
        lon=grouped_df['Longitude'],
        mode='markers',
        marker=dict(size=10, color='lightblue', opacity=0.7),  
        hovertemplate='%{text}<extra></extra>',  
        text=grouped_df.apply(lambda row: f"{row['Name']}, {row['City']}, {row['Country']}", axis=1),
    ))

    # Update the layout for the map
    fig.update_layout(
        mapbox=dict(
            style="carto-darkmatter",  
            center=dict(lat=center_lat, lon=center_lon),
            zoom=2,  
        ),
        showlegend=False,
        margin={"r": 0, "t": 30, "l": 0, "b": 0},
        width=800,
        height=800
    )

    # Display the map in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    st.write('---')
    st.header("Flights per Airport")

    # Selectbox for airport (based on the selected region)
    airport = st.selectbox("Select an airport", grouped_df['Name'].unique())

    airport_frequency_df = df_merge[df_merge['Name'] == airport]
    airport_frequency_df['STA_STD_ltc'] = pd.to_datetime(airport_frequency_df['STA_STD_ltc'], format='%Y-%m-%d %H:%M:%S')

    # Group by date and count the number of flights per day for arrivals and departures
    arrivals_df = airport_frequency_df[airport_frequency_df['LSV'] == 'Arrivals']
    departures_df = airport_frequency_df[airport_frequency_df['LSV'] == 'Departures']

    arrivals_count = arrivals_df.groupby(arrivals_df['STA_STD_ltc'].dt.date).size().reset_index(name='Aantal vluchten')
    arrivals_count.columns = ['Datum', 'Aantal vluchten']
    arrivals_count['Type'] = 'Arrival'  # Add Type column

    departures_count = departures_df.groupby(departures_df['STA_STD_ltc'].dt.date).size().reset_index(name='Aantal vluchten')
    departures_count.columns = ['Datum', 'Aantal vluchten']
    departures_count['Type'] = 'Departure'  # Add Type column

    # Combine arrivals and departures into one DataFrame
    combined_df = pd.concat([arrivals_count, departures_count])

    # Define custom color sequence for better contrast, red and blue
    color_sequence = ['#FF0000', '#0000FF']

    # Plotting with Plotly
    fig = px.line(combined_df, x='Datum', y='Aantal vluchten', color='Type',
                title=f'Aantal vluchten per dag voor {airport}',
                labels={'Datum': 'Datum', 'Aantal vluchten': 'Aantal vluchten'},
                color_discrete_sequence=color_sequence)  # Use custom color sequence

    # Rotate date labels for readability
    fig.update_xaxes(tickangle=45)

    # Adjust height and width to 800
    fig.update_layout(width=800)

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

with tab2:

    # selectbox voor vertrek of aankomst
    lsv = st.selectbox("Selecteer Vertrek of Aankomst", df_count['LSV'].unique())

    df_LSV_count = df_count[df_count['LSV'] == lsv]
    df_LSV_merge = df_merge[df_merge['LSV'] == lsv]


    # Center coordinates for the map
    center_lat = df_LSV_count['Latitude'].mean()
    center_lon = df_LSV_count['Longitude'].mean()

    # Function to determine the color based on delay
    def get_color(delay):
        if delay < 5:
            return 'green'
        elif 5 <= delay < 10:
            return 'orange'
        else:
            return 'red'

    # Create a list of colors based on the delay
    colors = df_LSV_count['delay_minutes'].apply(get_color)

    # Create the map
    fig = go.Figure()

    # Add the main data trace without showing its legend
    fig.add_trace(go.Scattermapbox(
        lat=df_LSV_count['Latitude'],
        lon=df_LSV_count['Longitude'],
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=10,
            color=colors,
            opacity=0.7,
        ),
        hovertemplate='%{text}<extra></extra>',  # Show text only on hover, no extra info
        text=df_LSV_count.apply(lambda row: f"{row['Name']}: {row['delay_minutes']} minutes delay. Based on {row['Count']} {lsv}", axis=1),
        showlegend=False  # Prevent the main data trace from appearing in the legend
    ))

    # Update the layout for the map
    fig.update_layout(
        mapbox=dict(
            style="carto-darkmatter",  # You can change to other styles
            center=dict(lat=center_lat, lon=center_lon),
            zoom=2,  # Adjust this zoom level as needed
            # 'bearing' and 'pitch' could be set for better orientation, but not necessary here
        ),
        showlegend=True,  # Show the legend for the custom traces
        margin={"r":0,"t":30,"l":0,"b":0},
        width=800,   # Set the desired width in pixels
        height=800   # Set the desired height in pixels
    )

    # Add traces for legend without plotting any actual points
    fig.add_trace(go.Scattermapbox(
            lat=[None],  # None means no point will be plotted
            lon=[None],
            mode='markers',
            marker=dict(color='green', size=10, symbol='circle'),
            name='Less than 5 min'  # Legend label
        ))

    fig.add_trace(go.Scattermapbox(
            lat=[None],
            lon=[None],
            mode='markers',
            marker=dict(color='orange', size=10, symbol='circle'),
            name='5-10 min'  # Legend label
        ))
        
    fig.add_trace(go.Scattermapbox(
            lat=[None],
            lon=[None],
            mode='markers',
            marker=dict(color='red', size=10, symbol='circle'),
            name='More than 10 min'  # Legend label
        ))

    # Display the map in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    st.write('---')

    # Create Streamlit app
    st.header("Airport Delay Analysis at Schiphol")

    # Create the scatter plot
    fig = px.scatter(df_LSV_merge,
                    x='STA_STD_ltc',
                    y='delay_minutes',
                    color='Gate_Changed',
                    title='Airport Delay Analysis at Schiphol',
                    labels={'STA_STD_ltc': 'Scheduled Arrival Time', 'delay_minutes': 'Delay (minutes)'},
                    color_discrete_sequence=['red', 'green'],  # Customize colors for True/False
                    symbol='Gate_Changed',
                    opacity=0.7,  # Different symbols for gate change
                    hover_data=['FLT']
                    )

    # Display the plot in Streamlit
    st.plotly_chart(fig)

    st.write('---')
    st.header("Average delay by gate adjustments")

    # Average delay for flights with and without gate changes
    vertraging_met_gate_verandering = df_LSV_merge[df_LSV_merge['Gate_Changed']]['delay_minutes'].mean()
    vertraging_zonder_gate_verandering = df_LSV_merge[~df_LSV_merge['Gate_Changed']]['delay_minutes'].mean()

    # Number of flights with and without gate changes
    aantal_met_gate_verandering = df_LSV_merge[df_LSV_merge['Gate_Changed']].shape[0]
    aantal_zonder_gate_verandering = df_LSV_merge[~df_LSV_merge['Gate_Changed']].shape[0]

    # Data for the chart
    labels = ['Met gate-verandering', 'Zonder gate-verandering']
    vertragingen = [vertraging_met_gate_verandering, vertraging_zonder_gate_verandering]

    # Create F-strings for the total flight count
    flight_count_text = [
        f'Totaal aantal vluchten: {aantal_met_gate_verandering}',
        f'Totaal aantal vluchten: {aantal_zonder_gate_verandering}'
    ]

    # Bar plot of average delay with Plotly
    fig = go.Figure()

    # Add bars for average delay
    fig.add_trace(go.Bar(
        x=labels,
        y=vertragingen,
        text=flight_count_text,  # Display the total flight count as text
        textposition='outside',  # Position text outside of the bars
        marker_color=['orange', 'blue'],
    ))

    # Update layout
    fig.update_layout(
        title='Gemiddelde vertraging bij gate-verandering',
        xaxis_title='Gate-verandering',
        yaxis_title='Gemiddelde vertraging (minuten)',
        yaxis=dict(range=[0, max(vertragingen) + 5]),  # Set y-axis range slightly higher than the highest value
        template='plotly_white'
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig)

    st.write('---')
    st.header("Top 10 Delays")
    chosen_airport = st.selectbox("Selecteer een luchthaven", df_LSV_merge['Name'].unique())
    df_name = df_LSV_merge[df_LSV_merge['Name'] == chosen_airport]
    col1, col2 = st.columns(2)

    df_vertraging = df_name[df_name['delay_minutes'] > 0]
    
   # Data groeperen per vliegtuigtype ('ACT') en de gemiddelde vertraging per vliegtuigtype berekenen
    gemiddelde_vertraging_per_act = df_vertraging.groupby('ACT')['delay_minutes'].mean().reset_index()

    # Kolomnamen aanpassen voor leesbaarheid
    gemiddelde_vertraging_per_act.columns = ['Vliegtuigtype', 'Gemiddelde vertraging (minuten)']

    # Sorteren op de gemiddelde vertraging om de grootste vertraging bovenaan te krijgen
    gemiddelde_vertraging_per_act = gemiddelde_vertraging_per_act.sort_values(by='Gemiddelde vertraging (minuten)', ascending=False)

    # Selecteer de eerste 20 vliegtuigtypes met de hoogste gemiddelde vertraging
    top_10_vertraging_per_act = gemiddelde_vertraging_per_act.head(10)

    # Plotten van de gemiddelde vertraging per vliegtuigtype (top 10) met Plotly, alleen vliegtuigtype
    fig = px.bar(top_10_vertraging_per_act, 
                title='Top 10 Vliegtuigtypes met Hoogste Gemiddelde Vertraging',
                x='Vliegtuigtype', 
                y='Gemiddelde vertraging (minuten)', 
                # Geen tekst (gate informatie) boven de balken
    )



    with col1:
        st.plotly_chart(fig)
    
    # Group data by gate ('GAT') and calculate average delay per gate
    gemiddelde_vertraging_per_gate = df_vertraging.groupby('GAT')['delay_minutes'].mean().reset_index().round(2)

    # Rename columns for readability
    gemiddelde_vertraging_per_gate.columns = ['Gate', 'Gemiddelde vertraging (minuten)']

    # Sort by average delay to get the highest delays on top
    gemiddelde_vertraging_per_gate = gemiddelde_vertraging_per_gate.sort_values(by='Gemiddelde vertraging (minuten)', ascending=False)

    # Select the top 10 gates with the highest average delay
    top_10_vertraging_per_gate = gemiddelde_vertraging_per_gate.head(10)

    # Create a bar plot of average delay per gate (top 10) with Plotly
    fig = px.bar(top_10_vertraging_per_gate, 
                 x='Gate', 
                 y='Gemiddelde vertraging (minuten)', 
                 title='Top 10 Gates met Hoogste Gemiddelde Vertraging',
                 labels={'Gemiddelde vertraging (minuten)': 'Gemiddelde vertraging (minuten)', 'Gate': 'Gate'},
                 text='Gemiddelde vertraging (minuten)')  # Show the delay above the bars

    # Update layout for better readability
    fig.update_layout(xaxis_title='Gate', yaxis_title='Gemiddelde vertraging (minuten)', xaxis_tickangle=-45)

    with col2:
        # Display the plot
        st.plotly_chart(fig)

with tab3:
    # Lees de data
    df = pd.read_csv("all_flights.csv")

    # Bereken de globale min en max waarden voor x en y assen
    x_min = df['Time (secs)'].min()
    x_max = df['Time (secs)'].max()
    altitude_max = df['[3d Altitude M]'].max()
    speed_max = df['SPEED (km/h)'].max()

    # Voeg een selectbox toe om een vlucht te kiezen, inclusief een "All Flights" optie
    flight_options = ["All Flights"] + list(df['Flight'].unique())
    selected_option = st.selectbox("Select a flight", flight_options)

    color_map = {
        'Flight 1': 'blue',
        'Flight 2': 'orange',
        'Flight 3': 'green',
        'Flight 4': 'red',
        'Flight 5': 'purple',
        'Flight 6': 'brown',
        'Flight 7': 'pink'
    }

    # Maak een subplot met 1 rij en 3 kolommen
    fig = make_subplots(
        rows=1, cols=3, 
        subplot_titles=(
            'Altitude and Time of the Flight(s)', 
            'Airspeed and Time of the Flight(s)',
            'Combined Altitude and Airspeed'
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": True}]]  # Gebruik secondary_y voor de derde grafiek
    )

    if selected_option == "All Flights":
        # Toon alle vluchten
        for flight in df['Flight'].unique():
            flight_data = df[df['Flight'] == flight]
            color = color_map.get(flight, 'gray')

            fig.add_trace(
                go.Scatter(x=flight_data['Time (secs)'], y=flight_data['[3d Altitude M]'], 
                        name=f'{flight} Altitude', mode='lines', legendgroup=f'{flight}', showlegend=True, line=dict(color=color)),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=flight_data['Time (secs)'], y=flight_data['SPEED (km/h)'], 
                        name=f'{flight} Airspeed', mode='lines', legendgroup=f'{flight}', showlegend=False, line=dict(color=color)),
                row=1, col=2
            )
            
            # Voeg beide y-assen toe aan de derde kolom (altitude op primaire as, airspeed op secundaire as)
            fig.add_trace(
                go.Scatter(x=flight_data['Time (secs)'], y=flight_data['[3d Altitude M]'], 
                        name=f'{flight} Altitude', mode='lines', legendgroup=f'{flight}', showlegend=False, line=dict(color=color)),
                row=1, col=3, secondary_y=False  # Altitude op de primaire as
            )
            fig.add_trace(
                go.Scatter(x=flight_data['Time (secs)'], y=flight_data['SPEED (km/h)'], 
                        name=f'{flight} Airspeed', mode='lines', legendgroup=f'{flight}', showlegend=False, line=dict(dash='dash', color=color)),
                row=1, col=3, secondary_y=True  # Airspeed op de secundaire as
            )
            
        title_text = "Flight Data Analysis - All Flights"
    else:
        # Toon alleen de geselecteerde vlucht
        flight_data = df[df['Flight'] == selected_option]
        color = color_map.get(selected_option, 'gray')  # Default to gray if flight not in color_map
        fig.add_trace(
            go.Scatter(x=flight_data['Time (secs)'], y=flight_data['[3d Altitude M]'], 
                    name=f'{selected_option} Altitude', mode='lines', legendgroup=f'{selected_option}', showlegend=True, line=dict(color=color)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=flight_data['Time (secs)'], y=flight_data['SPEED (km/h)'], 
                    name=f'{selected_option} Airspeed', mode='lines', legendgroup=f'{selected_option}', showlegend=False, line=dict(color=color)),
            row=1, col=2
        )
        
        # Voeg beide y-assen toe aan de derde kolom
        fig.add_trace(
            go.Scatter(x=flight_data['Time (secs)'], y=flight_data['[3d Altitude M]'], 
                    name=f'{selected_option} Altitude', mode='lines', legendgroup=f'{selected_option}', showlegend=False, line=dict(color=color)),
            row=1, col=3, secondary_y=False  # Altitude op de primaire as
        )
        fig.add_trace(
            go.Scatter(x=flight_data['Time (secs)'], y=flight_data['SPEED (km/h)'], 
                    name=f'{selected_option} Airspeed', mode='lines', legendgroup=f'{selected_option}', showlegend=False, line=dict(dash='dash', color=color)),
            row=1, col=3, secondary_y=True  # Airspeed op de secundaire as
        )
        
        title_text = f"Flight Data Analysis - {selected_option}"

    # Update de layout
    fig.update_layout(
        height=800,  # Verhoog de hoogte om ruimte te maken voor de legenda
        width=1100,  # Pas de breedte aan om ruimte te maken voor drie kolommen
        title_text=title_text,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,  # Plaats de legenda onder de grafieken
            xanchor="center",
            x=0.5,
            font=dict(size=10),
        )
    )

    # Update x-assen
    fig.update_xaxes(title_text="Time (secs)", range=[x_min, x_max], row=1, col=1)
    fig.update_xaxes(title_text="Time (secs)", range=[x_min, x_max], row=1, col=2)
    fig.update_xaxes(title_text="Time (secs)", range=[x_min, x_max], row=1, col=3)

    # Update y-assen
    fig.update_yaxes(title_text="Altitude (meters)", range=[0, altitude_max * 1.1], row=1, col=1)
    fig.update_yaxes(title_text="Airspeed (km/h)", range=[0, speed_max * 1.1], row=1, col=2)

    # Voor de derde grafiek (gecombineerde y-assen)
    fig.update_yaxes(title_text="Altitude (meters)", range=[0, altitude_max * 1.1], row=1, col=3, secondary_y=False)
    fig.update_yaxes(title_text="Airspeed (km/h)", range=[0, speed_max * 1.1], row=1, col=3, secondary_y=True)

    # Toon de plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    st.write('---')
    st.header("Flight Line")

    selected_line = st.selectbox("Select a flight", list(df['Flight'].unique()))

    df_flight_line = df[df['Flight'] == selected_line]

    # Define column names
    lat_column = '[3d Latitude]'
    lon_column = '[3d Longitude]'
    speed_column = 'SPEED (km/h)'
    altitude_column = '[3d Altitude M]'

    # Filter on unrealistic coordinates
    df_flight_line = df_flight_line[
        (df_flight_line[lat_column] >= -90) & 
        (df_flight_line[lat_column] <= 90) & 
        (df_flight_line[lon_column] >= -180) & 
        (df_flight_line[lon_column] <= 180)
    ]

    # Calculate the median of all speeds
    median_speed = df_flight_line[speed_column].median()

    # Create a new column for color (discrete) for markers only
    df_flight_line['Color'] = np.where(df_flight_line[speed_column] > median_speed, 'blue', 'red')

    # Create a scatter mapbox for the flight path with lines
    fig = go.Figure()

    # Add the flight path line (constant color) and colored markers
    fig.add_trace(go.Scattermapbox(
        lat=df_flight_line[lat_column],
        lon=df_flight_line[lon_column],
        mode='lines+markers',
        line=dict(width=2, color='white'),  # Set line color as gray or any constant color
        marker=dict(size=6, color=df_flight_line['Color'], showscale=False),  # Marker color based on speed
        hoverinfo='text',
        text=df_flight_line[speed_column].apply(lambda x: f"Snelheid: {x:.2f} Km/h"),  # Show speed on hover
        showlegend=False  # No need to show legend for this trace
    ))

    # Add invisible traces for legend
    fig.add_trace(go.Scattermapbox(
        lat=[None],  # None means no point will be plotted
        lon=[None],
        mode='markers',
        marker=dict(color='red', size=10, symbol='circle'),
        name='Minder snel'  # Legend label for red
    ))

    fig.add_trace(go.Scattermapbox(
        lat=[None],
        lon=[None],
        mode='markers',
        marker=dict(color='blue', size=10, symbol='circle'),
        name='Snel'  # Legend label for blue
    ))

    # Add statistics to the map
    stats_text = (
        f"<b>Minimum snelheid:</b> {df_flight_line[speed_column].min():.2f} Km/h<br>"
        f"<b>Maximum snelheid:</b> {df_flight_line[speed_column].max():.2f} Km/h<br>"
        f"<b>Gemiddelde snelheid:</b> {df_flight_line[speed_column].mean():.2f} Km/h<br>"
        f"<b>Mediaan snelheid:</b> {df_flight_line[speed_column].median():.2f} Km/h<br>"
        f"<b>Maximum hoogte:</b> {df_flight_line[altitude_column].max():.2f} M"
    )

    # Add annotations
    fig.update_layout(
        title="Flight Path",
        mapbox_style="carto-darkmatter",
        mapbox=dict(center={"lat": df_flight_line[lat_column].mean(), "lon": df_flight_line[lon_column].mean()}, zoom=4),
        width=700,
        height=800,
        annotations=[
            dict(
                x=0.95,  # Position in the x-axis (0 to 1)
                y=0.95,  # Position in the y-axis (0 to 1)
                xref="paper",
                yref="paper",
                text=stats_text,
                showarrow=False,
                bgcolor="rgba(255, 255, 255, 0.8)",  # White background with some transparency
                bordercolor="black",
                borderwidth=2,
                borderpad=4,
                font=dict(size=12, color="black"),
                align="left",
            )
        ]
    )

    # Show the map in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # Create a 3D scatter plot for the flight path with lines
    fig = go.Figure()

    # Add the flight path line (constant color) and colored markers in 3D
    fig.add_trace(go.Scatter3d(
        x=df_flight_line[lon_column],  # Longitude for x-axis
        y=df_flight_line[lat_column],  # Latitude for y-axis
        z=df_flight_line[altitude_column],  # Altitude for z-axis
        mode='lines+markers',
        line=dict(width=2, color='white'),  # Set line color as constant
        marker=dict(size=6, color=df_flight_line['Color'], showscale=False),  # Marker color based on speed
        hoverinfo='text',
        text=df_flight_line[speed_column].apply(lambda x: f"Snelheid: {x:.2f} Km/h"),  # Show speed on hover
        showlegend=False
    ))

    # Add invisible traces for legend
    fig.add_trace(go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode='markers',
        marker=dict(color='red', size=10, symbol='circle'),
        name='Minder snel'
    ))

    fig.add_trace(go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode='markers',
        marker=dict(color='blue', size=10, symbol='circle'),
        name='Snel'
    ))

    # Update the layout for 3D view
    fig.update_layout(
        title="3D Flight Path",
        scene=dict(
            xaxis_title="Longitude",
            yaxis_title="Latitude",
            zaxis_title="Altitude (M)",
            aspectmode='cube',  # Keep the aspect ratio cubic for a natural 3D look
        ),
        annotations=[
            dict(
                x=0.95,  # Position in the x-axis (0 to 1)
                y=0.95,  # Position in the y-axis (0 to 1)
                xref="paper",
                yref="paper",
                text=stats_text,
                showarrow=False,
                bgcolor="rgba(255, 255, 255, 0.8)",  # White background with some transparency
                bordercolor="black",
                borderwidth=2,
                borderpad=4,
                font=dict(size=12, color="black"),
                align="left",
            )
        ],
        width=700,
        height=800
    )

    # Show the 3D plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)


with tab4:   
    # Title for the tab
    st.title("Data Cleaning Process")

    # Introduce the cleaning process
    st.write("In this section, we demonstrate how the data was cleaned to prepare it for analysis.")

    # **1. Loading Data**
    st.subheader("1. Loading Data")
    loading_code = """
import pandas as pd

# Read the CSV file
df = pd.read_csv('airports-extended-clean.csv', sep=';')
"""
    st.code(loading_code, language='python')

    # show data head
    df_raw = pd.read_csv("airports-extended-clean.csv", sep=';')
    st.dataframe(df_raw.head())

    # **2. Filtering Data**
    st.subheader("2. Filtering Data")
    filtering_code = """
# Only use airports
df = df[df['Type'] == 'airport']

# Filter on OurAirports in the Source column
df = df[df['Source'] == 'OurAirports']
"""
    st.code(filtering_code, language='python')

    # **3. Data Type Conversion**
    st.subheader("3. Data Type Conversion")
    type_conversion_code = """
def comma_to_float(x):
    return x.replace(',', '.')

# Convert latitude and longitude from object to float
df['Latitude'] = df['Latitude'].apply(comma_to_float)
df['Longitude'] = df['Longitude'].apply(comma_to_float)

# Convert latitude and longitude to float
df['Latitude'] = df['Latitude'].astype(float)
df['Longitude'] = df['Longitude'].astype(float)
"""
    st.code(type_conversion_code, language='python')

    # **4. Merging Data**
    st.subheader("4. Merging Data")
    merging_code = """
# Read schedule_airport.csv
df_schedule = pd.read_csv('schedule_airport.csv')
"""
    st.code(merging_code, language='python')

    # show data head of schedule_airport.csv
    
    with zipfile.ZipFile('schedule_airport.zip', 'r') as z:
        # Open the CSV file within the ZIP file
        with z.open('schedule_airport.csv') as f:
            # Read the CSV file into a pandas DataFrame
            df_schedule = pd.read_csv(f)
    
    st.dataframe(df_schedule.head())
    merging_code = """
    # Merge the two dataframes on the ICAO column
merged_df = pd.merge(df_schedule, df, left_on='Org/Des', right_on='ICAO', how='inner')
"""
    st.code(merging_code, language='python')

    # **5. Datetime Conversion**
    st.subheader("5. Datetime Conversion")
    datetime_conversion_code = """
# Convert 'STD' (which contains the date) to datetime
merged_df['STD'] = pd.to_datetime(merged_df['STD'], format='%d/%m/%Y')

# Concatenate 'STD' (date) with 'STA_STD_ltc' and 'ATA_ATD_ltc' (time columns) to create full datetime objects
merged_df['STA_STD_ltc'] = pd.to_datetime(merged_df['STD'].astype(str) + ' ' + merged_df['STA_STD_ltc'], format='%Y-%m-%d %H:%M:%S')
merged_df['ATA_ATD_ltc'] = pd.to_datetime(merged_df['STD'].astype(str) + ' ' + merged_df['ATA_ATD_ltc'], format='%Y-%m-%d %H:%M:%S')
"""
    st.code(datetime_conversion_code, language='python')

    # **6. Delay Calculation**
    st.subheader("6. Delay Calculation")
    delay_calculation_code = """
# Calculate delay in minutes (ATA - STA)
merged_df['delay_minutes'] = (merged_df['ATA_ATD_ltc'] - merged_df['STA_STD_ltc']).dt.total_seconds() / 60

# Round the delay to 2 decimal places
merged_df['delay_minutes'] = merged_df['delay_minutes'].round(2)
"""
    st.code(delay_calculation_code, language='python')

    # **7. Cleaning Timezone Data**
    st.subheader("7. Cleaning Timezone Data")
    timezone_cleaning_code = """
# Split column Tz with delimiter '/' and take the first part, which is the region
merged_df['Tz'] = merged_df['Tz'].str.split('/').str[0]

# Rename the column Tz to Region
merged_df.rename(columns={'Tz': 'Region'}, inplace=True)

# Make the Region column a categorical type
merged_df['Region'] = merged_df['Region'].astype('category')
"""
    st.code(timezone_cleaning_code, language='python')

    # **8. Dropping Unnecessary Columns**
    st.subheader("8. Dropping Unnecessary Columns and fix some columns/values")
    drop_columns_code = """
# Drop the STD and DL1, IX1, DL2, IX2 columns
merged_df = merged_df.drop(columns=['STD', 'DL1', 'IX1', 'DL2', 'IX2', 'Type', 'Source', 'Identifier', 'IATA', 'RWC', 'Airport ID', 'Timezone', 'DST', 'ICAO'])


# drop value '-' from the column 'GAT' 
merged_df = merged_df[merged_df['GAT'] != '-']

# drop delays greater than 500 minutes and less than -100 minutes
merged_df = merged_df[(merged_df['delay_minutes'] < 500) & (merged_df['delay_minutes'] > -100)]

# add column that is either true or false if Gat equals Tar
merged_df['Gate_Changed'] = merged_df['GAT'] != merged_df['TAR']
"""
    st.code(drop_columns_code, language='python')

    # **9. Mapping Arrivals and Departures**
    st.subheader("9. Mapping Arrivals and Departures")
    mapping_code = """
# Map the LSV values to 'Arrivals' and 'Departures'
merged_df['LSV'] = merged_df['LSV'].map({'L': 'Arrivals', 'S': 'Departures'})

# Convert the 'LSV' column to a categorical type
merged_df['LSV'] = merged_df['LSV'].astype('category')
"""
    st.code(mapping_code, language='python')

    # **10. Filtering Arrivals and Departures**
    st.subheader("10. Filtering Arrivals and Departures")
    filtering_arrivals_departures_code = """
# Keep only arrivals
arrivals_df = merged_df[merged_df['LSV'] == 'Arrivals']  
departures_df = merged_df[merged_df['LSV'] == 'Departures']  
"""
    st.code(filtering_arrivals_departures_code, language='python')

    # **11. Calculating Average Delays**
    st.subheader("11. Calculating Average Delays")
    avg_delay_code = """
# Calculate average delays
average_delay_per_airport_arrival = departures_df.groupby('Name')['delay_minutes'].mean().reset_index().round(2)
average_delay_per_airport_departure = arrivals_df.groupby('Name')['delay_minutes'].mean().reset_index().round(2)
"""
    st.code(avg_delay_code, language='python')

    # **12. Preparing Datasets**
    st.subheader("12. Preparing Datasets")
    preparing_code = """
# Prepare datasets with all known airports
arrival_count = arrivals_df.groupby(['Name'], as_index=False).first()
arrival_count = arrival_count[['Name', 'Latitude', 'Longitude', 'Country', 'Org/Des', 'Region', 'LSV', 'City']]
airport_count = arrivals_df.groupby(['Name'], as_index=False).count()
airport_count = airport_count[['Name', 'Org/Des']]
airport_count.columns = ['Name', 'Count']

# Drop the Org/Des column from arrivals_df
arrivals_df = arrivals_df.drop(columns=['Org/Des'])

# Merge and sort count
arrival_count = pd.merge(arrival_count, airport_count, on='Name', how='inner')
arrival_count = arrival_count.sort_values(by='Count', ascending=False)
"""
    st.code(preparing_code, language='python')

    # **13. Merging Departures Datasets**
    st.subheader("13. Merging Departures Datasets")
    merging_departures_code = """
# Merge with previous data
arrival_count = pd.merge(arrival_count, average_delay_per_airport_arrival, on='Name', how='inner')

# Repeat for departures
departure_count = departures_df.groupby(['Name'], as_index=False).first()
departure_count = departure_count[['Name', 'Latitude', 'Longitude', 'Country', 'Org/Des', 'Region', 'LSV', 'City']]
airport_count = departures_df.groupby(['Name'], as_index=False).count()
airport_count = airport_count[['Name', 'Org/Des']]
airport_count.columns = ['Name', 'Count']
departures_df = departures_df.drop(columns=['Org/Des'])
departure_count = pd.merge(departure_count, airport_count, on='Name', how='inner')
departure_count = departure_count.sort_values(by='Count', ascending=False)
departure_count = pd.merge(departure_count, average_delay_per_airport_departure, on='Name', how='inner')
"""
    st.code(merging_departures_code, language='python')

    # **14. Final Filtering and Concatenation**
    st.subheader("14. Final Filtering and Concatenation")
    final_steps_code = """
# Define a custom threshold for "too few" entries
custom_threshold = 25

# Filter airports that have counts below this threshold
arrival_count = arrival_count[arrival_count['Count'] > custom_threshold]
arrivals_df = arrivals_df[arrivals_df['Name'].isin(arrival_count['Name'])]

# Same for departures
departure_count = departure_count[departure_count['Count'] > custom_threshold]
departures_df = departures_df[departures_df['Name'].isin(departure_count['Name'])]

# Put arrivals_df and departures_df together
airport_df_clean = pd.concat([arrivals_df, departures_df])
"""
    st.code(final_steps_code, language='python')

    # **15. Outputting Cleaned Data**
    st.subheader("15. Outputting Cleaned Data")
    output_code = """
# Write to CSV
airport_df_clean.to_csv('airport_merge_CLEAN.csv', index=False)

# Prepare airport count dataset and write to CSV
airport_count_clean = pd.concat([arrival_count, departure_count])
airport_count_clean.to_csv('airport_count_CLEAN.csv', index=False)
"""
    st.code(output_code, language='python')

    # **16. Final Cleaned Data**

    st.subheader("Final Cleaned Data (Count)")
    st.dataframe(df_count.head())

    st.subheader("Final Cleaned Data (Merge)")
    st.dataframe(df_merge.head())

    st.write('---')
    st.title("Flight Data Cleaning Process")
    cleaning_code = """
# Als eerst heb ik op deze manier de data voorbewerkt. Het is namelijk zo dat in sommige kolommen een * stond voor het getal waardoor die niet gezien
# werd als een float

import pandas as pd
import os

folder = 'case3'
files = os.listdir(folder)

# Only keep the files that start with '30'
files = [file for file in files if file.startswith('30')]

# Add the files to the folder path
list_with_paths = [os.path.join(folder, file) for file in files]

# Initialize flight number and an empty list to store dataframes
flight_number = 1
all_flights_data = []

for path in list_with_paths:
    vlucht = pd.read_excel(path, dtype=str)

    # Verwijder sterretjes uit een specifieke kolom
    vlucht['[3d Altitude M]'] = vlucht['[3d Altitude M]'].str.replace('*', '', regex=False)

    # Zet de kolom om naar een float voor numerieke berekeningen
    vlucht['[3d Altitude M]'] = pd.to_numeric(vlucht['[3d Altitude M]'], errors='coerce')
    vlucht['TRUE AIRSPEED (derived)'] = pd.to_numeric(vlucht['TRUE AIRSPEED (derived)'], errors='coerce')
    vlucht['Time (secs)'] = pd.to_numeric(vlucht['Time (secs)'], errors='coerce')

    # Pak het begin en eind van de vlucht
    vlucht_start = vlucht[vlucht['[3d Altitude M]'] == -1.2]
    vlucht_end = vlucht[vlucht['[3d Altitude M]'] == 6.4]

    # Save the first and last row
    vlucht_first_row = vlucht_start.iloc[-1]  # -1 because ascending is on the last row of the start
    vlucht_last_row = vlucht_end.iloc[0]      # 0 because landing is on the first row of the end

    # Drop the rows with altitude -1.2 and 6.4
    vlucht = vlucht[vlucht['[3d Altitude M]'] != -1.2]
    vlucht = vlucht[vlucht['[3d Altitude M]'] != 6.4]

    # Put the first row back at the beginning of the dataframe
    vlucht = pd.concat([vlucht_first_row.to_frame().T, vlucht])
    # Put the last row back at the end of the dataframe
    vlucht = pd.concat([vlucht, vlucht_last_row.to_frame().T])

    # Reset the index
    vlucht.reset_index(drop=True, inplace=True)

    # Add the flight number as a column (not index)
    vlucht['Flight'] = f'Flight {flight_number}'

    # Add the processed dataframe to the list
    all_flights_data.append(vlucht)

    # Increment flight number for the next iteration
    flight_number += 1

# Concatenate all flight dataframes into a single dataframe
all_flights = pd.concat(all_flights_data, ignore_index=True)

# Drop rows with missing values
all_flights = all_flights.dropna()

# Change the values in column TRUE AIRSPEED (derived) from knots to km/h
all_flights['TRUE AIRSPEED (derived)'] = (all_flights['TRUE AIRSPEED (derived)'] * 1.852).round(2)

# Rename the column TRUE AIRSPEED (derived) to SPEED (km/h)
all_flights = all_flights.rename(columns={'TRUE AIRSPEED (derived)': 'SPEED (km/h)'})

# Set the 'Flight' column as the index
all_flights.set_index('Flight', inplace=True)

# Write to CSV
all_flights.to_csv('all_flights.csv')
"""
    st.code(cleaning_code, language='python')
    cleaned_flight_data = pd.read_csv("all_flights.csv")
    st.dataframe(cleaned_flight_data.head())
    st.dataframe(cleaned_flight_data.tail())

with tab5:
    code = """
import plotly.graph_objs as go
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


        # Filter alleen Schiphol Airport
df = airport_df_clean[airport_df_clean['Name'].str.contains('Schiphol', case=False)]

        # Zet de tijdkolommen om naar datetime
df['STA_STD_ltc'] = pd.to_datetime(df['STA_STD_ltc'])
df['ATA_ATD_ltc'] = pd.to_datetime(df['ATA_ATD_ltc'])

        # Voeg een kolom toe met de vertraging in minuten
df['delay_minutes'] = (df['ATA_ATD_ltc'] - df['STA_STD_ltc']).dt.total_seconds() / 60

        # Voeg een kolom toe met de dag van de week
df['day_of_week'] = df['STA_STD_ltc'].dt.day_name()

        # Voeg een nieuwe kolom toe voor dinsdag of woensdag (1 als dinsdag of woensdag, anders 0)
df['is_tuesday_or_wednesday'] = df['day_of_week'].apply(lambda x: 1 if x in ['Tuesday', 'Wednesday'] else 0)

        # Voeg een nieuwe kolom toe voor vóór of na COVID-19 (1 voor vóór, 0 voor na)
df['before_covid'] = df['STA_STD_ltc'].apply(lambda x: 1 if x < pd.Timestamp('2020-03-01') else 0)

        # Voeg een kolom toe voor LSV (1 voor Arrivals, 0 voor Departures)
df['LSV_binary'] = df['LSV'].apply(lambda x: 1 if x == 'Arrivals' else 0)

        # Step 3: Groeperen op vluchtcode (FLT) en het gemiddelde van de vertraging berekenen
df_avg_delay = df.groupby('FLT')['delay_minutes'].mean().reset_index()

        # Step 4: Voeg een nieuwe kolom toe met de categorieën
df_avg_delay['delay_category'] = df_avg_delay['delay_minutes'].apply(
            lambda x: 1 if x > 15 else 0)  # 1 als vertraging > 15 minuten, anders 0

        # Step 5: Voeg de delay_category toe aan het oorspronkelijke dataframe op basis van FLT
df = pd.merge(df, df_avg_delay[['FLT', 'delay_category']], on='FLT', how='left')

        # Step 6: Kwadrateer de delay_category
df['delay_category'] = df['delay_category'] ** 2  # Delay category in het kwadraat

        # Step 7: Voeg een kolom toe met de 8e machtswortel van de vertraging in minuten
df['eighth_root_delay_minutes'] = np.power(df['delay_minutes'].clip(lower=0), 1 / 8)  # 8e machtswortel, clip voor negatieve waarden

        # Step 8: Definieer onafhankelijke en afhankelijke variabelen voor het regressiemodel
X = df[['delay_category', 'is_tuesday_or_wednesday', 'before_covid', 'LSV_binary']]  # Onafhankelijke variabelen
y = df['eighth_root_delay_minutes']  # Afhankelijke variabele (8e machtswortel van vertraging)

    # Voeg een constante toe aan het model
X = sm.add_constant(X)

    # Step 9: Splits de data in trainings- en testsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 10: Maak het lineaire regressiemodel aan
model = sm.OLS(y_train, X_train).fit()

        # Step 11: Bekijk de modelresultaten
#write model summary to a file
with open('model_summary.txt', 'w') as f:
    f.write(model.summary().as_text())

        # Step 12: Voorspel op de testset
predictions_eighth_root = model.predict(X_test)

        # Step 13: Verhef de voorspellingen tot de 8e macht om ze terug te transformeren naar de oorspronkelijke schaal
predictions = np.power(predictions_eighth_root, 8)

        # Step 14: Evalueer het model met de RMSE op de originele schaal
rmse = np.sqrt(np.mean((predictions - np.power(y_test, 8)) ** 2))  # Vergelijk met de 8e machts originele y_test
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Create the residual plot using Plotly
residuals = predictions_eighth_root - y_test 

fig = go.Figure()

# Add scatter plot for residuals
fig.add_trace(go.Scatter(
    x=predictions,
    y=residuals,
    mode='markers',
    marker=dict(color='blue', size=10, opacity=0.5),
    name='Residuals'
))

# Add horizontal line at y=0
fig.add_shape(type="line", x0=min(predictions), x1=max(predictions), y0=0, y1=0,
              line=dict(color='red', dash='dash'))

# Customize layout
fig.update_layout(
    title="Residual Plot",
    xaxis_title="Voorspelde vertraging (in minuten)",
    yaxis_title="Residuen (Voorspelde - Werkelijke)",
    height=500,  # Reduced plot height for better display
    width=800  # Adjusted plot width
)

# save the fig
fig.write_html('residual_plot.html')
"""
    import plotly.graph_objs as go
    import statsmodels.api as sm
    from sklearn.model_selection import train_test_split

            # Filter alleen Schiphol Airport
    df = df_merge[df_merge['Name'].str.contains('Schiphol', case=False)]

            # Zet de tijdkolommen om naar datetime
    df['STA_STD_ltc'] = pd.to_datetime(df['STA_STD_ltc'])
    df['ATA_ATD_ltc'] = pd.to_datetime(df['ATA_ATD_ltc'])

            # Voeg een kolom toe met de vertraging in minuten
    df['delay_minutes'] = (df['ATA_ATD_ltc'] - df['STA_STD_ltc']).dt.total_seconds() / 60

            # Voeg een kolom toe met de dag van de week
    df['day_of_week'] = df['STA_STD_ltc'].dt.day_name()

            # Voeg een nieuwe kolom toe voor dinsdag of woensdag (1 als dinsdag of woensdag, anders 0)
    df['is_tuesday_or_wednesday'] = df['day_of_week'].apply(lambda x: 1 if x in ['Tuesday', 'Wednesday'] else 0)

            # Voeg een nieuwe kolom toe voor vóór of na COVID-19 (1 voor vóór, 0 voor na)
    df['before_covid'] = df['STA_STD_ltc'].apply(lambda x: 1 if x < pd.Timestamp('2020-03-01') else 0)

            # Voeg een kolom toe voor LSV (1 voor Arrivals, 0 voor Departures)
    df['LSV_binary'] = df['LSV'].apply(lambda x: 1 if x == 'Arrivals' else 0)

            # Step 3: Groeperen op vluchtcode (FLT) en het gemiddelde van de vertraging berekenen
    df_avg_delay = df.groupby('FLT')['delay_minutes'].mean().reset_index()

            # Step 4: Voeg een nieuwe kolom toe met de categorieën
    df_avg_delay['delay_category'] = df_avg_delay['delay_minutes'].apply(
                lambda x: 1 if x > 15 else 0)  # 1 als vertraging > 15 minuten, anders 0

            # Step 5: Voeg de delay_category toe aan het oorspronkelijke dataframe op basis van FLT
    df = pd.merge(df, df_avg_delay[['FLT', 'delay_category']], on='FLT', how='left')

            # Step 6: Kwadrateer de delay_category
    df['delay_category'] = df['delay_category'] ** 2  # Delay category in het kwadraat

            # Step 7: Voeg een kolom toe met de 8e machtswortel van de vertraging in minuten
    df['eighth_root_delay_minutes'] = np.power(df['delay_minutes'].clip(lower=0), 1 / 8)  # 8e machtswortel, clip voor negatieve waarden

            # Step 8: Definieer onafhankelijke en afhankelijke variabelen voor het regressiemodel
    X = df[['delay_category', 'is_tuesday_or_wednesday', 'before_covid', 'LSV_binary']]  # Onafhankelijke variabelen
    y = df['eighth_root_delay_minutes']  # Afhankelijke variabele (8e machtswortel van vertraging)

        # Voeg een constante toe aan het model
    X = sm.add_constant(X)

        # Step 9: Splits de data in trainings- en testsets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Step 10: Maak het lineaire regressiemodel aan
    model = sm.OLS(y_train, X_train).fit()

            # Step 11: Bekijk de modelresultaten
    #streamlit model summary
    st.write(model.summary())

            # Step 12: Voorspel op de testset
    predictions_eighth_root = model.predict(X_test)

            # Step 13: Verhef de voorspellingen tot de 8e macht om ze terug te transformeren naar de oorspronkelijke schaal
    predictions = np.power(predictions_eighth_root, 8)

            # Step 14: Evalueer het model met de RMSE op de originele schaal
    rmse = np.sqrt(np.mean((predictions - np.power(y_test, 8)) ** 2))  # Vergelijk met de 8e machts originele y_test
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

    # Create the residual plot using Plotly
    residuals = predictions_eighth_root - y_test 

    fig = go.Figure()

    # Add scatter plot for residuals
    fig.add_trace(go.Scatter(
        x=predictions,
        y=residuals,
        mode='markers',
        marker=dict(color='lightblue', size=10, opacity=0.5),
        name='Residuals'
    ))

    # Add horizontal line at y=0
    fig.add_shape(type="line", x0=min(predictions), x1=max(predictions), y0=0, y1=0,
                line=dict(color='red', dash='dash'))

    # Customize layout
    fig.update_layout(
        title="Residual Plot",
        xaxis_title="Voorspelde vertraging (in minuten)",
        yaxis_title="Residuen (Voorspelde - Werkelijke)",
        height=500,  # Reduced plot height for better display
        width=800  # Adjusted plot width
    )

    # show fig
    st.plotly_chart(fig, use_container_width=True)
