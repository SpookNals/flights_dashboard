import streamlit as st
import pandas as pd
import plotly.graph_objects as go
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

center_lat = 0
center_lon = 0

tab1, tab2, tab3, tab4 = st.tabs(["Airports", "Vertragingen", "Vlucht Data", "Data Cleaning"])

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
        marker=dict(size=10, color='blue', opacity=0.7),  
        hovertemplate='%{text}<extra></extra>',  
        text=grouped_df.apply(lambda row: f"{row['Name']}, {row['City']}, {row['Country']}", axis=1),
    ))

    # Update the layout for the map
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",  
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

with tab2:

    # selectbox voor vertrek of aankomst
    lsv = st.selectbox("Selecteer Vertrek of Aankomst", df_count['LSV'].unique())

    df_LSV = df_count[df_count['LSV'] == lsv]

    # Center coordinates for the map
    center_lat = df_LSV['Latitude'].mean()
    center_lon = df_LSV['Longitude'].mean()

    # Function to determine the color based on delay
    def get_color(delay):
        if delay < 5:
            return 'green'
        elif 5 <= delay < 10:
            return 'orange'
        else:
            return 'red'

    # Create a list of colors based on the delay
    colors = df_LSV['delay_minutes'].apply(get_color)

    # Create the map
    fig = go.Figure()

    # Add the main data trace without showing its legend
    fig.add_trace(go.Scattermapbox(
        lat=df_LSV['Latitude'],
        lon=df_LSV['Longitude'],
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=10,
            color=colors,
            opacity=0.7,
        ),
        hovertemplate='%{text}<extra></extra>',  # Show text only on hover, no extra info
        text=df_LSV.apply(lambda row: f"{row['Name']}: {row['delay_minutes']} minutes delay. Based on {row['Count']} {lsv}", axis=1),
        showlegend=False  # Prevent the main data trace from appearing in the legend
    ))

    # Update the layout for the map
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",  # You can change to other styles
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

    # Maak een subplot met 1 rij en 2 kolommen
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Altitude and Time of the Flight(s)', 'Airspeed and Time of the Flight(s)'))

    if selected_option == "All Flights":
        # Toon alle vluchten
        for flight in df['Flight'].unique():
            flight_data = df[df['Flight'] == flight]
            color = color_map.get(flight, 'gray')

            fig.add_trace(
            go.Scatter(x=flight_data['Time (secs)'], y=flight_data['[3d Altitude M]'], 
                       name=f'{flight}', mode='lines', legendgroup=f'{flight}', showlegend=True, line=dict(color=color)),
            row=1, col=1
            )
            fig.add_trace(
            go.Scatter(x=flight_data['Time (secs)'], y=flight_data['SPEED (km/h)'], 
                       name=f'{flight}', mode='lines', legendgroup=f'{flight}', showlegend=False, line=dict(color=color)),
            row=1, col=2
            )
        title_text = "Flight Data Analysis - All Flights"
    else:
        # Toon alleen de geselecteerde vlucht
        flight_data = df[df['Flight'] == selected_option]
        color = color_map.get(selected_option, 'gray')  # Default to gray if flight not in color_map
        fig.add_trace(
            go.Scatter(x=flight_data['Time (secs)'], y=flight_data['[3d Altitude M]'], 
                    name=f'{selected_option}', mode='lines', legendgroup=f'{selected_option}', showlegend=True, line=dict(color=color)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=flight_data['Time (secs)'], y=flight_data['SPEED (km/h)'], 
                    name=f'{selected_option}', mode='lines', legendgroup=f'{selected_option}', showlegend=False, line=dict(color=color)),
            row=1, col=2
        )
        title_text = f"Flight Data Analysis - {selected_option}"

    # Update de layout
    fig.update_layout(
        height=800,  # Verhoog de hoogte om ruimte te maken voor de legenda
        width=700, 
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

    # Update y-assen
    fig.update_yaxes(title_text="Altitude (meters)", range=[0, altitude_max * 1.1], row=1, col=1)
    fig.update_yaxes(title_text="Airspeed (km/h)", range=[0, speed_max * 1.1], row=1, col=2)

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
        f"<b>Mediaan snelheid:</b> {df_flight_line[speed_column].median():.2f} Km/h"
    )

    # Add annotations
    fig.update_layout(
        title="Flight Path",
        mapbox_style="open-street-map",
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
    st.subheader("8. Dropping Unnecessary Columns")
    drop_columns_code = """
# Drop the STD and DL1, IX1, DL2, IX2 columns
merged_df = merged_df.drop(columns=['STD', 'DL1', 'IX1', 'DL2', 'IX2', 'Type', 'Source', 'Identifier', 'IATA', 'RWC', 'Airport ID', 'Timezone', 'DST', 'ICAO'])
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
    df_cleaned_count = pd.read_csv("airport_count_CLEAN.csv")
    df_cleaned_merge = pd.read_csv("airport_merge_CLEAN.csv")

    st.subheader("Final Cleaned Data (Count)")
    st.dataframe(df_cleaned_count.head())

    st.subheader("Final Cleaned Data (Merge)")
    st.dataframe(df_cleaned_merge.head())

    st.write('---')
    st.title("Flight Data Cleaning Process")
    cleaning_code = """
folder = 'case3'
files = os.listdir(folder)
list_with_paths = [os.path.join(folder, file) for file in files]

vlucht1 = pd.read_excel(list_with_paths[7], dtype=str)

# Verwijder sterretjes uit een specifieke kolom (vervang 'KolomNaam' met de juiste kolomnaam)
vlucht1['[3d Altitude M]'] = vlucht1['[3d Altitude M]'].str.replace('*', '', regex=False)

# Zet de kolom om naar een float (voor numerieke berekeningen)
vlucht1['[3d Altitude M]'] = pd.to_numeric(vlucht1['[3d Altitude M]'], errors='coerce')
vlucht1['TRUE AIRSPEED (derived)'] = pd.to_numeric(vlucht1['TRUE AIRSPEED (derived)'], errors='coerce')
vlucht1['Time (secs)'] = pd.to_numeric(vlucht1['Time (secs)'], errors='coerce')

# save the first and last row
vlucht1_first_row = vlucht1.iloc[0]
vlucht1_last_row = vlucht1.iloc[-1]

# drop the rows with altitude -1.2 and 6.4
vlucht1 = vlucht1[vlucht1['[3d Altitude M]'] != -1.2]
vlucht1 = vlucht1[vlucht1['[3d Altitude M]'] != 6.4]

# put the first row back in the beginnning of the dataframe
vlucht1 = pd.concat([vlucht1_first_row.to_frame().T, vlucht1])
# put the last row back in the end of the dataframe
vlucht1 = pd.concat([vlucht1, vlucht1_last_row.to_frame().T])
"""
    st.code(cleaning_code, language='python')
    cleaning_code = """
# Combineer alle vluchtdata in één DataFrame
all_flights = pd.concat([vlucht1, vlucht2, vlucht3, vlucht4, vlucht5, vlucht6, vlucht7], 
                            keys=['Flight 1', 'Flight 2', 'Flight 3', 'Flight 4', 'Flight 5', 'Flight 6', 'Flight 7'])
all_flights = all_flights.reset_index(level=0)
all_flights = all_flights.rename(columns={'level_0': 'Flight'})

# if a row has non values, drop it
all_flights = all_flights.dropna()

all_flights['TRUE AIRSPEED (derived)'] = all_flights['TRUE AIRSPEED (derived)'].astype(float)

# change the values in column TRUE AIRSPEED (derived) from knopen to km/h
all_flights['TRUE AIRSPEED (derived)'] = (all_flights['TRUE AIRSPEED (derived)'] * 1.852).round(2)
# rename the column TRUE AIRSPEED (derived) to SPEED (km/h)
all_flights = all_flights.rename(columns={'TRUE AIRSPEED (derived)': 'SPEED (km/h)'})

# write to csv file
all_flights.to_csv('all_flights.csv', index=False)

"""
    st.code(cleaning_code, language='python')
    cleaned_flight_data = pd.read_csv("all_flights.csv")
    st.dataframe(cleaned_flight_data.head())
    st.dataframe(cleaned_flight_data.tail())