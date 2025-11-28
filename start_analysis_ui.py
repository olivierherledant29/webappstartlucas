##################################################################################################################
# We want to write a ui in dash to :
# - select the boat ref from the boat_list_default - default to 'FRA' in a drop down
# - select the list of boats to take into account for the analysis from the boat_list_default - default to all boats - tick boxes
# - select the end time which will be in UTC - default to now time in UTC
# - select the amount of time before the end time to get the data from InfluxDB, default to 2 minutes
# - have a button to get the data from InfluxDB and get the start times in that time range
# - plot the last analyzed start in the time range for the selected boat ref
# - plot the last ttk start analysis for the selected boat ref
# - Use the functions from start_analysis_live.py to get the last analyzed start and plot the graphs

################################################################################################################### 
# Import necessary libraries
import dash
import dash_mantine_components as dmc
from dash import Dash, dcc, html, Input, Output, State
import plotly.graph_objs as go

from datetime import datetime, timedelta
import pandas as pd
import arrow 
import pickle
import os

from src.analysis.app.start_analysis_live import *

from influxdb_client import InfluxDBClient

from scipy.interpolate import RegularGridInterpolator

####################################################################################################################
# Load polar data
polar_csv = r"C:\Users\LucasDelcourt\Downloads\2511_m21_LAW2_LAB2_LARW2.kph.csv"

# Preload data
polar_df = pd.read_csv(polar_csv, index_col=0)
polar_df.index = polar_df.index.astype(float)
polar_df.columns = polar_df.columns.astype(float)

# Build interpolator
twa_grid = polar_df.index.values
tws_grid = polar_df.columns.values
speeds = polar_df.values
interp_func = RegularGridInterpolator((twa_grid, tws_grid), speeds, bounds_error=False, fill_value=0.0)

####################################################################################################################
# Define a default boat list
boat_list_default = ['AUS', 'BRA', 'CAN', 'GBR', 'GER', 'ITA', 'SUI', 'USA', 'DEN', 'FRA', 'ESP', 'NZL']

# Parameter to control TTK plot type
USE_SUBPLOT_TTK_PLOT = True  # Set to True for new subplot version, False for original

# Define a default start time which will be 30s from the current time in UTC in the format '2025-09-13T11:38:30.000Z'
start_time_default = (arrow.utcnow().shift(seconds = -120)).format('YYYY-MM-DDTHH:mm:ss') + '.000Z'
# Define a default end time which will be the current time in UTC in the format '2025-09-13T11:53:30.000Z'
end_time_default = arrow.utcnow().format('YYYY-MM-DDTHH:mm:ss') + '.000Z'

###################################################################################################################
## Get the data from the InfluxDB
org = '0c2a130d50b8facc'
token = '2vTlG__z6bc7bibptc1FE_gXRwK6761dmxW_sasiAC1qsNqwbAbAj0PJD9yRIQPR0bfwdl_4-S_5gIecgkfz_Q=='
url = 'https://data.sailgp.tech'

client = InfluxDBClient(url=url, token=token, org=org, timeout=1080_000, verify_ssl=False)

###################################################################################################################
# Define the default channel names list
channel_names = [
    "TWS_MHU_SGP_km_h_1",
    "TWA_MHU_SGP_deg",
    "TWD_MHU_SGP_deg",
    "TWS_BOW_SGP_km_h_1",
    "TWA_BOW_SGP_deg",
    "TWD_BOW_SGP_deg",
    "LATITUDE_GPS_unk",
    "LONGITUDE_GPS_unk",
    "BOAT_SPEED_km_h_1",
    "HEADING_deg",
    "DB_STOW_STATE_P_unk",
    "DB_STOW_STATE_S_unk",
    "RATE_YAW_deg_s_1",
    "TRK_RACE_NUM_unk",
    "TRK_LEG_NUM_unk",
    "TTS_s",
    "PC_TTS_s",
    "PC_TTK_s",
    "PC_DTL_m",
    "PC_START_LINE_PER_pct",
    "PC_START_RATIO_unk",
    "TIME_TC_START_s",
    "SEL_WAND_SGP_unk"
]

# Define the mark list
mark_list = ['M1', 'SL1', 'SL2']

# Define the Channel names list
channel_names_marks = [
    "TWS_MDSS_km_h_1",
    "TWD_MDSS_deg",
    "LATITUDE_MDSS_deg",
    "LONGITUDE_MDSS_deg",
]

###################################################################################################################
# Define the app

app = Dash(__name__, external_stylesheets=dmc.styles.ALL)

###################################################################################################################
# Define the app layout

app.layout = dmc.MantineProvider([
    # Data store to hold analysis results
    dcc.Store(id='analysis-data-store', data={}),
    
    dmc.Container([
        dmc.Title("SailGP Start Analysis Tool", order=1, style={"textAlign": "center", "marginBottom": "30px"}),
        
        # Practice Start Toggle
        dmc.Card([
            dmc.CardSection([
                dmc.Group([
                    dmc.Switch(
                        id='practice-start-toggle',
                        label="Practice Start Mode",
                        description="Use PC_TTC_s for start detection and analyze only reference boat",
                        checked=False,
                        size="md",
                        color="orange"
                    ),
                ], justify="center"),
            ], withBorder=True, inheritPadding=True, py="sm")
        ], withBorder=True, shadow="sm", radius="md", style={"marginBottom": "20px"}),
        
        # Data Selection Section
        dmc.Card([
            dmc.CardSection([
                dmc.Title("Data Selection", order=3, c="blue"),
                
                # Boat Reference Selection
                dmc.Group([
                    dmc.Stack([
                        dmc.Text("Reference Boat:", fw=500),
                        dcc.Dropdown(
                            id='boat-ref-dropdown',
                            options=[{'label': boat, 'value': boat} for boat in boat_list_default],
                            value='FRA',
                            style={'width': '150px'}
                        ),
                    ], gap="xs"),
                    
                    # Time Selection
                    dmc.Stack([
                        dmc.Text("End Time (UTC):", fw=500),
                        dcc.Input(
                            id='end-time-input',
                            type='text',
                            value=end_time_default,
                            style={'width': '200px'}
                        ),
                    ], gap="xs"),
                    
                    dmc.Stack([
                        dmc.Text("Minutes Before:", fw=500),
                        dcc.Input(
                            id='minutes-before-input',
                            type='number',
                            value=2,
                            min=1,
                            max=60,
                            style={'width': '100px'}
                        ),
                    ], gap="xs"),
                ], justify="flex-start", align="end", gap="xl"),
                
                # Boat Selection Checkboxes
                dmc.Space(h=20),
                dmc.Text("Select Boats for Analysis:", fw=500),
                dmc.Space(h=10),
                dmc.SimpleGrid([
                    dmc.Checkbox(
                        id=f'boat-checkbox-{boat}',
                        label=boat,
                        checked=True,
                        value=boat
                    ) for boat in boat_list_default
                ], cols=4, spacing="md"),
                
                # Action Buttons
                dmc.Space(h=30),
                dmc.Group([
                    dmc.Button(
                        "Fetch Data & Analyze",
                        id='fetch-analyze-btn',
                        color="blue",
                        size="lg",
                        loading=False
                    ),
                    dmc.Button(
                        "Clear Results",
                        id='clear-btn',
                        color="red",
                        variant="outline",
                        size="lg"
                    ),
                ], justify="center"),
                
            ], withBorder=True, inheritPadding=True, py="md")
        ], withBorder=True, shadow="sm", radius="md", style={"marginBottom": "20px"}),
        
        # Status Section
        dmc.Card([
            dmc.CardSection([
                dmc.Text(id='status-text', size="md", c="gray"),
            ], inheritPadding=True, py="xs")
        ], withBorder=True, shadow="sm", radius="md", style={"marginBottom": "20px"}),
        
        # Results Section
        dmc.Card([
            dmc.CardSection([
                dmc.Title("Analysis Results", order=3, c="green"),
                
                # Start Analysis Plot
                dmc.Space(h=20),
                dmc.Text("Start Analysis Plot", fw=500, size="lg"),
                dcc.Graph(id='start-analysis-graph', style={"height": "950px"}),  # Increased from 600px to accommodate 900px plot
                
                dmc.Space(h=30),
                dmc.Divider(),
                dmc.Space(h=20),
                
                # TTK Analysis Plot
                dmc.Text("TTK Analysis Plot", fw=500, size="lg"),
                dcc.Graph(id='ttk-analysis-graph', style={"height": "750px"}),  # Increased from 500px to accommodate 700px plot
                
            ], withBorder=True, inheritPadding=True, py="md")
        ], withBorder=True, shadow="sm", radius="md", id="results-section", style={"display": "none"}),
        
        # Export Section
        dmc.Card([
            dmc.CardSection([
                dmc.Title("Export Analysis Data", order=3, c="purple"),
                
                dmc.Space(h=20),
                dmc.Group([
                    dmc.Stack([
                        dmc.Text("Export Path:", fw=500),
                        dcc.Input(
                            id='export-path-input',
                            type='text',
                            value=os.getcwd(),  # Default to current working directory
                            style={'width': '400px'},
                            placeholder="Enter path to save files..."
                        ),
                    ], gap="xs"),
                    
                    dmc.Stack([
                        dmc.Space(h=22),  # Align button with input field
                        dmc.Button(
                            "Export Data",
                            id='export-btn',
                            color="purple",
                            size="md",
                            loading=False
                        ),
                    ], gap="xs"),
                ], justify="flex-start", align="end", gap="lg"),
                
                dmc.Space(h=15),
                dmc.Text(id='export-status-text', size="sm", c="gray"),
                
            ], withBorder=True, inheritPadding=True, py="md")
        ], withBorder=True, shadow="sm", radius="md", id="export-section", style={"display": "none", "marginTop": "20px"}),
        
    ], size="xl", px="md")
])

###################################################################################################################
# Callbacks

@app.callback(
    [Output('start-analysis-graph', 'figure'),
     Output('ttk-analysis-graph', 'figure'),
     Output('results-section', 'style'),
     Output('export-section', 'style'),
     Output('status-text', 'children'),
     Output('fetch-analyze-btn', 'loading'),
     Output('analysis-data-store', 'data')],
    [Input('fetch-analyze-btn', 'n_clicks'),
     Input('clear-btn', 'n_clicks')],
    [State('boat-ref-dropdown', 'value'),
     State('end-time-input', 'value'),
     State('minutes-before-input', 'value'),
     State('practice-start-toggle', 'checked')] + 
    [State(f'boat-checkbox-{boat}', 'checked') for boat in boat_list_default],
    prevent_initial_call=True
)
def update_analysis(fetch_clicks, clear_clicks, boat_ref, end_time, minutes_before, practice_mode, *boat_selections):
    ctx = dash.callback_context
    
    if not ctx.triggered:
        return go.Figure(), go.Figure(), {"display": "none"}, {"display": "none"}, "Ready to analyze", False, {}
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Clear button clicked
    if button_id == 'clear-btn':
        return go.Figure(), go.Figure(), {"display": "none"}, {"display": "none"}, "Results cleared", False, {}
    
    # Fetch and analyze button clicked
    if button_id == 'fetch-analyze-btn':
        try:
            # For practice mode, only analyze reference boat but keep same channels
            if practice_mode:
                selected_boats = [boat_ref]
                status_msg = f"Practice Mode: Analyzing only {boat_ref}"
            else:
                # Get selected boats for normal mode
                selected_boats = []
                for i, boat in enumerate(boat_list_default):
                    if boat_selections[i]:
                        selected_boats.append(boat)
                
                if not selected_boats:
                    return go.Figure(), go.Figure(), {"display": "none"}, {"display": "none"}, "Error: No boats selected", False, {}
                status_msg = f"Normal Mode: Analyzing {len(selected_boats)} boats"
            
            # Calculate start time
            try:
                end_time_arrow = arrow.get(end_time)
                start_time_arrow = end_time_arrow.shift(minutes=-minutes_before)
                start_time = start_time_arrow.format('YYYY-MM-DDTHH:mm:ss') + 'Z'
                end_time = end_time_arrow.format('YYYY-MM-DDTHH:mm:ss') + 'Z'
            except Exception as e:
                return go.Figure(), go.Figure(), {"display": "none"}, {"display": "none"}, f"Error: Invalid time format - {str(e)}", False, {}
            
            # Get boat reference data to find start times
            status_msg += f" | Fetching data from {start_time} to {end_time}..."
            
            # Get high resolution data for boat reference (always use same channel names)
            df_boat_ref = get_high_res_data_event(client, org, boat_ref, start_time, end_time, channel_names)
            
            if df_boat_ref.empty:
                return go.Figure(), go.Figure(), {"display": "none"}, {"display": "none"}, "Error: No data found for the specified time range", False, {}
            
            # For practice mode, replace TTS_s with PC_TTS_s after download for start detection
            if practice_mode and 'PC_TTS_s' in df_boat_ref.columns and 'TTS_s' in df_boat_ref.columns:
                df_boat_ref['TTS_s'] = df_boat_ref['PC_TTS_s']
            
            # Get start times (use normal function since we replaced the column)
            start_dict = get_start_times_boat_ref(df_boat_ref)
            
            if not start_dict:
                return go.Figure(), go.Figure(), {"display": "none"}, {"display": "none"}, "Error: No start times found in the specified range", False, {}
            
            # Get the last race with a start time
            last_race = max(start_dict.keys())
            
            # Get data for all selected boats for the last start
            start_dict = get_data_boat_for_starts(client, org, start_dict, selected_boats, channel_names)
            
            # For practice mode, replace TTS_s with PC_TTS_s in the boat data for consistency
            if practice_mode:
                for boat in selected_boats:
                    df_boat = start_dict[last_race][boat]
                    if 'PC_TTS_s' in df_boat.columns and 'TTS_s' in df_boat.columns:
                        df_boat['TTS_s'] = df_boat['PC_TTS_s']
            
            # Get marks data
            start_dict = get_data_marks_for_starts(org, client, start_dict, mark_list, channel_names_marks)
            
            # Extract metrics for the race
            all_metrics_df, all_analyzed_dfs_race, all_points_of_interest_race, line_analysis = extract_metrics_for_race(
                last_race, start_dict, interp_func, twa_grid, tws_grid
            )
            
            # Add M1 ranking
            all_metrics_df['M1_rank'] = all_metrics_df['M1_PC_TTS_s'].rank(ascending=False, method='min')

            if not practice_mode:
                print("Computing ideal time series...")
                
                # Compute ideal time series with power of 2
                ideal_time_series = compute_ideal_time_series_pow(last_race, {last_race: all_analyzed_dfs_race}, all_metrics_df, pow=2, channel_names=channel_names)

                # Remove all the rows with NaN values
                ideal_time_series = ideal_time_series.dropna()
                ideal_time_series['PC_START_RATIO_unk'] = ideal_time_series['PC_TTS_s'] / (ideal_time_series['PC_TTS_s'] - ideal_time_series['PC_TTK_s'])
                
                # Recompute stats for ideal time series
                ideal_analyzed, ideal_metrics = recompute_stats_for_ideal_time_series(ideal_time_series, line_analysis, interp_func)
                
                # Add ideal boat to the analyzed data and metrics for plotting
                all_analyzed_dfs_race['PON'] = ideal_analyzed
                ideal_metrics['boat'] = 'PON'
                ideal_metrics['M1_rank'] = 3.6  # Set ideal as the weighted rank
                all_metrics_df = pd.concat([all_metrics_df, ideal_metrics], ignore_index=True)
                
                # Create mock points of interest for ideal boat
                ideal_poi = get_points_of_interest(ideal_analyzed)
                all_points_of_interest_race['PON'] = ideal_poi
            
            # Store analysis data for export
            if practice_mode:
                analysis_data = {
                    'last_race': last_race,
                    'practice_mode': True,
                    'boat_ref': boat_ref,
                    'all_analyzed_dfs_race': {boat_ref: all_analyzed_dfs_race[boat_ref].to_dict('records')},
                    'all_metrics_df': all_metrics_df.to_dict('records')
                }
            else:
                analysis_data = {
                    'last_race': last_race,
                    'practice_mode': False,
                    'all_analyzed_dfs_race': {boat: df.to_dict('records') for boat, df in all_analyzed_dfs_race.items()},
                    'all_metrics_df': all_metrics_df.to_dict('records')
                }
            
            # Create plots based on mode
            if practice_mode:
                # For practice mode, only show TTK plot
                start_fig = go.Figure().add_annotation(text="Practice Mode: Only TTK analysis available", 
                                                     showarrow=False, x=0.5, y=0.5,
                                                     font=dict(size=16))
                
                # Create TTK analysis plot for reference boat
                if boat_ref in all_analyzed_dfs_race:
                    if USE_SUBPLOT_TTK_PLOT:
                        ttk_fig = plot_ttk_start_analysis_subplots_plotly(boat_ref, all_analyzed_dfs_race, all_points_of_interest_race)
                    else:
                        ttk_fig = plot_ttk_start_analysis_plotly(boat_ref, all_analyzed_dfs_race, all_points_of_interest_race)
                else:
                    ttk_fig = go.Figure().add_annotation(text="No TTK data available for reference boat", 
                                                       showarrow=False, x=0.5, y=0.5)
                
                status_msg = f"✓ Practice analysis completed for {boat_ref} in Race {last_race}"
            else:
                # Normal mode - both plots
                # Create start analysis plot
                start_fig = plot_start_analysis_with_ideal_plotly(last_race, all_metrics_df, line_analysis, 
                                                                 all_analyzed_dfs_race, all_points_of_interest_race)
                
                # Create TTK analysis plot for reference boat
                if boat_ref in all_analyzed_dfs_race:
                    if USE_SUBPLOT_TTK_PLOT:
                        ttk_fig = plot_ttk_start_analysis_subplots_plotly(boat_ref, all_analyzed_dfs_race, all_points_of_interest_race)
                    else:
                        ttk_fig = plot_ttk_start_analysis_plotly(boat_ref, all_analyzed_dfs_race, all_points_of_interest_race)
                else:
                    ttk_fig = go.Figure().add_annotation(text="No TTK data available for reference boat", 
                                                       showarrow=False, x=0.5, y=0.5)
                
                status_msg = f"✓ Analysis completed for Race {last_race} with {len(selected_boats)} boats"
            
            return start_fig, ttk_fig, {"display": "block"}, {"display": "block"}, status_msg, False, analysis_data
            
        except Exception as e:
            error_msg = f"Error during analysis: {str(e)}"
            return go.Figure(), go.Figure(), {"display": "none"}, {"display": "none"}, error_msg, False, {}
    
    return go.Figure(), go.Figure(), {"display": "none"}, {"display": "none"}, "Ready to analyze", False, {}

# Export callback
@app.callback(
    [Output('export-status-text', 'children'),
     Output('export-btn', 'loading')],
    [Input('export-btn', 'n_clicks')],
    [State('export-path-input', 'value'),
     State('analysis-data-store', 'data')],
    prevent_initial_call=True
)
def export_analysis_data(export_clicks, export_path, analysis_data):
    if not export_clicks or not analysis_data:
        return "No data to export", False
    
    try:
        # Check if path exists, create if it doesn't
        if not os.path.exists(export_path):
            try:
                os.makedirs(export_path, exist_ok=True)
            except Exception as e:
                return f"Error: Could not create directory - {str(e)}", False
        
        # Check if path is writable
        if not os.access(export_path, os.W_OK):
            return f"Error: No write permission for path - {export_path}", False
        
        # Generate timestamp
        timestamp = arrow.utcnow().format('YYYYMMDD_HHmmss')
        last_race = analysis_data.get('last_race', 'unknown')
        practice_mode = analysis_data.get('practice_mode', False)
        
        if practice_mode:
            # Handle practice mode export
            boat_ref = analysis_data.get('boat_ref', 'unknown')
            df_boat = pd.DataFrame(analysis_data['df_boat'])
            if '_time' in df_boat.columns:
                df_boat['_time'] = pd.to_datetime(df_boat['_time'])
                df_boat.set_index('_time', inplace=True)
            
            # Define file path for practice mode
            practice_filename = f"race_{last_race}_practice_{boat_ref}_{timestamp}.pkl"
            practice_path = os.path.join(export_path, practice_filename)
            
            # Save the pickle file
            with open(practice_path, 'wb') as f:
                pickle.dump(df_boat, f)
            
            success_msg = f"✓ Successfully exported practice mode data:\n• {practice_filename}\nto: {export_path}"
            return success_msg, False
        else:
            # Handle normal mode export
            # Convert back to DataFrames for export
            all_analyzed_dfs_race = {}
            for boat, records in analysis_data['all_analyzed_dfs_race'].items():
                df = pd.DataFrame(records)
                if '_time' in df.columns:
                    df['_time'] = pd.to_datetime(df['_time'])
                    df.set_index('_time', inplace=True)
                all_analyzed_dfs_race[boat] = df
            
            all_metrics_df = pd.DataFrame(analysis_data['all_metrics_df'])
            
            # Define file paths
            analyzed_dfs_filename = f"race_{last_race}_analyzed_dfs_{timestamp}.pkl"
            metrics_filename = f"race_{last_race}_metrics_{timestamp}.pkl"
            
            analyzed_dfs_path = os.path.join(export_path, analyzed_dfs_filename)
            metrics_path = os.path.join(export_path, metrics_filename)
            
            # Save the pickle files
            with open(analyzed_dfs_path, 'wb') as f:
                pickle.dump(all_analyzed_dfs_race, f)
            
            with open(metrics_path, 'wb') as f:
                pickle.dump(all_metrics_df, f)
            
            success_msg = f"✓ Successfully exported:\n• {analyzed_dfs_filename}\n• {metrics_filename}\nto: {export_path}"
            return success_msg, False
        
    except Exception as e:
        error_msg = f"Export failed: {str(e)}"
        return error_msg, False

###################################################################################################################
# Run the app

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=8058)

