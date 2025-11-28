# Activate the interactive matplotlib backend
# %matplotlib tk

import numpy as np
import pandas as pd


import warnings
import arrow

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import re

import utm

from matplotlib.path import Path

from scipy.signal import savgol_filter

warnings.filterwarnings('ignore')

####################################################################################################################################################################
## Functions for the boat data retrieval
####################################################################################################################################################################

def get_high_res_data_event(client, org, boat, start_time, end_time, channel_names):
    """
    Query high resolution data for a specific boat and event using a .flux template.
    """

    # Load the Flux query template
    with open("new_global_req_last_high_res_manoeuvres.flux", "r") as file:
        query = file.read()

    # Fill template placeholders
    query = query.replace("{start_time}", arrow.get(start_time).format("YYYY-MM-DDTHH:mm:ss") + "Z")
    query = query.replace("{stop_time}", arrow.get(end_time).format("YYYY-MM-DDTHH:mm:ss") + "Z")
    query = query.replace("{boat}", boat)

    # Build REGEX for channel names
    regex_pattern = r"/^(" + "|".join(re.escape(ch) for ch in channel_names) + r")/"
    query = query.replace("{REGEX}", regex_pattern)

    # Run the query
    result = client.query_api().query(org=org, query=query)

    # Parse FluxTables into DataFrames
    tables = []
    for table in result:
        rows = [record.values for record in table.records]
        if rows:
            df_table = pd.DataFrame(rows)
            df_table = df_table.drop(columns=["result", "table"], errors="ignore")
            df_table = df_table.set_index("_time")
            tables.append(df_table)

    # No data?
    if not tables:
        return pd.DataFrame()

    # Merge all tables
    df = pd.concat(tables, axis=1).reset_index()
    df["_time"] = pd.to_datetime(df["_time"])
    df.set_index("_time", inplace=True)

    # Add boat tag
    df["BOAT"] = boat

    # === Keep everything below exactly as in your original file ===
    # Unwrap TWA_BOW_SGP_deg
    if "TWA_BOW_SGP_deg" in df.columns:
        df["TWA_BOW_SGP_deg"] = np.unwrap(np.radians(df["TWA_BOW_SGP_deg"])) * 180 / np.pi

    # Unwrap TWA_MHU_SGP_deg
    if "TWA_MHU_SGP_deg" in df.columns:
        df["TWA_MHU_SGP_deg"] = np.unwrap(np.radians(df["TWA_MHU_SGP_deg"])) * 180 / np.pi

    # Interpolate missing values
    df = df.interpolate(method="linear").ffill().bfill()

    # Compute derivatives / smoothing
    if "BOAT_SPEED_km_h_1" in df.columns:
        df["BOAT_SPEED_m_s"] = df["BOAT_SPEED_km_h_1"] / 3.6

    # Calculate distance traveled (if GPS exists)
    if "LATITUDE_GPS_unk" in df.columns and "LONGITUDE_GPS_unk" in df.columns:
        df["lat_rad"] = np.radians(df["LATITUDE_GPS_unk"])
        df["lon_rad"] = np.radians(df["LONGITUDE_GPS_unk"])

        df["dlat"] = df["lat_rad"].diff()
        df["dlon"] = df["lon_rad"].diff()
        a = np.sin(df["dlat"] / 2) ** 2 + np.cos(df["lat_rad"]) * np.cos(df["lat_rad"].shift()) * np.sin(df["dlon"] / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        df["distance_m"] = 6371000 * c  # Earth radius

    # OPTIONAL: more processing depending on your original file…

    return df


def get_start_times_boat_ref(df_boat_ref):
    """
    Function to get the start times from the boat reference dataframe
    Input: df_boat_ref - dataframe with the boat reference data
    Output: start_dict - dictionary with the start times
    """
    # Detect the start times
    # Write a dictionnary with the race number as key and the start time as another key with the start time as a value
    start_dict = {}

    # This is defined as the times where PC_TTS_s is 0.0
    start_times = df_boat_ref[df_boat_ref["TTS_s"].abs() <= 0.05]

    for index, row in start_times.iterrows():
        race_num = int(row["TRK_RACE_NUM_unk"])
        if race_num not in start_dict:
            start_dict[race_num] = {"start_time": index}
        else:
            if "start_time" in start_dict[race_num]:
                if index > start_dict[race_num]["start_time"]:
                    start_dict[race_num]["start_time"] = index
            else:
                continue
    return start_dict

def get_data_boat_for_starts(client, org, start_dict, boat_list, channel_names):
    """
    Function to get the high res data for all the boats for 3 minute before and 2 minute after each start time
    Input: start_dict - dictionary with the start times
           boat_list - list of boats to get the data for
           channel_names - list of channel names to get the data for
    Output: start_dict - dictionary with the start times and the dataframes for each boat
    """
    # We are going to get the high res data for all the boats for 3 minute before and 2 minute after each start time
    for race_num in start_dict.keys():
        print(race_num)
        time = start_dict[race_num]["start_time"]
        start_time = time - pd.Timedelta(seconds=180)
        end_time = time + pd.Timedelta(seconds=180)
        print(f"Getting data for race {race_num} from {start_time} to {end_time}")
        print(boat_list)
        for boat in boat_list:
            print(f"Getting data for boat {boat}")
            df_boat = get_high_res_data_event(client, org, boat, start_time, end_time, channel_names)
            # Add the dataframe to the start_dict under the boat key
            if boat not in start_dict[race_num].keys():
                start_dict[race_num][boat] = df_boat
            else:
                print(f"Data for boat {boat} already exists for race {race_num}")
    return start_dict

####################################################################################################################################################################
## Functions for the mark data retrieval
####################################################################################################################################################################

def get_data_marks(org, client, start_time, end_time, mark, channel_names_marks):
    """
    Function to get data for specific marks
    Input: client - InfluxDB client
           org - organization name
           start_time - start time of the event
           end_time - end time of the event
           mark - mark name
           channel_names_marks - list of channel names to get the data for
    Output: df - dataframe with the data for the mark
    """
    # Returns a dictionary of dataframes for each mark
    with open("new_global_req_marks.flux", "r") as file:
        query = file.read()
        query = query.replace("{start_time}", arrow.get(start_time).format("YYYY-MM-DDTHH:mm:ss") + "Z")
        query = query.replace("{stop_time}", arrow.get(end_time).format("YYYY-MM-DDTHH:mm:ss") + "Z")
        query = query.replace("{boat}", mark)
        regex_pattern = r"/^(" + "|".join(re.escape(channel) for channel in channel_names_marks) + r")/"
        query = query.replace("{REGEX}", regex_pattern)
        result = client.query_api().query(org=org, query=query)
        
        # transform FluxTable as pandas series
        tables = [
            pd.DataFrame([record.values for record in table.records])
            .drop(columns=["result", "table"])
            .set_index("_time")
            for table in result
        ]
        if tables:
            df = pd.concat(tables, axis=1).reset_index()
        else:
            df = pd.DataFrame()
        df.dropna(inplace=True)
        df["mark"] = mark
        if len(df) == 0:
            return df
        else:
            # Convert the time column to datetime and set it as index
            df["_time"] = pd.to_datetime(df["_time"])
            df.set_index("_time", inplace=True)

    return df

def get_data_marks_for_starts(org, client, start_dict, mark_list, channel_names_marks):
    """
    Function to get the marks data for all the marks for 45 s before and 15 s after each start time
    Input: start_dict - dictionary with the start times
           mark_list - list of marks to get the data for
           channel_names_marks - list of channel names to get the data for
    Output: start_dict - dictionary with the start times and the dataframes for each mark
    """
    # Add the marks data to the start_dict
    for race_num in start_dict.keys():
        print(race_num)
        time = start_dict[race_num]["start_time"]
        start_time = time - pd.Timedelta(seconds=45)
        end_time = time + pd.Timedelta(seconds=15)
        print(f"Getting marks data for race {race_num} from {start_time} to {end_time}")
        for mark in mark_list:
            df_mark = get_data_marks(org, client, start_time, end_time, mark, channel_names_marks)
            # Rename the channels to include MDSS
            df_mark.rename(columns={
                "LATITUDE_deg": "LATITUDE_MDSS_deg",
                "LONGITUDE_deg": "LONGITUDE_MDSS_deg",
                "TWS_km_h_1": "TWS_MDSS_km_h_1",
                "TWD_deg": "TWD_MDSS_deg"
            }, inplace=True)
            # Add the dataframe to the start_dict under the mark key
            if mark not in start_dict[race_num].keys():
                start_dict[race_num][mark] = df_mark
            else:
                print(f"Data for mark {mark} already exists for race {race_num}")
    return start_dict

####################################################################################################################################################################
## Functions for the polars
####################################################################################################################################################################
def get_boat_speed_from_polar(twa, tws, interp_func, twa_grid, tws_grid):
    """
    Function to get the boat speed from the polar given the true wind angle and true wind speed
    Input:  twa - true wind angle in degrees
            tws - true wind speed in km/h
            interp_func - interpolation function for the polar
            twa_grid - grid of true wind angles used for the polar
            tws_grid - grid of true wind speeds used for the polar
    Output: boat speed in km/h
    """
    twa = abs(twa)
    if twa > 180:
        twa = 360 - twa

    # Clip TWA and TWS to within data bounds
    twa = np.clip(twa, twa_grid.min(), twa_grid.max())
    tws = np.clip(tws, tws_grid.min(), tws_grid.max())
    
    point = np.array([[twa, tws]])
    
    return float(interp_func(point))

def get_vmg(twa, tws, interp_func, twa_grid, tws_grid):
    """
    Function to get the vmg given the true wind angle and true wind speed
    Input: twa - true wind angle in degrees
              tws - true wind speed in km/h 
              Output: vmg in km/h
    """
    bsp = get_boat_speed_from_polar(twa, tws, interp_func, twa_grid, tws_grid)
    vmg = bsp * np.cos(np.radians(twa))
    return vmg

def get_optimal_twa_upwind(tws, interp_func, twa_grid, tws_grid):
    """
    Function to get the optimal true wind angle to sail upwind given the true wind speed
    Input: tws - true wind speed in km/h
    Output: optimal true wind angle in degrees
    """
    twa_values = np.linspace(0, 90, 181)  # TWA from 0 to 90 degrees
    vmg_values = np.zeros_like(twa_values)
    for twa in twa_values:
        vmg_values[np.where(twa_values == twa)[0][0]] = get_vmg(twa, tws, interp_func, twa_grid, tws_grid)
    optimal_twa = twa_values[np.argmax(vmg_values)]
    return optimal_twa

def get_optimal_twa_downwind(tws, interp_func, twa_grid, tws_grid):
    """
    Function to get the optimal true wind angle to sail downwind given the true wind speed
    Input: tws - true wind speed in km/h
    Output: optimal true wind angle in degrees
    """
    twa_values = np.linspace(90, 180, 181)  # TWA from 90 to 180 degrees
    vmg_values = np.zeros_like(twa_values)
    for twa in twa_values:
        vmg_values[np.where(twa_values == twa)[0][0]] = get_vmg(twa, tws, interp_func, twa_grid, tws_grid)
    optimal_twa = twa_values[np.argmin(vmg_values)]
    return optimal_twa

####################################################################################################################################################################
## Line analysis functions
####################################################################################################################################################################
def get_mark_position_and_wind(df_mark):
    """
    Function to get the average latitude, longitude, TWS and TWD of a mark from its dataframe
    Input: df_mark - dataframe with the mark data
    Output: lat - average latitude in degrees
            lon - average longitude in degrees
            tws - average true wind speed in km/h
            twd - average true wind direction in degrees
    """
    if df_mark.empty:
        return None, None
    lat = df_mark['LATITUDE_MDSS_deg'].mean() / 10000000.0
    lon = df_mark['LONGITUDE_MDSS_deg'].mean() / 10000000.0
    tws = df_mark['TWS_MDSS_km_h_1'].mean()
    twd = df_mark['TWD_MDSS_deg'].mean()

    return lat, lon, tws, twd

def rotate_coordinates(lon, lat, twd, ref_lon, ref_lat):
    """
    Function to rotate coordinates based on TWD and reference point
    Input: lon - longitude in degrees
           lat - latitude in degrees
           twd - true wind direction in degrees
           ref_lon - reference longitude in degrees
           ref_lat - reference latitude in degrees
    Output: rotated_x - rotated x coordinate
            rotated_y - rotated y coordinate
    """
    # Convert degrees to radians
    twd_rad = np.radians(twd)

    # Convert to UTM coordinates
    x, y, zone_number, zone_letter = utm.from_latlon(lat, lon)
    ref_x, ref_y, _, _ = utm.from_latlon(ref_lat, ref_lon)

    # Translate coordinates to the reference point
    translated_x = x - ref_x
    translated_y = y - ref_y

    # Rotation matrix
    rotated_y = translated_y * np.cos(twd_rad) + translated_x * np.sin(twd_rad)
    rotated_x = -translated_y * np.sin(twd_rad) + translated_x * np.cos(twd_rad)

    return rotated_x, rotated_y

def rotate_coordinates_df(df, TWD, reference_point_coords):
    """
    Function to rotate coordinates in a dataframe based on TWD and reference point
    Input: df - dataframe with LONGITUDE and LATITUDE columns
           TWD - true wind direction in degrees
           reference_point_coords - tuple with reference point (lat, lon)
    Output: df - dataframe with rotated x and y coordinates
    """
    ## Let's extract the UTM coordinates from the GPS data
    # Convert latitude and longitude to UTM coordinates
    utm_coords = [utm.from_latlon(lat, lon) for lat, lon in zip(df['LATITUDE_GPS_unk'] / 10000000.0, df['LONGITUDE_GPS_unk'] / 10000000.0)]
    utm_df = pd.DataFrame(utm_coords, columns=['UTM_Easting', 'UTM_Northing', 'Zone_Number', 'Zone_Letter'])
    
    ## Add UTM coordinates to the dataframe knowing utm_df doesn't have the same index as df
    df['EASTING'] = None
    df['NORTHING'] = None
    
    df['EASTING'] = utm_df['UTM_Easting'].values
    df['NORTHING'] = utm_df['UTM_Northing'].values
    
    # Reference point coordinates contains longitude and latitude in degrees
    # Convert reference point coordinates to UTM coordinates
    reference_point_utm = utm.from_latlon(reference_point_coords[0], reference_point_coords[1])
    reference_point_easting = reference_point_utm[0]
    reference_point_northing = reference_point_utm[1]
    
    
    df['NORTHING'] = df['NORTHING'] - reference_point_northing
    df['EASTING'] = df['EASTING'] - reference_point_easting
    
    ## We want to rotate the coordinates around the mark coordinates to align with the TWD
    df['y'] = (df['NORTHING'] * np.cos(np.deg2rad(TWD)) + df['EASTING'] * np.sin(np.deg2rad(TWD)))
    df['x'] = (-df['NORTHING'] * np.sin(np.deg2rad(TWD)) + df['EASTING'] * np.cos(np.deg2rad(TWD)))
    
    # Drop the UTM coordinates
    df.drop(columns=['EASTING', 'NORTHING'], inplace=True)
    
    return df

def compute_distance_and_time(x, y, ref_x, ref_y, tws, interp_func, twa_grid, tws_grid):
    """
    Function to compute distance and time from a point to a reference point based on TWS
    Input: x - x coordinate of the point
           y - y coordinate of the point
           ref_x - x coordinate of the reference point
           ref_y - y coordinate of the reference point
           tws - true wind speed in km/h
    Output: distance - distance to the reference point in meters
            time - time to reach the reference point in seconds
    """
    distance = np.sqrt((x - ref_x)**2 + (y - ref_y)**2)  # in meters
    angle = np.degrees(np.arctan2(y - ref_y, x - ref_x)) + 90 # in degrees
    boat_speed = get_boat_speed_from_polar(angle, tws, interp_func, twa_grid, tws_grid)  # in km/h
    if boat_speed == 0:
        return distance, None
    time = distance / (boat_speed / 3.6)  # in seconds
    return distance, time, boat_speed

def get_line_analyzed(df_sl1, df_sl2, df_m1, interp_func, twa_grid, tws_grid):
    """
    Function to analyze the start line and compute various parameters
    Input:  df_sl1 - dataframe with the SL1 mark data
            df_sl2 - dataframe with the SL2 mark data
            df_m1 - dataframe with the M1 mark data
    Output: dictionary with various computed parameters
            - avg_tws: average true wind speed in km/h
            - avg_twd: average true wind direction in degrees
            - m1: tuple with M1 latitude and longitude (because x and y of M1 is 0,0)
            - sl1: tuple with SL1 x and y coordinates
            - sl2: tuple with SL2 x and y coordinates
            - best_start_point: tuple with best start point x and y coordinates
            - sl1_layline_end: tuple with SL1 layline end x and y coordinates
            - sl2_layline_end: tuple with SL2 layline end x and y coordinates
            - sl1_biss_end: tuple with SL1 biss layline end x and y coordinates
            - line_points: numpy array with line points x, y, distance, time, speed
            - zones: dictionary with zones A, B, C and D x and y coordinates
    """
    # M1 is the 0, 0 point
    # Provide the lat, lon, tws and twd of M1
    m1_lat, m1_lon, m1_tws, m1_twd = get_mark_position_and_wind(df_m1)
    # SL1 and SL2 are the two points of the start line
    sl1_lat, sl1_lon, sl1_tws, sl1_twd = get_mark_position_and_wind(df_sl1)
    sl2_lat, sl2_lon, sl2_tws, sl2_twd = get_mark_position_and_wind(df_sl2)

    # Compute average TWS and TWD on the line
    line_tws = (sl1_tws + sl2_tws) / 2
    line_twd = (sl1_twd + sl2_twd) / 2

    # Compute average TWS and TWD between the line and M1
    avg_tws = (line_tws + m1_tws) / 2
    avg_twd = (line_twd + m1_twd) / 2

    # Get the x,y coordinates of M1, SL1 and SL2
    sl1_x, sl1_y = rotate_coordinates(sl1_lon, sl1_lat, avg_twd, m1_lon, m1_lat)
    sl2_x, sl2_y = rotate_coordinates(sl2_lon, sl2_lat, avg_twd, m1_lon, m1_lat)

    # Compute the distance and time from M1 to SL1 and SL2 for points every 10 meters along the line
    line_length = np.sqrt((sl2_x - sl1_x)**2 + (sl2_y - sl1_y)**2)
    num_points = int(line_length / 10) + 1
    line_points = []
    for i in range(num_points):
        point_x = sl1_x + (sl2_x - sl1_x) * i / (num_points - 1)
        point_y = sl1_y + (sl2_y - sl1_y) * i / (num_points - 1)
        distance, time, speed = compute_distance_and_time(point_x, point_y, 0.0, 0.0, avg_tws, interp_func, twa_grid, tws_grid)
        line_points.append((point_x, point_y, distance, time, speed))
    line_points = np.array(line_points)

    # Find the best start point based on the minimum time to reach M1
    valid_points = line_points[line_points[:, 3] != None]
    best_start_index = np.argmin(valid_points[:, 3])
    best_start_x, best_start_y, best_start_distance, best_start_time, best_start_speed = valid_points[best_start_index]

    # Compute the laylines end points of SL1 and SL2 and the best start point
    optimal_twa = get_optimal_twa_upwind(avg_tws, interp_func, twa_grid, tws_grid)
    # Compute the end point of the layline from SL1
    layline_distance = 900  # in meters
    layline_angle_rad = np.radians(optimal_twa - 90)  # Convert to radians and adjust for coordinate system
    sl1_layline_end_x = sl1_x + layline_distance * np.cos(layline_angle_rad)
    sl1_layline_end_y = sl1_y + layline_distance * np.sin(layline_angle_rad)
    # Compute the end point of the layline from SL2
    layline_angle_rad = np.radians(optimal_twa - 90)  # Convert to radians and adjust for coordinate system
    sl2_layline_end_x = sl2_x + layline_distance * np.cos(layline_angle_rad)
    sl2_layline_end_y = sl2_y + layline_distance * np.sin(layline_angle_rad)
    # Compute the end point of the line from SL1 with half the angle to the perpendicular
    layline_angle_rad = np.radians(optimal_twa - 90) / 2 # Convert to radians and adjust for coordinate system
    sl1_biss_end_x = sl1_x + layline_distance * np.cos(layline_angle_rad)
    sl1_biss_end_y = sl1_y + layline_distance * np.sin(layline_angle_rad)
    
    # Define zones A, B, C and D based on the laylines
    # Zone A
    # Corner 1: SL2
    zone_a_c1 = (sl2_x, sl2_y)
    # Corner 2: SL2 layline end point
    zone_a_c2 = (sl2_layline_end_x, sl2_layline_end_y)
    # Corner 3: SL1 layline end point
    zone_a_c3 = (sl1_layline_end_x, sl1_layline_end_y)
    # Corner 4: SL1
    zone_a_c4 = (sl1_x, sl1_y)
    zone_a_x = [zone_a_c1[0], zone_a_c2[0], zone_a_c3[0], zone_a_c4[0], zone_a_c1[0]]
    zone_a_y = [zone_a_c1[1], zone_a_c2[1], zone_a_c3[1], zone_a_c4[1], zone_a_c1[1]]
    # Zone B > Triangle
    # Corner 1: SL1
    zone_b_c1 = (sl1_x, sl1_y)
    # Corner 2: Best start point layline end point
    zone_b_c2 = (sl1_layline_end_x, sl1_layline_end_y)
    # Corner 3: SL1 biss layline end point
    zone_b_c3 = (sl1_biss_end_x, sl1_biss_end_y)
    zone_b_x = [zone_b_c1[0], zone_b_c2[0], zone_b_c3[0], zone_b_c1[0]]
    zone_b_y = [zone_b_c1[1], zone_b_c2[1], zone_b_c3[1], zone_b_c1[1]]
    # Zone C : Triangle
    # Corner 1: SL1
    zone_c_c1 = (sl1_x, sl1_y)
    # Corner 2: SL1 biss layline end point
    zone_c_c2 = (sl1_biss_end_x, sl1_biss_end_y)
    # Corner 3: Horizontal line end point
    zone_c_c3 = (sl1_x + 700, sl1_y)
    zone_c_x = [zone_c_c1[0], zone_c_c2[0], zone_c_c3[0], zone_c_c1[0]]
    zone_c_y = [zone_c_c1[1], zone_c_c2[1], zone_c_c3[1], zone_c_c1[1]]
    # Zone D : Area above the horizontal line going through SL1 and going right
    # Corner 1: SL1
    zone_d_c1 = (sl1_x, sl1_y)
    # Corner 2: Horizontal line end point
    zone_d_c2 = (sl1_x + 700, sl1_y)
    # Corner 3: A point far right and up
    zone_d_c3 = (sl1_x + 700, sl1_y + 300)
    # Corner 4: A point far up
    zone_d_c4 = (sl1_x, sl1_y + 300)
    zone_d_x = [zone_d_c1[0], zone_d_c2[0], zone_d_c3[0], zone_d_c4[0], zone_d_c1[0]]
    zone_d_y = [zone_d_c1[1], zone_d_c2[1], zone_d_c3[1], zone_d_c4[1], zone_d_c1[1]]

    return {
        "avg_tws": avg_tws,
        "avg_twd": avg_twd,
        "m1": (m1_lat, m1_lon),
        "sl1": (sl1_x, sl1_y),
        "sl2": (sl2_x, sl2_y),
        "best_start_point": (best_start_x, best_start_y),
        "sl1_layline_end": (sl1_layline_end_x, sl1_layline_end_y),
        "sl2_layline_end": (sl2_layline_end_x, sl2_layline_end_y),
        "sl1_biss_end": (sl1_biss_end_x, sl1_biss_end_y),
        "line_points": line_points,
        "zones": {
            "A": (zone_a_x, zone_a_y),
            "B": (zone_b_x, zone_b_y),
            "C": (zone_c_x, zone_c_y),
            "D": (zone_d_x, zone_d_y)
        }
    }

def get_zone_df(df, zone_a_x, zone_a_y, zone_b_x, zone_b_y, zone_c_x, zone_c_y, zone_d_x, zone_d_y):
    """
    Function to get the zone in which each point in the dataframe is located
    Input: df - dataframe with x and y columns
           zone_a_x - x coordinates of zone A
           zone_a_y - y coordinates of zone A
           zone_b_x - x coordinates of zone B
           zone_b_y - y coordinates of zone B
           zone_c_x - x coordinates of zone C
           zone_c_y - y coordinates of zone C
           zone_d_x - x coordinates of zone D
           zone_d_y - y coordinates of zone D
    Output: df - dataframe with an additional zone column
    """
    path_a = Path(np.column_stack((zone_a_x, zone_a_y)))
    path_b = Path(np.column_stack((zone_b_x, zone_b_y)))
    path_c = Path(np.column_stack((zone_c_x, zone_c_y)))
    path_d = Path(np.column_stack((zone_d_x, zone_d_y)))
    
    def determine_zone(row):
        point = (row['x'], row['y'])
        if path_a.contains_point(point):
            return 'A'
        elif path_b.contains_point(point):
            return 'B'
        elif path_c.contains_point(point):
            return 'C'
        elif path_d.contains_point(point):
            return 'D'
        else:
            return 'Outside'
    
    df['zone'] = df.apply(determine_zone, axis=1)
    return df

def distance_to_sl2_layline_df(df, sl2_x, sl2_y, sl2_layline_end_x, sl2_layline_end_y):
    """
    Function to compute distance to SL2 layline
    Input: df - dataframe with x and y columns
           sl2_x - x coordinate of SL2
           sl2_y - y coordinate of SL2
           sl2_layline_end_x - x coordinate of SL2 layline end
           sl2_layline_end_y - y coordinate of SL2 layline end
    Output: df - dataframe with an additional distance_to_sl2_layline column
    """
    # Line equation Ax + By + C = 0
    A = sl2_layline_end_y - sl2_y
    B = sl2_x - sl2_layline_end_x
    C = sl2_layline_end_x * sl2_y - sl2_x * sl2_layline_end_y
    
    df['distance_to_sl2_layline'] = (A * df['x'] + B * df['y'] + C) / np.sqrt(A**2 + B**2)
    df['distance_to_sl2_layline'] = -df['distance_to_sl2_layline']
    return df

def distance_to_sl_df(df, sl1_x, sl1_y, sl2_x, sl2_y):
    """
    Function to compute distance to start line and distance along the line
    Input:  df - dataframe with x and y columns
            sl1_x - x coordinate of SL1
            sl1_y - y coordinate of SL1
            sl2_x - x coordinate of SL2
            sl2_y - y coordinate of SL2
    Output: df - dataframe with additional distance_perp_line and distance_along_line columns
    """
    # Line equation Ax + By + C = 0 for the start line, this is the one that will determine the cross
    A = sl1_y - sl2_y
    B = sl2_x - sl1_x
    C = sl1_x * sl2_y - sl2_x * sl1_y
    df['distance_perp_line'] = (A * df['x'] + B * df['y'] + C) / np.sqrt(A**2 + B**2)
    # Orthogonal line to the start line going through SL2, this is the one we want the distance to
    A_ortho = -B
    B_ortho = A
    C_ortho = B * sl2_x - A * sl2_y
    df['distance_along_line'] = (A_ortho * df['x'] + B_ortho * df['y'] + C_ortho) / np.sqrt(A_ortho**2 + B_ortho**2)
    return df

def get_m1_polar_info_df(df, TWD, TWS, interp_func):
    """
    Function to get the boat speed target and boat speed target to M1 based on the polar
    Input:  df - dataframe with HEADING_deg and BOAT_SPEED_km_h_1 columns
            TWD - true wind direction in degrees
            TWS - true wind speed in km/h
            interp_func - interpolation function for the polar
    Output: df - dataframe with additional boat_speed_target, twa_to_M1, boat_speed_target_to_m1 and boat_speed_target_to_m1_ratio columns
    """
    # Calculate TWA
    df['twa'] = (TWD - df['HEADING_deg']) % 360
    df['twa'] = df['twa'].apply(lambda x: (x + 180) % 360 - 180)  # Normalize between -180 and 180

    twa_array = np.array(df['twa'])
    tws_array = np.array([TWS] * len(df))
    # Cap the twa and tws to the range of the polar
    twa_array = np.clip(twa_array, interp_func.grid[0].min(), interp_func.grid[0].max())
    tws_array = np.clip(tws_array, interp_func.grid[1].min(), interp_func.grid[1].max())
    # Get the boat speeds from the polar
    speeds = interp_func(np.column_stack([twa_array, tws_array]))
    # Apply the get_boat_speed_from_polar function to the TWA column
    df['boat_speed_target'] = speeds
    
    # Now get the twa to M1 based on the avg_twd of the line (everything is already rotated so M1 is at (0,0) and vertical is the TWD direction)
    df['twa_to_M1'] = np.degrees(np.arctan2(df['y'], df['x'])) + 90

    # Compute the boat speed target to M1 based on the polar
    twa_array = np.array(df['twa_to_M1'])
    # Cap the twa to the range of the polar
    twa_array = np.clip(twa_array, interp_func.grid[0].min(), interp_func.grid[0].max())
    # Get the boat speeds from the polar
    speeds = interp_func(np.column_stack([twa_array, tws_array]))
    # Apply the get_boat_speed_from_polar function to the TWA to M1 column
    df['boat_speed_target_to_m1'] = speeds

    # Compute the boat speed target to M1 ratio
    df['boat_speed_target_to_m1_ratio'] = df.apply(lambda row: row['BOAT_SPEED_km_h_1'] / row['boat_speed_target_to_m1'] if row['boat_speed_target_to_m1'] > 0 else np.nan, axis=1)
    
    return df

def closest_point_on_start_line_and_metrics(df_boat, avg_twd, avg_tws, sl1_x, sl1_y, sl2_x, sl2_y, interp_func):
    """
    Function to compute the closest point on the start line and various metrics
    At the moment the TTK is missing the acceleration part so it's just distance / boat speed target
    Input:  df_boat - dataframe with boat data including x, y, TWD_MHU_SGP_deg, TWS_MHU_SGP_km_h_1, TWD_BOW_SGP_deg, TWS_BOW_SGP_km_h_1
            avg_twd - average true wind direction in degrees
            avg_tws - average true wind speed in km/h
            sl1_x - x coordinate of SL1
            sl1_y - y coordinate of SL1
            sl2_x - x coordinate of SL2
            sl2_y - y coordinate of SL2
            interp_func - interpolation function for the polar
    Output: df_boat - dataframe with additional columns:
            - x_closest: x coordinate of the closest point on the start line
            - y_closest: y coordinate of the closest point on the start line
            - distance_to_closest: distance to the closest point on the start line in meters
            - twa_to_closest_in_start_frame: TWA to the closest point in the avg_TWD frame
            - twa_to_closest_in_MHU_frame: TWA to the closest point in the MHU wind frame
            - twa_to_closest_in_BOW_frame: TWA to the closest point in the BOW wind frame
            - boat_speed_target_to_closest_in_start_frame: boat speed target to the closest point in the avg_TWD frame in km/h
            - boat_speed_target_to_closest_MHU: boat speed target to the closest point in the MHU wind frame in km/h
            - boat_speed_target_to_closest_BOW: boat speed target to the closest point in the BOW wind frame in km/h
            - TTL_post_s_AVG: time to reach the closest point in seconds based on avg_TWS and TWA in avg_TWD frame
            - TTL_post_s_MHU: time to reach the closest point in seconds based on MHU TWS and TWA in MHU wind frame
            - TTL_post_s_BOW: time to reach the closest point in seconds based on BOW TWS and TWA in BOW wind frame
            - TTK_post_s_AVG: time to kill to reach the closest point in seconds based on avg_TWS and TWA in avg_TWD frame
            - TTK_post_s_MHU: time to kill to reach the closest point in seconds based on MHU TWS and TWA in MHU wind frame
            - TTK_post_s_BOW: time to kill to reach the closest point in seconds based on BOW TWS and TWA in BOW wind frame
            - twa_live_AVG: live TWA based on avg_TWD in degrees
            - twa_live_MHU: live TWA based on MHU TWD in degrees
            - twa_live_BOW: live TWA based on BOW TWD in degrees
            - delta_twa_AVG: delta TWA between live TWA and TWA to closest point in avg_TWD frame in degrees
            - delta_twa_MHU: delta TWA between live TWA and TWA to closest point in MHU wind frame in degrees
            - delta_twa_BOW: delta TWA between live TWA and TWA to closest point in BOW wind frame in degrees
            - delta_speed_to_point_AVG: delta speed to closest point in avg_TWD frame in km/h
            - delta_speed_to_point_MHU: delta speed to closest point in MHU wind frame in km/h
            - delta_speed_to_point_BOW: delta speed to closest point in BOW wind frame in km/h
    """
    # Compute the closest point on the start line
    dx = sl2_x - sl1_x
    dy = sl2_y - sl1_y

    t = ((df_boat['x'] - sl1_x) * dx + (df_boat['y'] - sl1_y) * dy) / (dx * dx + dy * dy)
    t = np.clip(t, 0, 1)

    df_boat['x_closest'] = sl1_x + t * dx
    df_boat['y_closest'] = sl1_y + t * dy
    df_boat['distance_to_closest'] = np.sqrt((df_boat['x'] - df_boat['x_closest'])**2 + (df_boat['y'] - df_boat['y_closest'])**2)
    # Compute TWA to closest point in the avg_TWD frame
    df_boat['twa_to_closest_in_start_frame'] = np.degrees(np.arctan2(df_boat['y'] - df_boat['y_closest'], df_boat['x'] - df_boat['x_closest'])) + 90
    # Compute the TWA in both MHU and BOW frames
    df_boat['twa_to_closest_in_MHU_frame'] = (df_boat['twa_to_closest_in_start_frame'] + df_boat['TWD_MHU_SGP_deg'] - avg_twd)
    df_boat['twa_to_closest_in_BOW_frame'] = (df_boat['twa_to_closest_in_start_frame'] + df_boat['TWD_BOW_SGP_deg'] - avg_twd)
    # Normalize between -180 and 180
    df_boat['twa_to_closest_in_start_frame'] = df_boat['twa_to_closest_in_start_frame'].apply(lambda x: (x + 180) % 360 - 180)
    df_boat['twa_to_closest_in_MHU_frame'] = df_boat['twa_to_closest_in_MHU_frame'].apply(lambda x: (x + 180) % 360 - 180)
    df_boat['twa_to_closest_in_BOW_frame'] = df_boat['twa_to_closest_in_BOW_frame'].apply(lambda x: (x + 180) % 360 - 180)
    # Compute the boat speed target to closest point based on the polar
    twa_array_AVG = np.array(df_boat['twa_to_closest_in_start_frame'])
    twa_array_MHU = np.array(df_boat['twa_to_closest_in_MHU_frame'])
    twa_array_BOW = np.array(df_boat['twa_to_closest_in_BOW_frame'])
    tws_array_AVG = np.array([avg_tws] * len(df_boat))
    tws_array_MHU = np.array(df_boat['TWS_MHU_SGP_km_h_1'])
    tws_array_BOW = np.array(df_boat['TWS_BOW_SGP_km_h_1'])
    # Cap the twa and tws to the range of the polar
    twa_array_AVG = np.clip(twa_array_AVG, interp_func.grid[0].min(), interp_func.grid[0].max())
    twa_array_MHU = np.clip(twa_array_MHU, interp_func.grid[0].min(), interp_func.grid[0].max())
    twa_array_BOW = np.clip(twa_array_BOW, interp_func.grid[0].min(), interp_func.grid[0].max())
    tws_array_AVG = np.clip(tws_array_AVG, interp_func.grid[1].min(), interp_func.grid[1].max())
    tws_array_MHU = np.clip(tws_array_MHU, interp_func.grid[1].min(), interp_func.grid[1].max())
    tws_array_BOW = np.clip(tws_array_BOW, interp_func.grid[1].min(), interp_func.grid[1].max())
    # Get the boat speeds from the polar
    speeds_AVG = interp_func(np.column_stack([twa_array_AVG, tws_array_AVG]))
    speeds_MHU = interp_func(np.column_stack([twa_array_MHU, tws_array_MHU]))
    speeds_BOW = interp_func(np.column_stack([twa_array_BOW, tws_array_BOW]))
    # Apply the get_boat_speed_from_polar function to the TWA column
    df_boat['boat_speed_target_to_closest_in_start_frame'] = speeds_AVG
    df_boat['boat_speed_target_to_closest_MHU'] = speeds_MHU
    df_boat['boat_speed_target_to_closest_BOW'] = speeds_BOW
    # Compute TTL_post_s
    df_boat['TTL_post_s_AVG'] = df_boat.apply(lambda row: row['distance_to_closest'] / (row['boat_speed_target_to_closest_in_start_frame'] / 3.6) if row['boat_speed_target_to_closest_in_start_frame'] > 0 else np.nan, axis=1)
    df_boat['TTL_post_s_MHU'] = df_boat.apply(lambda row: row['distance_to_closest'] / (row['boat_speed_target_to_closest_MHU'] / 3.6) if row['boat_speed_target_to_closest_MHU'] > 0 else np.nan, axis=1)
    df_boat['TTL_post_s_BOW'] = df_boat.apply(lambda row: row['distance_to_closest'] / (row['boat_speed_target_to_closest_BOW'] / 3.6) if row['boat_speed_target_to_closest_BOW'] > 0 else np.nan, axis=1)
    # Compute TTK_post_s
    df_boat['TTK_post_s_AVG'] = df_boat['PC_TTS_s'] - df_boat['TTL_post_s_AVG']
    df_boat['TTK_post_s_MHU'] = df_boat['PC_TTS_s'] - df_boat['TTL_post_s_MHU']
    df_boat['TTK_post_s_BOW'] = df_boat['PC_TTS_s'] - df_boat['TTL_post_s_BOW']
    # Compute the live TWA based on the TWD and heading
    df_boat['twa_live_MHU'] = (df_boat['TWD_MHU_SGP_deg'] - df_boat['HEADING_deg']) % 360
    df_boat['twa_live_BOW'] = (df_boat['TWD_BOW_SGP_deg'] - df_boat['HEADING_deg']) % 360
    df_boat['twa_live_MHU'] = df_boat['twa_live_MHU'].apply(lambda x: (x + 180) % 360 - 180)
    df_boat['twa_live_BOW'] = df_boat['twa_live_BOW'].apply(lambda x: (x + 180) % 360 - 180)
    # Compute delta_twa
    df_boat['delta_twa_to_closest_in_start_frame'] = df_boat['twa'] - df_boat['twa_to_closest_in_start_frame']
    df_boat['delta_twa_to_closest_MHU'] = df_boat['twa_live_MHU'] - df_boat['twa_to_closest_in_MHU_frame']
    df_boat['delta_twa_to_closest_BOW'] = df_boat['twa_live_BOW'] - df_boat['twa_to_closest_in_BOW_frame']
    # Compute delta_speed_to_point in m/s
    df_boat['delta_speed_to_closest_in_start_frame'] = (df_boat['boat_speed_target_to_closest_in_start_frame'] - df_boat['BOAT_SPEED_km_h_1'] * np.cos(np.radians(df_boat['delta_twa_to_closest_in_start_frame'])))
    df_boat['delta_speed_to_closest_MHU'] = (df_boat['boat_speed_target_to_closest_MHU'] - df_boat['BOAT_SPEED_km_h_1'] * np.cos(np.radians(df_boat['delta_twa_to_closest_MHU'])))
    df_boat['delta_speed_to_closest_BOW'] = (df_boat['boat_speed_target_to_closest_BOW'] - df_boat['BOAT_SPEED_km_h_1'] * np.cos(np.radians(df_boat['delta_twa_to_closest_BOW'])))
    # Compute the delta_TTK_s based on the delta speed to closest point being the delta speed to closest point divided by the boat_speed target to closest point multiplied by -0.1s (the time step between measurements)
    df_boat['delta_TTK_s_AVG'] = df_boat.apply(lambda row: -0.1 * (row['delta_speed_to_closest_in_start_frame'] / row['boat_speed_target_to_closest_in_start_frame']) if row['boat_speed_target_to_closest_in_start_frame'] > 0 else np.nan, axis=1)
    df_boat['delta_TTK_s_MHU'] = df_boat.apply(lambda row: -0.1 * (row['delta_speed_to_closest_MHU'] / row['boat_speed_target_to_closest_MHU']) if row['boat_speed_target_to_closest_MHU'] > 0 else np.nan, axis=1)
    df_boat['delta_TTK_s_BOW'] = df_boat.apply(lambda row: -0.1 * (row['delta_speed_to_closest_BOW'] / row['boat_speed_target_to_closest_BOW']) if row['boat_speed_target_to_closest_BOW'] > 0 else np.nan, axis=1)

    return df_boat

def analyze_boat_positions(df, line_analysis, interp_func):
    """
    Function to analyze boat positions with respect to the start line and M1
    Input:  df - dataframe with boat data including LATITUDE_deg, LONGITUDE_deg, HEADING_deg, BOAT_SPEED_km_h_1, TWD_MHU_SGP_deg, TWS_MHU_SGP_km_h_1, TWD_BOW_SGP_deg, TWS_BOW_SGP_km_h_1
            line_analysis - dictionary with line analysis data
            interp_func - interpolation function for the polar
    Output: df - dataframe with additional columns:
            - zone: zone in which the boat is located (A, B, C, D or Outside)
            - distance_to_sl2_layline: distance to SL2 layline in meters
            - distance_perp_line: distance perpendicular to the start line in meters
            - distance_along_line: distance along the start line in meters
            - twa: true wind angle based on avg_twd in degrees
            - boat_speed_target: boat speed target based on the polar, avg_tws and twa in km/h
            - twa_to_M1: true wind angle to M1 based on avg_twd in degrees
            - boat_speed_target_to_m1: boat speed target to M1 based on the polar, avg_tws and twa_to_M1 in km/h
            - all additional columns from closest_point_on_start_line_and_metrics function
    """
    zone_a_x, zone_a_y = line_analysis["zones"]["A"]
    zone_b_x, zone_b_y = line_analysis["zones"]["B"]
    zone_c_x, zone_c_y = line_analysis["zones"]["C"]
    zone_d_x, zone_d_y = line_analysis["zones"]["D"]
    sl1_x, sl1_y = line_analysis["sl1"]
    sl2_x, sl2_y = line_analysis["sl2"]
    sl2_layline_end_x, sl2_layline_end_y = line_analysis["sl2_layline_end"]
    avg_twd = line_analysis["avg_twd"]
    avg_tws = line_analysis["avg_tws"]

    df = rotate_coordinates_df(df, avg_twd, (line_analysis["m1"][0], line_analysis["m1"][1]))
    df = get_zone_df(df, zone_a_x, zone_a_y, zone_b_x, zone_b_y, zone_c_x, zone_c_y, zone_d_x, zone_d_y)
    df = distance_to_sl2_layline_df(df, sl2_x, sl2_y, sl2_layline_end_x, sl2_layline_end_y)
    df = distance_to_sl_df(df, sl1_x, sl1_y, sl2_x, sl2_y)
    df = get_m1_polar_info_df(df, avg_twd, avg_tws, interp_func)
    df = closest_point_on_start_line_and_metrics(df, avg_twd, avg_tws, sl1_x, sl1_y, sl2_x, sl2_y, interp_func)

    return df

def get_points_of_interest(df_boat_analyzed):
    """
    Function to get the points of interest from the analyzed boat dataframe
    Input:  df_boat_analyzed - dataframe with analyzed boat data including PC_TTS_s, distance_perp_line, twa, x, y
    Output: points_of_interest - list of points of interest indices
    """
    # Filter the dataframe to only take the last race_id, sometimes we have some of the previous race left and we want to discard it
    # Get the index of the max TTS_s
    if df_boat_analyzed['PC_TTS_s'].iloc[0] < 0 :
        max_tts_index = df_boat_analyzed['PC_TTS_s'].idxmax()
        df_boat_analyzed = df_boat_analyzed.loc[max_tts_index:]
    df_cross = df_boat_analyzed[(df_boat_analyzed['distance_perp_line'] >= 0) & (df_boat_analyzed['distance_perp_line'].shift(1) < 0) & (df_boat_analyzed['PC_TTS_s'] <= 150) & (df_boat_analyzed['PC_TTS_s'] >= 0)]
    if df_cross.empty:
        point_1 = np.nan
    else:
        point_1 = df_cross.index[-1]
    # Last Manoeuvre Point
    # df_last_manoeuvre = df_boat_analyzed[(df_boat_analyzed['twa'] < 0) & (df_boat_analyzed['twa'].shift(-1) >= 0) & (df_boat_analyzed['PC_TTS_s'] >= 0)]
    # Try a different approach to get the last manoeuvre point
    # This is the last point where delta_x is positive and the next point delta_x is negative and TTS is positive
    df_last_manoeuvre = df_boat_analyzed[(df_boat_analyzed['x'].diff() > 0) & (df_boat_analyzed['x'].diff().shift(-1) <= 0) & (df_boat_analyzed['PC_TTS_s'] >= 0)]
    if df_last_manoeuvre.empty:
        point_2 = np.nan
    else:
        point_2 = df_last_manoeuvre.index[-1]
    point_3 = df_boat_analyzed[df_boat_analyzed['PC_TTS_s'] <= 10].index[0]
    point_4 = df_boat_analyzed[df_boat_analyzed['PC_TTS_s'] <= 5].index[0]
    point_5 = df_boat_analyzed[df_boat_analyzed['PC_TTS_s'] <= 0].index[0]
    point_6 = df_boat_analyzed[df_boat_analyzed['PC_TTS_s'] <= -5].index[0]
    df_M1_reach = df_boat_analyzed[(df_boat_analyzed['x'] < 0) & (df_boat_analyzed['y'] < 0) & (df_boat_analyzed['PC_TTS_s'] <= -5)]
    if df_M1_reach.empty:
        point_7 = df_boat_analyzed.index[-1]
    else:
        point_7 = df_M1_reach.index[0]
    points_of_interest = [point_1, point_2, point_3, point_4, point_5, point_6, point_7]
    return points_of_interest

def extract_metrics_from_poi(df_boat_analyzed, line_analysis, points_of_interest):
    """
    Function to extract metrics from points of interest
    Input:  df_boat_analyzed - dataframe with analyzed boat data including various metrics
            line_analysis - dictionary with line analysis data
            points_of_interest - list of points of interest indices
    Output: metrics_df - dataframe with extracted metrics
    """
    metrics = {}
    # Point 1
    point_1 = points_of_interest[0]
    metrics['point_1'] = {}
    metrics['point_1']['PC_TTS_s'] = df_boat_analyzed.at[point_1, 'PC_TTS_s'] if not pd.isna(point_1) else np.nan
    metrics['point_1']['distance_to_sl2_layline'] = df_boat_analyzed.at[point_1, 'distance_to_sl2_layline'] if not pd.isna(point_1) else np.nan
    # TWS 1 min before point 1
    time_1_min_before = point_1 - pd.Timedelta(minutes=1) if not pd.isna(point_1) else points_of_interest[4] # Take the gun time if point 1 is nan
    df_1_min_before = df_boat_analyzed[df_boat_analyzed.index <= time_1_min_before]
    if not df_1_min_before.empty:
        metrics['point_1']['TWS_MHU_1_min_before'] = df_1_min_before.iloc[-1]['TWS_MHU_SGP_km_h_1']
        metrics['point_1']['TWS_BOW_1_min_before'] = df_1_min_before.iloc[-1]['TWS_BOW_SGP_km_h_1']
    else:
        metrics['point_1']['TWS_MHU_1_min_before'] = np.nan
        metrics['point_1']['TWS_BOW_1_min_before'] = np.nan
    # Point 2
    point_2 = points_of_interest[1]
    metrics['point_2'] = {}
    metrics['point_2']['PC_TTS_s'] = df_boat_analyzed.at[point_2, 'PC_TTS_s'] if not pd.isna(point_2) else np.nan
    metrics['point_2']['PC_DTL_m'] = df_boat_analyzed.at[point_2, 'PC_DTL_m'] if not pd.isna(point_2) else np.nan
    metrics['point_2']['PC_TTK_s'] = df_boat_analyzed.at[point_2, 'PC_TTK_s'] if not pd.isna(point_2) else np.nan
    metrics['point_2']['PC_START_RATIO_unk'] = df_boat_analyzed.at[point_2, 'PC_START_RATIO_unk'] if not pd.isna(point_2) else np.nan
    metrics['point_2']['zone'] = df_boat_analyzed.at[point_2, 'zone'] if not pd.isna(point_2) else np.nan
    metrics['point_2']['distance_to_sl2_layline'] = df_boat_analyzed.at[point_2, 'distance_to_sl2_layline'] if not pd.isna(point_2) else np.nan
    # Point 3, 4, 5, 6
    for i, point in enumerate(points_of_interest[2:], start=3):
        metrics[f'point_{i}'] = {}
        metrics[f'point_{i}']['BOAT_SPEED_km_h_1'] = df_boat_analyzed.at[point, 'BOAT_SPEED_km_h_1']
        metrics[f'point_{i}']['boat_speed_target_to_m1_ratio'] = df_boat_analyzed.at[point, 'boat_speed_target_to_m1_ratio']
        metrics[f'point_{i}']['twa'] = df_boat_analyzed.at[point, 'twa']
        metrics[f'point_{i}']['twa_to_M1'] = df_boat_analyzed.at[point, 'twa_to_M1']
        metrics[f'point_{i}']['delta_twa_to_M1'] = df_boat_analyzed.at[point, 'twa_to_M1'] - df_boat_analyzed.at[point, 'twa']
        line_length = np.sqrt((line_analysis["sl2"][0] - line_analysis["sl1"][0])**2 + (line_analysis["sl2"][1] - line_analysis["sl1"][1])**2)
        metrics[f'point_{i}']['distance_along_sl_ratio'] = df_boat_analyzed.at[point, 'distance_along_line'] / line_length
    # Point 7
    point_7 = points_of_interest[6]
    metrics['point_7'] = {}
    metrics['point_7']['PC_TTS_s'] = df_boat_analyzed.at[point_7, 'PC_TTS_s']
    # Transform the metrics dictionary into a dataframe with the metrics as columns
    # We will rename the columns to have point_1_PC_TTS_s, point_1_distance_to_sl2_layline, etc.
    metrics_df = pd.DataFrame()
    for point, point_metrics in metrics.items():
        for metric, value in point_metrics.items():
            metrics_df.at[0, f'{point}_{metric}'] = value

    # Rename the columns by replacing :
    # - point_1_ by Cross_
    # - point_2_ by Last_Manoeuvre_
    # - point_3_ by Gun_-10s_
    # - point_4_ by Gun_-5s_
    # - point_5_ by Gun_
    # - point_6_ by Gun_+5s_
    # - point_7_ by M1_
    metrics_df = metrics_df.rename(columns=lambda x: x.replace('point_1_', 'Cross_').replace('point_2_', 'Last_Manoeuvre_').replace('point_3_', 'Gun_-10s_').replace('point_4_', 'Gun_-5s_').replace('point_5_', 'Gun_').replace('point_6_', 'Gun_+5s_').replace('point_7_', 'M1_'))

    return metrics_df

def extract_metrics_for_boat_race(race, boat, start_dict, interp_func, line_analysis):
    """
    Function to extract metrics for a specific boat in a specific race
    Input:  race - race number
            boat - boat name
            start_dict - dictionary with start data
            interp_func - interpolation function for the polar
            line_analysis - dictionary with line analysis data
    Output: metrics_df - dataframe with extracted metrics
            df_boat_analyzed - dataframe with analyzed boat data
            points_of_interest - list of points of interest indices
    """
    df_boat = start_dict[race][boat]
    df_boat_analyzed = analyze_boat_positions(df_boat, line_analysis, interp_func)
    points_of_interest = get_points_of_interest(df_boat_analyzed)
    metrics_df = extract_metrics_from_poi(df_boat_analyzed, line_analysis, points_of_interest)
    metrics_df['boat'] = boat
    metrics_df['race'] = race
    return metrics_df, df_boat_analyzed, points_of_interest

def extract_metrics_for_race(race, start_dict, interp_func, twa_grid, tws_grid):
    """
    Function to extract metrics for all boats in a specific race
    Input:  race - race number
            start_dict - dictionary with start data
            interp_func - interpolation function for the polar
    Output: all_metrics_df - dataframe with extracted metrics for all boats
            all_analyzed_dfs_race - dictionary with analyzed boat data for all boats
            all_points_of_interest_race - dictionary with points of interest for all boats
            line_analysis - dictionary with line analysis data
    """
    line_analysis = get_line_analyzed(start_dict[race]['SL1'], start_dict[race]['SL2'], start_dict[race]['M1'], interp_func, twa_grid, tws_grid)
    boats = [key for key in start_dict[race].keys() if key not in ['start_time', 'SL1', 'SL2', 'M1']]
    all_metrics_df = pd.DataFrame()
    all_analyzed_dfs_race = {}
    all_points_of_interest_race = {}
    for boat in boats:
        try:
            metrics_df, df_boat_analyzed, points_of_interest = extract_metrics_for_boat_race(race, boat, start_dict, interp_func, line_analysis)
            all_metrics_df = pd.concat([all_metrics_df, metrics_df], ignore_index=True)
            all_analyzed_dfs_race[boat] = df_boat_analyzed
            all_points_of_interest_race[boat] = points_of_interest
        except Exception as e:
            print(f'Error processing boat {boat} in race {race}: {e}')
    return all_metrics_df, all_analyzed_dfs_race, all_points_of_interest_race, line_analysis

def compute_ideal_time_series_pow(race, all_analyzed_dfs, all_stats, pow, channel_names):
    """
    Function to compute the ideal time series for a given race based on M1 ranking
    Input:  race - race number
            all_analyzed_dfs - dictionary with analyzed boat data for all races
            all_stats - dataframe with boat statistics including M1_rank
            pow - power to which the weight is raised
    Output: ideal_time_series - dataframe with ideal time series
    """
    boats = all_stats[all_stats['race'] == race]['boat'].values
    weights = {}
    for boat in boats:
        M1_rank = all_stats[(all_stats['race'] == race) & (all_stats['boat'] == boat)]['M1_rank'].values[0]
        weights[boat] = (13 - M1_rank) ** pow
    total_weight = sum(weights.values())
    # Create a time index that is the union of all boats time indexes
    time_index = pd.Index([])
    for boat in boats:
        df_boat_analyzed = all_analyzed_dfs[race][boat]
        time_index = time_index.union(df_boat_analyzed.index)
    ideal_time_series = pd.DataFrame(index=time_index)
    # For each channel, compute the weighted average
    channels = channel_names #['PC_TTS_s', 'BOAT_SPEED_km_h_1', 'twa', 'x', 'y']
    for channel in channels:
        ideal_time_series[channel] = 0.0
        for boat in boats:
            df_boat_analyzed = all_analyzed_dfs[race][boat]
            df_boat_analyzed_channel = df_boat_analyzed[[channel]].reindex(time_index)
            ideal_time_series[channel] += df_boat_analyzed_channel[channel] * weights[boat] / total_weight
    return ideal_time_series

## NEED TO CREATE A FUNCTION TO RECOMPUTE ALL THE STATS FOR THE IDEAL TIME SERIES BASED ON THE EXISTING FUNCTIONS
def recompute_stats_for_ideal_time_series(ideal_time_series, line_analysis, interp_func):
    """
    Function to recompute all the stats for the ideal time series
    Input:  ideal_time_series - dataframe with ideal time series including PC_TTS_s, BOAT_SPEED_km_h_1, x, y
            line_analysis - dictionary with line analysis data
            interp_func - interpolation function for the polar
    Output: ideal_time_series_analyzed - dataframe with analyzed ideal time series
    """
    analyzed_ideal_time_series = analyze_boat_positions(ideal_time_series, line_analysis, interp_func)
    poi_ideal = get_points_of_interest(analyzed_ideal_time_series)
    metrics_ideal_df = extract_metrics_from_poi(analyzed_ideal_time_series, line_analysis, poi_ideal)
    return analyzed_ideal_time_series, metrics_ideal_df


####################################################################################################################################################################
## Plot functions for the ui
####################################################################################################################################################################

def plot_start_analysis_with_ideal_plotly(race, stats_start, line_analysis, all_analyzed_dfs_race, all_points_of_interest_race):
    """
    Function to plot the start analysis with ideal time series using Plotly
    Input:  race - race number
            stats_start - dataframe with boat statistics including M1_rank
            line_analysis - dictionary with line analysis data
            all_analyzed_dfs_race - dictionary with analyzed boat data for all boats
            all_points_of_interest_race - dictionary with points of interest for all boats
    Output: fig - Plotly figure object
    """
    # Boats to plot
    winner_boat = stats_start[stats_start['M1_rank'] == 1]['boat'].values[0]
    second_boat = stats_start[stats_start['M1_rank'] == 2]['boat'].values[0]
    boats_to_plot = ['FRA', winner_boat, 'PON'] if winner_boat != 'FRA' else ['FRA', second_boat, 'PON']
    boat_colors = {'FRA': 'darkcyan', winner_boat: 'red', second_boat: 'orange', 'PON': 'black'}

    fig = go.Figure()

    # Start line
    fig.add_trace(go.Scatter(x=[line_analysis["sl1"][0], line_analysis["sl2"][0]],
                             y=[line_analysis["sl1"][1], line_analysis["sl2"][1]],
                             mode='lines', line=dict(color='black', width=2), name='Start Line'))

    # M1 point
    fig.add_trace(go.Scatter(x=[0], y=[0], mode='markers',
                             marker=dict(color='red', size=10), name='M1'))

    # Laylines
    fig.add_trace(go.Scatter(x=[line_analysis["sl1"][0], line_analysis["sl1_layline_end"][0]],
                             y=[line_analysis["sl1"][1], line_analysis["sl1_layline_end"][1]],
                             mode='lines', line=dict(color='magenta', dash='dash'), name='Layline SL1'))
    fig.add_trace(go.Scatter(x=[line_analysis["sl2"][0], line_analysis["sl2_layline_end"][0]],
                             y=[line_analysis["sl2"][1], line_analysis["sl2_layline_end"][1]],
                             mode='lines', line=dict(color='magenta', dash='dash'), name='Layline SL2'))
    fig.add_trace(go.Scatter(x=[line_analysis["sl1"][0], line_analysis["sl1_biss_end"][0]],
                             y=[line_analysis["sl1"][1], line_analysis["sl1_biss_end"][1]],
                             mode='lines', line=dict(color='green', dash='dash'), name='Layline Middle'))

    # Horizontal line
    fig.add_trace(go.Scatter(x=[line_analysis["sl1"][0], line_analysis["sl1"][0] + 700],
                             y=[line_analysis["sl1"][1], line_analysis["sl1"][1]],
                             mode='lines', line=dict(color='blue', dash='dash'), name='Horizontal Line'))

    # Zones
    for zone, color in zip(['A', 'B', 'C', 'D'], ['red', 'blue', 'green', 'yellow']):
        fig.add_trace(go.Scatter(x=line_analysis["zones"][zone][0],
                                 y=line_analysis["zones"][zone][1],
                                 fill='toself', fillcolor=color, opacity=0.3,
                                 line=dict(color=color), name=f'Zone {zone}'))

    # Line points with color scale
    fig.add_trace(go.Scatter(x=line_analysis["line_points"][:, 0],
                             y=line_analysis["line_points"][:, 1],
                             mode='markers',
                             marker=dict(size=5, color=line_analysis["line_points"][:, 3],
                                         colorscale='Reds', colorbar=dict(title='TTS (s)')),
                             name='Line Points'))
    
    # Best start point
    fig.add_trace(go.Scatter(x=[line_analysis["best_start_point"][0]], y=[line_analysis["best_start_point"][1]],
                             mode='markers', marker=dict(color='yellow', size=15), name='Best Start Point'))

    # Boats and annotations
    min_pc_tts = 0.0
    for boat in boats_to_plot:
        points_of_interest = all_points_of_interest_race[boat]
        df_boat_analyzed = all_analyzed_dfs_race[boat]
        df_boat_analyzed = df_boat_analyzed[(df_boat_analyzed['PC_TTS_s'] <= 120)]
        df_boat_analyzed = df_boat_analyzed.loc[:points_of_interest[-1] + pd.Timedelta(seconds=10)]
        min_pc_tts = min(min_pc_tts, df_boat_analyzed['PC_TTS_s'].min())

        # Boat trajectory
        fig.add_trace(go.Scatter(x=df_boat_analyzed['x'], y=df_boat_analyzed['y'],
                                 mode='lines', line=dict(color=boat_colors[boat], width=2),
                                 opacity=0.6, name=f'Boat {boat}'))

        # Cross point
        if points_of_interest[0] in df_boat_analyzed.index:
            cross_x = df_boat_analyzed.at[points_of_interest[0], 'x']
            cross_y = df_boat_analyzed.at[points_of_interest[0], 'y']
            fig.add_trace(go.Scatter(x=[cross_x], y=[cross_y], mode='markers',
                                     marker=dict(symbol='x', size=12, color=boat_colors[boat]),
                                     name=f'{boat} Cross'))
            fig.add_annotation(x=line_analysis["sl2"][0] - 50,
                               y = line_analysis["sl2"][1] - (75 if boat == 'FRA' else (50 if boat == 'PON' else 25)),
                               text=f'TTS: {df_boat_analyzed.at[points_of_interest[0], "PC_TTS_s"]:.1f}s',
                               showarrow=False, font=dict(color=boat_colors[boat]))

        # Last manoeuvre
        if points_of_interest[1] in df_boat_analyzed.index:
            last_x = df_boat_analyzed.at[points_of_interest[1], 'x']
            last_y = df_boat_analyzed.at[points_of_interest[1], 'y']
            fig.add_trace(go.Scatter(x=[last_x], y=[last_y], mode='markers',
                                     marker=dict(symbol='square', size=12, color=boat_colors[boat]),
                                     name=f'{boat} Last Manoeuvre'))
            fig.add_annotation(x=line_analysis["sl1"][0] + 400,
                               y=line_analysis["sl1"][1] + (45 if boat == 'FRA' else (130 if boat == 'PON' else 215)),
                               text=f'TTS: {df_boat_analyzed.at[points_of_interest[1], "PC_TTS_s"]:.1f}s<br>'
                                    f'TTK: {df_boat_analyzed.at[points_of_interest[1], "PC_TTK_s"]:.1f}s<br>'
                                    f'Start Ratio: {df_boat_analyzed.at[points_of_interest[1], "PC_START_RATIO_unk"]:.2f}',
                               showarrow=False, font=dict(color=boat_colors[boat]))

        # Gun point and speed annotation
        gun_x = df_boat_analyzed.at[points_of_interest[4], 'x']
        gun_y = df_boat_analyzed.at[points_of_interest[4], 'y']
        fig.add_trace(go.Scatter(x=[gun_x], y=[gun_y], mode='markers',
                                 marker=dict(size=9, color=boat_colors[boat]),
                                 name=f'{boat} Gun'))
        fig.add_annotation(x=line_analysis["sl1"][0],
                           y=line_analysis["sl1"][1] + (25 if boat == 'FRA' else (50 if boat == 'PON' else 75)),
                           text=f'BS: {df_boat_analyzed.at[points_of_interest[4], "BOAT_SPEED_km_h_1"]:.1f} km/h',
                           showarrow=False, font=dict(color=boat_colors[boat]))

        # Gun -10s, -5s, +5s
        for idx, size, label in zip([2, 3, 5], [5, 7, 7], ['Gun -10s', 'Gun -5s', 'Gun +5s']):
            fig.add_trace(go.Scatter(x=[df_boat_analyzed.at[points_of_interest[idx], 'x']],
                                     y=[df_boat_analyzed.at[points_of_interest[idx], 'y']],
                                     mode='markers',
                                     marker=dict(size=size, color=boat_colors[boat]),
                                     name=f'{boat} {label}'))

    # Layout
    fig.update_layout(title=f'Pre Start - Race {race} - Boats: {", ".join(boats_to_plot)}',
                      xaxis=dict(title='X (m)', range=[-200, 1000], showgrid=True),
                      yaxis=dict(title='Y (m)', range=[-500, 700], showgrid=True),
                      legend=dict(x=0.01, y=0.99),
                      showlegend=False,  # Hide the legend
                      width=900, height=900)

    return fig  # Return figure instead of fig.show()


def plot_ttk_start_analysis_plotly(boat, all_analyzed_dfs_race, points_of_interest_race):
    df_boat_analyzed_to_plot = all_analyzed_dfs_race[boat]
    point_of_interest = points_of_interest_race[boat]

    # Filter data
    min_time = max(df_boat_analyzed_to_plot.index.min(),
                   max(point_of_interest[1],
                       df_boat_analyzed_to_plot[df_boat_analyzed_to_plot['PC_TTS_s'] <= 120].index.min()))
    df_boat_analyzed_to_plot = df_boat_analyzed_to_plot[df_boat_analyzed_to_plot.index >= min_time]
    df_boat_analyzed_to_plot = df_boat_analyzed_to_plot[df_boat_analyzed_to_plot['PC_TTS_s'] >= 0.0]

    # Compute stats
    max_TTK_plot = df_boat_analyzed_to_plot[['TTK_post_s_AVG', 'TTK_post_s_MHU', 'TTK_post_s_BOW']].max().max()
    min_TTK_plot = df_boat_analyzed_to_plot[['TTK_post_s_AVG', 'TTK_post_s_MHU', 'TTK_post_s_BOW']].min().min()

    TTK_integrate_AVG = df_boat_analyzed_to_plot['TTK_post_s_AVG'].iloc[0] - df_boat_analyzed_to_plot['TTK_post_s_AVG'].iloc[-1]
    TTK_integrate_MHU = df_boat_analyzed_to_plot['TTK_post_s_MHU'].iloc[0] - df_boat_analyzed_to_plot['TTK_post_s_MHU'].iloc[-1]
    TTK_integrate_BOW = df_boat_analyzed_to_plot['TTK_post_s_BOW'].iloc[0] - df_boat_analyzed_to_plot['TTK_post_s_BOW'].iloc[-1]

    TTK_integrate_action_AVG = -np.sum(df_boat_analyzed_to_plot['delta_TTK_s_AVG'])
    TTK_integrate_action_MHU = -np.sum(df_boat_analyzed_to_plot['delta_TTK_s_MHU'])
    TTK_integrate_action_BOW = -np.sum(df_boat_analyzed_to_plot['delta_TTK_s_BOW'])

    delta_TTK_AVG = TTK_integrate_action_AVG - TTK_integrate_AVG
    delta_TTK_MHU = TTK_integrate_action_MHU - TTK_integrate_MHU
    delta_TTK_BOW = TTK_integrate_action_BOW - TTK_integrate_BOW

    ratio_TTK_AVG = TTK_integrate_action_AVG / TTK_integrate_AVG if TTK_integrate_AVG != 0 else np.nan
    ratio_TTK_MHU = TTK_integrate_action_MHU / TTK_integrate_MHU if TTK_integrate_MHU != 0 else np.nan
    ratio_TTK_BOW = TTK_integrate_action_BOW / TTK_integrate_BOW if TTK_integrate_BOW != 0 else np.nan

    # Create figure
    fig = go.Figure()

    # Horizontal line at 0
    fig.add_shape(type="line", x0=df_boat_analyzed_to_plot.index.min(), x1=df_boat_analyzed_to_plot.index.max(),
                  y0=0, y1=0, line=dict(color="black", dash="dash"))

    # Add traces
    fig.add_trace(go.Scatter(x=df_boat_analyzed_to_plot.index, y=df_boat_analyzed_to_plot['PC_TTK_s'],
                             mode='lines', line=dict(color='black'), name='PC_TTK_s'))
    fig.add_trace(go.Scatter(x=df_boat_analyzed_to_plot.index, y=df_boat_analyzed_to_plot['TTK_post_s_AVG'],
                             mode='lines', line=dict(color='blue'), name='TTK_post_s_AVG'))
    fig.add_trace(go.Scatter(x=df_boat_analyzed_to_plot.index, y=df_boat_analyzed_to_plot['TTK_post_s_MHU'],
                             mode='lines', line=dict(color='red'), name='TTK_post_s_MHU'))
    fig.add_trace(go.Scatter(x=df_boat_analyzed_to_plot.index, y=df_boat_analyzed_to_plot['TTK_post_s_BOW'],
                             mode='lines', line=dict(color='green'), name='TTK_post_s_BOW'))

    # Conditional colors for annotations
    color_delta_avg = "green" if delta_TTK_AVG > 0 else "red"
    color_ratio_avg = "green" if ratio_TTK_AVG > 1 else "red"
    color_delta_mhu = "green" if delta_TTK_MHU > 0 else "red"
    color_ratio_mhu = "green" if ratio_TTK_MHU > 1 else "red"
    color_delta_bow = "green" if delta_TTK_BOW > 0 else "red"
    color_ratio_bow = "green" if ratio_TTK_BOW > 1 else "red"

    # Add annotations
    fig.add_annotation(xref='paper', yref='paper', x=0.05, y=0.95,
                       text=f"ΔTTK AVG= {delta_TTK_AVG:.2f}s", font=dict(color=color_delta_avg, size=14), showarrow=False)
    fig.add_annotation(xref='paper', yref='paper', x=0.05, y=0.90,
                       text=f"Ratio AVG= {ratio_TTK_AVG:.2f}", font=dict(color=color_ratio_avg, size=14), showarrow=False)

    fig.add_annotation(xref='paper', yref='paper', x=0.35, y=0.95,
                       text=f"ΔTTK MHU= {delta_TTK_MHU:.2f}s", font=dict(color=color_delta_mhu, size=14), showarrow=False)
    fig.add_annotation(xref='paper', yref='paper', x=0.35, y=0.90,
                       text=f"Ratio MHU= {ratio_TTK_MHU:.2f}", font=dict(color=color_ratio_mhu, size=14), showarrow=False)

    fig.add_annotation(xref='paper', yref='paper', x=0.65, y=0.95,
                       text=f"ΔTTK BOW= {delta_TTK_BOW:.2f}s", font=dict(color=color_delta_bow, size=14), showarrow=False)
    fig.add_annotation(xref='paper', yref='paper', x=0.65, y=0.90,
                       text=f"Ratio BOW= {ratio_TTK_BOW:.2f}", font=dict(color=color_ratio_bow, size=14), showarrow=False)

    # Layout
    fig.update_layout(
        title=f'Boat TTK Analysis Over Time - Boat {boat}',
        xaxis=dict(title='Time'),
        yaxis=dict(title='TTK (s)', range=[min_TTK_plot - 5, max_TTK_plot + 5]),
        width=1000, height=700,
        legend=dict(x=0.01, y=0.1),
        template='plotly_white'
    )

    return fig  # Return figure instead of fig.show()

def plot_ttk_start_analysis_subplots_plotly(boat, all_analyzed_dfs_race, points_of_interest_race):
    """
    Function to plot TTK start analysis with subplots using Plotly
    First subplot: PC_TTK_s only
    Second subplot: delta_TTK_s_(selected_wand) and PC_TTK_s.diff()
    Input:  boat - boat name
            all_analyzed_dfs_race - dictionary with analyzed boat data for all boats
            points_of_interest_race - dictionary with points of interest for all boats
    Output: fig - Plotly figure object with subplots
    """
    df_boat_analyzed_to_plot = all_analyzed_dfs_race[boat]
    point_of_interest = points_of_interest_race[boat]

    # Filter data
    min_time = max(df_boat_analyzed_to_plot.index.min(),
                   max(point_of_interest[1],
                       df_boat_analyzed_to_plot[df_boat_analyzed_to_plot['PC_TTS_s'] <= 120].index.min()))
    df_boat_analyzed_to_plot = df_boat_analyzed_to_plot[df_boat_analyzed_to_plot.index >= min_time]
    df_boat_analyzed_to_plot = df_boat_analyzed_to_plot[df_boat_analyzed_to_plot['PC_TTS_s'] >= 0.0]

    # Determine selected wand based on SEL_WAND_SGP_unk
    if 'SEL_WAND_SGP_unk' in df_boat_analyzed_to_plot.columns:
        # Get the most common value or last value
        wand_values = df_boat_analyzed_to_plot['SEL_WAND_SGP_unk'].dropna()
        if not wand_values.empty:
            selected_wand_value = wand_values.mode().iloc[0] if not wand_values.mode().empty else wand_values.iloc[-1]
            if selected_wand_value == 2:
                selected_wand = 'MHU'
            elif selected_wand_value == 1:
                selected_wand = 'BOW'
            else:
                selected_wand = 'MHU'  # Default fallback
        else:
            selected_wand = 'MHU'  # Default fallback
    else:
        selected_wand = 'MHU'  # Default fallback

    # Compute TTK metrics
    TTK_integrate_PC = df_boat_analyzed_to_plot['PC_TTK_s'].iloc[0] - df_boat_analyzed_to_plot['PC_TTK_s'].iloc[-1]
    
    delta_ttk_column = f'delta_TTK_s_{selected_wand}'
    if delta_ttk_column in df_boat_analyzed_to_plot.columns:
        TTK_integrate_action_selected = -np.sum(df_boat_analyzed_to_plot[delta_ttk_column])
        delta_TTK_selected = TTK_integrate_action_selected - TTK_integrate_PC
        ratio_TTK_selected = TTK_integrate_action_selected / TTK_integrate_PC if TTK_integrate_PC != 0 else np.nan
    else:
        TTK_integrate_action_selected = np.nan
        delta_TTK_selected = np.nan
        ratio_TTK_selected = np.nan

    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('PC_TTK_s', 'Delta TTK Analysis'),
        vertical_spacing=0.1,
        shared_xaxes=True
    )

    # First subplot: PC_TTK_s only
    fig.add_trace(
        go.Scatter(
            x=df_boat_analyzed_to_plot.index, 
            y=df_boat_analyzed_to_plot['PC_TTK_s'],
            mode='lines', 
            line=dict(color='black', width=2), 
            name='PC_TTK_s'
        ),
        row=1, col=1
    )

    # Add horizontal line at 0 for first subplot
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)

    # Second subplot: delta_TTK_s and PC_TTK_s.diff()
    # Plot delta_TTK_s_(selected_wand) in orange
    if delta_ttk_column in df_boat_analyzed_to_plot.columns:
        fig.add_trace(
            go.Scatter(
                x=df_boat_analyzed_to_plot.index, 
                y=df_boat_analyzed_to_plot[delta_ttk_column],
                mode='lines', 
                line=dict(color='orange', width=2), 
                name=f'Delta TTK {selected_wand}'
            ),
            row=2, col=1
        )

    # Plot PC_TTK_s.diff()
    pc_ttk_diff = df_boat_analyzed_to_plot['PC_TTK_s'].diff()
    fig.add_trace(
        go.Scatter(
            x=df_boat_analyzed_to_plot.index, 
            y=pc_ttk_diff,
            mode='lines', 
            line=dict(color='blue', width=2), 
            name='PC_TTK_s Diff'
        ),
        row=2, col=1
    )

    # Add horizontal line at 0 for second subplot
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

    # Conditional colors for annotations (red/green based on performance)
    color_delta_selected = "green" if delta_TTK_selected > 0 else "red"
    color_ratio_selected = "green" if ratio_TTK_selected > 1 else "red"

    # Add TTK metrics annotations positioned next to the legend
    fig.add_annotation(xref='paper', yref='paper', x=0.25, y=0.55,
                       text=f"ΔTTK {selected_wand}= {delta_TTK_selected:.2f}s", 
                       font=dict(color=color_delta_selected, size=14), showarrow=False)
    fig.add_annotation(xref='paper', yref='paper', x=0.25, y=0.45,
                       text=f"Ratio {selected_wand}= {ratio_TTK_selected:.2f}", 
                       font=dict(color=color_ratio_selected, size=14), showarrow=False)

    # Update layout
    fig.update_layout(
        title=f'TTK Analysis Subplots - Boat {boat} (Selected Wand: {selected_wand})',
        height=700,
        showlegend=True,
        legend=dict(x=0.01, y=0.5, yanchor='middle'),  # Position legend between plots on the left
        template='plotly_white'
    )

    # Update x-axis labels
    fig.update_xaxes(title_text="Time", row=2, col=1)
    
    # Update y-axis labels
    fig.update_yaxes(title_text="PC_TTK_s (s)", row=1, col=1)
    fig.update_yaxes(title_text="Delta Values", row=2, col=1)

    return fig

