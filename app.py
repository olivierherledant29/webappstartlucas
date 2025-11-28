import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import arrow

from influxdb_client import InfluxDBClient
from scipy.interpolate import RegularGridInterpolator

from start_analysis_live import (
    get_high_res_data_event,
    get_start_times_boat_ref,
    get_data_boat_for_starts,
    get_data_marks_for_starts,
    extract_metrics_for_race,
    compute_ideal_time_series_pow,
    recompute_stats_for_ideal_time_series,
    get_points_of_interest,
    plot_start_analysis_with_ideal_plotly,
    plot_ttk_start_analysis_subplots_plotly
)


st.write("Loading polar file...")
polar_df = pd.read_csv("polar.csv", index_col=0)
polar_df.index = polar_df.index.astype(float)
polar_df.columns = polar_df.columns.astype(float)

twa_grid = polar_df.index.values
tws_grid = polar_df.columns.values
interp_func = RegularGridInterpolator((twa_grid, tws_grid), polar_df.values, bounds_error=False, fill_value=0.0)



st.title("SailGP Start Analysis (Streamlit)")

boat_list_default = ['AUS','BRA','CAN','GBR','GER','ITA','SUI','USA','DEN','FRA','ESP','NZL']

boat_ref = st.selectbox("Boat reference", boat_list_default, index=boat_list_default.index("FRA"))

end_time = st.text_input("End time UTC", arrow.utcnow().format("YYYY-MM-DDTHH:mm:ss") + 'Z')

minutes_before = st.number_input("Minutes before end", min_value=1, max_value=30, value=2)

selected_boats = st.multiselect("Select boats", boat_list_default, default=boat_list_default)

practice_mode = st.checkbox("Practice mode")

run_button = st.button("Run analysis")



if run_button:

    client = InfluxDBClient(
        url="https://data.sailgp.tech",
        token="TON_TOKEN_ICI",
        org="0c2a130d50b8facc",
        verify_ssl=False
    )

    end = arrow.get(end_time)
    start = end.shift(minutes=-minutes_before)

    st.write(f"Fetching reference boat data from {start} to {end}...")

    df_ref = get_high_res_data_event(
        client, "0c2a130d50b8facc",
        boat_ref,
        start, end,
        channel_names=[
            "LATITUDE_GPS_unk","LONGITUDE_GPS_unk",
            "BOAT_SPEED_km_h_1",
            "TWS_BOW_SGP_km_h_1","TWA_BOW_SGP_deg","TWD_BOW_SGP_deg",
            "TWS_MHU_SGP_km_h_1","TWA_MHU_SGP_deg",
            "TRK_RACE_NUM_unk","TTS_s","PC_TTS_s","PC_TTK_s"
        ]
    )

    start_dict = get_start_times_boat_ref(df_ref)
    last_race = max(start_dict.keys())

    st.success(f"Race detected: {last_race}")



    start_dict = get_data_boat_for_starts(
        client, "0c2a130d50b8facc",
        start_dict, selected_boats,
        channel_names=[
            "LATITUDE_GPS_unk","LONGITUDE_GPS_unk",
            "BOAT_SPEED_km_h_1",
            "TWS_BOW_SGP_km_h_1","TWA_BOW_SGP_deg",
            "TWS_MHU_SGP_km_h_1"
        ]
    )

    start_dict = get_data_marks_for_starts(
        "0c2a130d50b8facc",
        client,
        start_dict,
        ["M1","SL1","SL2"],
        ["TWS_MDSS_km_h_1","TWD_MDSS_deg","LATITUDE_MDSS_deg","LONGITUDE_MDSS_deg"]
    )



    all_metrics_df, all_analyzed_dfs_race, all_poi_race, line_analysis = extract_metrics_for_race(
        last_race, start_dict, interp_func, twa_grid, tws_grid
    )

    start_fig = plot_start_analysis_with_ideal_plotly(
        last_race,
        all_metrics_df,
        line_analysis,
        all_analyzed_dfs_race,
        all_poi_race
    )

    ttk_fig = plot_ttk_start_analysis_subplots_plotly(
        boat_ref,
        all_analyzed_dfs_race,
        all_poi_race
    )

    st.plotly_chart(start_fig, use_container_width=True)
    st.plotly_chart(ttk_fig, use_container_width=True)
