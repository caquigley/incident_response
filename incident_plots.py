#import yaml
#%%
import sys
import os
import requests
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from obspy import UTCDateTime
from obspy.geodetics import gps2dist_azimuth
from obspy.geodetics import kilometers2degrees
from obspy import read_events
from matplotlib.colors import LightSource, to_rgba
import cartopy.crs as ccrs
import geopandas as gpd
from matplotlib.colors import ListedColormap, BoundaryNorm
from pyproj import Geod, Proj
import matplotlib.patheffects as pe

from incident_functions import (pull_earthquakes, pull_stations, pull_streams,
                                calculate_shaking, make_profile, add_inset_map,
                                legend_handles, add_basemap)

'''
To run this file, need to be in cartopy environment

python incident_plots.py ak0216xu2rod

'''

#############################################################
#----------STEP 1: INPUTS-----------------------------------
#############################################################


#Put in event ID
if len(sys.argv) > 1:
    event_id = sys.argv[1]
    MMI_plot = sys.argv[2]
    print(f"Event ID for analysis: {event_id}")
else:
    print("No input provided, need event ID to run")

#event_id = 'us6000rsy1' #'aka2026ukynnb' 5.6 today, #'us7000asvb' 7.8 Sandpoint #'us7000qd1y' 2025 Sandpoint #'ak018fcnsk91' #(2018 Anchorage), 'ak01613v15nv' (2016 Iniskin) #'us6000tz1y' (Australia) #'us7000ti1p' (M6.5 Rat islands) # 'aka2026tuxgky' (5.5)  us6000tza4 (Rat islands),  us6000u18k (Panama)

#What plots to make
plot_map = True
plot_cross_section = True
plot_accelerations = True
plot_mmi = True


#Inputs for reading in event----------------
event_id  = str(event_id)
lat = '61.218056' #Anchorage latitude
lon = '-149.900278' #Anchorage longitude
max_rad = '4000' #circle that covers all of Alaska, 19000 for globe
min_mag = '5.0' #min mag to search over
event_loc = 'Alaska' #Alaska, Global
start = str(UTCDateTime.now() - (60*60*24*365)*10) #pull events from last ten years
end = str(UTCDateTime.now())

#Inputs for stations-------------------
max_rad_stations = 300 #radius of circle to search for stations from in km
subset_dist = 200 #radius of circle of stations subset from original list
    
#Inputs for accelrations plot---------------
MMI_plot = int(MMI_plot) #MMI of plot to be used

#Historical inputs--------
max_rad_hist = '600'
min_mag_hist = '3.0' #3.0

#Aftershock inputs---------
max_rad_aftershocks = '100'
min_mag_aftershocks = '2.0'

if __name__ == "__main__":

    #############################################################
    #----------STEP 2: PULL DATA-----------------------------------
    #############################################################
   
    #Pull earthquakes to find event-------
    eq_df = pull_earthquakes(lat, lon, max_rad, min_mag, start, end)

    #Pull out event of interest -----------------------------
    temp = eq_df.loc[eq_df['event_id'] == event_id].head(1)
    if len(temp) == 1:
        print('Event found!')

    
    #Create directory to save figures-------------------
    dir_path = "./events/"+event_id
    os.makedirs(dir_path, exist_ok=True)

    #Define metrics for the earthquake ----------------------
    eq_lat = temp['latitude'].to_numpy()[0]
    eq_lon = temp['longitude'].to_numpy()[0]
    eq_depth = temp['depth'].to_numpy()[0]
    eq_mag = temp['magnitude'].to_numpy()[0]
    eq_time = str(temp['time_utc'].to_numpy()[0])

    #Pull all stations in region -----------------------------
    stations_df, inv = pull_stations(eq_lat, eq_lon, eq_time, eq_depth, max_rad_stations)

    #Create subset of stations based on closest station------------------------------
    min_dist = stations_df['distance'].min()
    first_p = stations_df['p_arrival'].min()
    closest_stations = min_dist + subset_dist #200
    stations_subset = pd.DataFrame(stations_df[stations_df['distance']<= closest_stations])
    stations_subset = stations_subset.sort_values(by="distance", ascending=True)
    
    #Try and pull seismic data -----------------------------
    networks = stations_subset['network'].to_numpy()
    stations = stations_subset['station'].to_numpy()
    channels = stations_subset['channel'].to_numpy()

    st, inv2, df = pull_streams(networks, stations, channels, eq_time)
    st1 = st.copy() # save version for later analysis

    #Merge all station data--------------
    available_stations = pd.merge(df, stations_subset, on=['station', 'channel', 'network'])
    available_stations = available_stations.drop_duplicates().reset_index()
    available_stations = available_stations.sort_values(by = 'distance')

    #Create traces for plotting in map figure----------------
    st.remove_sensitivity(inventory = inv2)  
    st.taper(max_percentage=0.05)
    st.filter("bandpass", freqmin=1, freqmax=18, 
                corners=2, zerophase=True)
    st = st.slice(UTCDateTime(eq_time), UTCDateTime(eq_time)+ first_p +120) #90

    #Pull historical earthquakes-----------------------------
    print('Pulling historical earthquakes for plotting...')
    start = '2000-01-01' 
    end = str(UTCDateTime(eq_time)-(60*60*24))
    
    try:
        hist_df = pull_earthquakes(str(eq_lat), str(eq_lon), max_rad_hist, min_mag_hist, start, end)
    except ValueError:
        print('No historical earthquakes from comcat data request.')
        hist_df = None
    except requests.exceptions.HTTPError as e:
        print(f'HTTP error pulling historical earthquakes: {e}')
        print('Trying to pull fewer earthquakes')
        start = str(UTCDateTime(eq_time)-86400*365*5) #pull event from last 5 years
        min_mag_hist = '4.0'
        hist_df = pull_earthquakes(str(eq_lat), str(eq_lon), max_rad_hist, min_mag_hist, start, end)
        

    #Pull aftershocks --------------------------
    print('Pulling aftershocks for plotting...')
    start = str(UTCDateTime(eq_time)+1*60) #one minute after
    end = str(UTCDateTime(eq_time)+86400*10) #10 days after
    if UTCDateTime(end) - UTCDateTime.now() > 0:
        end  = str(UTCDateTime.now())
    try:
        aftershock_df = pull_earthquakes(str(eq_lat), str(eq_lon), max_rad_aftershocks, min_mag_aftershocks, start, end)
    except ValueError:
        print('No aftershocks from comcat data request.')
        aftershock_df = []
    except requests.exceptions.HTTPError as e:
        print(f'HTTP error pulling aftershocks: {e}')
        aftershock_df = []

    #Calculate shaking-------------------------
    #------------------------------------------
    parrivals = available_stations['p_arrival'].values
    result = calculate_shaking(st1, inv2, parrivals, eq_time)
    
    stations = available_stations['station'].values
    channels = available_stations['channel'].values

    durations = []
    starts = []
    ends = []
    max_mmis = []
    for i in range(len(stations)):
        tr = st1[i]
        temp = pd.DataFrame(result[result['station']==stations[i]])
        temp = pd.DataFrame(temp[temp['channel']==channels[i]])
        temp = pd.DataFrame(temp[temp['mmi']== MMI_plot])
        start_shaking = temp['start_shaking'].values[0]
        end_shaking = temp['end_shaking'].values[0]
        duration = temp['duration'].values[0]
        max_mmi = temp['max_mmi'].values[0]
        durations.append(duration)
        starts.append(start_shaking)
        ends.append(end_shaking)
        max_mmis.append(max_mmi)

    #Add information to available stations
    available_stations['duration'] = durations
    available_stations['start_shaking'] = starts
    available_stations['end_shaking'] = ends
    available_stations['max_mmi'] = max_mmis

    #Recalculate duration times based on start/end shaking times on each channel
    df = available_stations.copy()          

    # Work only on rows with a usable shaking window
    good = df[df['duration'] != 0].copy()

    if len(good) > 0:
        good['start_shaking'] = pd.to_datetime([t.datetime for t in good['start_shaking']])
        good['end_shaking'] = pd.to_datetime([t.datetime for t in good['end_shaking']])

        # Instrument type (BHZ -> BH, HNE -> HN)
        good['instrument'] = good['channel'].str[:2]

        # Step 1: full shaking window for each station + instrument
        inst = (good.groupby(['station', 'instrument'], as_index=False)
                    .agg(inst_start=('start_shaking', 'min'),
                        inst_end=('end_shaking', 'max')))
        inst['inst_duration'] = (inst['inst_end'] - inst['inst_start']).dt.total_seconds()

        # Step 2: for each station, keep the instrument with the longest window
        best = inst.loc[inst.groupby('station')['inst_duration'].idxmax()]
        best = best.rename(columns={'inst_duration': 'max_duration',
                                    'inst_start': 'max_dur_start',
                                    'inst_end': 'max_dur_end',
                                    'instrument': 'max_dur_instrument'})
        best = best[['station', 'max_duration', 'max_dur_start',
                    'max_dur_end', 'max_dur_instrument']]
    else:
        best = pd.DataFrame(columns=['station', 'max_duration', 'max_dur_start',
                                    'max_dur_end', 'max_dur_instrument'])

    # Merge onto the FULL dataframe so zero-duration stations are kept
    df = df.drop(columns=['max_duration', 'max_dur_start', 'max_dur_end', 'max_dur_instrument'],
                errors='ignore')
    df = df.merge(best, on='station', how='left')

    # Stations with no valid window: duration 0, start/end stay NaT
    df['max_duration'] = df['max_duration'].fillna(0)
    df['max_dur_start'] = df['max_dur_start'].fillna(0)
    df['max_dur_end'] = df['max_dur_end'].fillna(0)
    df['max_dur_instrument'] = df['max_dur_instrument'].fillna(0)

    available_stations = df
    available_stations['max_mmi_all'] = (
        available_stations.groupby('station')['max_mmi'].transform('max'))

    #Filter back down to vertical stations for plotting---------
    available_stations = available_stations[
            available_stations['channel'].str.endswith('Z')].copy()

    #Drop stations that have exceedingly long/or short shaking durations
    #available_stations_filt = pd.DataFrame(available_stations[available_stations['duration'] <= 3*np.median(durations)])
    available_stations_filt = available_stations

    #Read in fault file----------------------------
    print('Reading in Alaska fault files')
    gdf = gpd.read_file('./fault_files/AEC_Faults_Simplified/Faults_simple.shp')
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    gdf = gdf.to_crs('EPSG:4326')     # in case the file isn't in lon/lat (check gdf.crs)

    #Read in towns-------------------------------
    towns = pd.read_csv('./points_of_interest.csv')
    km = []
    deg = []
    for i in range(len(towns)):
        dist, az, baz = gps2dist_azimuth(eq_lat, eq_lon, towns['lat'].values[i],
                                         towns['lon'].values[i])
        km.append(dist/1000)
        deg.append(kilometers2degrees(dist/1000))
    towns['distance_km'] = km
    towns['distance_deg'] = deg

    #Create legend handles----------------
    (depth_handles_hist, depth_handles_aftershocks, mag_handles, 
            fault_handle, sta_handle, sta_handle2) = legend_handles()
    
    #%%

    if plot_map == True:
        #--------------------------------------------------
        #----------MAP FIGURE------------------------------

        print('Starting map figure...')

        # Clarify whether to plot aftershocks and historical data-------------------
        plot_aftershocks = False
        plot_historical = False
        if len(aftershock_df) == 0:
            plot_aftershocks = False
        if len(hist_df) == 0:
            plot_historical = False

        #Set width of map------------
        dlat = 1.5 #2.5

        #Pull out towns only in map area---------------------------
        towns_filt = pd.DataFrame(towns[towns['distance_deg'] < dlat])

        #Find min distance from each station to a town----------------
    
        min_tdist = []
        for i in range(len(available_stations)):
            sta_lat = available_stations['sta_lat'].values[i]
            sta_lon = available_stations['sta_lon'].values[i]
            temp_dist = []
            for k in range(len(towns_filt)):
                town_lat = towns_filt['lat'].values[k]
                town_lon = towns_filt['lon'].values[k]
                dist, az, baz = gps2dist_azimuth(sta_lat, sta_lon, town_lat, town_lon)
                temp_dist.append(dist/1000)

            min_tdist.append(np.min(np.array(temp_dist)))

        
        available_stations['min_town_dist'] = min_tdist

        num_streams = 3
        dist_sep = 50 #50 km 
        dist_targets = []
        for i in range(num_streams):
            dist_targets.append(min_dist + i*dist_sep)
        df_targets = pd.DataFrame({'target': dist_targets}).sort_values('target')

        # Stations for plotting-----------------
        plot_stations = pd.merge_asof(
            df_targets, 
            available_stations, 
            left_on='target', 
            right_on='distance', 
            direction='nearest'
        )
        plot_stations = plot_stations.drop_duplicates(subset=['station'])
        '''
        available_stations['min_town_dist'] = min_tdist

        num_streams = 3 #4,3
        dist_sep = 50   #50     # km between target distances
        dist_tol = 10 #50       # km: only consider stations within +/- this of a target
        town_scale = 20      # km: a town distance of this much counts as a "full" penalty
        w_dist, w_town = 0.5, 0.5   # relative priority: raise w_town to favor towns more

        dist_targets = [min_dist + i * dist_sep for i in range(num_streams)]

        selected = []
        used = set()
        for target in dist_targets:
            cand = available_stations[~available_stations['station'].isin(used)].copy()
            if cand.empty:
                break

            cand['dist_err'] = (cand['distance'] - target).abs()

            # Prefer stations inside the tolerance window; if none, fall back to the closest one
            pool = cand[cand['dist_err'] <= dist_tol]
            if pool.empty:
                pool = cand.nsmallest(1, 'dist_err')
            pool = pool.copy()

            # Lower score = better. Both terms are roughly 0-1 so the weights are comparable.
            pool['score'] = (w_dist * pool['dist_err'] / dist_tol
                            + w_town * pool['min_town_dist'].fillna(town_scale * 10) / town_scale)

            best = pool.loc[pool['score'].idxmin()].copy()
            best['target'] = target
            selected.append(best)
            used.add(best['station'])
 
        plot_stations = pd.DataFrame(selected).reset_index(drop=True)
        plot_stations = plot_stations.drop_duplicates(subset=['station'])  
        '''
        #-------------------------------------------------
        #Create figure------------------------------------  
        
        fig = plt.figure(figsize=(8, 10), dpi=150)
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.5], hspace=0.1) #0.25

        #Create basemap from personalized function
        ax, region = add_basemap(fig, gs[0], eq_lon, eq_lat, dlat)
        left, right, bottom, top = region
        PC = ccrs.PlateCarree()
    

        # Plot historical data--------------------------
        if plot_historical == True:
            #Plot historical earthquakes---------------------
            depth = hist_df["depth"].values
            mag = hist_df["magnitude"].values
            depth_colors = np.where(depth < 35, 'whitesmoke',#"firebrick",
                            np.where(depth < 100, "mediumslateblue","slateblue"))#"gold", "#4a0010"))
            alphas = np.where(mag >= 4.5, 0.8, 0.25)   # 0.25fade small events; adjust to taste
            face = [to_rgba(c, a) for c, a in zip(depth_colors, alphas)]
            edge = [to_rgba("k", a) for a in alphas]

            ax.scatter(hist_df["longitude"], hist_df["latitude"], s=1.5 * 2**mag,
                    c=face, edgecolors=edge, linewidths=0.8,
                    transform=PC, zorder=3)

        if plot_aftershocks == True:
            #Plot aftershocks ---------------------
            depth = aftershock_df["depth"].values
            mag = aftershock_df["magnitude"].values

            depth_colors = np.where(depth < 35, "firebrick",
                            np.where(depth < 100, "gold", "#4a0010"))
            alphas = np.where(mag >= 4.5, 1.0, 1.0)   # fade small events; adjust to taste
            face = [to_rgba(c, a) for c, a in zip(depth_colors, alphas)]
            edge = [to_rgba("k", a) for a in alphas]

            ax.scatter(aftershock_df["longitude"], aftershock_df["latitude"], s=1.5 * 2**mag,
                    c=face, edgecolors=edge, linewidths=0.8,
                    transform=PC, zorder=6)

        #Plot faults-----------------
        if event_loc == 'Alaska': #only have faults for alaska for now...
            ax.add_geometries(gdf.geometry, crs=ccrs.PlateCarree(),
                            facecolor='none', edgecolor='saddlebrown', #dimgrey
                            alpha=0.9, linewidth=1.5, zorder=2)
            # Plot towns-------------------------
            for i in range(len(towns_filt)):
                town_lon = towns_filt['lon'].values[i]
                town_lat = towns_filt['lat'].values[i]
                town_name = towns_filt['name'].values[i]
                ax.scatter(town_lon, town_lat, marker="s", s=80,
                                c="navy", edgecolors="k", linewidths=1.0, 
                                transform=PC, zorder=4)
                ax.text(town_lon, town_lat - 0.05, town_name, fontsize=8, transform = PC,
                            fontname = 'Arial', va = 'top', #edgecolor = 'black',
                            ha = 'center', color = 'black', fontweight = 'bold',
                            zorder = 20)

        #Plot earthquake-----------------------
        ax.scatter(eq_lon, eq_lat, s = 1000, color = 'red', edgecolors='black', 
                linewidths=1.5, marker = "o",
                transform=PC, zorder=30)

        # Plot stations-------------------------
        ax.scatter(stations_df["sta_lon"], stations_df["sta_lat"], marker="v", s=70,
                c="gray", edgecolors="k", linewidths=1.0, 
                transform=PC, zorder=3)

        
        
        #Handle legend inputs----------------
        if plot_historical == True:
            handles = depth_handles_hist + mag_handles + [sta_handle] + [sta_handle2] + [fault_handle]
        elif plot_historical == False:
            if plot_aftershocks == True:
                handles = depth_handles_aftershocks+ [sta_handle] + [sta_handle2] + mag_handles + [fault_handle]
            elif plot_aftershocks == False:
                handles = [sta_handle] + [sta_handle2] + [fault_handle]

        #Plot legend
        leg = ax.legend(handles=handles, loc='lower right', frameon=True,
                        framealpha=1.0, 
                        prop={'family': 'arial', 'size': 7},
                        labelspacing=0.6, borderpad=0.4)
        leg.set_zorder(30)   # keep it above the relief and scatter layers

        # Inset map --------------------------
        ax_in = add_inset_map(ax, event_loc, eq_lon, eq_lat, region)
        
        #----------------------------------------------
        # Traces panel---------------------------------
        #----------------------------------------------

        ax2 = fig.add_subplot(gs[1])
        index = plot_stations['og_index'].values

        #Plot traces-----------------
        stas = []
        for i in range(len(index)):
            tr = st[index[i]]
            ax2.plot(tr.times(), (tr.data/np.max(np.abs(tr.data)))-i*2, color = 'black', alpha = 0.8)
            ax2.text(np.max(tr.times()), -i*2, tr.stats.station, va = 'bottom', 
                    ha = 'right', color = 'silver', fontweight = 'bold')
            
            stas.append(tr.stats.station)

        ax2.grid(alpha = 0.3)
        ax2.axvline(0, color = 'red', linestyle = '--')
        ax2.scatter(0,0.75, marker = 'o', color = 'red', edgecolor = 'black', s = 300, zorder = 20)
        ax2.set_xlabel("time since earthquake origin (s)")
        ax2.set_yticks([])
        ax2.set_ylabel('distance from earthquake', labelpad=15)

        #Add vertical line on left side of plot---------------
        ax2.annotate(
            "",
            xy=(-0.02, -2*(len(plot_stations)-1)),
            xytext=(-0.02, 0),
            xycoords=ax2.get_yaxis_transform(),   # x in axes fraction, y in data
            textcoords=ax2.get_yaxis_transform(),
            annotation_clip=False,
            arrowprops=dict(arrowstyle="-|>", color="red", lw=2, mutation_scale=20),
        )


        #Add stations to original map----------
        for i in range(len(plot_stations)):
            #temp = stations_df.loc[stations_df['station'] == stas[i]].head(1)
            sta_lon = plot_stations['sta_lon'].values[i]
            sta_lat = plot_stations['sta_lat'].values[i]
            name = plot_stations['station'].values[i]
            # Line from epicenter to station
            ax.plot([eq_lon, sta_lon], [eq_lat, sta_lat],
                    color='firebrick', linewidth=1.2, alpha=0.8,
                    transform=ccrs.Geodetic(), zorder=3.5)
            ax.scatter(sta_lon, sta_lat, marker="v", s=200,
                    c="firebrick", edgecolors="k", linewidths=1.0, 
                    transform=PC, zorder=4)

            ax.text(sta_lon, sta_lat+0.05, name, fontsize=9, transform = PC,
                    fontname = 'Arial', va = 'bottom', #edgecolor = 'black',
                    ha = 'center', color = 'firebrick', fontweight = 'bold',
                    zorder = 20)

        # Plot Date/Magnitude text----------------------
        t = UTCDateTime(eq_time)
        date_str = t.strftime('%Y %b. %d')     # 2026 Oct. 15
        ax.text(0.02, 0.98, date_str + ' M' + str(eq_mag), transform=ax.transAxes,
                fontsize=17, va='top', ha='left', fontweight='bold', fontname='Arial',
                bbox=dict(facecolor='white', alpha=1.0, edgecolor='black'), zorder=20)
        
        # Draw plot together correctly-----------------------------
        fig.canvas.draw()                     # forces cartopy to apply its aspect ratio
        pos_map = ax.get_position()
        pos_bot = ax2.get_position()
        gap = 0.0   # figure-fraction gap between panels; try 0.01-0.03 if you want some room
        new_y0 = pos_map.y0 - gap - pos_bot.height
        ax2.set_position([pos_map.x0, new_y0, pos_map.width, pos_bot.height])

        #Save figure
        fig.savefig(dir_path+'/map_figure.png', dpi=720, bbox_inches='tight')

        plt.show()
   
    #%%
    #############################################
    # Cros section plot-------------------------

    if plot_cross_section == True:
        print('Plotting cross section...')
        dlat = 3.0
        plot_aftershocks = False
        plot_historical = True
        
        #--------------------------------
        #Pull in slab 2 data

        #Pull in SLAB 2 depths--------------
        path = './slab2/alu_slab2_dep_02.23.18.xyz'
        df = pd.read_csv(path, names = ['lon','lat','depth'])
        df = df.dropna(subset=['depth'])
        df['depth'] = df['depth']*-1

        #Pull in SLAB 2 strikes--------------
        path = './slab2/alu_slab2_str_02.23.18.xyz'
        df1 = pd.read_csv(path, names = ['lon','lat','strike'])
        df1 = df1.dropna(subset=['strike'])

        #Pull in SLAB 2 thickness--------------
        path = './slab2/alu_slab2_thk_02.23.18.xyz'
        df2 = pd.read_csv(path, names = ['lon','lat','thick'])
        df2 = df2.dropna(subset=['thick'])

        df3 = pd.merge(df, df1, on=['lat', 'lon'])
        df3 = pd.merge(df3, df2, on=['lat', 'lon'])

        #Convert to correct longitude
        temp_vals = df3['lon'].values
        lons = []
        for i in range(len(temp_vals)):
            temp = temp_vals[i]
            if temp >180:
                lons.append(temp - 360)
            elif temp <=180:
                lons.append(temp)
        df3['lon'] = lons
        df3['depth_top'] = df3['depth']
        df3['depth_bottom'] = df3['depth']+ df3['thick']

        #Round eq_lat/eq_lon for comparison
        lat_round = np.round(eq_lat, decimals = 2)
        lon_round = np.round(eq_lon, decimals = 2)

        # Use the nearest slab nodes for strike, even if the event is outside the grid
        lon_d = (df3['lon'].values - eq_lon + 180) % 360 - 180            # dateline-safe
        lat_d = df3['lat'].values - eq_lat
        dist = np.hypot(lon_d * np.cos(np.deg2rad(eq_lat)), lat_d)        # approx degrees

        # Search to find nearest grid point
        search_deg = 2.0                                                  # how far away to look
        if dist.min() <= search_deg:
            near = dist <= max(0.2, dist.min() + 0.2)                     # ~your old +/-0.2 deg box when inside
            strike = np.deg2rad(df3['strike'].values[near])
            mean_strike = np.rad2deg(np.arctan2(np.sin(strike).mean(), np.cos(strike).mean())) % 360  # circular mean
            print(f'Using slab strike {mean_strike:.0f} from nodes {dist.min():.2f} deg away')
        else:
            print('No slab nearby, assigning default strike of 270.')
            mean_strike = 270 #Assigned so that default cross section is north south

        geod = Geod(ellps='WGS84')

        #----------------------------------------
        # Calculate the profile-----------

        az, half_len = mean_strike - 90, 250  # azimuth in degrees, half-length in km
        lon1, lat1, _ = geod.fwd(eq_lon, eq_lat, az + 180, half_len * 1000)
        lon2, lat2, _ = geod.fwd(eq_lon, eq_lat, az, half_len * 1000)

        start = (lon1, lat1)       # A
        end = (lon2, lat2)         # A'
        half_width = 40            # km on each side of the line

        project, L, swath_lon, swath_lat = make_profile(start, end, half_width)

        along, perp = project(hist_df['longitude'].values, hist_df['latitude'].values)
        mask = (np.abs(perp) <= half_width) & (along >= 0) & (along <= L)
        print(f'{mask.sum()} earthquakes within {half_width} km of the profile')

        #----------------------------------------
        # Make the map---------------------------
        fig = plt.figure(figsize=(8, 10), dpi=150)
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.5], hspace=0.1) #0.25
        ax, region = add_basemap(fig, gs[0], eq_lon, eq_lat, dlat)
        left, right, bottom, top = region
        PC = ccrs.PlateCarree()

        ax_xs = fig.add_subplot(gs[1])                      # cross-section


        if plot_historical == True:
            #Plot historical earthquakes---------------------
            depth = hist_df["depth"].values
            mag = hist_df["magnitude"].values

            depth_colors = np.where(depth < 35, 'whitesmoke',#"firebrick", lightgrey, whitesmoke
                            np.where(depth < 100, "mediumslateblue","darkslateblue"))#"gold", "#4a0010")), darkgrey, dimgrey
            alphas = np.where(mag >= 4.5, 0.8, 0.8)   # 0.25fade small events; adjust to taste
            face = [to_rgba(c, a) for c, a in zip(depth_colors, alphas)]
            edge = [to_rgba("k", a) for a in alphas]

            ax.scatter(hist_df["longitude"], hist_df["latitude"], s=1.5 * 2**mag, #3
                    c=face, edgecolors=edge, linewidths=0.8,
                    transform=PC, zorder=3)

        if plot_aftershocks == True:
            #Plot historical earthquakes---------------------
            depth = aftershock_df["depth"].values
            mag = aftershock_df["magnitude"].values

            depth_colors = np.where(depth < 35, "firebrick",
                            np.where(depth < 100, "gold", "#4a0010"))
            alphas = np.where(mag >= 4.5, 1.0, 1.0)   # fade small events; adjust to taste
            face = [to_rgba(c, a) for c, a in zip(depth_colors, alphas)]
            edge = [to_rgba("k", a) for a in alphas]

            ax.scatter(aftershock_df["longitude"], aftershock_df["latitude"], s=1.5 * 2**mag,
                    c=face, edgecolors=edge, linewidths=0.8,
                    transform=PC, zorder=30)
            
        #Plot faults-----------------
        if event_loc == 'Alaska': #only have faults for alaska for now...
            ax.add_geometries(gdf.geometry, crs=ccrs.PlateCarree(),
                            facecolor='none', edgecolor='saddlebrown', #dimgrey
                            alpha=0.9, linewidth=1.5, zorder=2)

        #Plot earthquake-----------------------
        ax.scatter(eq_lon, eq_lat, s = 800, color = 'red', edgecolors='black', 
                linewidths=1.5, marker = "o",
                transform=PC, zorder=20)

        #Plot legend------------
        if plot_historical == True:
            handles = depth_handles_hist + mag_handles  + [fault_handle]
        elif plot_historical == False:
            if plot_aftershocks == True:
                handles =   depth_handles_aftershocks + mag_handles + [fault_handle]
            elif plot_aftershocks == False:
                handles =  [fault_handle]
        leg = ax.legend(handles=handles, loc='lower right', frameon=True,
                        framealpha=1.0, 
                        prop={'family': 'arial', 'size': 7},
                        labelspacing=0.6, borderpad=0.4)
        leg.set_zorder(10)   # keep it above the relief and scatter layers

        # Inset locator map ----------
        # Bounds are in axes fraction: [x0, y0, width, height] of the main map
        ax_in = add_inset_map(ax, event_loc, eq_lon, eq_lat, region)

        # Profile line and swath on the map
        ax.plot([lon1, lon2], [lat1, lat2], color='cyan', linewidth=2,
                transform=PC, zorder=6) #ccrs.Geodetic
        ax.plot(swath_lon, swath_lat, color='cyan', linewidth=2, linestyle='--',
                transform=PC, zorder=6)

        for (lo, la), lab in [(start, 'A'), (end, "A'")]:
            ax.text(lo, la, lab, transform=PC, fontsize=14, fontweight='bold',
                    ha='center', va='bottom', zorder=7,
                    path_effects=[pe.withStroke(linewidth=3, foreground='white')])
            
        mag = hist_df['magnitude'].values
        depth = hist_df['depth'].values

        #----------------------------------------
        # Cross-section portion------------------
        #----------------------------------------

        ax_xs.scatter(along[mask], depth[mask], s=3 * 2**mag[mask],
                    c=np.array(face, dtype=object)[mask].tolist(),      # same colors as map
                    edgecolors=np.array(edge, dtype=object)[mask].tolist(),
                    linewidths=0.8, zorder=3)

        # Mainshock, if it falls within the swath
        e_along, e_perp = project(eq_lon, eq_lat)
        if abs(e_perp) <= half_width and 0 <= e_along <= L:
            ax_xs.scatter(e_along, eq_depth, marker='o', s=600, color='red',
                        edgecolors='k', linewidths=1.5, zorder=40)

        ax_xs.set_xlim(0, L)
        ax_xs.invert_yaxis()                                  # depth increases downward
        ax_xs.set_ylabel('depth (km)')
        ax_xs.set_xlabel('distance along profile (km)')
        ax_xs.grid(alpha=0.3)

        ax_xs.text(0.01, 1.02, 'A', transform=ax_xs.transAxes, fontsize=14,
                fontweight='bold', ha='left', va='bottom')
        ax_xs.text(0.99, 1.02, "A'", transform=ax_xs.transAxes, fontsize=14,
                fontweight='bold', ha='right', va='bottom')

        #----------------------------------------------------------------
        # Add slab contours

        # slab_df columns: 'lon', 'lat', 'depth'  (rename to match yours)
        slab_df = df3
        slab_lon = slab_df['lon'].values
        slab_lat = slab_df['lat'].values
        slab_depth = slab_df['depth'].values
        slab_thick = slab_df['thick'].values

        # Project and select the swath
        s_along, s_perp = project(slab_lon, slab_lat)
        m = (np.abs(s_perp) <= half_width) & (s_along >= 0) & (s_along <= L)
        s_along, s_depth, s_thick = s_along[m], slab_depth[m], slab_thick[m]

        # Collapse to one depth per distance bin
        bin_km = 10                                              # match to your grid spacing
        edges = np.arange(0, L + bin_km, bin_km)
        idx = np.digitize(s_along, edges) - 1

        centers, med, lo, hi = [], [], [], []
        thick_list = []
        for b in range(len(edges) - 1):
            d = s_depth[idx == b]
            thick = s_thick[idx == b]
            if d.size:
                centers.append((edges[b] + edges[b + 1]) / 2)
                med.append(np.median(d))
                lo.append(d.min())
                hi.append(d.max())
                thick_list.append(thick.min())

        centers, med, lo, hi = map(np.array, (centers, med, lo, hi))

        have_slab = len(centers) > 0
        plot_arrow = have_slab
        z0 = 35
        x_end = L

        def extend_right(x, y, x_end):
            """Continue a line horizontally at its last depth out to x_end."""
            if x[-1] < x_end:
                x = np.r_[x, x_end]
                y = np.r_[y, y[-1]]
            return x, y

        def extend_left(x, y):
            """Continue a line horizontally at its first depth back to x = 0."""
            if x[0] > 0:
                x = np.r_[0, x]
                y = np.r_[y[0], y]
            return x, y

        if not have_slab:
            # No slab data in the swath: schematic flat 35 km plate only
            ax_xs.fill_between([0, x_end], 0, z0, color='tab:orange', alpha=0.3,
                            edgecolor='none', zorder=2, label='North American Plate')
            ax_xs.plot([0, x_end], [z0, z0], color='k', linewidth=2, zorder=10)

        else: #Plot the slab and the North American Plate
            x, y = centers, lo
            s = y - z0
            i = np.where(s[:-1] * s[1:] < 0)[0]          # may be empty: no crossing is fine
            x_cross = x[i] + (z0 - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i])
            xs = np.insert(x, i + 1, x_cross)
            ys = np.insert(y, i + 1, z0)
            boundary = np.minimum(ys, z0)

            ocean_side = lo[-1] < z0                      # slab data end shallow = trench/ocean side

            xs_f = np.r_[0, xs]
            b_f = np.r_[boundary[0], boundary]
            if not ocean_side:                            # slab still deep: NA plate continues flat
                xs_f = np.r_[xs_f, x_end]
                b_f = np.r_[b_f, z0]

            #Color the North American Plate
            ax_xs.fill_between(xs_f, 0, b_f, color='tab:orange', alpha=0.3,
                            edgecolor='none', zorder=2, label='North American Plate')
            ax_xs.plot(xs_f, np.where(b_f >= z0, z0, np.nan),
                    color='k', linewidth=2, zorder=10)
            '''
            #Extend left and right side if the data ends early
            x1, y1 = extend_left(centers, lo)
            x2, y2 = extend_left(centers + 40, lo + 35)

            if ocean_side:
                x1, y1 = extend_right(x1, y1, x_end)
                x2, y2 = extend_right(x2, y2, x_end)

            ax_xs.fill(np.r_[x1, x2[::-1]], np.r_[y1, y2[::-1]],
                    facecolor='steelblue', alpha=0.3, edgecolor='none',
                    zorder=2, label='Pacific Plate')
            ax_xs.plot(x1, y1, color='k', linewidth=2, zorder=10)
            ax_xs.plot(x2, y2, color='k', linewidth=2, zorder=10)
            '''
            # Top and bottom surfaces of the Pacific plate, starting where Slab 2.0 data start
            x1, y1 = centers, lo
            x2, y2 = centers + 40, lo + 35

            # Only extend flat to the right if the data end at the shallow (ocean) side
            if ocean_side:
                x1, y1 = extend_right(x1, y1, x_end)
                x2, y2 = extend_right(x2, y2, x_end)

            ax_xs.fill(np.r_[x1, x2[::-1]], np.r_[y1, y2[::-1]],
                    facecolor='steelblue', alpha=0.3, edgecolor='none',
                    zorder=2, label='Pacific Plate')
            ax_xs.plot(x1, y1, color='k', linewidth=2, zorder=10)
            ax_xs.plot(x2, y2, color='k', linewidth=2, zorder=10)

            # Dotted line closing the plate where the Slab 2.0 data end on the left
            ax_xs.plot([x1[0], x2[0]], [y1[0], y2[0]],
                    color='k', linewidth=2, linestyle=':', zorder=10)
            # Ocean only where the plate was extended past the slab data
            if ocean_side and centers[-1] < x_end:
                ax_xs.fill_between([centers[-1], x_end], 0, lo[-1],
                                color='lightskyblue', alpha=0.4, edgecolor='none',
                                zorder=1, label='Pacific Ocean')

        # Y-limit that can't crash on an empty swath and never gets too shallow
        y_candidates = [60, eq_depth]
        if mask.any():
            y_candidates.append(depth[mask].max())
        ax_xs.set_ylim(max(y_candidates), 0)
        
        #---------------------------------------------
        # Add arrow and text--------------------------
        plot_arrow = False
        if plot_arrow == True:
            # Bottom edge of the pacific plate
            xb, yb = centers + 40, lo + 35
            # Arrow runs right-to-left: tail at xa_tail, tip at xa_tip (tip is to the left)
            xa_tail, xa_tip = 500, 400           # km along profile; adjust to taste
            ya_tail, ya_tip = np.interp([xa_tail, xa_tip], xb, yb)

            offset = 10   # km below the plate

            ax_xs.annotate("",
                        xy=(xa_tip, ya_tip + offset),        # arrow tip (left)
                        xytext=(xa_tail, ya_tail + offset),  # arrow tail (right)
                        arrowprops=dict(arrowstyle="-|>", color="firebrick",
                                        lw=3, mutation_scale=25),
                        annotation_clip=False, zorder=10)

            ax_xs.text((xa_tail + xa_tip) / 2, (ya_tail + ya_tip) / 2 + offset + 10,
                    'plate motion', color='firebrick', ha='center', va='top', fontsize=12)

        #--------------------------------------------
        # Set x and y limits------------------------   
        ax_xs.set_xlim(0, x_end)    

       # Add legend ------------------
        ax_xs.legend(frameon=True, #loc = 'lower right'
                        framealpha=1.0, 
                        prop={'family': 'arial', 'size': 8},
                        labelspacing=0.6, borderpad=0.4)
    

        #Add text to top of figure----------------------
        t = UTCDateTime(eq_time)
        date_str = t.strftime('%Y %b. %d')     # 2026 Oct. 15

        ax.text(0.02, 0.98, date_str + ' M' + str(eq_mag), transform=ax.transAxes,
                fontsize=17, va='top', ha='left', fontweight='bold', fontname='Arial',
                bbox=dict(facecolor='white', alpha=1.0, edgecolor='black'), zorder=20)
        
        ax.text(0.02, 0.90, 'Circles show M3+ historical seismicity since 2000', transform=ax.transAxes,
                fontsize=10, va='top', ha='left', fontweight = 'bold', style = 'italic',
                fontname = 'Arial', color = 'black', zorder = 20,
                bbox=dict(facecolor='white', alpha=0.6))

        #Connect top and bottom figures
        fig.canvas.draw()                     # forces cartopy to apply its aspect ratio
        pos_map = ax.get_position()
        pos_bot = ax_xs.get_position()
        gap = 0.03   # figure-fraction gap between panels; try 0.01-0.03 if you want some room
        new_y0 = pos_map.y0 - gap - pos_bot.height
        ax_xs.set_position([pos_map.x0, new_y0, pos_map.width, pos_bot.height])

        #Save figure
        fig.savefig(dir_path+'/cross_section_figure.png', dpi=720, bbox_inches='tight')

        plt.show()
#________________________________________________________________________________
#________________________________________________________________________________
#________________________________________________________________________________
    #%%
    if plot_accelerations == True:
        print('Making accelerations plot...')

        plot_aftershocks = False
        plot_historical = False
        if aftershock_df is None:
            plot_aftershocks = False
        if hist_df is None:
            plot_historical = False
        
        dlat = 2.5
        num_streams = 4 #5
        dist_sep = 30

        dist_tol = 40   # km: how far from a target distance a station may be and still count

        dist_targets = [min_dist + i * dist_sep for i in range(num_streams)]

        # One row per station, flagged by whether it recorded any shaking
        stations = available_stations.drop_duplicates(subset='station').copy()
        stations['has_shaking'] = stations['max_duration'].fillna(0) > 0

        selected = []
        used = set()
        for target in dist_targets:
            cand = stations[~stations['station'].isin(used)].copy()
            if cand.empty:
                break
            cand['dist_err'] = (cand['distance'] - target).abs()

            # Tiers, tried in order; the first non-empty one wins
            tiers = [
                cand[cand['has_shaking'] & (cand['dist_err'] <= dist_tol)],  # shaking, close to target
                cand[cand['dist_err'] <= dist_tol],                          # any station, close to target
                cand[cand['has_shaking']],                                   # shaking, but far from target
                cand,                                                        # anything left
            ]
            pool = next(t for t in tiers if not t.empty)

            best = pool.loc[pool['dist_err'].idxmin()].copy()
            best['target'] = target
            selected.append(best)
            used.add(best['station'])

        plot_stations = pd.DataFrame(selected).reset_index(drop=True)

        '''
        #Prioritize stations that have shaking recorded
        temp = pd.DataFrame(available_stations[available_stations['max_duration']!= 0])
        temp = temp.drop_duplicates(subset=['station'])
        if len(temp) >= num_streams:
            available_stations_filt = temp
        '''
    # Find stations with adequate distance seperation-------------------
        '''
        dist_targets = []
        for i in range(num_streams):
            dist_targets.append(min_dist + i*dist_sep)

        df_targets = pd.DataFrame({'target': dist_targets}).sort_values('target')

        # Stations for plotting-----------------
        plot_stations = pd.merge_asof(
            df_targets, 
            available_stations, 
            left_on='target', 
            right_on='distance', 
            direction='nearest'
        )
        plot_stations = plot_stations.drop_duplicates(subset=['station'])
        '''

        '''
        Option to have a weighted choosing of stations, based on proximity to town and distance
        towns_filt = pd.DataFrame(towns[towns['distance_deg'] < dlat + 0.5])

        #Find min distance from each station to a town
    
        min_tdist = []
        for i in range(len(available_stations)):
            sta_lat = available_stations['sta_lat'].values[i]
            sta_lon = available_stations['sta_lon'].values[i]
            temp_dist = []
            for k in range(len(towns_filt)):
                town_lat = towns_filt['lat'].values[k]
                town_lon = towns_filt['lon'].values[k]
                dist, az, baz = gps2dist_azimuth(sta_lat, sta_lon, town_lat, town_lon)
                temp_dist.append(dist/1000)

            min_tdist.append(np.min(np.array(temp_dist)))

        available_stations['min_town_dist'] = min_tdist

        num_streams = 4
        dist_sep = 50   #50     # km between target distances
        dist_tol = 50        # km: only consider stations within +/- this of a target
        town_scale = 10      # km: a town distance of this much counts as a "full" penalty
        w_dist, w_town = 0.5, 0.5   # relative priority: raise w_town to favor towns more

        dist_targets = [min_dist + i * dist_sep for i in range(num_streams)]

        selected = []
        used = set()
        for target in dist_targets:
            cand = available_stations[~available_stations['station'].isin(used)].copy()
            if cand.empty:
                break

            cand['dist_err'] = (cand['distance'] - target).abs()

            # Prefer stations inside the tolerance window; if none, fall back to the closest one
            pool = cand[cand['dist_err'] <= dist_tol]
            if pool.empty:
                pool = cand.nsmallest(1, 'dist_err')
            pool = pool.copy()

            # Lower score = better. Both terms are roughly 0-1 so the weights are comparable.
            pool['score'] = (w_dist * pool['dist_err'] / dist_tol
                            + w_town * pool['min_town_dist'].fillna(town_scale * 10) / town_scale)

            best = pool.loc[pool['score'].idxmin()].copy()
            best['target'] = target
            selected.append(best)
            used.add(best['station'])
 
        plot_stations = pd.DataFrame(selected).reset_index(drop=True)
        '''
            
        # Make plot-----------------------
        fig = plt.figure(figsize=(8, 10), dpi=150)
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.5], hspace=0.1) #0.25
        ax, region = add_basemap(fig, gs[0], eq_lon, eq_lat, dlat)
        left, right, bottom, top = region
        PC = ccrs.PlateCarree()


        #Plot faults-----------------
        if event_loc == 'Alaska': #only have faults for alaska for now...

            ax.add_geometries(gdf.geometry, crs=ccrs.PlateCarree(),
                            facecolor='none', edgecolor='saddlebrown', #dimgrey
                            alpha=0.9, linewidth=1.5, zorder=2)

        #Plot earthquake-----------------------
        ax.scatter(eq_lon, eq_lat, s = 1000, color = 'red', edgecolors='black', 
                linewidths=1.5, marker = "o",
                transform=PC, zorder=5)
        
        # Plot stations-------------------------
        sc = ax.scatter(available_stations["sta_lon"], available_stations["sta_lat"], marker="v", s=150,
                c=available_stations['max_duration'], cmap = 'Reds', vmin = 0, 
                vmax = (np.max(available_stations['max_duration'].values)),
                edgecolors="k", linewidths=1.0, 
                transform=PC, zorder=3)
        # Add colorbar for stations
        cax = ax.inset_axes([1.03, 0.0, 0.03, 1.0])
        cbar = fig.colorbar(sc, cax=cax)
        cbar.set_label("duration of MMI "+str(MMI_plot)+"+ shaking (seconds)", fontsize=11)

        '''
        #Option to add towns, seems to be too much for this one
        # Plot towns-------------------------
        for i in range(len(towns_filt)):
            town_lon = towns_filt['lon'].values[i]
            town_lat = towns_filt['lat'].values[i]
            town_name = towns_filt['name'].values[i]
            ax.scatter(town_lon, town_lat, marker="s", s=100,
                            c="gold", edgecolors="k", linewidths=1.0, 
                            transform=PC, zorder=4)
            ax.text(town_lon, town_lat - 0.1, town_name, fontsize=10, transform = PC,
                        fontname = 'Arial', va = 'top', #edgecolor = 'black',
                        ha = 'center', color = 'black', fontweight = 'bold',
                        zorder = 20)
        '''

        #Plot legend------------
        if plot_historical == True:
            handles = depth_handles_hist + mag_handles + [sta_handle]  + [fault_handle]
        elif plot_historical == False:
            if plot_aftershocks == True:
                handles = depth_handles_aftershocks + [sta_handle]  + mag_handles + [fault_handle]
            elif plot_aftershocks == False:
                handles = [sta_handle] + [fault_handle]

        leg = ax.legend(handles=handles, loc='lower right', frameon=True,
                        framealpha=1.0, 
                        prop={'family': 'arial', 'size': 7},
                        labelspacing=0.6, borderpad=0.4)
        leg.set_zorder(10)   # keep it above the relief and scatter layers

        
        # Inset locator map ----------
        # Bounds are in axes fraction: [x0, y0, width, height] of the main map
        ax_in = add_inset_map(ax, event_loc, eq_lon, eq_lat, region)

        #----------------------------------------------
        # Traces panel---------------------------------

        ax2 = fig.add_subplot(gs[1])
        index = plot_stations['og_index'].values
        durations = plot_stations['max_duration'].values
        '''
        st3 = st1.copy()
        st3 = st3.remove_response(inventory = inv2, output = 'ACC')
        median_dur = np.median(durations)

        st3 = st3.slice(starttime = UTCDateTime(eq_time), endtime=UTCDateTime(eq_time)+ first_p + 120)

        data = []
        max_vals = []
        for i in range(len(index)):
            tr = st3[index[i]]
            percent_grav = (tr.data/9.8)*100
            data.append(percent_grav)
            max_vals.append(np.max(percent_grav))
        scale = np.max(max_vals)

        stas = []
        for i in range(len(index)):
            tr = st3[index[i]]
            #start_shaking = plot_stations['start_shaking'].values[i]
            #end_shaking = plot_stations['end_shaking'].values[i]
            start_shaking = plot_stations['max_dur_start'].values[i]
            end_shaking = plot_stations['max_dur_end'].values[i]
            duration = durations[i]
            temp_data = data[i]/scale
            #ax2.plot(tr.times(), (tr.data/np.max(np.abs(tr.data)))-i*2, color = 'black', alpha = 0.8)
            #ax2.plot(tr.times(), (temp_data)-i*2, color = 'black', alpha = 0.8)
            ax2.plot(tr.times(), (temp_data)-i*2, color = 'black', alpha = 0.8)
            ax2.text(np.max(tr.times()), -i*2, tr.stats.station, va = 'bottom', 
                    ha = 'right', color = 'silver', fontweight = 'bold')
            
            ax2.text(np.max(tr.times())*0.75, -i*2+0.3, str(int(durations[i]))+" s", va = 'bottom', #/2
                    ha = 'right', color = 'red', fontweight = 'bold')
            
            if start_shaking == 0:
                print('No MMI', MMI_plot, 'for station', stations[i])
            else:
                trt = tr.slice(starttime = UTCDateTime(start_shaking), endtime=UTCDateTime(end_shaking))
                trt_data = ((trt.data/9.8)*100)/scale
                dif = abs(tr.stats.starttime - trt.stats.starttime)
                #ax2.plot(trt.times() + dif, (-2*i)+trt.data/np.max(np.abs(tr.data)), color = 'red')
                ax2.plot(trt.times() + dif, (-2*i)+trt_data, color = 'red')
                #ax2.plot(trt.times() + dif, (-2*i)+trt_data, color = 'red')
            
            stas.append(tr.stats.station)
        '''
        spacing = 2.0
        fill_frac = 0.8

        st3 = st1.copy()
        st3 = st3.remove_response(inventory=inv2, output='ACC')
        st3 = st3.slice(starttime=UTCDateTime(eq_time),
                        endtime=UTCDateTime(eq_time) + first_p + 120)

        # %g data with each trace's baseline offset removed AFTER slicing
        data, offsets = [], []
        for i in range(len(index)):
            d = (st3[index[i]].data / 9.8) * 100
            off = np.median(d)
            data.append(d - off)
            offsets.append(off)

        scale = max(np.max(np.abs(d)) for d in data)
        amp = spacing * fill_frac / scale

        label_tf = ax2.get_yaxis_transform()   # x in axes fraction, y in data units

        for i in range(len(index)):
            tr = st3[index[i]]
            y0 = -i * spacing

            start_shaking = plot_stations['max_dur_start'].values[i]
            end_shaking = plot_stations['max_dur_end'].values[i]

            ax2.plot(tr.times(), y0 + data[i] * amp, color='black', alpha=0.8, lw=0.8)

            # Labels at fixed axes-fraction x, fixed offset above the baseline
            ax2.text(0.99, y0 + 0.15 * spacing, tr.stats.station, transform=label_tf,
                    va='bottom', ha='right', color='silver', fontweight='bold')
            ax2.text(0.80, y0 + 0.15 * spacing, f"{int(durations[i])} s", transform=label_tf,
                    va='bottom', ha='right', color='red', fontweight='bold')

            if pd.isnull(start_shaking) or start_shaking == 0:
                print('No MMI', MMI_plot, 'for station', tr.stats.station)
            else:
                trt = tr.slice(starttime=UTCDateTime(str(start_shaking)),
                            endtime=UTCDateTime(str(end_shaking)))
                dif = abs(tr.stats.starttime - trt.stats.starttime)
                trt_data = ((trt.data / 9.8) * 100 - offsets[i]) * amp   # same offset as black trace
                ax2.plot(trt.times() + dif, y0 + trt_data, color='red', lw=0.8)

        n = len(index)
        ax2.set_ylim(-(n - 1) * spacing - spacing / 2, (spacing / 2)+0.2)
        ax2.set_yticks([-i * spacing for i in range(n)])

        ax2.set_yticklabels([])                 # station names are already printed on the traces
        ax2.grid(alpha = 0.3)
        ax2.axvline(0, color = 'red', linestyle = '--')
        ax2.scatter(0,0.75,marker = 'o', color = 'red', edgecolor = 'black', s = 300, zorder = 20)
        ax2.set_xlabel("time since earthquake origin (s)")
        #ax2.set_yticks([])
        ax2.set_ylabel('distance from earthquake', labelpad=15)

        ax2.annotate(
            "",
            xy=(-0.02, -2*(len(plot_stations)-1)),
            xytext=(-0.02, 0),
            xycoords=ax2.get_yaxis_transform(),   # x in axes fraction, y in data
            textcoords=ax2.get_yaxis_transform(),
            annotation_clip=False,
            arrowprops=dict(arrowstyle="-|>", color="red", lw=2, mutation_scale=20),
        )


        #Add stations to original map----------
        for i in range(len(plot_stations)):
            #temp = stations_df.loc[stations_df['station'] == stas[i]].head(1)
            sta_lon = plot_stations['sta_lon'].values[i]
            sta_lat = plot_stations['sta_lat'].values[i]
            name = plot_stations['station'].values[i]
            # Line from epicenter to station
            
            ax.plot([eq_lon, sta_lon], [eq_lat, sta_lat],
                    color='gray', linewidth=1.2, alpha=0.8,
                    transform=ccrs.Geodetic(), zorder=3.5)
            
            ax.text(sta_lon, sta_lat+0.05, name, fontsize=9, transform = PC,
                    fontname = 'Arial', va = 'bottom', #edgecolor = 'black',
                    ha = 'center', color = 'firebrick', fontweight = 'bold',
                    zorder = 20)
            #ax.scatter(sta_lon, sta_lat, marker="v", s=200,
                    #c="firebrick", edgecolors="k", linewidths=1.0, 
                    #transform=PC, zorder=4)

        #Plot day/time and magnitude-----------------
        t = UTCDateTime(eq_time)
        date_str = t.strftime('%Y %b. %d')     # 2026 Oct. 15

        ax.text(0.02, 0.98, date_str + ' M' + str(eq_mag), transform=ax.transAxes,
                fontsize=17, va='top', ha='left', fontweight='bold', fontname='Arial',
                bbox=dict(facecolor='white', alpha=1.0, edgecolor='black'), zorder=20)

        #Plot MMI
        ax.text(0.02, 0.90, 'Duration of MMI '+str(MMI_plot)+'+ shaking', transform=ax.transAxes,
                fontsize=17, va='top', ha='left', fontweight = 'bold', style = 'italic',
                fontname = 'Arial', color = 'red', zorder = 20)

        # Draw plot together correctly-----------------------------
        fig.canvas.draw()                     # forces cartopy to apply its aspect ratio
        pos_map = ax.get_position()
        pos_bot = ax2.get_position()
        gap = 0.0   # figure-fraction gap between panels; try 0.01-0.03 if you want some room
        new_y0 = pos_map.y0 - gap - pos_bot.height
        ax2.set_position([pos_map.x0, new_y0, pos_map.width, pos_bot.height])

        #Save figue-------
        fig.savefig(dir_path+'/accelerations_figure.png', dpi=720, bbox_inches='tight')

        plt.show()

#________________________________________________________________________________
#________________________________________________________________________________
#________________________________________________________________________________
#%%
    if plot_mmi == True:
        print('Making MMI plot...')
        dlat = 2.0
        plot_aftershocks = False
        plot_historical = False
        if aftershock_df is None:
            plot_aftershocks = False
        if hist_df is None:
            plot_historical = False

        ############################################
        mmi_labels = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X']
        mmi_colors = [
            '#FFFFFF',  # I
            '#ECEDF3',  # II
            '#BEC3DC',  # III
            '#A0D7E4',  # IV
            '#8FCB90',  # V
            '#F8E92E',  # VI
            '#F5C022',  # VII
            '#E98A2A',  # VIII
            '#E05529',  # IX
            '#D9252B',  # X
        ]

        mmi_cmap = ListedColormap(mmi_colors, name='MMI')

        # Bin edges at half-integers so each integer value falls in the middle of its bin
        bounds = np.arange(0.5, 10.6, 1)                 # 0.5, 1.5, ..., 10.5
        mmi_norm = BoundaryNorm(bounds, mmi_cmap.N)

        #Make figure
        fig = plt.figure(figsize=(8, 10), dpi=150)
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.5], hspace=0.1) #0.25
        ax, region = add_basemap(fig, gs[0], eq_lon, eq_lat, dlat)
        left, right, bottom, top = region
        PC = ccrs.PlateCarree()
    

        #Plot faults-----------------

        if event_loc == 'Alaska': #only have faults for alaska for now...

            ax.add_geometries(gdf.geometry, crs=ccrs.PlateCarree(),
                            facecolor='none', edgecolor='saddlebrown', #dimgrey
                            alpha=0.9, linewidth=1.5, zorder=2)

        #Plot earthquake-----------------------
        ax.scatter(eq_lon, eq_lat, s = 1000, color = 'red', edgecolors='black', 
                linewidths=1.5, marker = "o",
                transform=PC, zorder=5)

        # Plot stations-------------------------

        sc = ax.scatter(available_stations_filt["sta_lon"], available_stations_filt["sta_lat"], marker="v", s=100,
                        c=available_stations_filt["max_mmi_all"], cmap=mmi_cmap, norm=mmi_norm,
                        edgecolors="k", linewidths=1.0, transform=PC, zorder=3)

        cax = ax.inset_axes([1.03, 0.0, 0.03, 1.0])
        cbar = fig.colorbar(sc, cax=cax, ticks=np.arange(1, 11))
        cbar.ax.set_yticklabels(mmi_labels)
        cbar.set_label('Modified Mercalli Intensity')

        #Plot legend------------
        if plot_historical == True:
            handles = depth_handles_hist + mag_handles + [sta_handle]  + [fault_handle]
        elif plot_historical == False:
            if plot_aftershocks == True:
                handles = depth_handles_aftershocks + [sta_handle]  + mag_handles + [fault_handle]
            elif plot_aftershocks == False:
                handles = [sta_handle] + [fault_handle]

        leg = ax.legend(handles=handles, loc='lower right', frameon=True,
                        framealpha=1.0, 
                        prop={'family': 'arial', 'size': 7},
                        labelspacing=0.6, borderpad=0.4)
        leg.set_zorder(10)   # keep it above the relief and scatter layers
        

        # ---------- Inset locator map ----------
        # Bounds are in axes fraction: [x0, y0, width, height] of the main map
        ax_in = add_inset_map(ax, event_loc, eq_lon, eq_lat, region)


        # Plot Date/Magnitude text----------------------

        t = UTCDateTime(eq_time)
        date_str = t.strftime('%Y %b. %d')     # 2026 Oct. 15

        ax.text(0.02, 0.98, date_str + ' M' + str(eq_mag), transform=ax.transAxes,
                fontsize=17, va='top', ha='left', fontweight='bold', fontname='Arial',
                bbox=dict(facecolor='white', alpha=1.0, edgecolor='black'), zorder=20)

        #Plot MMI
        ax.text(0.02, 0.90, 'Max MMI recorded', transform=ax.transAxes,
                fontsize=17, va='top', ha='left', fontweight = 'bold', style = 'italic',
                fontname = 'Arial', color = 'red', zorder = 20)
        
        fig.savefig(dir_path+'/mmi_figure.png', dpi=720, bbox_inches='tight')

        plt.show()
