#import yaml
import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.geodetics import gps2dist_azimuth
from obspy.geodetics import kilometers2degrees
from obspy import read_events
from obspy.taup import TauPyModel
from obspy import Stream
from obspy.clients.fdsn.header import FDSNNoDataException
from obspy import Inventory

import matplotlib.ticker as mticker
from matplotlib.colors import LightSource, to_rgba
from matplotlib.patches import Rectangle
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.geodesic import Geodesic
import pygmt  # only used to grab the relief grid; any xarray grid (ETOPO, GEBCO) works
import tempfile
from matplotlib.lines import Line2D
import geopandas as gpd
import matplotlib.image as mpimg
from matplotlib.colors import ListedColormap, BoundaryNorm
from pyproj import Geod, Proj
import matplotlib.patheffects as pe

from incident_functions import (pull_earthquakes, pull_stations, pull_streams,
                                make_region, exceedance_episodes, check_mmi, 
                                calculate_shaking, make_profile)

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
    print(f"Event ID for analysis: {event_id}")
else:
    print("No input provided, need event ID to run")

#event_id =  'ak018fcnsk91' #(2018 Anchorage), 'ak01613v15nv' (2016 Iniskin) #'us6000tz1y' (Australia) #'us7000ti1p' (M6.5 Rat islands) # 'aka2026tuxgky' (5.5)  us6000tza4 (Rat islands),  us6000u18k (Panama)

#What plots to make
plot_map = True
plot_cross_section = True
plot_mmi = True
plot_accelerations = True

#Inputs for reading in event----------------
event_id  = str(event_id)
lat = '61.218056' #Anchorage latitude
lon = '-149.900278' #Anchorage longitude
max_rad = '4000' #circle that covers all of Alaska,19000 for globe
min_mag = '6.0' #min mag to search over, 4
event_loc = 'Alaska' #Alaska, Global
start = str(UTCDateTime.now() - (60*60*24*365)*20) #pull events from last two years
end = str(UTCDateTime.now())

#Inputs for stations
max_rad_stations = 600 #600 #km, radius of circle to pull stations from
subset_dist = 200 #200
    
#Inputs for accelrations plot--------
MMI_plot = 4

#Historical inputs---
max_rad_hist = '600'
min_mag_hist = '3.0' #3.0

#Aftershock inputs----
max_rad_aftershocks = '100'
min_mag_aftershocks = '2.0'

if __name__ == "__main__":

    '''
    #############################################################
    #----------STEP 1: LOAD INPUTS FROM PARAMETER FILE-----------
    #############################################################
    with open(sys.argv[1]) as f:
        params = yaml.safe_load(f)
    
    (event_id, plot_map, plot_mmi, plot_accelerations, plot_cross_section) = read_params(params)
    '''

    #############################################################
    #----------STEP 3: PULL DATA-----------------------------------
    #############################################################
   
    
    eq_df = pull_earthquakes(lat, lon, max_rad, min_mag, start, end)

    #Pull out event of interest -----------------------------
    temp = eq_df.loc[eq_df['event_id'] == event_id].head(1)
    if len(temp) == 1:
        print('Event found!')

    
    #Create directory-------------------
    dir_path = "./"+event_id
    os.makedirs(dir_path, exist_ok=True)

    #Define metrics for the earthquake ----------------------
    eq_lat = temp['latitude'].to_numpy()[0]
    eq_lon = temp['longitude'].to_numpy()[0]
    eq_depth = temp['depth'].to_numpy()[0]
    eq_mag = temp['magnitude'].to_numpy()[0]
    eq_time = str(temp['time_utc'].to_numpy()[0])

    #Pull all stations in region -----------------------------
    stations_df, inv = pull_stations(eq_lat, eq_lon, eq_time, eq_depth, max_rad_stations)

    #Create subset of stations based on closest station
    min_dist = stations_df['distance'].min()
    first_p = stations_df['p_arrival'].min()

    closest_stations = min_dist + subset_dist #200
    stations_subset = pd.DataFrame(stations_df[stations_df['distance']<= closest_stations])


    stations_subset = stations_subset.sort_values(by="distance", ascending=True)
    networks = stations_subset['network'].to_numpy()
    stations = stations_subset['station'].to_numpy()
    channels = stations_subset['channel'].to_numpy()


    #See what stations are available -----------------------------
    st, inv2 = pull_streams(networks, stations, channels, eq_time)
    st1 = st.copy() # save version for later analysis

    st.remove_sensitivity(inventory = inv2)  
    st.taper(max_percentage=0.05)
    #st.detrend('demean')
    st.filter("bandpass", freqmin=1, freqmax=18, 
                corners=2, zerophase=True)

    st = st.slice(UTCDateTime(eq_time), UTCDateTime(eq_time)+ first_p +60) #90

    #Pull out stations/channels from actual stations available
    stas = []
    chans = []
    nets = []
    index = []
    for i in range(len(st)):
        stas.append(st[i].stats.station)
        chans.append(st[i].stats.channel)
        nets.append(st[i].stats.network)
        index.append(i)

    data = {
            'station': stas,
            'channel': chans,
            'network': nets,
            'og_index': index,
            }

    df = pd.DataFrame(data)
    available_stations = pd.merge(df, stations_subset, on=['station', 'channel', 'network'])
    available_stations = available_stations.drop_duplicates().reset_index()
    available_stations = available_stations.sort_values(by = 'distance')


    #Pull historical earthquakes-----------------------------
    print('Pulling historical earthquakes for plotting...')
    start = '1990-01-01'
    end = str(UTCDateTime(eq_time)-(60*60*24))
    hist_df = pull_earthquakes(str(eq_lat), str(eq_lon), max_rad_hist, min_mag_hist, start, end)

    #Pull aftershocks --------------------------
    print('Pulling aftershocks for plotting...')
    start = str(UTCDateTime(eq_time)+1*60) #one minute after
    end = str(UTCDateTime(eq_time)+86400*10) #10 days after
    aftershock_df = pull_earthquakes(str(eq_lat), str(eq_lon), max_rad_aftershocks, min_mag_aftershocks, start, end)

    #Calculate shaking-------------------------
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

    #Drop stations that have exceedingly long/or short shaking durations
    available_stations_filt = pd.DataFrame(available_stations[available_stations['duration'] <= 3*np.median(durations)])


    #Read in logo---------------------------------
    print('Reading in logo file')
    logo = mpimg.imread('./logos/EC_black_with_rectangle.png')

    #Read in fault file----------------------------
    print('Reading in Alaska fault files')
    gdf = gpd.read_file('./fault_files/AEC_Faults_Simplified/Faults_simple.shp')
    #print(len(gdf), gdf.geometry.is_empty.sum(), gdf.geometry.isna().sum())   # how many bad rows?
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    gdf = gdf.to_crs('EPSG:4326')     # in case the file isn't in lon/lat (check gdf.crs)


    #Create handles for legend plotting------------------------

    # Depth handles historical
    depth_handles_hist = [
        Line2D([], [], marker='o', linestyle='', markersize=9,
            markerfacecolor='whitesmoke', markeredgecolor='k', label='< 35 km'),
        Line2D([], [], marker='o', linestyle='', markersize=9,
            markerfacecolor='darkgray', markeredgecolor='k',label='35-100 km'),
        Line2D([], [], marker='o', linestyle='', markersize=9,
            markerfacecolor="dimgrey", markeredgecolor='k', label='> 100 km'),
    ]

    # Depth handles aftershocks
    depth_handles_aftershocks = [
        Line2D([], [], marker='o', linestyle='', markersize=9,
                    markerfacecolor='firebrick', markeredgecolor='k', label='< 35 km'),
        Line2D([], [], marker='o', linestyle='', markersize=9,
                    markerfacecolor='gold', markeredgecolor='k',label='35-100 km'),
        Line2D([], [], marker='o', linestyle='', markersize=9,
                    markerfacecolor='#4a0010', markeredgecolor='k', label='> 100 km'),
            ]

    # Magnitude (size) entries: markersize is in points, scatter's s is points^2
    mag_handles = [
        Line2D([], [], marker='o', linestyle='', markerfacecolor='lightgray',
            markeredgecolor='k', markersize=np.sqrt(3 * 2**m), label=f'M{m}')
        for m in (4, 6, 7)
    ]

    #Fault handle
    fault_handle = Line2D([], [], linestyle='-', linewidth=2,
                    color='saddlebrown', label='faults')
    
    # Station handles
    sta_handle = Line2D([], [], marker='v', linestyle='', markersize=12,
                        markerfacecolor='gray', markeredgecolor='k',
                        markeredgewidth=1.5, label='seismic stations')

    sta_handle2 = Line2D([], [], marker='v', linestyle='', markersize=12,
                        markerfacecolor='firebrick', markeredgecolor='k',
                        markeredgewidth=1.5, label='plotted stations')



    if plot_map == True:
        #--------------------------------------------------
        #----------MAP FIGURE------------------------------

        print('Starting map figure...')

        # Find stations with adequate distance seperation for trace plots-------------------
        plot_aftershocks = False
        plot_historical = False

        dlat = 2.5
        num_streams = 5
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

    
        fig = plt.figure(figsize=(8, 10), dpi=150)
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.5], hspace=0.1) #0.25

        left, right, bottom, top = region = make_region(eq_lon, eq_lat, dlat=dlat)

        proj = ccrs.Mercator(central_longitude=eq_lon)      # was ccrs.Mercator()
        ax = fig.add_subplot(gs[0], projection=proj)

        PC = ccrs.PlateCarree()


        ax.set_extent(region, crs=ccrs.PlateCarree())


        # ---------- Shaded relief background ----------
        #grid = pygmt.datasets.load_earth_relief(resolution="01m", region=region).sortby("lat")
        grid = pygmt.datasets.load_earth_relief(resolution= '15s',#'30s',#'10m', 
                                                            region=region, 
                                                            registration=None, 
                                                            data_source='igpp', 
                                                            use_srtm=False).sortby("lat") 
        elev = grid.values
        lon, lat = grid.lon.values, grid.lat.values


        # hillshade
        ls = LightSource(azdeg=315, altdeg=45)
        cell = 463
        dx = cell * np.cos(np.deg2rad(lat.mean()))
        #dx = cell * np.cos(np.deg2rad(60))
        shade = ls.hillshade(elev, vert_exag=10, #3, 20
                            dx=dx, dy=cell)

        base = np.ones(elev.shape + (3,))
        base[elev < 0] = [0.80, 0.83, 0.87]
        rgb = base * (0.55 + 0.45 * shade)[..., None]

        
        img_crs = ccrs.PlateCarree(central_longitude=eq_lon)
        ax.imshow(rgb, origin="lower",
                extent=[lon.min() - eq_lon, lon.max() - eq_lon, lat.min(), lat.max()],
                transform=img_crs, zorder=0, regrid_shape=2000, interpolation='nearest')

        # ---------- Coastlines and borders ----------
        ax.coastlines(resolution="10m", linewidth=0.6, zorder=1)
        ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=1.5, zorder=1)

        if plot_historical == True:
            #Plot historical earthquakes---------------------
            depth = hist_df["depth"].values
            mag = hist_df["magnitude"].values

            depth_colors = np.where(depth < 35, 'whitesmoke',#"firebrick",
                            np.where(depth < 100, "darkgray","dimgrey"))#"gold", "#4a0010"))
            alphas = np.where(mag >= 4.5, 0.8, 0.25)   # 0.25fade small events; adjust to taste
            face = [to_rgba(c, a) for c, a in zip(depth_colors, alphas)]
            edge = [to_rgba("k", a) for a in alphas]

            ax.scatter(hist_df["longitude"], hist_df["latitude"], s=3 * 2**mag,
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

            ax.scatter(aftershock_df["longitude"], aftershock_df["latitude"], s=3 * 2**mag,
                    c=face, edgecolors=edge, linewidths=0.8,
                    transform=PC, zorder=6)
            

        #Plot faults-----------------

        if event_loc == 'Alaska': #only have faults for alaska for now...

            ax.add_geometries(gdf.geometry, crs=ccrs.PlateCarree(),
                            facecolor='none', edgecolor='saddlebrown', #dimgrey
                            alpha=0.8, linewidth=1.5, zorder=2)


        #Plot earthquake-----------------------
        ax.scatter(eq_lon, eq_lat, s = 1000, color = 'red', edgecolors='black', 
                linewidths=1.5, marker = "o",
                transform=PC, zorder=5)

        # Plot stations-------------------------

        ax.scatter(stations_df["sta_lon"], stations_df["sta_lat"], marker="v", s=70,
                c="gray", edgecolors="k", linewidths=1.0, 
                transform=PC, zorder=3)


        if plot_historical == True:
            handles = depth_handles_hist + mag_handles + [sta_handle] + [sta_handle2] + [fault_handle]
        elif plot_historical == False:
            if plot_aftershocks == True:
                handles = depth_handles_aftershocks+ [sta_handle] + [sta_handle2] + mag_handles + [fault_handle]
            elif plot_aftershocks == False:
                handles = [sta_handle] + [sta_handle2] + [fault_handle]

        leg = ax.legend(handles=handles, loc='lower right', frameon=True,
                        framealpha=1.0, 
                        prop={'family': 'arial', 'size': 7},
                        labelspacing=0.6, borderpad=0.4)
        leg.set_zorder(10)   # keep it above the relief and scatter layers

        # ---------- Lat/lon labels on left and bottom only ----------
        gl = ax.gridlines(draw_labels=True, crs=PC, linewidth=0)   # linewidth=0 -> no grid lines
        gl.top_labels = True
        gl.bottom_labels = False
        gl.right_labels = False

        # ---------- Inset locator map ----------
        # Bounds are in axes fraction: [x0, y0, width, height] of the main map

        if event_loc == 'Global':
            ax_in = ax.inset_axes([0.75, 0.8, 0.25, 0.30],
                            projection=ccrs.Orthographic(central_longitude=eq_lon, #eq_lon
                                                        central_latitude=eq_lat)) #eq_lat
            ax_in.set_global()
            ax_in.add_feature(cfeature.OCEAN, facecolor='lightsteelblue')
            ax_in.add_feature(cfeature.LAND, facecolor='0.9')
            ax_in.coastlines(resolution='110m', linewidth=0.5)
            ax_in.gridlines(linewidth=0.3, color='gray', alpha=0.6)

        if event_loc == 'Alaska':
            anc_lat = 61.218056 #Anchorage latitude
            anc_lon = -149.900278 #Anchorage longitude
            ax_in = ax.inset_axes([0.75, 0.8, 0.25, 0.30],
                                projection=ccrs.Orthographic(central_longitude=anc_lon, #eq_lon
                                                            central_latitude=anc_lat)) #eq_lat
            #ax_in = ax.inset_axes([0.90, 0.7, 0.30, 0.36], projection=ccrs.Geodetic())
            ax_in.set_extent([-180, -140, 50, 70], crs=ccrs.PlateCarree())
            ax_in.add_feature(cfeature.LAND, facecolor='0.9')
            ax_in.add_feature(cfeature.OCEAN, facecolor='lightsteelblue')
            ax_in.coastlines(resolution='50m', linewidth=0.5)
            ax_in.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.6,
                        edgecolor='k', zorder=2)

            # Box showing the extent of the main map
        box_lons = [left, right, right, left, left]
        box_lats = [bottom, bottom, top, top, bottom]
        ax_in.plot(box_lons, box_lats, color='red', linewidth=1.5,
                    transform=ccrs.Geodetic())

        ax_in.plot(box_lons, box_lats, color='red', linewidth=1.5, transform=ccrs.Geodetic())
        ax_in.set_zorder(10)   # keep it above the relief and scatter layers


        #----------------------------------------------
        # Traces panel---------------------------------
        #----------------------------------------------

        ax2 = fig.add_subplot(gs[1])
        index = plot_stations['og_index'].values

        stas = []
        for i in range(len(index)):
            tr = st[index[i]]
            ax2.plot(tr.times(), (tr.data/np.max(np.abs(tr.data)))-i*2, color = 'black', alpha = 0.8)
            ax2.text(np.max(tr.times()), -i*2, tr.stats.station, va = 'bottom', 
                    ha = 'right', color = 'red', fontweight = 'bold')
            
            stas.append(tr.stats.station)

        ax2.grid(alpha = 0.3)
        ax2.axvline(0, color = 'red', linestyle = '--')
        ax2.scatter(0,0.75,marker = '*', color = 'red', edgecolor = 'black', s = 300, zorder = 20)
        ax2.set_xlabel("time since earthquake origin (s)")
        ax2.set_yticks([])
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
            # Line from epicenter to station
            ax.plot([eq_lon, sta_lon], [eq_lat, sta_lat],
                    color='firebrick', linewidth=1.2, alpha=0.8,
                    transform=ccrs.Geodetic(), zorder=3.5)
            ax.scatter(sta_lon, sta_lat, marker="v", s=200,
                    c="firebrick", edgecolors="k", linewidths=1.0, 
                    transform=PC, zorder=4)
            
        #Add AEC logo-------------------

        # Preserve the logo's aspect ratio
        h, w = logo.shape[:2]
        logo_width = 0.25                                  # fraction of the map's width
        fig_w, fig_h = fig.get_size_inches()
        map_pos = ax.get_position()                        # after your canvas.draw() step
        map_aspect = (map_pos.height * fig_h) / (map_pos.width * fig_w)
        logo_height = logo_width * (h / w) / map_aspect    # height in axes fraction

        ax_logo = ax.inset_axes([0.0, 0.0, logo_width, logo_height])
        ax_logo.imshow(logo)
        ax_logo.axis('off')
        ax_logo.set_zorder(10)

        # Plot Date/Magnitude text----------------------
        ax.text(0.02, 0.98, eq_time[0:10]+' M'+str(eq_mag), transform=ax.transAxes,
                fontsize=17, va='top', ha='left', fontweight = 'bold', fontname = 'Arial',
                bbox=dict(facecolor='white', alpha = 1.0, edgecolor='black'),zorder = 20)

        # Draw plot together correctly-----------------------------

        fig.canvas.draw()                     # forces cartopy to apply its aspect ratio
        pos_map = ax.get_position()
        pos_bot = ax2.get_position()
        gap = 0.0   # figure-fraction gap between panels; try 0.01-0.03 if you want some room
        new_y0 = pos_map.y0 - gap - pos_bot.height
        ax2.set_position([pos_map.x0, new_y0, pos_map.width, pos_bot.height])

        #Save figure
        fig.savefig(dir_path+'/map_figure.png', dpi=720)

        plt.show()
#________________________________________________________________________________
#________________________________________________________________________________
#________________________________________________________________________________

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

        #Find slab parameters in vicinity of earthquake
        slab2_loc = pd.DataFrame(df3[df3['lat'] <= lat_round + 0.2])
        slab2_loc = pd.DataFrame(slab2_loc[slab2_loc['lat'] >= lat_round - 0.2])
        slab2_loc = pd.DataFrame(slab2_loc[slab2_loc['lon'] <= lon_round + 0.2])
        slab2_loc = pd.DataFrame(slab2_loc[slab2_loc['lon'] >= lon_round - 0.2])

        if len(slab2_loc) > 0: #the earthquake occurred in the lat/lon grid of the slab
            mean_strike = np.mean(slab2_loc['strike'].values)

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
        proj = ccrs.Mercator(central_longitude=eq_lon)      # was ccrs.Mercator()
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.5], hspace=0.1)
        ax = fig.add_subplot(gs[0], projection=proj)   # your existing map axes
        ax_xs = fig.add_subplot(gs[1])                      # cross-section

        left, right, bottom, top = region = make_region(eq_lon, eq_lat, dlat=dlat)


        #ax = fig.add_subplot(gs[0], projection=proj)

        PC = ccrs.PlateCarree()


        ax.set_extent(region, crs=ccrs.PlateCarree())
        #ax.set_extent(region, crs = proj)


        # ---------- Shaded relief background ----------
        #grid = pygmt.datasets.load_earth_relief(resolution="01m", region=region).sortby("lat")
        grid = pygmt.datasets.load_earth_relief(resolution= '15s',#'30s',#'10m', 
                                                            region=region, 
                                                            registration=None, 
                                                            data_source='igpp', 
                                                            use_srtm=False).sortby("lat") 
        elev = grid.values
        lon, lat = grid.lon.values, grid.lat.values


        # hillshade
        ls = LightSource(azdeg=315, altdeg=45)
        cell = 463
        dx = cell * np.cos(np.deg2rad(lat.mean()))
        #dx = cell * np.cos(np.deg2rad(60))
        shade = ls.hillshade(elev, vert_exag=10, #3, 20
                            dx=dx, dy=cell)

        base = np.ones(elev.shape + (3,))
        base[elev < 0] = [0.80, 0.83, 0.87]
        rgb = base * (0.55 + 0.45 * shade)[..., None]

        #ax.imshow(rgb, origin="lower", extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                #transform=PC, zorder=0, regrid_shape=2000,interpolation='nearest')

        img_crs = ccrs.PlateCarree(central_longitude=eq_lon)
        ax.imshow(rgb, origin="lower",
                extent=[lon.min() - eq_lon, lon.max() - eq_lon, lat.min(), lat.max()],
                transform=img_crs, zorder=0, regrid_shape=2000, interpolation='nearest')

        # ---------- Coastlines and borders ----------
        ax.coastlines(resolution="10m", linewidth=0.6, zorder=1)
        ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=1.5, zorder=1)

        if plot_historical == True:
            #Plot historical earthquakes---------------------
            depth = hist_df["depth"].values
            mag = hist_df["magnitude"].values

            depth_colors = np.where(depth < 35, 'whitesmoke',#"firebrick", lightgrey
                            np.where(depth < 100, "darkgray","dimgrey"))#"gold", "#4a0010")), darkgrey, dimgrey
            alphas = np.where(mag >= 4.5, 1, 1)   # 0.25fade small events; adjust to taste
            face = [to_rgba(c, a) for c, a in zip(depth_colors, alphas)]
            edge = [to_rgba("k", a) for a in alphas]

            ax.scatter(hist_df["longitude"], hist_df["latitude"], s=1.5 * 2**mag, #3
                    c=face, edgecolors=edge, linewidths=0.8,
                    transform=PC, zorder=3)
            
            # Depth (color) entries
            depth_handles = [
                Line2D([], [], marker='o', linestyle='', markersize=9,
                    markerfacecolor='whitesmoke', markeredgecolor='k', label='< 35 km'),
                Line2D([], [], marker='o', linestyle='', markersize=9,
                    markerfacecolor='darkgray', markeredgecolor='k',label='35-100 km'),
                Line2D([], [], marker='o', linestyle='', markersize=9,
                    markerfacecolor="dimgrey", markeredgecolor='k', label='> 100 km'),
            ]

            # Magnitude (size) entries: markersize is in points, scatter's s is points^2
            mag_handles = [
                Line2D([], [], marker='o', linestyle='', markerfacecolor='lightgray',
                    markeredgecolor='k', markersize=np.sqrt(1.5 * 2**m), label=f'M{m}')
                for m in (7,6,4)
            ]

        if plot_aftershocks == True:
            #Plot historical earthquakes---------------------
            depth = aftershock_df["depth"].values
            mag = aftershock_df["magnitude"].values

            depth_colors = np.where(depth < 35, "firebrick",
                            np.where(depth < 100, "gold", "#4a0010"))
            alphas = np.where(mag >= 4.5, 1.0, 1.0)   # fade small events; adjust to taste
            face = [to_rgba(c, a) for c, a in zip(depth_colors, alphas)]
            edge = [to_rgba("k", a) for a in alphas]

            ax.scatter(aftershock_df["longitude"], aftershock_df["latitude"], s=3 * 2**mag,
                    c=face, edgecolors=edge, linewidths=0.8,
                    transform=PC, zorder=6)
            

        #Plot faults-----------------

        if event_loc == 'Alaska': #only have faults for alaska for now...

            ax.add_geometries(gdf.geometry, crs=ccrs.PlateCarree(),
                            facecolor='none', edgecolor='saddlebrown', #dimgrey
                            alpha=0.8, linewidth=1.5, zorder=2)

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

        # ---------- Lat/lon labels on left and bottom only ----------
        gl = ax.gridlines(draw_labels=True, crs=PC, linewidth=0)   # linewidth=0 -> no grid lines
        gl.top_labels = True
        gl.bottom_labels = False
        gl.right_labels = False

        # ---------- Inset locator map ----------
        # Bounds are in axes fraction: [x0, y0, width, height] of the main map

        if event_loc == 'Global':
            ax_in = ax.inset_axes([0.75, 0.8, 0.25, 0.30],
                            projection=ccrs.Orthographic(central_longitude=eq_lon, #eq_lon
                                                        central_latitude=eq_lat)) #eq_lat
            ax_in.set_global()
            ax_in.add_feature(cfeature.OCEAN, facecolor='lightsteelblue')
            ax_in.add_feature(cfeature.LAND, facecolor='0.9')
            ax_in.coastlines(resolution='110m', linewidth=0.5)
            ax_in.gridlines(linewidth=0.3, color='gray', alpha=0.6)


        if event_loc == 'Alaska':
            anc_lat = 61.218056 #Anchorage latitude
            anc_lon = -149.900278 #Anchorage longitude
            ax_in = ax.inset_axes([0.75, 0.8, 0.25, 0.30],
                                projection=ccrs.Orthographic(central_longitude=anc_lon, #eq_lon
                                                            central_latitude=anc_lat)) #eq_lat
            #ax_in = ax.inset_axes([0.90, 0.7, 0.30, 0.36], projection=ccrs.Geodetic())
            ax_in.set_extent([-180, -140, 50, 70], crs=ccrs.PlateCarree())
            ax_in.add_feature(cfeature.LAND, facecolor='0.9')
            ax_in.add_feature(cfeature.OCEAN, facecolor='lightsteelblue')
            ax_in.coastlines(resolution='50m', linewidth=0.5)
            ax_in.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.6,
                        edgecolor='k', zorder=2)

            # Box showing the extent of the main map
        box_lons = [left, right, right, left, left]
        box_lats = [bottom, bottom, top, top, bottom]
        ax_in.plot(box_lons, box_lats, color='red', linewidth=1.5,
                    transform=ccrs.Geodetic())

        ax_in.plot(box_lons, box_lats, color='red', linewidth=1.5, transform=ccrs.Geodetic())

            # Or mark the epicenter instead of (or in addition to) the box
        #ax_in.scatter(eq_lon, eq_lat, marker='*', s=150, color='red',
                        #edgecolors='k', linewidths=0.8, transform=PC)

        ax_in.set_zorder(10)   # keep it above the relief and scatter layers



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

        '''
        x1, y1 = centers, lo
        x2, y2 = centers + 40, lo + 35

        ax_xs.fill(np.r_[x1, x2[::-1]], np.r_[y1, y2[::-1]],
                facecolor='tab:blue', alpha=0.3, edgecolor='none',
                zorder=2, label='Pacific Plate')
        
        #Plot boundaries of slab--------------------
        ax_xs.plot(x1, y1, color='k', linewidth=2, zorder=10)
        ax_xs.plot(x2, y2, color='k', linewidth=2, zorder=10)
        '''

        #Plot North American plate---------------------
        x, y = centers, lo          # upper slab surface (swap in `med` if you prefer)
        z0 = 35                     # depth of the horizontal line (km)

        # Find where the slab crosses z0 (sign change between neighboring points)
        s = y - z0
        i = np.where(s[:-1] * s[1:] < 0)[0]
        x_cross = x[i] + (z0 - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i])   # linear interpolation

        # Insert the crossing points into the line
        xs = np.insert(x, i + 1, x_cross)
        ys = np.insert(y, i + 1, z0)

        # Boundary: slab where shallower than z0, flat at z0 where deeper
        boundary = np.minimum(ys, z0)

        xs_f = np.r_[0, xs, L]
        b_f = np.r_[boundary[0], boundary, boundary[-1]]
        ax_xs.fill_between(xs_f, 0, b_f, color='tab:orange', alpha=0.3, edgecolor='none', zorder=2, label = 'North American Plate')

        # Horizontal portion only
        ax_xs.plot(xs_f, np.where(b_f >= z0, z0, np.nan),
                color='k', linewidth=2, linestyle='-', zorder=10)

        #---------------------------------------------
        # Add arrow and text--------------------------

        # Bottom edge of the blue plate
        xb, yb = centers + 40, lo + 35

        # Where along the profile the arrow starts and ends (km); adjust to taste
        xa0, xa1 = 200, 300
        ya0, ya1 = np.interp([xa0, xa1], xb, yb)

        offset = 15   # km below the plate (larger = further below on the inverted y-axis)

        # Make the tip point down-dip, i.e. toward the deeper end
        if ya1 < ya0:
            xa0, xa1, ya0, ya1 = xa1, xa0, ya1, ya0

        ax_xs.annotate("",
                    xy=(xa1, ya1 + offset),        # arrow tip
                    xytext=(xa0, ya0 + offset),    # arrow tail
                    arrowprops=dict(arrowstyle="-|>", color= "firebrick",
                                    lw=3, mutation_scale=25),
                    annotation_clip=False, zorder=10)

        ax_xs.text((xa0), (ya0 + ya1) / 2 + offset + 5, 'plate motion',
                    color='firebrick', ha='center', va='top', fontsize = 12)
        

        x_end = L          # right edge of the cross-section (profile length)

        def extend_right(x, y, x_end):
            """Continue a line horizontally at its last depth out to x_end."""
            if x[-1] < x_end:
                x = np.r_[x, x_end]
                y = np.r_[y, y[-1]]
            return x, y

        # Top and bottom surfaces of the Pacific plate, extended flat to the right
        x1, y1 = extend_right(centers, lo, x_end)
        x2, y2 = extend_right(centers + 40, lo + 35, x_end)

        ax_xs.fill(np.r_[x1, x2[::-1]], np.r_[y1, y2[::-1]],
                facecolor='tab:blue', alpha=0.3, edgecolor='none',
                zorder=2, label='Pacific Plate')

        ax_xs.plot(x1, y1, color='k', linewidth=2, zorder=10)
        ax_xs.plot(x2, y2, color='k', linewidth=2, zorder=10)

        ax_xs.set_xlim(0, x_end)    # replaces the hardcoded 490


        #--------------------------------------------
        # Set x and y limits------------------------

        ax_xs.set_ylim(np.max(depth[mask]), 0)
        #ax_xs.set_xlim(0,490)
        ax_xs.legend(loc='lower right', frameon=True,
                        framealpha=1.0, 
                        prop={'family': 'arial', 'size': 8},
                        labelspacing=0.6, borderpad=0.4)
        
        #Add AEC logo-------------------

        # Preserve the logo's aspect ratio
        h, w = logo.shape[:2]
        logo_width = 0.25                                  # fraction of the map's width
        fig_w, fig_h = fig.get_size_inches()
        map_pos = ax.get_position()                        # after your canvas.draw() step
        map_aspect = (map_pos.height * fig_h) / (map_pos.width * fig_w)
        logo_height = logo_width * (h / w) / map_aspect    # height in axes fraction

        ax_logo = ax.inset_axes([0.0, 0.0, logo_width, logo_height])
        ax_logo.imshow(logo)
        ax_logo.axis('off')
        ax_logo.set_zorder(10)


        #Add text to top of figure----------------------
        ax.text(0.02, 0.98, eq_time[0:10]+' M'+str(eq_mag), transform=ax.transAxes,
                fontsize=17, va='top', ha='left', fontweight = 'bold', fontname = 'Arial',
                bbox=dict(facecolor='white', alpha = 1.0, edgecolor='black'),zorder = 20)

        fig.canvas.draw()                     # forces cartopy to apply its aspect ratio
        pos_map = ax.get_position()
        pos_bot = ax_xs.get_position()
        gap = 0.03   # figure-fraction gap between panels; try 0.01-0.03 if you want some room
        new_y0 = pos_map.y0 - gap - pos_bot.height
        ax_xs.set_position([pos_map.x0, new_y0, pos_map.width, pos_bot.height])
        
        fig.savefig(dir_path+'/cross_section_figure.png', dpi=720)

        plt.show()

#________________________________________________________________________________
#________________________________________________________________________________
#________________________________________________________________________________

    if plot_accelerations == True:
        print('Making accelerations plot...')

        plot_historical = False
        plot_aftershocks = False
        dlat = 2.0
        num_streams = 5
        dist_sep = 15

    # Find stations with adequate distance seperation-------------------
        
        dist_targets = []
        for i in range(num_streams):
            dist_targets.append(min_dist + i*dist_sep)

        df_targets = pd.DataFrame({'target': dist_targets}).sort_values('target')

        # Stations for plotting-----------------
        plot_stations = pd.merge_asof(
            df_targets, 
            available_stations_filt, 
            left_on='target', 
            right_on='distance', 
            direction='nearest'
        )
        plot_stations = plot_stations.drop_duplicates(subset=['station'])


            
        # Make plot-----------------------
    
        

        fig = plt.figure(figsize=(8, 10), dpi=150)
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.5], hspace=0.1) #0.25



        left, right, bottom, top = region = make_region(eq_lon, eq_lat, dlat=dlat)

        proj = ccrs.Mercator(central_longitude=eq_lon)      # was ccrs.Mercator()
        ax = fig.add_subplot(gs[0], projection=proj)

        PC = ccrs.PlateCarree()


        ax.set_extent(region, crs=ccrs.PlateCarree())


        # ---------- Shaded relief background ----------
        #grid = pygmt.datasets.load_earth_relief(resolution="01m", region=region).sortby("lat")
        grid = pygmt.datasets.load_earth_relief(resolution= '15s',#'30s',#'10m', 
                                                            region=region, 
                                                            registration=None, 
                                                            data_source='igpp', 
                                                            use_srtm=False).sortby("lat") 
        elev = grid.values
        lon, lat = grid.lon.values, grid.lat.values


        # hillshade
        ls = LightSource(azdeg=315, altdeg=45)
        cell = 463
        dx = cell * np.cos(np.deg2rad(lat.mean()))
        #dx = cell * np.cos(np.deg2rad(60))
        shade = ls.hillshade(elev, vert_exag=10, #3, 20
                            dx=dx, dy=cell)

        base = np.ones(elev.shape + (3,))
        base[elev < 0] = [0.80, 0.83, 0.87]
        rgb = base * (0.55 + 0.45 * shade)[..., None]

        #ax.imshow(rgb, origin="lower", extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                #transform=PC, zorder=0, regrid_shape=2000,interpolation='nearest')

        img_crs = ccrs.PlateCarree(central_longitude=eq_lon)
        ax.imshow(rgb, origin="lower",
                extent=[lon.min() - eq_lon, lon.max() - eq_lon, lat.min(), lat.max()],
                transform=img_crs, zorder=0, regrid_shape=2000, interpolation='nearest')

        # ---------- Coastlines and borders ----------
        ax.coastlines(resolution="10m", linewidth=0.6, zorder=1)
        ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=1.5, zorder=1)



        #Plot faults-----------------

        if event_loc == 'Alaska': #only have faults for alaska for now...

            ax.add_geometries(gdf.geometry, crs=ccrs.PlateCarree(),
                            facecolor='none', edgecolor='saddlebrown', #dimgrey
                            alpha=0.8, linewidth=1.5, zorder=2)

        


        #Plot earthquake-----------------------
        ax.scatter(eq_lon, eq_lat, s = 1000, color = 'red', edgecolors='black', 
                linewidths=1.5, marker = "o",
                transform=PC, zorder=5)

        # Plot stations-------------------------

        sc = ax.scatter(available_stations_filt["sta_lon"], available_stations_filt["sta_lat"], marker="v", s=150,
                c=available_stations_filt['duration'], cmap = 'Reds', vmin = 0, vmax = 1.5*(np.median(available_stations_filt['duration'].values)), edgecolors="k", linewidths=1.0, 
                transform=PC, zorder=3)

        cax = ax.inset_axes([1.03, 0.0, 0.03, 1.0])
        cbar = fig.colorbar(sc, cax=cax)
        cbar.set_label("duration of MMI "+str(MMI_plot)+"+ shaking (seconds)", fontsize=11)

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

        # ---------- Lat/lon labels on left and bottom only ----------
        gl = ax.gridlines(draw_labels=True, crs=PC, linewidth=0)   # linewidth=0 -> no grid lines
        gl.top_labels = True
        gl.bottom_labels = False
        gl.right_labels = False

        # ---------- Inset locator map ----------
        # Bounds are in axes fraction: [x0, y0, width, height] of the main map

        if event_loc == 'Global':
            ax_in = ax.inset_axes([0.75, 0.8, 0.25, 0.30],
                            projection=ccrs.Orthographic(central_longitude=eq_lon, #eq_lon
                                                        central_latitude=eq_lat)) #eq_lat
            ax_in.set_global()
            ax_in.add_feature(cfeature.OCEAN, facecolor='lightsteelblue')
            ax_in.add_feature(cfeature.LAND, facecolor='0.9')
            ax_in.coastlines(resolution='110m', linewidth=0.5)
            ax_in.gridlines(linewidth=0.3, color='gray', alpha=0.6)


        if event_loc == 'Alaska':
            anc_lat = 61.218056 #Anchorage latitude
            anc_lon = -149.900278 #Anchorage longitude
            ax_in = ax.inset_axes([0.75, 0.8, 0.25, 0.30],
                                projection=ccrs.Orthographic(central_longitude=anc_lon, #eq_lon
                                                            central_latitude=anc_lat)) #eq_lat
            #ax_in = ax.inset_axes([0.90, 0.7, 0.30, 0.36], projection=ccrs.Geodetic())
            ax_in.set_extent([-180, -135, 50, 70], crs=ccrs.PlateCarree())
            ax_in.add_feature(cfeature.LAND, facecolor='0.9')
            ax_in.add_feature(cfeature.OCEAN, facecolor='lightsteelblue')
            ax_in.coastlines(resolution='50m', linewidth=0.5)
            ax_in.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.6,
                        edgecolor='k', zorder=2)

            # Box showing the extent of the main map
        box_lons = [left, right, right, left, left]
        box_lats = [bottom, bottom, top, top, bottom]
        ax_in.plot(box_lons, box_lats, color='red', linewidth=1.5,
                    transform=ccrs.Geodetic())

        ax_in.plot(box_lons, box_lats, color='red', linewidth=1.5, transform=ccrs.Geodetic())

        ax_in.set_zorder(10)   # keep it above the relief and scatter layers


        #----------------------------------------------
        # Traces panel---------------------------------

        ax2 = fig.add_subplot(gs[1])
        index = plot_stations['og_index'].values
        durations = plot_stations['duration'].values
        st3 = st1.copy()
        st3 = st3.slice(starttime = UTCDateTime(eq_time), endtime=UTCDateTime(eq_time)+first_p+(3*np.median(durations)))

        stas = []
        for i in range(len(index)):
            tr = st3[index[i]]
            start_shaking = plot_stations['start_shaking'].values[i]
            end_shaking = plot_stations['end_shaking'].values[i]
            duration = durations[i]
            ax2.plot(tr.times(), (tr.data/np.max(np.abs(tr.data)))-i*2, color = 'black', alpha = 0.8)
            ax2.text(np.max(tr.times()), -i*2, tr.stats.station, va = 'bottom', 
                    ha = 'right', color = 'silver', fontweight = 'bold')
            
            ax2.text(np.max(tr.times())/2, -i*2+0.3, str(int(durations[i]))+" s", va = 'bottom', 
                    ha = 'right', color = 'red', fontweight = 'bold')
            
            if start_shaking == 0:
                print('No MMI', MMI_plot, 'for station', stations[i])
            else:
                trt = tr.slice(starttime = UTCDateTime(start_shaking), endtime=UTCDateTime(end_shaking))
                dif = abs(tr.stats.starttime - trt.stats.starttime)
                ax2.plot(trt.times() + dif, (-2*i)+trt.data/np.max(np.abs(tr.data)), color = 'red')
            
            stas.append(tr.stats.station)

        ax2.grid(alpha = 0.3)
        ax2.axvline(0, color = 'red', linestyle = '--')
        ax2.scatter(0,0.75,marker = '*', color = 'red', edgecolor = 'black', s = 300, zorder = 20)
        ax2.set_xlabel("time since earthquake origin (s)")
        ax2.set_yticks([])
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
            # Line from epicenter to station
            
            ax.plot([eq_lon, sta_lon], [eq_lat, sta_lat],
                    color='gray', linewidth=1.2, alpha=0.8,
                    transform=ccrs.Geodetic(), zorder=3.5)
            #ax.scatter(sta_lon, sta_lat, marker="v", s=200,
                    #c="firebrick", edgecolors="k", linewidths=1.0, 
                    #transform=PC, zorder=4)
            
        #Add AEC logo-------------------

        # Preserve the logo's aspect ratio
        h, w = logo.shape[:2]
        logo_width = 0.25                                  # fraction of the map's width
        fig_w, fig_h = fig.get_size_inches()
        map_pos = ax.get_position()                        # after your canvas.draw() step
        map_aspect = (map_pos.height * fig_h) / (map_pos.width * fig_w)
        logo_height = logo_width * (h / w) / map_aspect    # height in axes fraction

        ax_logo = ax.inset_axes([0.0, 0.0, logo_width, logo_height])
        ax_logo.imshow(logo)
        ax_logo.axis('off')
        ax_logo.set_zorder(10)

        # Plot Date/Magnitude text----------------------
        ax.text(0.02, 0.98, eq_time[0:10]+' M'+str(eq_mag), transform=ax.transAxes,
                fontsize=17, va='top', ha='left', fontweight = 'bold', fontname = 'Arial',
                bbox=dict(facecolor='white', alpha = 1.0, edgecolor='black'),zorder = 20)

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

        
        fig.savefig(dir_path+'/accelerations_figure.png', dpi=720)

        plt.show()


#________________________________________________________________________________
#________________________________________________________________________________
#________________________________________________________________________________

    if plot_mmi == True:
        print('Making MMI plot...')
        dlat = 2.0
        plot_historical = False
        plot_aftershocks = False

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


        fig = plt.figure(figsize=(8, 10), dpi=150)
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.5], hspace=0.1) #0.25



        left, right, bottom, top = region = make_region(eq_lon, eq_lat, dlat=dlat)

        proj = ccrs.Mercator(central_longitude=eq_lon)      # was ccrs.Mercator()
        ax = fig.add_subplot(gs[0], projection=proj)

        PC = ccrs.PlateCarree()


        ax.set_extent(region, crs=ccrs.PlateCarree())


        # ---------- Shaded relief background ----------
        #grid = pygmt.datasets.load_earth_relief(resolution="01m", region=region).sortby("lat")
        grid = pygmt.datasets.load_earth_relief(resolution= '15s',#'30s',#'10m', 
                                                            region=region, 
                                                            registration=None, 
                                                            data_source='igpp', 
                                                            use_srtm=False).sortby("lat") 
        elev = grid.values
        lon, lat = grid.lon.values, grid.lat.values


        # hillshade
        ls = LightSource(azdeg=315, altdeg=45)
        cell = 463
        dx = cell * np.cos(np.deg2rad(lat.mean()))
        #dx = cell * np.cos(np.deg2rad(60))
        shade = ls.hillshade(elev, vert_exag=10, #3, 20
                            dx=dx, dy=cell)

        base = np.ones(elev.shape + (3,))
        base[elev < 0] = [0.80, 0.83, 0.87]
        rgb = base * (0.55 + 0.45 * shade)[..., None]

        #ax.imshow(rgb, origin="lower", extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                #transform=PC, zorder=0, regrid_shape=2000,interpolation='nearest')

        img_crs = ccrs.PlateCarree(central_longitude=eq_lon)
        ax.imshow(rgb, origin="lower",
                extent=[lon.min() - eq_lon, lon.max() - eq_lon, lat.min(), lat.max()],
                transform=img_crs, zorder=0, regrid_shape=2000, interpolation='nearest')

        # ---------- Coastlines and borders ----------
        ax.coastlines(resolution="10m", linewidth=0.6, zorder=1)
        ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=1.5, zorder=1)



        #Plot faults-----------------

        if event_loc == 'Alaska': #only have faults for alaska for now...

            ax.add_geometries(gdf.geometry, crs=ccrs.PlateCarree(),
                            facecolor='none', edgecolor='saddlebrown', #dimgrey
                            alpha=0.8, linewidth=1.5, zorder=2)

        


        #Plot earthquake-----------------------
        ax.scatter(eq_lon, eq_lat, s = 1000, color = 'red', edgecolors='black', 
                linewidths=1.5, marker = "o",
                transform=PC, zorder=5)

        # Plot stations-------------------------

        sc = ax.scatter(available_stations_filt["sta_lon"], available_stations_filt["sta_lat"], marker="v", s=150,
                c=available_stations_filt['duration'], cmap = 'Reds', vmin = 0, vmax = 1.5*(np.median(available_stations_filt['duration'].values)), edgecolors="k", linewidths=1.0, 
                transform=PC, zorder=3)

        sc = ax.scatter(available_stations_filt["sta_lon"], available_stations_filt["sta_lat"], marker="v", s=100,
                        c=available_stations_filt["max_mmi"], cmap=mmi_cmap, norm=mmi_norm,
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

        # ---------- Lat/lon labels on left and bottom only ----------
        gl = ax.gridlines(draw_labels=True, crs=PC, linewidth=0)   # linewidth=0 -> no grid lines
        gl.top_labels = True
        gl.bottom_labels = False
        gl.right_labels = False

        # ---------- Inset locator map ----------
        # Bounds are in axes fraction: [x0, y0, width, height] of the main map

        if event_loc == 'Global':
            ax_in = ax.inset_axes([0.75, 0.8, 0.25, 0.30],
                            projection=ccrs.Orthographic(central_longitude=eq_lon, #eq_lon
                                                        central_latitude=eq_lat)) #eq_lat
            ax_in.set_global()
            ax_in.add_feature(cfeature.OCEAN, facecolor='lightsteelblue')
            ax_in.add_feature(cfeature.LAND, facecolor='0.9')
            ax_in.coastlines(resolution='110m', linewidth=0.5)
            ax_in.gridlines(linewidth=0.3, color='gray', alpha=0.6)


        if event_loc == 'Alaska':
            anc_lat = 61.218056 #Anchorage latitude
            anc_lon = -149.900278 #Anchorage longitude
            ax_in = ax.inset_axes([0.75, 0.8, 0.25, 0.30],
                                projection=ccrs.Orthographic(central_longitude=anc_lon, #eq_lon
                                                            central_latitude=anc_lat)) #eq_lat
            #ax_in = ax.inset_axes([0.90, 0.7, 0.30, 0.36], projection=ccrs.Geodetic())
            ax_in.set_extent([-180, -135, 50, 70], crs=ccrs.PlateCarree())
            ax_in.add_feature(cfeature.LAND, facecolor='0.9')
            ax_in.add_feature(cfeature.OCEAN, facecolor='lightsteelblue')
            ax_in.coastlines(resolution='50m', linewidth=0.5)
            ax_in.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.6,
                        edgecolor='k', zorder=2)

            # Box showing the extent of the main map
        box_lons = [left, right, right, left, left]
        box_lats = [bottom, bottom, top, top, bottom]
        ax_in.plot(box_lons, box_lats, color='red', linewidth=1.5,
                    transform=ccrs.Geodetic())

        ax_in.plot(box_lons, box_lats, color='red', linewidth=1.5, transform=ccrs.Geodetic())

        ax_in.set_zorder(10)   # keep it above the relief and scatter layers


        # Plot Date/Magnitude text----------------------
        ax.text(0.02, 0.98, eq_time[0:10]+' M'+str(eq_mag), transform=ax.transAxes,
                fontsize=17, va='top', ha='left', fontweight = 'bold', fontname = 'Arial',
                bbox=dict(facecolor='white', alpha = 1.0, edgecolor='black'),zorder = 20)

        #Plot MMI
        ax.text(0.02, 0.90, 'Max MMI recorded', transform=ax.transAxes,
                fontsize=17, va='top', ha='left', fontweight = 'bold', style = 'italic',
                fontname = 'Arial', color = 'red', zorder = 20)
        
        fig.savefig(dir_path+'/mmi_figure.png', dpi=720)

        plt.show()






