#%%
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

from incident_functions import (pull_earthquakes, pull_stations, pull_streams,
                                make_region, calculate_shaking)



#%%
event_id = 'us7000ti1p' #'ak018fcnsk91' (Anchorage 2018) #'us6000tz1y' (Australia) #'us7000ti1p' (M6.5 Rat islands) # 'aka2026tuxgky' (5.5)  us6000tza4 (Rat islands),  us6000u18k (Panama)

#Pull all earthquakes in Alaska from last week----------------
lat = '61.218056' #Anchorage latitude
lon = '-149.900278' #Anchorage longitude
max_rad = '13000' #circle that covers all of Alaska, 3000, 9000, 19000 for globe
min_mag = '6.0' #min mag to search over, 4
event_loc = 'Global' #Alaska, Global
dlat = 2.5
plot_aftershocks = True
plot_historical = False
start = str(UTCDateTime.now() - (60*60*24*365)*10) #pull events from last seven days
end = str(UTCDateTime.now())
eq_df = pull_earthquakes(lat, lon, max_rad, min_mag, start, end)

#Pull out event of interest -----------------------------
temp = eq_df.loc[eq_df['event_id'] == event_id].head(1)
if len(temp) == 1:
    print('Event found!')

eq_lat = temp['latitude'].to_numpy()[0]
eq_lon = temp['longitude'].to_numpy()[0]
eq_depth = temp['depth'].to_numpy()[0]
eq_mag = temp['magnitude'].to_numpy()[0]
eq_time = str(temp['time_utc'].to_numpy()[0])

#Pull all stations in region -----------------------------
max_rad = 600
stations_df, inv = pull_stations(eq_lat, eq_lon, eq_time, eq_depth, max_rad)

#Create subset of stations based on closest station
min_dist = stations_df['distance'].min()
first_p = stations_df['p_arrival'].min()
closest_stations = min_dist + 200
stations_subset = pd.DataFrame(stations_df[stations_df['distance']<= closest_stations])


stations_subset = stations_subset.sort_values(by="distance", ascending=True)
networks = stations_subset['network'].values
stations = stations_subset['station'].values
channels = stations_subset['channel'].values


#See what stations are available -----------------------------
st, inv2 = pull_streams(networks, stations, channels, eq_time)
st1 = st.copy() # save version for later analysis
#%%
st.remove_sensitivity(inventory = inv2)  
st.taper(max_percentage=0.05)
st.detrend('demean')
st.filter("bandpass", freqmin=1, freqmax=18, 
            corners=2, zerophase=True)

st = st.slice(UTCDateTime(eq_time), UTCDateTime(eq_time)+ first_p +60) #90

#%%
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
available_stations = pd.merge(df, stations_subset, on=['station', 'channel'])
available_stations = available_stations.drop_duplicates().reset_index()
available_stations = available_stations.sort_values(by = 'distance')


# Find stations with adequate distance seperation-------------------
num_streams = 5
dist_sep = 50
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
#df.drop_duplicates(subset=['email'], inplace=True)

#%%
#plot_historical = True
if plot_historical == True:
    #Pull historical earthquakes-----------------------------
    print('Pulling historical earthquakes for plotting...')
    max_rad = '600'
    min_mag = '4.5'
    start = '1900-00-00'
    end = str(UTCDateTime.now()-(60*60*24))
    hist_df = pull_earthquakes(str(eq_lat), str(eq_lon), max_rad, min_mag, start, end)

#plot_aftershocks = True
if plot_aftershocks == True:
    print('Pulling aftershocks for plotting...')
    max_rad = '90'
    min_mag = '3.0'
    start = str(UTCDateTime(eq_time)+1*60) #one minute after
    end = str(UTCDateTime(eq_time)+86400*10) #10 days after
    aftershock_df = pull_earthquakes(str(eq_lat), str(eq_lon), max_rad, min_mag, start, end)



#%%
#event_loc = 'Alaska'
print('Starting plot generation...')

#--------------------------------------------------
#----------MAP FIGURE------------------------------

fig = plt.figure(figsize=(8, 10), dpi=150)
gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.5], hspace=0.1) #0.25

#Define area of map--------
left, right, bottom, top = region = make_region(eq_lon, eq_lat, dlat=dlat)

proj = ccrs.Mercator(central_longitude=eq_lon)      # was ccrs.Mercator()
ax = fig.add_subplot(gs[0], projection=proj)

PC = ccrs.PlateCarree()
ax.set_extent(region, crs=ccrs.PlateCarree())


# ---------- Shaded relief background ----------

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
dx = cell * np.cos(np.deg2rad(60))
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

    depth_colors = np.where(depth < 35, 'lightgrey',#"firebrick",
                    np.where(depth < 100, "darkgrey","dimgrey"))#"gold", "#4a0010"))
    alphas = np.where(mag >= 4.5, 0.8, 0.25)   # 0.25fade small events; adjust to taste
    face = [to_rgba(c, a) for c, a in zip(depth_colors, alphas)]
    edge = [to_rgba("k", a) for a in alphas]

    ax.scatter(hist_df["longitude"], hist_df["latitude"], s=3 * 2**mag,
            c=face, edgecolors=edge, linewidths=0.8,
            transform=PC, zorder=3)
    
    # Depth (color) entries
    depth_handles = [
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
    
    # Depth (color) entries
    depth_handles = [
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

#Plot faults-----------------

if event_loc == 'Alaska': #only have faults for alaska for now...

    gdf = gpd.read_file('./fault_files/AEC_Faults_Simplified/Faults_simple.shp')
    print(len(gdf), gdf.geometry.is_empty.sum(), gdf.geometry.isna().sum())   # how many bad rows?

    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    gdf = gdf.to_crs('EPSG:4326')     # in case the file isn't in lon/lat (check gdf.crs)

    ax.add_geometries(gdf.geometry, crs=ccrs.PlateCarree(),
                    facecolor='none', edgecolor='saddlebrown', #dimgrey
                    alpha=0.8, linewidth=1.5, zorder=2)

fault_handle = Line2D([], [], linestyle='-', linewidth=2,
                    color='saddlebrown', label='faults')


#Plot earthquake-----------------------
ax.scatter(eq_lon, eq_lat, s = 1000, color = 'red', edgecolors='black', 
           linewidths=1.5, marker = "o",
           transform=PC, zorder=5)

# Plot stations-------------------------

ax.scatter(stations_df["sta_lon"], stations_df["sta_lat"], marker="v", s=70,
           c="gray", edgecolors="k", linewidths=1.0, 
           transform=PC, zorder=3)

#Plot legend------------

# Station entry
sta_handle = Line2D([], [], marker='v', linestyle='', markersize=12,
                    markerfacecolor='gray', markeredgecolor='k',
                    markeredgewidth=1.5, label='seismic stations')

sta_handle2 = Line2D([], [], marker='v', linestyle='', markersize=12,
                    markerfacecolor='firebrick', markeredgecolor='k',
                    markeredgewidth=1.5, label='plotted stations')

if plot_historical == True:
    handles = depth_handles + mag_handles + [sta_handle] + [sta_handle2] + [fault_handle]
elif plot_historical == False:
    if plot_aftershocks == True:
        handles = [sta_handle] + [sta_handle2] + mag_handles + [fault_handle]
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

    # Or mark the epicenter instead of (or in addition to) the box
#ax_in.scatter(eq_lon, eq_lat, marker='*', s=150, color='red',
                #edgecolors='k', linewidths=0.8, transform=PC)

ax_in.set_zorder(10)   # keep it above the relief and scatter layers


#----------------------------------------------
# Traces panel---------------------------------

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
ax2.scatter(0,0.75,marker = '*', color = 'red', edgecolor = 'black', s = 300)
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


logo = mpimg.imread('./logos/EC_black_with_rectangle.png')


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

#ax2.set_position([pos_map.x0, pos_bot.y0, pos_map.width, pos_bot.height])

#fig.tight_layout()
#fig.savefig('/Users/cadequigley/Downloads/comm_team_figure_example4.png', dpi=720)

plt.show()


# %%



#parrivals = available_stations['p_arrival'].values
#result = calculate_shaking(st1, inv2, parrivals)

#MMI_plot = 5

# %%
