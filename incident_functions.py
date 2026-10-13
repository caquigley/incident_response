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



#---------------------------------------------
#--------FUNCTIONS FOR PULLING DATA FROM PARAMETER FILE----------
#---------------------------------------------

def read_params(params):
    '''
    Read parameters from .yaml file

    Inputs: 
        params: input parameter (input_parameters.yaml) file read into python

    Returns:
        variables needed for computation

    '''
    #Network inputs----------
    event_id = params["inputs"]["event_id"] 
    plot_map = params["inputs"]["plot_map"]  
    plot_mmi = params["inputs"]["plot_mmi"]
    plot_accelerations = params["inputs"]["plot_accelerations"]
    plot_cross_section =  params["inputs"]["plot_cross_section"]


    return (event_id, plot_map, plot_mmi, plot_accelerations, plot_cross_section)


#---------------------------------------------
#--------FUNCTIONS FOR PULLING DATA----------
#---------------------------------------------


def pull_earthquakes(lat, lon, max_rad, min_mag, start, end):

    """
    Pulls in earthquakes from a region based on lat, lon, timing, and magnitude.
    It also returns other values of interest about the event for array 
    processing, such as backazimuth, slowness, and epicentral distance to the
    event.
    
    Parameters:
        lat: latitude of array/station (str)
        lon: longitude of array/station (str)
        max_rad: maximum radius of earthquakes in kilometers (str)
        start: start time in UTC format (str)
        end: end time in UTC format (str)
        min_mag: minimum magnitude of earthquakes (str)
        array_name: name of array/station (str)
        velocity_model: name of velocity model (ex. 'iasp91', 'ak135')
        
    Returns:
        pandas DataFrame:
           'event_id': event id from USGS catalog
           'depth': depth of earthquake in km
           'magnitude': magnitude of earthquake
           'latitude': earthquake latitude
           'longitude': earthquake longitude
           'time_utc': origin time in UTC
           'time_ak': origin time in AK
           'distance': epicentral distance to event in km
           'backazimuth': backazimuth from array/station to earthquake
           'array': name of station/array
           'slowness': surface slowness (s/km)
           'trace_vel': surface trace velocity (km/s)
           'incident_angle': angle from vertical of first arriving wave (degrees)
           'p_arrival': arrival time of p-wave (seconds)             
    """
    
    ##Pull data in from FDSNWS: https://earthquake.usgs.gov/fdsnws/event/1/
    url = ('https://earthquake.usgs.gov/fdsnws/event/1/query?format=quakeml&starttime='+start+'&endtime='
           +end+'&latitude='+lat+'&longitude='+lon+'&maxradiuskm='+max_rad+'&minmagnitude='+min_mag+'')

    catalog = read_events(url)
    depths = []
    magnitudes = []
    latitudes = []
    longitudes = []
    times_utc = []
    times_ak = []
    names = []

    # Extract data from each event
    for event in catalog:
        # Extract depth
        depth = event.origins[0].depth
        if depth is not None:
            depth = event.origins[0].depth / 1000  # Depth is in meters, convert to kilometers
        if depth is None:
            depth = 0
        # Extract magnitude
        magnitude = event.magnitudes[0].mag
    
        # Extract latitude and longitude
        latitude = event.origins[0].latitude
        longitude = event.origins[0].longitude
    
        # Extract time
        time = event.origins[0].time

        # Extract event_id
        resource_id = event.resource_id.id
        name = resource_id.split('?')[-1]
        name = name[:-15]
        name= name[8:]

        
        # Append data to lists
        depths.append(depth)
        magnitudes.append(magnitude)
        latitudes.append(latitude)
        longitudes.append(longitude)
        times_utc.append(time)
        times_ak.append(time - 60*60*8)  # conversion to AK time
        names.append(name)

    # Combine into DataFrame
    data = {
        'event_id': names,
        'depth': depths,
        'magnitude': magnitudes,
        'latitude': latitudes,
        'longitude': longitudes,
        'time_utc': times_utc,
        'time_ak': times_ak,
    }

    df = pd.DataFrame(data)
    return df


# Pull stations------------------------
def pull_stations(eq_lat, eq_lon, eq_time, eq_depth, max_rad):
    '''
    Pull stations in vicinity of earthquake of interest

    eq_lat: earthquake latitute (float)
    eq_lon: earthquake longitude (float)
    eq_time: earthquake time (str)
    eq_depth: earthquake depth (float)

    '''
    print('Finding stations in vicinity of event...')
    model = TauPyModel(model='ak135')
    client = Client("EARTHSCOPE")


    # PULL ALL STATIONS NEARBY----------
    inv = client.get_stations(
        starttime=UTCDateTime(eq_time)-(60*60*24*7), #stations in last week
        endtime=UTCDateTime(eq_time)+(60*60*24*7), # stations week after
        latitude=eq_lat, longitude=eq_lon,
        channel='BHZ,SHZ,HHZ,DHZ,HNZ', #NEED TO UPDATE TO ALL CHANNELS OF INTEREST
        maxradius=kilometers2degrees(max_rad),
        level='channel', #channel, station
        )
    
    ## PULL INFORMATION OUT OF INVENTORY-------------------------
    lat_list = []
    lon_list = []
    distance_list = []
    
    network_list = []
    station_list = []
    channels_list = []
    p_arrivals = []
    az_list = []


    for network in inv:
        network_label  = network.code
        for station in network:
            for channel in station:
                network_list.append(network_label)
                lat_list.append(station.latitude)
                lon_list.append(station.longitude)
                station_list.append(station.code)
                dist, az, baz = gps2dist_azimuth(eq_lat, eq_lon, 
                                                station.latitude,
                                                station.longitude)
                az_list.append(az)
                distance_list.append(dist/1000)
                
                #Calculate p-arrival time-------------
                dist_deg = kilometers2degrees(dist/1000)
                arrivals = model.get_travel_times(source_depth_in_km=eq_depth,
                                            distance_in_degree=dist_deg)
                                            #phase_list = ["P","p"])
                arr = arrivals[0]
                p_arrivals.append(arr.time)

                channels_list.append(channel.code)

            #Pull out channels available at station-------------
            #channels = []
            #for channel in station:
                #channels.append(channel.code)

            

        data = {
            'network': network_list,
            'station': station_list,
            'channel': channels_list,
            'distance': distance_list,
            'sta_lat': lat_list,
            'sta_lon': lon_list,
            'p_arrival': p_arrivals,
            'azimuth': az_list,
        }

    df = pd.DataFrame(data)
    print('Found', len(df), 'stations.')

    return df, inv

def pull_streams(networks, stations, channels, eq_time):

    print('Pulling station data...')
    client = Client("EARTHSCOPE")

    st = Stream()
    inv = Inventory()
    failed_stations = []
    START = UTCDateTime(eq_time) - 60*1 #two minutes before
    END = START + 60*11 #ten minutes after

    for i in range(len(stations)):
        net = networks[i]
        sta = stations[i]
        chan = channels[i]
        loc = '*'
        #channel = channels[i] 
        #for k in range(len(channel)):
            #chan = channel[k]

        try:
            st += client.get_waveforms(net, sta, loc, chan, START, END)
            inv += client.get_stations(network=net, station=sta, channel=chan,
                                    location=loc, 
                                    starttime=START,
                                    endtime=END, 
                                    level='response')
        except FDSNNoDataException:
            print(f"No data for station {sta} and channel {chan}")
            failed_stations.append(sta)
        except Exception as e:
            print(f"Error for station {sta}: {e}")
            failed_stations.append(sta)

       
    st.merge(fill_value='latest')
    st.trim(START, END, pad='true', fill_value=0)
    #st.remove_sensitivity(inventory = inv)  
    #st.taper(max_percentage=0.05)
    #st.detrend('demean')
    #st.filter("bandpass", freqmin=1, freqmax=18, 
                #corners=2, zerophase=True)
    #st = st.slice(UTCDateTime(eq_time), UTCDateTime(eq_time)+ first_p +60) #90

    order = {(n, s): i for i, (n, s) in enumerate(zip(networks, stations))}
    st.traces.sort(key=lambda tr: (order.get((tr.stats.network, tr.stats.station), len(order)),
                                   tr.stats.channel))

    return st, inv


#---------------------------------------------
#--------FUNCTIONS FOR MAPMAKING--------------
#---------------------------------------------

def make_region(lon0, lat0, dlat=2.5, aspect=1.0, max_ratio=10):
    """
    dlat:      half-height of the region in degrees latitude
    aspect:    desired width/height of the region on the map (1 = square)
    max_ratio: cap on dlon/dlat so the region doesn't explode near the poles
    """
    ratio = aspect / max(np.cos(np.deg2rad(lat0)), 1 / max_ratio)
    dlon = dlat * ratio                              # half-width in degrees longitude

    left, right = lon0 - dlon, lon0 + dlon           # may exceed ±180, keep continuous
    bottom = max(lat0 - dlat, -90)
    top = min(lat0 + dlat, 90)
    return [left, right, bottom, top]

#---------------------------------------------
#--------FUNCTIONS FOR CALCULATING SHAKING----------
#---------------------------------------------

def exceedance_episodes(data, dt, threshold, max_gap=2.0, min_count=3):
    idx = np.where(np.abs(data) > threshold)[0]
    if idx.size == 0:
        return []

    breaks = np.where(np.diff(idx) * dt > max_gap)[0]
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]

    episodes = []
    for s, e in zip(starts, ends):
        seg = data[s:e + 1]
        n_pos = np.count_nonzero(seg > threshold)
        n_neg = np.count_nonzero(seg < -threshold)
        if n_pos + n_neg >= min_count and n_pos > 0 and n_neg > 0:
            episodes.append((s, e, n_pos + n_neg))
    return episodes
'''
def exceedance_episodes(data, dt, threshold, max_gap=2.0, min_count=3):
    """
    Group samples above `threshold` into episodes of near-continuous shaking.

    max_gap:   gaps shorter than this (s) are bridged, so shaking that dips below
               the threshold briefly is still one episode
    min_count: minimum number of exceeding samples for an episode to count,
               which discards isolated spikes
    Returns a list of (start_idx, end_idx, n_exceed) tuples.
    """
    idx = np.where(data > threshold)[0]
    if idx.size == 0:
        return []

    # Split wherever the time between consecutive exceedances is larger than max_gap
    breaks = np.where(np.diff(idx) * dt > max_gap)[0]
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]

    episodes = []
    for s, e in zip(starts, ends):
        n = np.count_nonzero((idx >= s) & (idx <= e))
        if n >= min_count:
            episodes.append((s, e, n))
    return episodes
'''
def check_mmi(percent_g):
    #Define thesholds for MMI shaking------------------
    if percent_g <= 0.17:
        MMI = 1
    elif percent_g > 0.17 and percent_g <= 1.4:
        MMI = 3
    elif percent_g > 1.4 and percent_g <= 3.9:
        MMI = 4
    elif percent_g > 3.9 and percent_g <= 9.2:
        MMI = 5
    elif percent_g > 9.2 and percent_g <= 18:
        MMI = 6
    elif percent_g > 18 and percent_g <= 34:
        MMI = 7
    elif percent_g > 34 and percent_g <= 65:
        MMI = 8
    elif percent_g > 65 and percent_g <= 124:
        MMI = 9
    elif percent_g > 124:
        MMI = 10
    return MMI


def calculate_shaking(st2, inv, p_arrival, eq_time):

    st = st2.copy()
    st.remove_response(inventory=inv, output="ACC")
    #st.detrend('demean')
    #st.filter('bandpass')
    #st.filter("lowpass", freq=10,  #0.001
                #corners=2, zerophase=True)

    thresholds = [0.17, 1.4, 3.9, 9.2, 18, 34, 65, 124]
    mmi = [3, 4, 5, 6, 7, 8, 9, 10]
    df_list = []
    
    #st = st_av + st_av2
    for i in range(len(st)):
        tr1 = st[i]
        #print('Starting', tr1.stats.station)

        #Slice trace so that it is only looking at real signal--------------
        start = UTCDateTime(eq_time) + p_arrival[i] - 10 #ten seconds before p-arrival
        end = start + 10+ 5*60 #two minutes after p_arrival

        tr = tr1.slice(starttime=start, endtime=end)

        #temp_data = np.abs((tr.data/9.8)*100) #new version
        temp_data = (tr.data/9.8)*100 #old version
        #temp_data = temp_data[:-3000]

        #Loop through MMI thresholds-------------
        start_list = [] #start for each MMI
        end_list = [] #end for each MMI
        duration_list = []
        max_shaking = []
        max_mmi = []
        station_list = []
        channel_list = []

        '''
        for k in range(len(thresholds)):
            indices = np.where(temp_data > thresholds[k])[0]
            
            if indices.size > 0:
                last_index = indices[-1]
                first_index = indices[0]
                print("Values above MMI "+str(mmi[k]))
                start_list.append( start + tr.times()[first_index])
                end_list.append(start + tr.times()[last_index])
                duration_list.append(abs(tr.times()[first_index] - tr.times()[last_index]))
            else:
                print("No value above " +str(mmi[k]))
                start_list.append(0)
                end_list.append(0)
                duration_list.append(0)
        '''

        for k in range(len(thresholds)):
            episodes = exceedance_episodes(temp_data, tr.stats.delta, thresholds[k],
                                        max_gap=2.0, min_count=5)
            if not episodes:
                start_list.append(0)
                end_list.append(0)
                duration_list.append(0)
            else: # find the maximum episode
                times = tr.times()
                # Longest episode = the main shaking
                s, e, n = max(episodes, key=lambda ep: ep[1] - ep[0])

                start_list.append(start + times[s])
                end_list.append(start + times[e])
                duration_list.append(times[e] - times[s])
        # Find indices where condition is met
            station_list.append(tr.stats.station)
            channel_list.append(tr.stats.channel)
            max_shaking.append(np.max(temp_data))
            max_mmi.append(check_mmi(np.max(temp_data)))

        data = {
                'station': station_list,
                'channel': channel_list,
                'start_shaking': start_list,
                'end_shaking': end_list,
                'duration': duration_list,
                'mmi': mmi,
                'max_shaking': max_shaking,
                'max_mmi': max_mmi
            }

        df = pd.DataFrame(data)
        df_list.append(df)

    result = pd.concat(df_list, ignore_index=True)

    return result

#---------------------------------------------
#--------FUNCTIONS FOR CROSS-SECTION----------
#---------------------------------------------

def make_profile(start, end, half_width_km=30):
    """
    start, end: (lon, lat) of profile endpoints (A and A')
    Returns project(lons, lats) -> (along_km, perp_km), profile length (km),
    and the swath outline as lon/lat arrays.
    """
    geod = Geod(ellps='WGS84')
    lon1, lat1 = start
    lon2, lat2 = end

    # Center the local projection on the profile midpoint
    az12, _, dist = geod.inv(lon1, lat1, lon2, lat2)
    lon0, lat0, _ = geod.fwd(lon1, lat1, az12, dist / 2)
    proj = Proj(proj='aeqd', lat_0=lat0, lon_0=lon0, datum='WGS84')   # meters

    xs, ys = proj(lon1, lat1)
    xe, ye = proj(lon2, lat2)
    L = np.hypot(xe - xs, ye - ys)
    ux, uy = (xe - xs) / L, (ye - ys) / L                # unit vector along profile

    def project(lons, lats):
        x, y = proj(np.asarray(lons, float), np.asarray(lats, float))
        dx, dy = x - xs, y - ys
        along = (dx * ux + dy * uy) / 1000                # km from A along the line
        perp = (-dx * uy + dy * ux) / 1000                # km off the line (+ = left)
        return along, perp

    # Swath outline for the map
    a = np.array([0, L / 1000, L / 1000, 0, 0])
    p = np.array([-1, -1, 1, 1, -1]) * half_width_km
    x = xs + (a * ux - p * uy) * 1000
    y = ys + (a * uy + p * ux) * 1000
    swath_lon, swath_lat = proj(x, y, inverse=True)

    return project, L / 1000, swath_lon, swath_lat