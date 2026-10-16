# incident_plots

## What
Code for creating plots following a large earthquake in Alaska.

## How
This code does the following in the specified order:
1. Finds event information from the comprehensive earthquake catalog (COMCAT) based 
on a input event. 
2. Pulls streams of the event from obspy.
3. Calculates the shaking duration for a specified MMI as a function of percent gravity.
5. Plotting tools to assess different aspects of the event
- Overview map with earthquake epicenter and traces of nearby stations
- Map with historical seismicity and cross-section of the event relative to historical
seismicity.
- Map of shaking duration for nearby stations and some example stations.
- Maximum MMI recorded at nearby stations.


## Installation

Create a fork of this repository to your github. Then, create a local repository through your terminal. For example:

```python
git clone <github_link_to_repo>
```

A conda environment with the needed dependencies are provided in environment.yml. This can be created with the following command in your local repository:

```python
conda env create -f environment.yml
```


## Usage

Once your python environment is installed, this can be activated as:

```python
conda activate incident_plots
```

In your local repository, the main code can be run as:

```python
python incident_plots.py event_id MMI
```
Where event_id is the event_id from COMCAT (USGS) and MMI is an integer of the MMI shaking intensity desired. 

As an example, here is how you would run theses plots for the 2025 M7.3 Sandpoint Earthquake
and MMI 3 shaking duration:

```python
python incident_plots.py us7000qd1y 3
```


## Repository Contents

| File | Description |
|------|-------------|
| `incident_plots.py` | Main script for pulling data and making plots. |
| `incident_functions.py` | Helper functions used in the main script. |
| `environment.yml` | Conda environment file for creating a local environment. |
| `./events` | Folder containing event subfolders named by event_id. Plots are put into each subfolder. |
| `./slab2` | Folder containing SLAB2.0 grid files used for cross-section figures. |
| `./logos` | Folder containing AEC logos. |
| `./logos` | Folder containing fault files. |
| `points_of_interest.csv` | Towns in Alaska for plotting. |


