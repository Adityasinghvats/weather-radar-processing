import xarray as xr
import numpy as np
import wradlib.georef as georef
import wradlib.util as util
import pyproj
from osgeo import _osr
from osgeo import osr
print("GDal is working")

file_path = "2024062612321800dBZ.vol.nc"
ds = xr.open_dataset(file_path, decode_times=False)

# access useful variables
radar_lat = ds['latitude'].values
radar_lon = ds['longitude'].values
radar_altitude = ds['altitude'].values if 'altitude' in ds.variables else 0
# print(radar_lon, radar_lat, radar_altitude)

# access the data arrays
# based on relfectivity dimensions is (elevation, azi, bin)
reflectivity = ds['reflectivity'].values
elevation_angles = ds['elevation'].values
azimuth_angles = ds['azi'].values
# print("Ref:", reflectivity)
# print("Ele:", elevation_angles)
# print("Azi:",azimuth_angles)

# range information
srange = ds['srange'].values #start range 
resolution = ds['resolution'].values #range resolution(size of each bin)
# print("range", srange)
# print("res", resolution)

num_elevation_angles = reflectivity.shape[0]
num_azimuths = reflectivity.shape[1]
num_range_gates = reflectivity.shape[2]
print(num_elevation_angles, num_azimuths, num_range_gates)

# Construct the array of range values for the *center* of each gate
# The range to the center of bin 'k' is srange + (k + 0.5) * resolution
# ranges = srange + (np.arange(num_range_gates) + 0.5) * resolution

sitecoords = (float(np.squeeze(radar_lon)), float(np.squeeze(radar_lat)), float(np.squeeze(radar_altitude)))
print(f"Radar site coordinates: {sitecoords}")
trg_crs = osr.SpatialReference()
trg_crs.ImportFromEPSG(4326)  # WGS84


cloud_coordinates = []
reflectivity_threshold = 40
max_range_km = 250
max_range_meters = max_range_km * 1000


# loop through each elevation sweep
for i in range(num_elevation_angles):
    current_elevation = elevation_angles[i]
    current_srange = srange[i]
    current_resolution = resolution[i]

    # Calculate the ranges for the bins
    ranges_for_sweep = current_srange + (np.arange(num_range_gates) + 0.5) * current_resolution

    # Loop through each azimuth ray in the sweep
    for j in range(azimuth_angles.shape[1]):
        for k in range(num_range_gates):
            current_reflectivity = reflectivity[i, j, k]
            current_range = ranges_for_sweep[k]
            current_azi = azimuth_angles[i, j]

            if current_reflectivity > reflectivity_threshold and current_range <= max_range_meters:
                current_range_array = np.array([float(current_range)])
                current_azi_array = np.array([float(current_azi)])
                current_elevation_array = np.array([float(current_elevation)])

                # Convert polar coordinates to geographic coordinates
                coords = georef.polar.spherical_to_proj(
                    current_range_array,
                    current_azi_array,
                    current_elevation_array,
                    sitecoords,
                    crs=trg_crs
                )

                # Extract coordinates from the 1D array
                # coords contains [lon, lat, alt] for a single point
                cloud_lon = coords[0]  # First element is longitude
                cloud_lat = coords[1]  # Second element is latitude
                cloud_alt = coords[2]  # Third element is altitude

                cloud_coordinates.append({
                    'latitude': cloud_lat,
                    'longitude': cloud_lon,
                    'altitude': cloud_alt,
                    'reflectivity': current_reflectivity
                })

# After the loops, save the results if needed
if cloud_coordinates:
    import pandas as pd
    df = pd.DataFrame(cloud_coordinates)
    df.to_csv('radar_points.csv', index=False)
    print(f"Found {len(cloud_coordinates)} points above threshold")