import numpy as np
import pandas as pd
import xarray as xr
import wradlib as wrl
from osgeo import osr
import wradlib.georef as georef
import json

# Load the NetCDF file
ds = xr.open_dataset('kolkata_radar_data.nc')

# Print available variables to debug
print("Available variables in NetCDF file:")
print(list(ds.variables.keys()))
print("\nDataset info:")
print(ds)

# Extract coordinates and data with correct variable names
try:
    azimuth_angles = ds['azimuth'].values  # Changed from 'azi' to 'azimuth'
    print(f"Azimuth shape: {azimuth_angles.shape}")
except KeyError:
    print("Available coordinate variables:")
    for var in ds.coords:
        print(f"  {var}: {ds[var].shape}")
    # Try alternative names
    possible_azi_names = ['azimuth', 'azi', 'az', 'azimuth_angle']
    azimuth_angles = None
    for name in possible_azi_names:
        if name in ds.variables:
            azimuth_angles = ds[name].values
            print(f"Found azimuth data as '{name}': {azimuth_angles.shape}")
            break

try:
    elevation_angles = ds['elevation'].values  # This should be correct
    print(f"Elevation shape: {elevation_angles.shape}")
except KeyError:
    print("Elevation variable not found. Available variables:")
    for var in ds.variables:
        print(f"  {var}")
    # Try alternative names
    possible_elev_names = ['elevation', 'elev', 'el', 'elevation_angle']
    elevation_angles = None
    for name in possible_elev_names:
        if name in ds.variables:
            elevation_angles = ds[name].values
            print(f"Found elevation data as '{name}': {elevation_angles.shape}")
            break

try:
    reflectivity_data = ds['reflectivity'].values  # This should be correct
    print(f"Reflectivity shape: {reflectivity_data.shape}")
except KeyError:
    print("Reflectivity variable not found. Available variables:")
    # Try alternative names
    possible_refl_names = ['reflectivity', 'refl', 'dbz', 'Z', 'DBZH']
    reflectivity_data = None
    for name in possible_refl_names:
        if name in ds.variables:
            reflectivity_data = ds[name].values
            print(f"Found reflectivity data as '{name}': {reflectivity_data.shape}")
            break

# Extract radar metadata
try:
    radar_lat = ds['latitude'].values
    radar_lon = ds['longitude'].values
    print(f"Radar coordinates shape - Lat: {radar_lat.shape}, Lon: {radar_lon.shape}")
except KeyError:
    # Try alternative coordinate names
    coord_names = list(ds.coords.keys())
    print(f"Available coordinates: {coord_names}")
    
    # Extract radar location from attributes if available
    if hasattr(ds, 'attrs'):
        radar_lat = ds.attrs.get('radar_latitude', 22.5697)
        radar_lon = ds.attrs.get('radar_longitude', 88.3697)
        print(f"Using radar location from attributes: {radar_lat}, {radar_lon}")

# Extract range information
try:
    if 'range' in ds.variables:
        range_data = ds['range'].values
        print(f"Range data shape: {range_data.shape}")
    elif 'ranges' in ds.variables:
        range_data = ds['ranges'].values
        print(f"Range data shape: {range_data.shape}")
    else:
        print("Range data not found in variables. Will calculate from attributes.")
        range_data = None
except Exception as e:
    print(f"Error loading range data: {e}")
    range_data = None

# Check dimensions and create proper coordinate arrays
print(f"\nData dimensions:")
if azimuth_angles is not None:
    print(f"Azimuth: {azimuth_angles.shape}")
if elevation_angles is not None:
    print(f"Elevation: {elevation_angles.shape}")
if reflectivity_data is not None:
    print(f"Reflectivity: {reflectivity_data.shape}")
if range_data is not None:
    print(f"Range: {range_data.shape}")

# Get data dimensions
if reflectivity_data is not None:
    data_shape = reflectivity_data.shape
    print(f"Reflectivity data shape: {data_shape}")
    
    # Determine the data structure
    if len(data_shape) == 3:
        num_elevation_angles, num_azimuths, num_range_gates = data_shape
        print(f"3D data: {num_elevation_angles} elevations, {num_azimuths} azimuths, {num_range_gates} range gates")
    elif len(data_shape) == 2:
        num_azimuths, num_range_gates = data_shape
        num_elevation_angles = 1
        print(f"2D data: {num_azimuths} azimuths, {num_range_gates} range gates")
    else:
        print(f"Unexpected data shape: {data_shape}")

# Marshall-Palmer Z-R relationship
def calc_rainfall_rate(reflectivity_dbz):
    """Calculate rainfall rate using Marshall-Palmer equation"""
    if np.isnan(reflectivity_dbz) or reflectivity_dbz <= 0:
        return 0.0
    
    # Marshall-Palmer: Z = 200 * R^1.6
    # Solving for R: R = (Z/200)^(1/1.6)
    z_linear = 10.0 ** (reflectivity_dbz / 10.0)
    rainfall_rate = (z_linear / 200.0) ** (1.0 / 1.6)
    
    return rainfall_rate

# Set up coordinate system and radar parameters
radar_altitude = 30.0  # meters above sea level (approximate for Kolkata)

# Handle different coordinate array formats
if isinstance(radar_lat, np.ndarray):
    if radar_lat.size == 1:
        radar_lat = float(radar_lat.item())
        radar_lon = float(radar_lon.item())
    else:
        # Take the center point if it's an array
        radar_lat = float(np.nanmean(radar_lat))
        radar_lon = float(np.nanmean(radar_lon))

print(f"Using radar location: {radar_lat:.4f}°N, {radar_lon:.4f}°E")

sitecoords = (float(radar_lon), float(radar_lat), float(radar_altitude))
print(f"Radar site coordinates: {sitecoords}")

trg_crs = osr.SpatialReference()
trg_crs.ImportFromEPSG(4326)  # WGS84

# Set up range array if not available
if range_data is None:
    # Create range array based on typical radar parameters
    max_range_km = 250  # km
    range_resolution = 1.0  # km
    range_data = np.arange(0, max_range_km, range_resolution) * 1000  # Convert to meters
    print(f"Created range array: {len(range_data)} gates, max range: {max_range_km} km")

# Process radar data
cloud_coordinates = []
max_range_km = 250
max_range_meters = max_range_km * 1000

print(f"Processing radar data...")
print(f"Max range: {max_range_km} km")

# Handle different data structures
if len(reflectivity_data.shape) == 3:
    # 3D data: loop through elevation sweeps
    for sweep_idx in range(num_elevation_angles):
        current_elevation = elevation_angles[sweep_idx] if elevation_angles.ndim > 0 else 0.5
        print(f"Processing sweep {sweep_idx + 1}/{num_elevation_angles}, elevation: {current_elevation:.2f}°")
        
        sweep_refl = reflectivity_data[sweep_idx, :, :]
        
        for azi_idx in range(num_azimuths):
            if azimuth_angles is not None:
                current_azimuth = azimuth_angles[azi_idx] if azimuth_angles.ndim > 0 else azi_idx * (360.0 / num_azimuths)
            else:
                current_azimuth = azi_idx * (360.0 / num_azimuths)
            
            for range_idx in range(num_range_gates):
                current_reflectivity = sweep_refl[azi_idx, range_idx]
                current_range = range_data[range_idx] if range_idx < len(range_data) else range_idx * 1000
                
                if (not np.isnan(current_reflectivity) and 
                    current_range <= max_range_meters):
                    
                    # Convert to geographic coordinates
                    coords = georef.polar.spherical_to_proj(
                        current_range,
                        current_azimuth,
                        current_elevation,
                        sitecoords,
                        crs=trg_crs
                    )
                    
                    if len(coords) >= 2:
                        cloud_lon = coords[0]  # Longitude
                        cloud_lat = coords[1]  # Latitude
                        cloud_alt = coords[2] if len(coords) > 2 else radar_altitude
                        
                        rainfall_rate = calc_rainfall_rate(current_reflectivity)
                        
                        cloud_coordinates.append({
                            'latitude': float(cloud_lat),
                            'longitude': float(cloud_lon),
                            'altitude': float(cloud_alt),
                            'reflectivity': float(current_reflectivity),
                            'rainfall_rate': float(rainfall_rate),
                            'sweep': sweep_idx,
                            'elevation': float(current_elevation),
                            'azimuth': float(current_azimuth),
                            'range_km': float(current_range / 1000)
                        })

elif len(reflectivity_data.shape) == 2:
    # 2D data: single elevation sweep
    print("Processing 2D data (single sweep)")
    current_elevation = elevation_angles[0] if elevation_angles is not None and len(elevation_angles) > 0 else 0.5
    
    for azi_idx in range(num_azimuths):
        if azimuth_angles is not None:
            current_azimuth = azimuth_angles[azi_idx] if azimuth_angles.ndim > 0 else azi_idx * (360.0 / num_azimuths)
        else:
            current_azimuth = azi_idx * (360.0 / num_azimuths)
        
        for range_idx in range(num_range_gates):
            current_reflectivity = reflectivity_data[azi_idx, range_idx]
            current_range = range_data[range_idx] if range_idx < len(range_data) else range_idx * 1000
            
            if (not np.isnan(current_reflectivity) and 
                current_range <= max_range_meters):
                
                # Convert to geographic coordinates
                coords = georef.polar.spherical_to_proj(
                    current_range,
                    current_azimuth,
                    current_elevation,
                    sitecoords,
                    crs=trg_crs
                )
                
                if len(coords) >= 2:
                    cloud_lon = coords[0]  # Longitude
                    cloud_lat = coords[1]  # Latitude
                    cloud_alt = coords[2] if len(coords) > 2 else radar_altitude
                    
                    rainfall_rate = calc_rainfall_rate(current_reflectivity)
                    
                    cloud_coordinates.append({
                        'latitude': float(cloud_lat),
                        'longitude': float(cloud_lon),
                        'altitude': float(cloud_alt),
                        'reflectivity': float(current_reflectivity),
                        'rainfall_rate': float(rainfall_rate),
                        'sweep': 0,
                        'elevation': float(current_elevation),
                        'azimuth': float(current_azimuth),
                        'range_km': float(current_range / 1000)
                    })

# Save results
if cloud_coordinates:
    df = pd.DataFrame(cloud_coordinates)
    
    # Save to CSV
    output_file = 'radar_points_corrected.csv'
    df.to_csv(output_file, index=False)
    
    # Save to JSON with specified structure
    json_output_file = 'radar_points_corrected.json'
    
    # Create JSON array with the specified structure
    json_data = []
    for _, row in df.iterrows():
        json_data.append({
            'latitude': row['latitude'],
            'longitude': row['longitude'],
            'reflectivity': row['reflectivity'],
            'rainfall_rate': row['rainfall_rate']
        })
    
    # Write JSON file
    with open(json_output_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"\nProcessing complete!")
    print(f"Found {len(cloud_coordinates)} points above threshold")
    print(f"Data saved to: {output_file}")
    print(f"JSON data saved to: {json_output_file}")
    print(f"\nData summary:")
    print(f"  Latitude range: {df['latitude'].min():.4f} to {df['latitude'].max():.4f}")
    print(f"  Longitude range: {df['longitude'].min():.4f} to {df['longitude'].max():.4f}")
    print(f"  Reflectivity range: {df['reflectivity'].min():.1f} to {df['reflectivity'].max():.1f} dBZ")
    print(f"  Rainfall range: {df['rainfall_rate'].min():.3f} to {df['rainfall_rate'].max():.3f} mm/hr")
    
    # Check for Alipore Observatory
    alipore_lat, alipore_lon = 22.5333, 88.3333
    distances = np.sqrt((df['latitude'] - alipore_lat)**2 + (df['longitude'] - alipore_lon)**2)
    closest_idx = distances.idxmin()
    closest_distance = distances.min() * 111  # Convert to km
    
    print(f"\nClosest point to Alipore Observatory:")
    print(f"  Distance: {closest_distance:.2f} km")
    print(f"  Location: {df.loc[closest_idx, 'latitude']:.4f}°N, {df.loc[closest_idx, 'longitude']:.4f}°E")
    print(f"  Reflectivity: {df.loc[closest_idx, 'reflectivity']:.1f} dBZ")
    print(f"  Rainfall: {df.loc[closest_idx, 'rainfall_rate']:.2f} mm/hr")
    
else:
    print("No data points found above the reflectivity threshold.")
    print("Try lowering the threshold or check the data quality.")