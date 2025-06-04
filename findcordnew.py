import numpy as np
import pandas as pd
import xarray as xr
import os

def check_radar_coverage():
    """Check if Alipore Observatory is within radar coverage"""
    
    # Kolkata radar location
    radar_lat = 22.5697
    radar_lon = 88.3697
    
    # Alipore Observatory location
    alipore_lat = 22.53605
    alipore_lon = 88.33089
    
    # Calculate distance
    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371  # Earth's radius in km
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = (np.sin(dlat/2)**2 + 
             np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2)
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c
    
    distance = haversine_distance(radar_lat, radar_lon, alipore_lat, alipore_lon)
    
    print(f"Kolkata Radar: {radar_lat:.4f}°N, {radar_lon:.4f}°E")
    print(f"Alipore Observatory: {alipore_lat:.4f}°N, {alipore_lon:.4f}°E")
    print(f"Distance: {distance:.2f} km")
    
    # Check if within radar range (assuming 250km max range)
    max_range = 250
    if distance <= max_range:
        print(f"✓ Alipore is within radar range ({max_range} km)")
    else:
        print(f"✗ Alipore is outside radar range ({max_range} km)")
    
    return distance

def find_nearest_points_to_alipore(csv_file):
    """Find the nearest radar data points to Alipore Observatory"""
    
    try:
        df = pd.read_csv(csv_file)
        
        alipore_lat = 22.5333
        alipore_lon = 88.3333
        
        # Calculate distances to all points
        df['distance_to_alipore'] = np.sqrt(
            (df['latitude'] - alipore_lat)**2 + 
            (df['longitude'] - alipore_lon)**2
        )
        
        # Find nearest points
        nearest_points = df.nsmallest(10, 'distance_to_alipore')
        
        print(f"\n10 Nearest radar points to Alipore Observatory:")
        print("-" * 80)
        for idx, row in nearest_points.iterrows():
            distance_km = row['distance_to_alipore'] * 111  # Approximate conversion to km
            print(f"Lat: {row['latitude']:.4f}, Lon: {row['longitude']:.4f}, "
                  f"Distance: {distance_km:.2f} km, "
                  f"Reflectivity: {row['reflectivity_dbz']:.1f} dBZ, "
                  f"Rainfall: {row['rainfall_rate_mm_hr']:.2f} mm/hr")
        
        return nearest_points
        
    except FileNotFoundError:
        print(f"File {csv_file} not found. Please run the main processor first.")
        return None

def check_data_coverage():
    """Check the geographical coverage of processed data"""
    
    try:
        # Try to load the NetCDF file first
        ds = xr.open_dataset('kolkata_radar_data.nc')
        
        lat_min = float(ds.latitude.min())
        lat_max = float(ds.latitude.max())
        lon_min = float(ds.longitude.min())
        lon_max = float(ds.longitude.max())
        
        print(f"\nRadar data coverage:")
        print(f"Latitude range: {lat_min:.4f}° to {lat_max:.4f}°N")
        print(f"Longitude range: {lon_min:.4f}° to {lon_max:.4f}°E")
        
        # Check if Alipore is within bounds
        alipore_lat = 22.5333
        alipore_lon = 88.3333
        
        lat_in_range = lat_min <= alipore_lat <= lat_max
        lon_in_range = lon_min <= alipore_lon <= lon_max
        
        print(f"\nAlipore Observatory (22.5333°N, 88.3333°E):")
        print(f"Within latitude range: {'✓' if lat_in_range else '✗'}")
        print(f"Within longitude range: {'✓' if lon_in_range else '✗'}")
        
        if lat_in_range and lon_in_range:
            print("✓ Alipore should be in the data coverage area")
        else:
            print("✗ Alipore is outside the data coverage area")
            
        return ds
            
    except FileNotFoundError:
        print("NetCDF file not found. Cannot check coverage.")
        return None

def find_alipore_data_in_netcdf(tolerance=0.01):
    """Search for data near Alipore Observatory in NetCDF file"""
    alipore_lat = 22.5333
    alipore_lon = 88.3333
    
    try:
        # Load NetCDF data
        ds = xr.open_dataset('kolkata_radar_data.nc')
        
        # Load raw data before filtering
        lats = ds.latitude.values
        lons = ds.longitude.values
        refl = ds.reflectivity.values
        
        print(f"\nSearching for data near Alipore Observatory...")
        print(f"Search tolerance: ±{tolerance}° ({tolerance * 111:.1f} km)")
        
        # Find points within tolerance
        lat_mask = np.abs(lats - alipore_lat) <= tolerance
        lon_mask = np.abs(lons - alipore_lon) <= tolerance
        combined_mask = lat_mask & lon_mask
        
        if np.any(combined_mask):
            num_points = np.sum(combined_mask)
            print(f"✓ Found {num_points} points near Alipore")
            
            nearby_lats = lats[combined_mask]
            nearby_lons = lons[combined_mask]
            nearby_refl = refl[combined_mask]
            
            # Show valid reflectivity values
            valid_refl = nearby_refl[np.isfinite(nearby_refl)]
            
            if len(valid_refl) > 0:
                print(f"Valid reflectivity values: {len(valid_refl)} points")
                print(f"Reflectivity range: {valid_refl.min():.1f} to {valid_refl.max():.1f} dBZ")
                print(f"Mean reflectivity: {valid_refl.mean():.1f} dBZ")
                
                # Show closest points
                distances = np.sqrt((nearby_lats - alipore_lat)**2 + (nearby_lons - alipore_lon)**2)
                closest_idx = np.argmin(distances)
                
                print(f"\nClosest point to Alipore:")
                print(f"  Lat: {nearby_lats.flat[closest_idx]:.6f}°N")
                print(f"  Lon: {nearby_lons.flat[closest_idx]:.6f}°E")
                print(f"  Distance: {distances.flat[closest_idx] * 111:.2f} km")
                print(f"  Reflectivity: {nearby_refl.flat[closest_idx]:.1f} dBZ")
                
            else:
                print("⚠ No valid reflectivity data found near Alipore")
                print("All nearby points have NaN or invalid reflectivity values")
        else:
            print("✗ No data points found near Alipore Observatory")
            print("Trying larger search radius...")
            
            # Try with larger tolerance
            larger_tolerance = tolerance * 5
            lat_mask_large = np.abs(lats - alipore_lat) <= larger_tolerance
            lon_mask_large = np.abs(lons - alipore_lon) <= larger_tolerance
            combined_mask_large = lat_mask_large & lon_mask_large
            
            if np.any(combined_mask_large):
                print(f"Found {np.sum(combined_mask_large)} points within {larger_tolerance}° radius")
                
                # Find the closest point in this larger area
                lat_diff = lats[combined_mask_large] - alipore_lat
                lon_diff = lons[combined_mask_large] - alipore_lon
                distances = np.sqrt(lat_diff**2 + lon_diff**2)
                closest_idx = np.argmin(distances)
                
                closest_lat = lats[combined_mask_large].flat[closest_idx]
                closest_lon = lons[combined_mask_large].flat[closest_idx]
                closest_refl = refl[combined_mask_large].flat[closest_idx]
                closest_dist_km = distances.flat[closest_idx] * 111
                
                print(f"Closest available point:")
                print(f"  Lat: {closest_lat:.6f}°N")
                print(f"  Lon: {closest_lon:.6f}°E") 
                print(f"  Distance: {closest_dist_km:.2f} km")
                print(f"  Reflectivity: {closest_refl:.1f} dBZ")
            else:
                print("No data points found even with larger search radius")
        
        return combined_mask
        
    except FileNotFoundError:
        print("NetCDF file 'kolkata_radar_data.nc' not found.")
        print("Please run the radar processing script first.")
        return None
    except Exception as e:
        print(f"Error reading NetCDF file: {e}")
        return None

def analyze_radar_geometry():
    """Analyze why Alipore might not be visible to the radar"""
    
    radar_lat = 22.5697
    radar_lon = 88.3697
    alipore_lat = 22.5333
    alipore_lon = 88.3333
    
    # Calculate azimuth from radar to Alipore
    lat_diff = alipore_lat - radar_lat
    lon_diff = alipore_lon - radar_lon
    
    azimuth = np.degrees(np.arctan2(lon_diff, lat_diff))
    if azimuth < 0:
        azimuth += 360
    
    # Calculate distance
    distance_deg = np.sqrt(lat_diff**2 + lon_diff**2)
    distance_km = distance_deg * 111
    
    print(f"\nRadar Geometry Analysis:")
    print(f"Azimuth from radar to Alipore: {azimuth:.1f}°")
    print(f"Distance: {distance_km:.2f} km")
    
    # Check if this is a near-field issue
    if distance_km < 10:
        print("⚠ NEAR-FIELD ISSUE: Alipore is very close to the radar")
        print("  • Radar beam may pass overhead at low elevation angles")
        print("  • Ground clutter filtering may remove near-field data")
        print("  • Minimum detection range limitations")
    
    return azimuth, distance_km

if __name__ == "__main__":
    print("CHECKING ALIPORE OBSERVATORY COVERAGE")
    print("=" * 50)
    
    # Check distance from radar
    distance = check_radar_coverage()
    
    # Analyze radar geometry
    azimuth, dist_km = analyze_radar_geometry()
    
    # Check data coverage
    ds = check_data_coverage()
    
    # Look for Alipore data in NetCDF
    find_alipore_data_in_netcdf(tolerance=0.01)
    
    # Look for nearest points in processed files
    csv_files = [
        'kolkata_radar_rainfall_tropical_filtered.csv',
        'kolkata_radar_rainfall_marshall_palmer_filtered.csv',
        'kolkata_radar_rainfall_tropical.csv'
    ]
    
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            print(f"\nChecking {csv_file}:")
            find_nearest_points_to_alipore(csv_file)
            break
    else:
        print("\nNo processed CSV files found.")
        print("Please run the main radar processing script first.")