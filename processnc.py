import numpy as np
import pandas as pd
import xarray as xr
import json
from datetime import datetime
import os

class RadarDataProcessor:
    def __init__(self, netcdf_file='kolkata_radar_data.nc'):
        """
        Initialize processor for NetCDF radar data
        """
        self.netcdf_file = netcdf_file
        self.load_data()
    
    def load_data(self):
        """Load NetCDF radar data"""
        if not os.path.exists(self.netcdf_file):
            raise FileNotFoundError(f"NetCDF file '{self.netcdf_file}' not found")
        
        print(f"Loading radar data from {self.netcdf_file}...")
        self.ds = xr.open_dataset(self.netcdf_file)
        print(f"Dataset loaded successfully")
        print(f"Dimensions: {dict(self.ds.dims)}")
        print(f"Variables: {list(self.ds.data_vars.keys())}")
    
    def marshall_palmer_equation(self, reflectivity_dbz):
        """
        Calculate rainfall rate using Marshall-Palmer equation
        
        The Marshall-Palmer relationship: Z = a * R^b
        Where:
        - Z = reflectivity factor (mm^6/m^3)
        - R = rainfall rate (mm/hr)
        - a = 200 (standard coefficient)
        - b = 1.6 (standard exponent)
        
        Solving for R: R = (Z/a)^(1/b) = (Z/200)^(1/1.6) = (Z/200)^0.625
        
        First convert dBZ to Z: Z = 10^(dBZ/10)
        """
        # Marshall-Palmer coefficients
        a = 200.0  # mm^6/m^3 per (mm/hr)^1.6
        b = 1.6    # dimensionless exponent
        
        # Convert dBZ to linear reflectivity factor Z (mm^6/m^3)
        z_linear = 10.0 ** (reflectivity_dbz / 10.0)
        
        # Calculate rainfall rate using Marshall-Palmer equation
        # R = (Z/a)^(1/b)
        rainfall_rate = (z_linear / a) ** (1.0 / b)
        
        # Handle invalid values
        rainfall_rate = np.where(
            np.isfinite(reflectivity_dbz) & (reflectivity_dbz > 0),
            rainfall_rate,
            0.0  # Set to 0 for invalid reflectivity values
        )
        
        return rainfall_rate
    
    def alternative_zr_relations(self, reflectivity_dbz, relation_type='tropical'):
        """
        Alternative Z-R relationships for different precipitation types
        
        Parameters:
        - reflectivity_dbz: Radar reflectivity in dBZ
        - relation_type: Type of Z-R relationship
          * 'marshall_palmer': Z = 200*R^1.6 (default)
          * 'tropical': Z = 300*R^1.4 (for tropical rainfall)
          * 'convective': Z = 486*R^1.37 (for convective storms)
          * 'stratiform': Z = 223*R^1.53 (for stratiform rain)
        """
        
        # Z-R relationship coefficients
        zr_coefficients = {
            'marshall_palmer': {'a': 200.0, 'b': 1.6},
            'tropical': {'a': 300.0, 'b': 1.4},
            'convective': {'a': 486.0, 'b': 1.37},
            'stratiform': {'a': 223.0, 'b': 1.53}
        }
        
        if relation_type not in zr_coefficients:
            relation_type = 'marshall_palmer'
        
        a = zr_coefficients[relation_type]['a']
        b = zr_coefficients[relation_type]['b']
        
        # Convert dBZ to linear reflectivity
        z_linear = 10.0 ** (reflectivity_dbz / 10.0)
        
        # Calculate rainfall rate
        rainfall_rate = (z_linear / a) ** (1.0 / b)
        
        # Handle invalid values
        rainfall_rate = np.where(
            np.isfinite(reflectivity_dbz) & (reflectivity_dbz > 0),
            rainfall_rate,
            0.0
        )
        
        return rainfall_rate
    
    def select_lowest_elevation_sweep(self):
        """
        Select the sweep with the lowest elevation angle for surface rainfall estimation
        """
        if 'sweep_number' in self.ds.dims:
            num_sweeps = self.ds.dims['sweep_number']
            print(f"Found {num_sweeps} sweeps in the dataset")
            
            # Find elevation angles for each sweep
            elevations = []
            for i in range(num_sweeps):
                sweep_data = self.ds.isel(sweep_number=i)
                if 'elevation' in sweep_data.coords:
                    elevation = float(sweep_data.elevation)
                else:
                    # Assume typical elevation progression: 0.5°, 1.0°, 1.5°, etc.
                    elevation = (i + 1) * 0.5
                elevations.append(elevation)
                print(f"  Sweep {i}: Elevation = {elevation:.2f}°")
            
            # Find sweep with minimum elevation
            min_elevation_idx = np.argmin(elevations)
            min_elevation = elevations[min_elevation_idx]
            
            print(f"Selected sweep {min_elevation_idx} with lowest elevation: {min_elevation:.2f}°")
            
            # Extract the lowest elevation sweep
            selected_sweep = self.ds.isel(sweep_number=min_elevation_idx)
            
            return selected_sweep, min_elevation_idx, min_elevation
        else:
            print("No sweep dimension found, using entire dataset")
            return self.ds, 0, 0.5
    
    def process_to_dataframe(self, min_reflectivity=-10, max_reflectivity=60, zr_relation='marshall_palmer'):
        """
        Convert NetCDF data to pandas DataFrame using only the lowest elevation sweep
        
        Parameters:
        - min_reflectivity: Minimum reflectivity threshold (dBZ)
        - max_reflectivity: Maximum reflectivity threshold (dBZ) - removes non-meteorological targets
        - zr_relation: Z-R relationship to use for rainfall calculation
        """
        print("Converting NetCDF data to DataFrame...")
        print("Using only the lowest elevation sweep for surface rainfall estimation...")
        
        # Select the lowest elevation sweep
        selected_sweep, sweep_idx, elevation_angle = self.select_lowest_elevation_sweep()
        
        # Extract data arrays from the selected sweep
        lats = selected_sweep.latitude.values
        lons = selected_sweep.longitude.values
        refl = selected_sweep.reflectivity.values
        
        print(f"Selected sweep data shape: {refl.shape}")
        
        # Calculate altitude for each point
        # Get radar location
        radar_lat = self.ds.attrs.get('radar_latitude', 22.5697)
        radar_lon = self.ds.attrs.get('radar_longitude', 88.3697)
        radar_alt = self.ds.attrs.get('radar_altitude', 30.0)  # meters above sea level
    
        # Calculate altitude using elevation angle and distance from radar
        print("Calculating altitude for each data point...")
        
        # Flatten arrays for DataFrame
        lats_flat = lats.flatten()
        lons_flat = lons.flatten()
        refl_flat = refl.flatten()
    
        # Calculate distance from radar and altitude
        altitudes = []
        for i in range(len(lats_flat)):
            if np.isfinite(lats_flat[i]) and np.isfinite(lons_flat[i]):
                # Calculate distance from radar (in km)
                lat_diff = lats_flat[i] - radar_lat
                lon_diff = lons_flat[i] - radar_lon
                distance_km = np.sqrt(lat_diff**2 + lon_diff**2) * 111.0  # Convert degrees to km
    
                # Calculate altitude using elevation angle
                # altitude = radar_altitude + distance * tan(elevation_angle)
                elevation_rad = np.radians(elevation_angle)
                altitude = radar_alt + (distance_km * 1000 * np.tan(elevation_rad))  # Convert to meters
                altitudes.append(altitude)
            else:
                altitudes.append(np.nan)
    
        # Create DataFrame
        df = pd.DataFrame({
            'latitude': lats_flat,
            'longitude': lons_flat,
            'altitude': altitudes,  # Add altitude column
            'reflectivity_dbz': refl_flat,
            'sweep_number': sweep_idx,
            'elevation_angle': elevation_angle
        })
        
        # Filter out invalid data and high reflectivity values
        valid_mask = (
            np.isfinite(df['reflectivity_dbz']) & 
            (df['reflectivity_dbz'] >= min_reflectivity) &
            (df['reflectivity_dbz'] <= max_reflectivity) &  # Remove high reflectivity values
            np.isfinite(df['latitude']) & 
            np.isfinite(df['longitude']) &
            np.isfinite(df['altitude'])  # Also filter valid altitude
        )
        
        df = df[valid_mask].copy()
        
        print(f"Valid data points: {len(df)}")
        print(f"Reflectivity range: {df['reflectivity_dbz'].min():.1f} to {df['reflectivity_dbz'].max():.1f} dBZ")
        print(f"Altitude range: {df['altitude'].min():.1f} to {df['altitude'].max():.1f} meters")
        print(f"Using elevation angle: {elevation_angle:.2f}° (sweep {sweep_idx})")
        
        if len(df) == 0:
            print("Warning: No valid data points found!")
            return df
        
        # Calculate rainfall rate using specified Z-R relationship
        print(f"Calculating rainfall rates using {zr_relation} relationship...")
        
        if zr_relation == 'marshall_palmer':
            df['rainfall_rate_mm_hr'] = self.marshall_palmer_equation(df['reflectivity_dbz'])
        else:
            df['rainfall_rate_mm_hr'] = self.alternative_zr_relations(
                df['reflectivity_dbz'], zr_relation
            )

        # Add additional calculated fields
        df['rainfall_intensity_category'] = self.categorize_rainfall_intensity(df['rainfall_rate_mm_hr'])
        
        # Add metadata
        if 'time' in selected_sweep.coords:
            df['timestamp'] = pd.to_datetime(selected_sweep.time.values)
        elif 'time' in self.ds.coords:
            df['timestamp'] = pd.to_datetime(self.ds.time.values[0])
        else:
            df['timestamp'] = datetime.now()

        if hasattr(self.ds, 'attrs'):
            df.attrs = {
                'radar_latitude': radar_lat,
                'radar_longitude': radar_lon,
                'radar_altitude': radar_alt,
                'max_range_km': self.ds.attrs.get('max_range_km', 250),
                'zr_relation': zr_relation,
                'min_reflectivity': min_reflectivity,
                'max_reflectivity': max_reflectivity,
                'selected_sweep': sweep_idx,
                'elevation_angle': elevation_angle,
                'sweep_selection': 'lowest_elevation',
                'created': datetime.now().isoformat()
            }

        # Sort by rainfall rate (highest first)
        df = df.sort_values('rainfall_rate_mm_hr', ascending=False)

        print(f"Rainfall rate statistics:")
        print(f"  Min: {df['rainfall_rate_mm_hr'].min():.2f} mm/hr")
        print(f"  Max: {df['rainfall_rate_mm_hr'].max():.2f} mm/hr")
        print(f"  Mean: {df['rainfall_rate_mm_hr'].mean():.2f} mm/hr")
        print(f"  Median: {df['rainfall_rate_mm_hr'].median():.2f} mm/hr")

        return df
    
    def categorize_rainfall_intensity(self, rainfall_rates):
        """
        Categorize rainfall intensity based on rate
        
        Categories based on IMD classification:
        - No rain: 0 mm/hr
        - Light: 0.1 - 2.5 mm/hr
        - Moderate: 2.5 - 10 mm/hr
        - Heavy: 10 - 50 mm/hr
        - Very Heavy: 50+ mm/hr
        """
        categories = np.full(len(rainfall_rates), 'No Rain', dtype=object)
        
        categories[rainfall_rates >= 0.1] = 'Light'
        categories[rainfall_rates >= 2.5] = 'Moderate'
        categories[rainfall_rates >= 10.0] = 'Heavy'
        categories[rainfall_rates >= 50.0] = 'Very Heavy'
        
        return categories
    
    def save_to_csv(self, df, filename='kolkata_radar_rainfall.csv'):
        """Save DataFrame to CSV file"""
        print(f"Saving data to CSV: {filename}")
        
        # Round numerical values for cleaner output
        df_export = df.copy()
        df_export['latitude'] = df_export['latitude'].round(6)
        df_export['longitude'] = df_export['longitude'].round(6)
        df_export['altitude'] = df_export['altitude'].round(1)  # Round altitude to 1 decimal place
        df_export['reflectivity_dbz'] = df_export['reflectivity_dbz'].round(2)
        df_export['rainfall_rate_mm_hr'] = df_export['rainfall_rate_mm_hr'].round(3)
        
        df_export.to_csv(filename, index=False)
        print(f"CSV file saved: {filename}")
        
        # Save summary statistics
        summary_filename = filename.replace('.csv', '_summary.csv')
        summary_stats = df.groupby('rainfall_intensity_category').agg({
            'rainfall_rate_mm_hr': ['count', 'mean', 'min', 'max'],
            'reflectivity_dbz': ['mean', 'min', 'max'],
            'altitude': ['mean', 'min', 'max']  # Add altitude statistics
        }).round(3)
        
        summary_stats.to_csv(summary_filename)
        print(f"Summary statistics saved: {summary_filename}")
        
        return filename
    
    def save_to_json(self, df, filename='kolkata_radar_rainfall.json', format_type='records'):
        """
        Save DataFrame to JSON file
        
        Parameters:
        - format_type: 'records', 'geojson', or 'summary'
        """
        print(f"Saving data to JSON: {filename}")
        
        if format_type == 'geojson':
            # Create GeoJSON format
            features = []
            for _, row in df.iterrows():
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [row['longitude'], row['latitude']]
                    },
                    "properties": {
                        "reflectivity_dbz": round(row['reflectivity_dbz'], 2),
                        "rainfall_rate_mm_hr": round(row['rainfall_rate_mm_hr'], 3),
                        "intensity_category": row['rainfall_intensity_category'],
                        "elevation_angle": round(row['elevation_angle'], 2),
                        "sweep_number": int(row['sweep_number']),
                        "timestamp": row['timestamp'].isoformat() if pd.notna(row['timestamp']) else None
                    }
                }
                features.append(feature)
            
            geojson_data = {
                "type": "FeatureCollection",
                "metadata": {
                    "title": "Kolkata Weather Radar Rainfall Data (Lowest Elevation)",
                    "description": "Surface rainfall rates from lowest elevation sweep using Marshall-Palmer equation",
                    "radar_location": {
                        "latitude": df.attrs.get('radar_latitude', 22.5697),
                        "longitude": df.attrs.get('radar_longitude', 88.3697)
                    },
                    "elevation_angle": df.attrs.get('elevation_angle', 0.5),
                    "sweep_number": df.attrs.get('selected_sweep', 0),
                    "total_points": len(features),
                    "created": datetime.now().isoformat()
                },
                "features": features
            }
            
            with open(filename, 'w') as f:
                json.dump(geojson_data, f, indent=2, default=str)
        
        elif format_type == 'summary':
            # Create summary JSON
            summary_data = {
                "metadata": {
                    "title": "Kolkata Weather Radar Rainfall Summary (Lowest Elevation)",
                    "total_points": len(df),
                    "radar_location": {
                        "latitude": df.attrs.get('radar_latitude', 22.5697),
                        "longitude": df.attrs.get('radar_longitude', 88.3697)
                    },
                    "elevation_angle": df.attrs.get('elevation_angle', 0.5),
                    "sweep_number": df.attrs.get('selected_sweep', 0),
                    "created": datetime.now().isoformat()
                },
                "statistics": {
                    "reflectivity": {
                        "min_dbz": float(df['reflectivity_dbz'].min()),
                        "max_dbz": float(df['reflectivity_dbz'].max()),
                        "mean_dbz": float(df['reflectivity_dbz'].mean())
                    },
                    "rainfall": {
                        "min_mm_hr": float(df['rainfall_rate_mm_hr'].min()),
                        "max_mm_hr": float(df['rainfall_rate_mm_hr'].max()),
                        "mean_mm_hr": float(df['rainfall_rate_mm_hr'].mean())
                    }
                },
                "intensity_distribution": df['rainfall_intensity_category'].value_counts().to_dict(),
                "sample_data": df.head(10).to_dict('records')
            }
            
            with open(filename, 'w') as f:
                json.dump(summary_data, f, indent=2, default=str)
        
        else:  # records format
            # Convert DataFrame to JSON records
            df_export = df.copy()
            df_export['latitude'] = df_export['latitude'].round(6)
            df_export['longitude'] = df_export['longitude'].round(6)
            df_export['reflectivity_dbz'] = df_export['reflectivity_dbz'].round(2)
            df_export['rainfall_rate_mm_hr'] = df_export['rainfall_rate_mm_hr'].round(3)
            
            data = {
                "metadata": {
                    "title": "Kolkata Weather Radar Rainfall Data (Lowest Elevation)",
                    "total_points": len(df_export),
                    "elevation_angle": df.attrs.get('elevation_angle', 0.5),
                    "sweep_number": df.attrs.get('selected_sweep', 0),
                    "created": datetime.now().isoformat()
                },
                "data": df_export.to_dict('records')
            }
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        
        print(f"JSON file saved: {filename}")
        return filename

def main():
    """Main function to process NetCDF radar data"""
    
    try:
        # Initialize processor
        processor = RadarDataProcessor('kolkata_radar_data.nc')
        
        # Process data to DataFrame (using lowest elevation sweep only)
        print("\n" + "="*50)
        print("PROCESSING RADAR DATA - LOWEST ELEVATION SWEEP")
        print("="*50)
        
        # You can change the Z-R relationship here:
        # Options: 'marshall_palmer', 'tropical', 'convective', 'stratiform'
        zr_relation = 'tropical'  # Good for Indian monsoon conditions
        
        df = processor.process_to_dataframe(
            min_reflectivity=0,     # Include low reflectivity values
            max_reflectivity=60,    # Remove high reflectivity values (non-meteorological targets)
            zr_relation=zr_relation
        )
        
        if len(df) == 0:
            print("No valid data to process!")
            return
        
        # Save to CSV
        print("\n" + "="*50)
        print("SAVING TO CSV")
        print("="*50)
        csv_file = processor.save_to_csv(df, f'kolkata2_radar_rainfall_{zr_relation}_lowest_elevation.csv')
        
        # Save to JSON (multiple formats)
        print("\n" + "="*50)
        print("SAVING TO JSON")
        print("="*50)
        
        # Print final summary
        print("\n" + "="*50)
        print("PROCESSING COMPLETE")
        print("="*50)
        print(f"Files created:")
        print(f"  • CSV: {csv_file}")
        # print(f"  • JSON: {json_file}")
        # print(f"  • GeoJSON: {geojson_file}")
        # print(f"  • Summary: {summary_file}")
        print(f"\nData summary:")
        print(f"  • Total points: {len(df):,}")
        print(f"  • Selected sweep: {df.attrs.get('selected_sweep', 0)}")
        print(f"  • Elevation angle: {df.attrs.get('elevation_angle', 0.5):.2f}°")
        print(f"  • Reflectivity range: {df['reflectivity_dbz'].min():.1f} to {df['reflectivity_dbz'].max():.1f} dBZ")
        print(f"  • Rainfall range: {df['rainfall_rate_mm_hr'].min():.3f} - {df['rainfall_rate_mm_hr'].max():.3f} mm/hr")
        print(f"  • Altitude range: {df['altitude'].min():.1f} to {df['altitude'].max():.1f} meters")  # Add altitude range
        print(f"  • Z-R relationship: {zr_relation}")
        print(f"  • Max reflectivity threshold: 60 dBZ")
        
        # Show intensity distribution
        intensity_counts = df['rainfall_intensity_category'].value_counts()
        print(f"\nRainfall intensity distribution:")
        for category, count in intensity_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  • {category}: {count:,} points ({percentage:.1f}%)")
        
    except Exception as e:
        print(f"Error processing radar data: {e}")
        import traceback
        traceback.print_exc()


        

if __name__ == "__main__":
    main()