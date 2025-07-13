import numpy as np
import xarray as xr
import xml.etree.ElementTree as ET
import struct
import zlib
from datetime import datetime
import os
import re

class RadarVolumeProcessor:
    def __init__(self, radar_lat=22.5697, radar_lon=88.3697, max_range_km=250):
        """
        Initialize radar processor for Kolkata radar
        Default coordinates for Kolkata radar station
        """
        self.radar_lat = radar_lat
        self.radar_lon = radar_lon
        self.max_range_km = max_range_km
        self.earth_radius = 6371.0  # km
        
    def parse_vol_file(self, filepath):
        """Parse a single .vol file and extract metadata and data"""
        with open(filepath, 'rb') as f:
            content = f.read()
        
        # Convert to string for easier processing
        try:
            content_str = content.decode('utf-8', errors='ignore')
        except:
            content_str = content.decode('latin-1', errors='ignore')
        
        # Find XML header end
        xml_end_marker = '<!-- END XML -->'
        xml_end_pos = content_str.find(xml_end_marker)
        
        if xml_end_pos == -1:
            raise ValueError("Could not find XML header end marker")
        
        xml_content = content_str[:xml_end_pos].strip()
        
        # Fix XML content if needed
        xml_content = self._fix_xml_content(xml_content)
        
        # Parse XML metadata
        try:
            root = ET.fromstring(xml_content)
            volume_info = {
                'version': root.get('version'),
                'datetime': root.get('datetime'),
                'type': root.get('type'),
                'owner': root.get('owner', '')
            }
        except ET.ParseError as e:
            print(f"XML Parse Error: {e}")
            print(f"XML Content: {xml_content[:200]}...")
            # Use defaults if XML parsing fails
            volume_info = {
                'version': '5.34.27',
                'datetime': '2025-04-27T00:22:33',
                'type': 'vol',
                'owner': ''
            }
        
        # Get binary content
        binary_start = xml_end_pos + len(xml_end_marker)
        binary_content = content[binary_start:]
        
        # Parse binary blobs
        blobs = self._parse_blobs(binary_content)
        
        return volume_info, blobs
    
    def _fix_xml_content(self, xml_content):
        """Fix common XML formatting issues"""
        # Remove any non-printable characters at the beginning
        xml_content = xml_content.lstrip()
        
        # Ensure XML starts with proper declaration or root element
        if not xml_content.startswith('<?xml') and not xml_content.startswith('<volume'):
            # Find the first < character
            start_pos = xml_content.find('<')
            if start_pos > 0:
                xml_content = xml_content[start_pos:]
        
        # Make sure we have a complete volume element
        if '<volume' in xml_content and '</volume>' not in xml_content:
            # Find the volume tag and make it self-closing if needed
            volume_match = re.search(r'<volume[^>]*>', xml_content)
            if volume_match:
                volume_tag = volume_match.group(0)
                if not volume_tag.endswith('/>'):
                    # Make it self-closing
                    volume_tag = volume_tag[:-1] + '/>'
                    xml_content = volume_tag
        
        return xml_content
    
    def _parse_blobs(self, binary_content):
        """Parse binary BLOB data from vol file"""
        blobs = {}
        
        # Convert to string for searching
        try:
            content_str = binary_content.decode('utf-8', errors='ignore')
        except:
            content_str = binary_content.decode('latin-1', errors='ignore')
        
        # Find all BLOB sections
        blob_pattern = r'<BLOB\s+blobid="(\d+)"\s+size="(\d+)"\s+compression="([^"]*)"[^>]*>'
        blob_matches = re.finditer(blob_pattern, content_str)
        
        for match in blob_matches:
            blob_id = int(match.group(1))
            size = int(match.group(2))
            compression = match.group(3)
            
            # Find the start of blob data (after the BLOB tag and any whitespace)
            data_start = match.end()
            while data_start < len(content_str) and content_str[data_start] in ['\n', '\r', ' ', '\t']:
                data_start += 1
            
            # Find the end of blob data (before </BLOB>)
            blob_end_pattern = r'</BLOB>'
            end_match = re.search(blob_end_pattern, content_str[data_start:])
            
            if end_match:
                data_end = data_start + end_match.start()
            else:
                data_end = data_start + size
            
            # Extract binary data
            blob_data = binary_content[data_start:data_end]
            
            # Handle compression
            if compression == 'qt':
                try:
                    blob_data = zlib.decompress(blob_data)
                except Exception as e:
                    print(f"Warning: Could not decompress blob {blob_id}: {e}")
                    # Try other decompression methods or keep raw
                    pass
            
            blobs[blob_id] = {
                'size': size,
                'compression': compression,
                'data': blob_data,
                'original_size': len(blob_data)
            }
            
            print(f"Found blob {blob_id}: size={size}, compression={compression}, actual_size={len(blob_data)}")
        
        return blobs
    
    def extract_radar_data(self, blobs):
        """Extract radar reflectivity data from blobs"""
        print(f"Available blobs: {list(blobs.keys())}")
        
        # Try different blob IDs for radar data
        data_blob = None
        for blob_id in [1, 0]:  # Try blob 1 first, then blob 0
            if blob_id in blobs:
                data_blob = blobs[blob_id]['data']
                print(f"Using blob {blob_id} for radar data (size: {len(data_blob)})")
                break
        
        if data_blob is None:
            raise ValueError("No suitable radar data blob found")
        
        # Parse the binary radar data
        radar_data = self._parse_radar_data(data_blob)
        
        return radar_data
    
    def _parse_radar_data(self, data):
        """Parse binary radar data - improved implementation"""
        print(f"Parsing radar data of size: {len(data)} bytes")
        
        # Try to interpret the data structure
        sweeps = []
        
        # Method 1: Try structured parsing
        try:
            sweeps = self._parse_structured_data(data)
            if sweeps:
                print(f"Successfully parsed {len(sweeps)} sweeps using structured method")
                return sweeps
        except Exception as e:
            print(f"Structured parsing failed: {e}")
        
        # Method 2: Fallback to simple interpretation
        try:
            sweep = self._create_sweep_from_raw_data(data)
            if sweep:
                sweeps = [sweep]
                print("Created single sweep from raw data")
                return sweeps
        except Exception as e:
            print(f"Raw data parsing failed: {e}")
        
        raise ValueError("Could not parse radar data with any method")
    
    def _parse_structured_data(self, data):
        """Try to parse data assuming it has a known structure"""
        sweeps = []
        
        # Common radar data structures
        possible_structures = [
            {'azimuths': 360, 'ranges': 500},
            {'azimuths': 360, 'ranges': 250},
            {'azimuths': 720, 'ranges': 250},
            {'azimuths': 180, 'ranges': 500},
        ]
        
        for struct_params in possible_structures:
            try:
                num_azimuths = struct_params['azimuths']
                num_ranges = struct_params['ranges']
                expected_size = num_azimuths * num_ranges
                
                if len(data) >= expected_size:
                    # Try different data types
                    for dtype in [np.uint8, np.int8, np.uint16, np.int16]:
                        try:
                            element_size = np.dtype(dtype).itemsize
                            if len(data) >= expected_size * element_size:
                                refl_data = np.frombuffer(data[:expected_size * element_size], dtype=dtype)
                                refl_data = refl_data[:expected_size].reshape((num_azimuths, num_ranges))
                                
                                # Convert to reasonable dBZ values
                                if dtype in [np.uint8, np.int8]:
                                    refl_dbz = (refl_data.astype(np.float32) - 64) / 2.0
                                else:
                                    refl_dbz = refl_data.astype(np.float32) / 100.0
                                
                                # Check if values are reasonable
                                valid_data = ~np.isnan(refl_dbz)
                                if np.any(valid_data):
                                    refl_range = np.ptp(refl_dbz[valid_data])
                                    if 1 < refl_range < 100:  # Reasonable reflectivity range
                                        sweep = {
                                            'elevation': 0.5,
                                            'reflectivity': refl_dbz,
                                            'num_azimuths': num_azimuths,
                                            'num_ranges': num_ranges,
                                            'data_size': expected_size * element_size
                                        }
                                        sweeps.append(sweep)
                                        print(f"Found valid sweep: {num_azimuths}x{num_ranges}, dtype={dtype}")
                                        return sweeps
                        except Exception:
                            continue
            except Exception:
                continue
        
        return sweeps
    
    def _create_sweep_from_raw_data(self, data):
        """Create sweep from raw data using best guess parameters"""
        # Use conservative estimates
        num_azimuths = 360
        
        # Calculate ranges based on available data
        for dtype in [np.uint8, np.int8]:
            element_size = np.dtype(dtype).itemsize
            total_elements = len(data) // element_size
            
            if total_elements >= num_azimuths:
                num_ranges = total_elements // num_azimuths
                
                if num_ranges > 10:  # Must have reasonable number of range gates
                    try:
                        size = num_azimuths * num_ranges * element_size
                        refl_data = np.frombuffer(data[:size], dtype=dtype)
                        refl_data = refl_data[:num_azimuths * num_ranges].reshape((num_azimuths, num_ranges))
                        
                        # Convert to dBZ
                        refl_dbz = (refl_data.astype(np.float32) - 64) / 2.0
                        refl_dbz[refl_data == 0] = np.nan  # No data values
                        
                        print(f"Created sweep: {num_azimuths}x{num_ranges}, dtype={dtype}")
                        return {
                            'elevation': 0.5,
                            'reflectivity': refl_dbz,
                            'num_azimuths': num_azimuths,
                            'num_ranges': num_ranges,
                            'data_size': size
                        }
                    except Exception as e:
                        print(f"Failed to create sweep with dtype {dtype}: {e}")
                        continue
        
        return None
    
    def calculate_coordinates(self, sweep_data):
        """Calculate lat/lon coordinates for each radar gate"""
        elevation = np.radians(sweep_data['elevation'])
        num_azimuths = sweep_data['num_azimuths']
        num_ranges = sweep_data['num_ranges']
        
        # Create azimuth and range arrays
        azimuths = np.linspace(0, 360, num_azimuths, endpoint=False)
        ranges = np.linspace(0, self.max_range_km, num_ranges)
        
        # Create meshgrid
        az_mesh, range_mesh = np.meshgrid(azimuths, ranges, indexing='ij')
        
        # Convert to radians
        az_rad = np.radians(az_mesh)
        
        # Calculate coordinates using radar beam geometry
        # Account for Earth's curvature and beam elevation
        
        # Ground range (accounting for elevation)
        ground_range = range_mesh * np.cos(elevation)
        
        # Height above radar
        height = range_mesh * np.sin(elevation) + (range_mesh**2) / (2 * self.earth_radius)
        
        # Calculate lat/lon using spherical geometry
        lat_rad = np.radians(self.radar_lat)
        lon_rad = np.radians(self.radar_lon)
        
        # Convert ground range to angular distance
        angular_dist = ground_range / self.earth_radius
        
        # Calculate new latitude
        lat_new = np.arcsin(
            np.sin(lat_rad) * np.cos(angular_dist) +
            np.cos(lat_rad) * np.sin(angular_dist) * np.cos(az_rad)
        )
        
        # Calculate new longitude
        lon_new = lon_rad + np.arctan2(
            np.sin(az_rad) * np.sin(angular_dist) * np.cos(lat_rad),
            np.cos(angular_dist) - np.sin(lat_rad) * np.sin(lat_new)
        )
        
        # Convert back to degrees
        lat_deg = np.degrees(lat_new)
        lon_deg = np.degrees(lon_new)
        
        return lat_deg, lon_deg, height, azimuths, ranges
    
    def process_vol_files(self, vol_files):
        """Process multiple vol files and combine into single dataset"""
        all_sweeps = []
        timestamps = []
        
        for vol_file in vol_files:
            print(f"\nProcessing {vol_file}...")
            try:
                volume_info, blobs = self.parse_vol_file(vol_file)
                radar_data = self.extract_radar_data(blobs)
                
                # Parse timestamp
                if volume_info['datetime']:
                    try:
                        # Handle different datetime formats
                        dt_str = volume_info['datetime'].replace('Z', '+00:00')
                        timestamp = datetime.fromisoformat(dt_str)
                        timestamps.append(timestamp)
                    except:
                        # Use current time as fallback
                        timestamps.append(datetime.now())
                
                all_sweeps.extend(radar_data)
                print(f"Successfully processed {vol_file}")
                
            except Exception as e:
                print(f"Error processing {vol_file}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if not all_sweeps:
            raise ValueError("No valid radar data found in any file")
        
        return self.create_netcdf_dataset(all_sweeps, timestamps)
    
    def create_netcdf_dataset(self, sweeps, timestamps):
        """Create xarray dataset with radar data"""
        datasets = []
        
        for i, sweep in enumerate(sweeps):
            if sweep is None:
                continue
                
            # Calculate coordinates
            lats, lons, heights, azimuths, ranges = self.calculate_coordinates(sweep)
            
            # Create dataset for this sweep
            ds = xr.Dataset({
                'reflectivity': (['azimuth', 'range'], sweep['reflectivity']),
                'latitude': (['azimuth', 'range'], lats),
                'longitude': (['azimuth', 'range'], lons),
                'height': (['azimuth', 'range'], heights),
            }, coords={
                'azimuth': azimuths,
                'range': ranges,
                'elevation': sweep['elevation'],
                'sweep_number': i
            })
            
            # Add timestamp if available
            if i < len(timestamps):
                ds = ds.assign_coords(time=timestamps[i])
            
            datasets.append(ds)
        
        if len(datasets) == 1:
            combined_ds = datasets[0]
        else:
            # Combine along sweep dimension
            combined_ds = xr.concat(datasets, dim='sweep_number')
        
        # Add metadata
        combined_ds.attrs.update({
            'title': 'Weather Radar Data - Kolkata',
            'institution': 'India Meteorological Department',
            'source': 'Doppler Weather Radar',
            'radar_latitude': self.radar_lat,
            'radar_longitude': self.radar_lon,
            'max_range_km': self.max_range_km,
            'created': datetime.now().isoformat()
        })
        
        # Add variable attributes
        combined_ds['reflectivity'].attrs.update({
            'long_name': 'Radar Reflectivity',
            'units': 'dBZ',
            'description': 'Radar reflectivity factor'
        })
        
        combined_ds['latitude'].attrs.update({
            'long_name': 'Latitude',
            'units': 'degrees_north',
            'standard_name': 'latitude'
        })
        
        combined_ds['longitude'].attrs.update({
            'long_name': 'Longitude',
            'units': 'degrees_east',
            'standard_name': 'longitude'
        })
        
        combined_ds['height'].attrs.update({
            'long_name': 'Height above mean sea level',
            'units': 'km',
            'description': 'Height of radar beam above MSL'
        })
        
        return combined_ds

def main():
    """Main function to process radar files"""
    
    # Initialize processor
    processor = RadarVolumeProcessor()
    
    # List of vol files to process
    vol_files = [
        '2025042623000000V.vol',
        '2025042623000000W.vol', 
        '2025042623000000dBZ.vol'
    ]
    
    # Check if files exist
    existing_files = [f for f in vol_files if os.path.exists(f)]
    if not existing_files:
        print("No vol files found in current directory")
        return
    
    print(f"Found {len(existing_files)} vol files to process")
    
    try:
        # Process files
        dataset = processor.process_vol_files(existing_files)
        
        # Save to NetCDF
        output_file = '2025042623000000.nc'
        dataset.to_netcdf(output_file)
        print(f"\nRadar data saved to {output_file}")
        
        # Print summary
        print("\nDataset Summary:")
        print(f"Dimensions: {dict(dataset.dims)}")
        print(f"Variables: {list(dataset.data_vars.keys())}")
        print(f"Coordinate ranges:")
        print(f"  Latitude: {dataset.latitude.min().values:.3f} to {dataset.latitude.max().values:.3f}")
        print(f"  Longitude: {dataset.longitude.min().values:.3f} to {dataset.longitude.max().values:.3f}")
        
        # Check for valid reflectivity data
        valid_refl = dataset.reflectivity.values[~np.isnan(dataset.reflectivity.values)]
        if len(valid_refl) > 0:
            print(f"  Reflectivity: {valid_refl.min():.1f} to {valid_refl.max():.1f} dBZ")
        else:
            print("  Reflectivity: No valid data found")
        
    except Exception as e:
        print(f"Error processing radar data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()