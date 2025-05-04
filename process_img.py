from netCDF4 import Dataset
import numpy as np

# Open the NetCDF file
file_name = "2024062612321800dBZ.vol.nc"
try:
    dataset = Dataset(file_name, mode="r")
    print("File opened successfully")
    # print(dataset)
    # print(dataset.variables)
    # print(dataset.variables.values())

    # List all variables in the file
    print("Variables:", dataset.variables.keys())
    # print(dataset.variables.items())
    print("Dims of ref",dataset.variables['reflectivity'].get_dims)
    print("Shape of ref",dataset.variables['reflectivity'].shape)

    if "latitude" in dataset.variables and "longitude" in dataset.variables and "reflectivity" in dataset.variables:
        latitudes = dataset.variables["latitude"][:]
        longitudes = dataset.variables["longitude"][:]
        reflectivity = dataset.variables["reflectivity"][:]

        reflectivity = np.ma.masked_array(reflectivity, mask=np.isnan(reflectivity))
        
        target_lat = 25.6125
        target_lon = 85.1283

        lat_idx = np.abs(latitudes - target_lat).argmin()
        lon_idx = np.abs(longitudes - target_lon).argmin()

        # Extract the reflectivity value at the closest latitude and longitude
        reflectivity_value = reflectivity[lat_idx, lon_idx]
        # print(f"Size of reflectivity value is {len(reflectivity_value)}")
        # print(f"Reflectivity at ({target_lat}, {target_lon}): {reflectivity_value}")

    # Access a specific variable (example: 'reflectivity')
    if "reflectivity" in dataset.variables:
        reflectivity = dataset.variables["erange"][:]
        print("Reflectivity data:", reflectivity)

    # Close the file
    dataset.close()
except Exception as e:
    print(f"Error reading NetCDF file: {e}")